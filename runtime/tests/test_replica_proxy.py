"""The replicas-mode proxy keeps a conversation's next turn on its replica,
balances everything else and passes streams through."""

import asyncio
import json

import pytest

from runtime import replica_proxy as proxy


def _chat(*contents, system="You are terse."):
    messages = [{"role": "system", "content": system}]
    for index, content in enumerate(contents):
        messages.append(
            {"role": "user" if index % 2 == 0 else "assistant", "content": content}
        )
    return json.dumps({"model": "m", "messages": messages}).encode()


def _chat_affinity(*contents, **kwargs):
    return proxy.request_affinity(
        "/v1/chat/completions", _chat(*contents, **kwargs), {}
    )


def _completion_affinity(prompt):
    body = json.dumps({"model": "m", "prompt": prompt}).encode()
    return proxy.request_affinity("/v1/completions", body, {})


def test_a_next_turn_extends_its_previous_turn_but_a_repeated_first_turn_does_not():
    first = _chat_affinity("hello")
    repeated = _chat_affinity("hello")
    next_turn = _chat_affinity("hello", "hi", "how are you")
    other_system = _chat_affinity("hello", "hi", "more", system="Be verbose.")

    assert first.key in next_turn.prefixes
    assert first.key not in repeated.prefixes
    assert first.key not in other_system.prefixes


def test_router_spreads_first_turns_and_keeps_next_turns_on_their_replica():
    router = proxy.StickyRouter(2)

    assert router.choose(_chat_affinity("a")) == 0
    assert router.choose(_chat_affinity("b")) == 1
    router.in_flight[0] = 3
    assert router.choose(_chat_affinity("c")) == 1
    assert router.choose(_chat_affinity("a", "x", "a2")) == 0
    assert router.choose(_chat_affinity("b", "y", "b2", "z", "b3")) == 1


def test_identical_prompts_side_by_side_use_every_replica():
    router = proxy.StickyRouter(2)
    replicas = {router.choose(_chat_affinity("same")) for _ in range(4)}

    assert replicas == {0, 1}


def test_completion_prompts_follow_only_a_shorter_recorded_prompt():
    router = proxy.StickyRouter(2)
    base = "x" * 5000

    first = router.choose(_completion_affinity(base))
    again = router.choose(_completion_affinity(base))
    extended = router.choose(_completion_affinity(base + " and more"))
    tokens = router.choose(_completion_affinity(list(range(1500))))
    token_turn = router.choose(_completion_affinity(list(range(2100))))

    assert first != again
    assert extended == again
    assert token_turn == tokens


def test_affinity_header_and_previous_response_pin_their_replica():
    router = proxy.StickyRouter(2)
    pinned = proxy.request_affinity(
        "/v1/chat/completions", _chat("a"), {"X-LIL-Affinity": "session-7"}
    )
    replica = router.choose(pinned)
    other = proxy.request_affinity(
        "/v1/chat/completions", _chat("b"), {"X-LIL-Affinity": "session-7"}
    )
    router.remember_response("resp_1", 1)
    follow_up = proxy.request_affinity(
        "/v1/responses",
        json.dumps({"input": "next", "previous_response_id": "resp_1"}).encode(),
        {},
    )

    assert router.choose(other) == replica
    assert router.choose(follow_up) == 1
    assert proxy.request_affinity("/v1/chat/completions", b"not json", {}).key is None


def test_metrics_merge_labels_each_replica_under_one_header():
    replica = (
        "# HELP vllm:num_requests_running Running requests.\n"
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{model_name="m"} {value}\n'
        "# HELP vllm:e2e Latency.\n"
        "# TYPE vllm:e2e histogram\n"
        'vllm:e2e_bucket{le="1.0"} 2\n'
        "vllm:e2e_count 2\n"
    )

    merged = proxy.merge_metrics(
        [replica.replace("{value}", "1"), replica.replace("{value}", "4")]
    ).splitlines()

    assert merged.count("# TYPE vllm:num_requests_running gauge") == 1
    assert 'vllm:num_requests_running{replica="0",model_name="m"} 1' in merged
    assert 'vllm:num_requests_running{replica="1",model_name="m"} 4' in merged
    family = merged.index("# TYPE vllm:e2e histogram")
    assert merged[family + 1 : family + 5] == [
        'vllm:e2e_bucket{replica="0",le="1.0"} 2',
        'vllm:e2e_count{replica="0"} 2',
        'vllm:e2e_bucket{replica="1",le="1.0"} 2',
        'vllm:e2e_count{replica="1"} 2',
    ]


def test_proxy_streams_keeps_conversations_and_reports_all_replicas():
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    hits = []
    healthy = [True, True]

    def upstream(index):
        async def chat(request):
            body = await request.json()
            hits.append((index, body["messages"][-1]["content"]))
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for part in ("a", "b", "[DONE]"):
                await response.write(f"data: {part}\n\n".encode())
            await response.write_eof()
            return response

        async def health(_request):
            return web.Response(status=200 if healthy[index] else 503)

        async def metrics(_request):
            return web.Response(
                text="# TYPE vllm:num_requests_running gauge\n"
                f"vllm:num_requests_running {index}\n"
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        app.router.add_get("/health", health)
        app.router.add_get("/metrics", metrics)
        return app

    async def scenario():
        servers = [TestServer(upstream(index)) for index in range(2)]
        for server in servers:
            await server.start_server()
        app = proxy.build_app([f"127.0.0.1:{server.port}" for server in servers])
        try:
            async with TestClient(TestServer(app)) as client:

                async def chat(*contents):
                    response = await client.post(
                        "/v1/chat/completions", data=_chat(*contents)
                    )
                    assert response.status == 200
                    assert response.headers["Content-Type"] == "text/event-stream"
                    return await response.text()

                assert await chat("one") == "data: a\n\ndata: b\n\ndata: [DONE]\n\n"
                await chat("two")
                await chat("one", "ok", "one again")
                await chat("two", "ok", "two again")
                assert (await client.get("/health")).status == 200
                metrics = await (await client.get("/metrics")).text()
                healthy[1] = False
                assert (await client.get("/health")).status == 503
                return metrics
        finally:
            for server in servers:
                await server.close()

    metrics = asyncio.run(scenario())

    assert hits == [(0, "one"), (1, "two"), (0, "one again"), (1, "two again")]
    assert 'vllm:num_requests_running{replica="1"} 1' in metrics.splitlines()
