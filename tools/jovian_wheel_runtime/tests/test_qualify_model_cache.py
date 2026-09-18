"""Reject cache-qualification claims unsupported by isolated runtime evidence."""

import copy
import importlib.util
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "qualify_model_cache",
    Path(__file__).resolve().parents[1] / "qualify_model_cache.py",
)
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


def snapshot(*, gpu=0, external=0, completed=0, loaded=0, stored=0):
    return {
        "vllm": {
            "vllm:prefix_cache_hits_total": gpu,
            "vllm:external_prefix_cache_hits_total": external,
            "vllm:request_success_total": completed,
            "vllm:num_requests_running": 0,
            "vllm:num_requests_waiting": 0,
        },
        "cache": {
            "lmcache_mp_l2_prefetch_load_completed_total": loaded,
            "lmcache_mp_l2_store_completed_objects_total": stored,
        },
    }


def response(word="COBALT"):
    return {"choices": [{"finish_reason": "stop", "message": {"content": word}}]}


def identity(started="2026-09-18T02:00:00Z", image="sha256:" + "a" * 64):
    return {
        "id": "test-container",
        "image": image,
        "started_at": started,
        "running": True,
    }


def test_metric_parser_and_counter_aliases():
    values = qualify.parse_metrics(
        "# HELP example Ignored\n"
        'vllm:request_success_total{engine="0",reason="stop"} 2\n'
        'vllm:request_success_total{engine="0",reason="length"} 1 123\n'
        "latency +Inf\n"
        "ignored NaN\n"
        'valid 2.5e3 # {trace_id="abc"} 1.0\n'
    )
    assert values == {"vllm:request_success_total": 3, "valid": 2500}
    assert qualify.metric(values, "completed") == 3
    values["vllm_request_success_total"] = 3
    with pytest.raises(ValueError, match="Ambiguous"):
        qualify.metric(values, "completed")


@pytest.mark.parametrize("after", [{}, {"vllm:prefix_cache_hits_total": 2}])
def test_counter_reset_cannot_look_like_a_cache_miss(after):
    with pytest.raises(ValueError, match="reset/disappearance"):
        qualify.delta({"vllm:prefix_cache_hits_total": 4}, after, "gpu_hits")


def test_missing_counters_are_not_evidence_of_zero_hits():
    with pytest.raises(ValueError, match="Missing evidence"):
        qualify.delta({}, {}, "external_hits")
    assert (
        qualify.delta(
            {}, {"vllm:external_prefix_cache_hits_total": 4096}, "external_hits"
        )
        == 4096
    )


@pytest.mark.parametrize(
    "stage,values",
    [
        ("cold", {}),
        ("apc", {"gpu": 8192}),
        ("prefix", {"gpu": 8192}),
        ("prefix", {"gpu": 16128, "external": 256}),
        ("l1", {"external": 8192}),
        ("l2", {"external": 8192, "loaded": 4}),
    ],
)
def test_valid_isolated_cache_stages(stage, values):
    evidence = qualify.check_evidence(
        stage, snapshot(), snapshot(completed=1, **values), 4096
    )
    assert evidence["gpu_hit_tokens"] == values.get("gpu", 0)
    assert evidence["external_hit_tokens"] == values.get("external", 0)


@pytest.mark.parametrize(
    "stage,values",
    [
        ("cold", {"gpu": 128}),
        ("cold", {"external": 1}),
        ("apc", {"gpu": 8192, "external": 8192}),
        ("apc", {"gpu": 128}),
        ("prefix", {"gpu": 128, "external": 16384}),
        ("l1", {"gpu": 8192, "external": 8192}),
        ("l1", {"external": 128}),
        ("l1", {"external": 8192, "loaded": 1}),
        ("l2", {"external": 8192}),
        ("l2", {"loaded": 1}),
    ],
)
def test_false_restore_claims_fail(stage, values):
    with pytest.raises(ValueError):
        qualify.check_evidence(stage, snapshot(), snapshot(completed=1, **values), 4096)


@pytest.mark.parametrize("completed", [0, 2])
def test_unrelated_traffic_invalidates_request_metrics(completed):
    with pytest.raises(ValueError, match="one completed"):
        qualify.check_evidence(
            "l1", snapshot(), snapshot(completed=completed, external=8192), 4096
        )


@pytest.mark.parametrize(
    "values", [{}, {"vllm:num_requests_running": 1}, {"vllm:num_requests_running": 0}]
)
def test_missing_or_busy_endpoint_metrics_prevent_reset(values):
    with pytest.raises(ValueError, match="demonstrably idle"):
        qualify.require_idle(values)


@pytest.mark.parametrize("status", [404, 500])
def test_reset_endpoint_errors_do_not_become_cache_misses(status, monkeypatch):
    client = qualify.Client("http://127.0.0.1:8000", "http://127.0.0.1:8001/metrics", 1)
    monkeypatch.setattr(client, "snapshot", snapshot)

    def fail(url, *, post):
        assert post
        assert "reset_external=false" in url
        assert "reset_running_requests=false" in url
        raise urllib.error.HTTPError(url, status, "test", {}, None)

    monkeypatch.setattr(client, "call", fail)
    if status == 404:
        with pytest.raises(ValueError, match="VLLM_SERVER_DEV_MODE=1"):
            client.reset_gpu()
    else:
        with pytest.raises(urllib.error.HTTPError):
            client.reset_gpu()


def test_same_process_or_changed_image_is_not_a_restart_control():
    previous = {role: identity() for role in ("serving", "cache")}
    with pytest.raises(ValueError, match="restarting serving"):
        qualify.require_restart(previous, previous)
    fresh = {role: identity(started="2026-09-18T03:00:00Z") for role in previous}
    qualify.require_restart(previous, fresh)
    fresh["cache"] = identity()
    with pytest.raises(ValueError, match="restarting cache"):
        qualify.require_restart(previous, fresh)
    fresh["cache"] = identity(started="later", image="sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="changed the cache image"):
        qualify.require_restart(previous, fresh)


def test_factual_checks_do_not_compare_stochastic_output_sequences():
    qualify.require_answer(response(" cobalt. "), "COBALT")
    with pytest.raises(ValueError, match="Expected factual answer"):
        qualify.require_answer(response("AMBER"), "COBALT")
    with pytest.raises(ValueError, match="truncated"):
        value = response()
        value["choices"][0]["finish_reason"] = "length"
        qualify.require_answer(value, "COBALT")
    payload = qualify.request_fixture("model", "test-salt", {})
    assert payload["temperature"] == 1
    assert payload["cache_salt"] == "test-salt"
    assert payload["chat_template_kwargs"] == {}


class FakeClient:
    base = "http://localhost:1"
    timeout = 1

    def __init__(self):
        self.values = snapshot()
        self.calls = []
        self.resets = 0

    def snapshot(self):
        return copy.deepcopy(self.values)

    def reset_gpu(self):
        self.resets += 1
        return {"response": {"success": True}, "attempts": 1}

    def call(self, url, payload, *, post=False):
        self.calls.append((url, payload, post))
        idx = len(self.calls)
        self.values["vllm"]["vllm:request_success_total"] += 1
        if idx == 2:
            self.values["vllm"]["vllm:prefix_cache_hits_total"] += 8192
        if idx >= 3:
            self.values["vllm"]["vllm:external_prefix_cache_hits_total"] += 8192
        self.values["cache"]["lmcache_mp_l2_store_completed_objects_total"] += 4
        return json.dumps(
            response(
                "AMBER"
                if payload["messages"][-1]["content"].endswith("BY")
                else "COBALT"
            )
        )

    def completed_snapshot(self, before):
        return self.snapshot()


def args(directory, command="prime"):
    return SimpleNamespace(
        command=command,
        output_dir=directory,
        model="model",
        persistent=True,
        minimum_reused_tokens=4096,
        container="dedicated",
        cache_container=None,
        thinking_kwargs={},
        fixture="text",
    )


def test_prime_records_cold_native_prefix_external_and_changed_suffix(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(qualify, "container_identity", lambda _: identity())
    client = FakeClient()
    report = qualify.run(args(tmp_path / "receipts"), client)
    assert report["status"] == "qualified"
    assert client.resets == 3
    assert [item["cache_stage"] for item in report["stages"]] == [
        "cold",
        "prefix",
        "l1",
        "l1",
    ]
    assert report["request_sha256"] == qualify.fingerprint(report["request"])
    assert len({item[1]["cache_salt"] for item in client.calls}) == 1
    assert all(item[1]["temperature"] == 1 for item in client.calls)
    assert all(item[0].endswith("/v1/chat/completions") for item in client.calls)
    assert "persistent_publication" in report
    assert report["stages"][1]["reuse_kind"] == "native_gpu_only"
    assert (tmp_path / "receipts/prime.json").exists()
    with pytest.raises(FileExistsError):
        qualify.run(args(tmp_path / "receipts"), client)


def test_native_prefix_with_external_tail_is_not_reported_as_isolated(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(qualify, "container_identity", lambda _: identity())

    class MixedPrefixClient(FakeClient):
        def call(self, url, payload, *, post=False):
            result = super().call(url, payload, post=post)
            if len(self.calls) == 2:
                self.values["vllm"]["vllm:external_prefix_cache_hits_total"] += 256
            return result

    report = qualify.run(args(tmp_path / "mixed"), MixedPrefixClient())
    assert report["status"] == "qualified"
    stage = report["stages"][1]
    assert stage["reuse_kind"] == "native_gpu_with_external_tail"
    assert stage["evidence"]["gpu_hit_tokens"] == 8192
    assert stage["evidence"]["external_hit_tokens"] == 256


@pytest.mark.parametrize("reverse", [False, True])
def test_vision_prefix_preserves_the_image_before_long_shared_text(reverse):
    import base64
    import struct
    import zlib

    payload = qualify.request_fixture("model", "vision-salt", {}, "vision")
    content = payload["messages"][1]["content"]
    assert [item["type"] for item in content] == ["image_url", "text"]
    assert len(content[1]["text"]) > 100000
    image = qualify.image_fixture(reverse=reverse)
    if not reverse:
        assert image == content[0]["image_url"]["url"]
    png = base64.b64decode(image.split(",", 1)[1])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack("!2I", png[16:24]) == (512, 256)
    length = struct.unpack("!I", png[33:37])[0]
    pixels = zlib.decompress(png[41 : 41 + length])
    left, right = ((0, 0, 255), (255, 0, 0)) if reverse else ((255, 0, 0), (0, 0, 255))
    assert pixels[1:4] == bytes(left)
    assert pixels[1 + 256 * 3 : 4 + 256 * 3] == bytes(right)


@pytest.mark.parametrize("cross_image_fault", [None, "false_hit", "wrong_answer"])
def test_vision_prime_and_persistent_restore_validate_image_dependent_answers(
    tmp_path, monkeypatch, cross_image_fault
):
    monkeypatch.setattr(qualify, "container_identity", lambda _: identity())
    options = args(tmp_path / "vision")
    options.fixture = "vision"
    client = FakeClient()
    original_call = client.call

    def vision_call(url, payload, *, post):
        # Reuse the metric transitions, but derive the expected answer from
        # the question rather than the text-only fixture's lookup table.
        forwarded = copy.deepcopy(payload)
        right = "right half" in payload["messages"][-1]["content"]
        forwarded["messages"][-1]["content"] = (
            "Look up item BY" if right else "Look up item AX"
        )
        original_call(url, forwarded, post=post)
        assert payload["messages"][1]["content"][0]["type"] == "image_url"
        reversed_image = payload["messages"][1]["content"][0]["image_url"][
            "url"
        ] == qualify.image_fixture(reverse=True)
        if reversed_image and cross_image_fault != "false_hit":
            client.values["vllm"]["vllm:external_prefix_cache_hits_total"] -= 8192
        if reversed_image and cross_image_fault != "wrong_answer":
            right = not right
        return json.dumps(response("BLUE" if right else "RED"))

    client.call = vision_call
    if cross_image_fault:
        message = (
            "Cold request reused"
            if cross_image_fault == "false_hit"
            else "Expected factual answer"
        )
        with pytest.raises(ValueError, match=message):
            qualify.run(options, client)
        saved = json.loads((options.output_dir / "prime.json").read_text())
        assert saved["status"] == "failed"
        assert saved["stages"][-1]["name"] == "external_changed_image"
        return
    prime = qualify.run(options, client)
    assert [stage["expected_answer"] for stage in prime["stages"]] == [
        "RED",
        "RED",
        "RED",
        "BLUE",
        "BLUE",
        "RED",
    ]
    assert client.resets == 5
    assert client.calls[2][1]["cache_salt"] == client.calls[4][1]["cache_salt"]
    assert client.calls[2][1]["messages"][-1] == client.calls[4][1]["messages"][-1]
    monkeypatch.setattr(
        qualify, "container_identity", lambda _: identity(started="after-restart")
    )
    restored = FakeClient()

    def restore_call(url, payload, *, post):
        assert "right half" in payload["messages"][-1]["content"]
        assert payload["messages"][1]["content"][0]["type"] == "image_url"
        restored.values = snapshot(completed=1, external=8192, loaded=4)
        return json.dumps(response("BLUE"))

    restored.call = restore_call
    options.command = "restore"
    report = qualify.run(options, restored)
    assert report["status"] == "qualified"
    assert report["stages"][0]["expected_answer"] == "BLUE"


def test_failure_keeps_response_and_metric_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(qualify, "container_identity", lambda _: identity())
    client = FakeClient()
    client.call = lambda *a, **k: json.dumps(response("WRONG"))
    with pytest.raises(ValueError, match="Expected factual answer"):
        qualify.run(args(tmp_path / "receipts"), client)
    saved = json.loads((tmp_path / "receipts/prime.json").read_text())
    assert saved["status"] == "failed"
    assert saved["stages"][0]["response"] == response("WRONG")
    assert "before" in saved["stages"][0] and "after" in saved["stages"][0]


def test_restore_requires_unchanged_prompt_and_both_processes_restarted(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(qualify, "container_identity", lambda _: identity())
    qualify.run(args(tmp_path / "receipts"), FakeClient())
    with pytest.raises(ValueError, match="restarting serving"):
        qualify.run(args(tmp_path / "receipts", "restore"), FakeClient())
    path = tmp_path / "receipts/prime.json"
    state = json.loads(path.read_text())
    state["request"]["cache_salt"] = "modified"
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="modified"):
        qualify.run(args(tmp_path / "receipts", "restore"), FakeClient())


def test_persistent_restore_requires_observed_external_tokens_and_l2_reads(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(qualify, "container_identity", lambda _: identity())
    qualify.run(args(tmp_path / "receipts"), FakeClient())
    monkeypatch.setattr(
        qualify,
        "container_identity",
        lambda _: identity(started="2026-09-18T03:00:00Z"),
    )
    client = FakeClient()

    def restore(url, payload, *, post):
        assert payload["messages"][-1]["content"] == "Look up item BY"
        client.values = snapshot(completed=1, external=8192, loaded=8)
        return json.dumps(response("AMBER"))

    client.call = restore
    report = qualify.run(args(tmp_path / "receipts", "restore"), client)
    assert report["status"] == "qualified"
    assert len(report["stages"]) == 1
    assert report["stages"][0]["evidence"] == {
        "gpu_hit_tokens": 0,
        "external_hit_tokens": 8192,
        "l2_loaded_objects": 8,
    }
    assert (tmp_path / "receipts/restore.json").exists()
    with pytest.raises(FileExistsError):
        qualify.run(args(tmp_path / "receipts", "restore"), client)


def test_actual_reset_keeps_external_objects_and_does_not_abort_requests(monkeypatch):
    client = qualify.Client("http://localhost:1", "http://localhost:2/metrics", 1)
    monkeypatch.setattr(client, "snapshot", lambda: snapshot())
    calls = []

    def call(url, **kwargs):
        calls.append((url, kwargs))
        return '{"success":true}'

    monkeypatch.setattr(client, "call", call)
    result = client.reset_gpu()
    assert result["response"]["success"] is True
    assert calls == [
        (
            "http://localhost:1/reset_prefix_cache?reset_external=false&reset_running_requests=false",
            {"post": True},
        )
    ]
