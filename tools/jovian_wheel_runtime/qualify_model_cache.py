"""Check model output and distinguish GPU, CPU, and persistent prefix restores.

Run only against a dedicated serving endpoint. The ``prime`` command sends cold,
GPU-prefix, and external-restore requests. ``restore`` reuses the saved request
after the operator restarts both serving and its CPU cache process. This tool
never starts/stops containers or clears external cache objects. Sampling uses
temperature 1; output checks verify facts, not identical stochastic continuations.
The dedicated vLLM server must enable ``VLLM_SERVER_DEV_MODE=1`` for its GPU
prefix-reset API. LMCache's HTTP application exposes metrics at
``http://CACHE_HTTP_HOST:CACHE_HTTP_PORT/metrics``; it disables the separate
Prometheus listener.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import struct
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zlib
from pathlib import Path

SCHEMA = "lil-model-cache-qualification/v1"
COUNTERS = {
    "gpu_hits": ("vllm:prefix_cache_hits_total", "vllm_prefix_cache_hits_total"),
    "external_hits": (
        "vllm:external_prefix_cache_hits_total",
        "vllm_external_prefix_cache_hits_total",
    ),
    "completed": ("vllm:request_success_total", "vllm_request_success_total"),
    "running": ("vllm:num_requests_running", "vllm_num_requests_running"),
    "waiting": ("vllm:num_requests_waiting", "vllm_num_requests_waiting"),
    "l2_stored": (
        "lmcache_mp_l2_store_completed_objects_total",
        "lmcache_mp_l2_store_completed_objects_chunks_total",
    ),
    "l2_loaded": (
        "lmcache_mp_l2_prefetch_load_completed_total",
        "lmcache_mp_l2_prefetch_load_completed_chunks_total",
    ),
}
SAMPLE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([^\s]+)(?:\s+\d+)?(?:\s+#.*)?$"
)


def parse_metrics(body: str) -> dict[str, float]:
    """Aggregate one dedicated endpoint's samples without mixing metric families."""
    result: dict[str, float] = {}
    for line in body.splitlines():
        match = SAMPLE.fullmatch(line.strip())
        if not match:
            continue
        name, raw = match.groups()
        value = float(raw)
        if not math.isfinite(value):
            continue
        result[name] = result.get(name, 0.0) + value
    return result


def metric(values: dict[str, float], key: str) -> float | None:
    matches = [values[name] for name in COUNTERS[key] if name in values]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous metric aliases for {key}")
    return matches[0] if matches else None


def delta(before: dict, after: dict, key: str, *, required: bool = True) -> float:
    left, right = metric(before, key), metric(after, key)
    if left is None and right is None:
        if required:
            raise ValueError(f"Missing evidence: {key} counters were not exported")
        return 0.0
    if right is None or (left is not None and right < left):
        raise ValueError(f"Counter reset/disappearance during request: {key}")
    return right - (left or 0.0)


def require_idle(values: dict) -> None:
    for key in ("running", "waiting"):
        value = metric(values, key)
        if value is None or value != 0:
            raise ValueError(
                f"Dedicated endpoint is not demonstrably idle: {key}={value}"
            )


def require_answer(response: dict, expected: str) -> None:
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("The factual response is missing or was truncated")
    content = choices[0].get("message", {}).get("content") or ""
    if not re.fullmatch(rf"\s*{re.escape(expected)}[.!]?\s*", content, re.IGNORECASE):
        raise ValueError(f"Expected factual answer {expected!r}, received {content!r}")


def check_evidence(stage: str, before: dict, after: dict, minimum: int) -> dict:
    completed = delta(before["vllm"], after["vllm"], "completed")
    if completed != 1:
        raise ValueError(f"Expected one completed request, observed {completed}")
    gpu = delta(before["vllm"], after["vllm"], "gpu_hits")
    external = delta(before["vllm"], after["vllm"], "external_hits")
    loaded = delta(before["cache"], after["cache"], "l2_loaded", required=stage == "l2")
    if stage == "cold" and (gpu != 0 or external != 0):
        raise ValueError(
            f"Cold request reused cached tokens: GPU={gpu}, external={external}"
        )
    if stage == "apc" and (gpu < minimum or external != 0):
        raise ValueError(
            f"Native GPU reuse was not isolated: GPU={gpu}, external={external}"
        )
    if stage in {"l1", "l2"} and (gpu != 0 or external < minimum):
        raise ValueError(
            f"External restore was not proven: GPU={gpu}, external={external}"
        )
    if stage == "l1" and loaded != 0:
        raise ValueError("The CPU L1 test fetched objects from L2")
    if stage == "l2" and loaded <= 0:
        raise ValueError("Persistent restore did not load any L2 objects")
    if stage not in {"cold", "apc", "l1", "l2"}:
        raise ValueError(f"Unknown cache stage {stage!r}")
    return {
        "gpu_hit_tokens": gpu,
        "external_hit_tokens": external,
        "l2_loaded_objects": loaded,
    }


def container_identity(name: str) -> dict:
    # Do not collect container ENV, which may contain credentials.
    template = "[{{json .Id}},{{json .Image}},{{json .State.StartedAt}},{{json .State.Running}}]"
    fields = json.loads(
        subprocess.check_output(
            ["docker", "inspect", "--format", template, name], text=True, timeout=30
        )
    )
    if fields[3] is not True:
        raise ValueError(f"Container {name!r} is not running")
    return dict(zip(("id", "image", "started_at", "running"), fields))


def require_restart(before: dict, after: dict) -> None:
    for role in ("serving", "cache"):
        previous, present = before[role], after[role]
        if previous["image"] != present["image"]:
            raise ValueError(f"Persistent restore changed the {role} image")
        if previous["started_at"] == present["started_at"]:
            raise ValueError(f"Persistent restore requires restarting {role}")


def image_fixture() -> str:
    """Encode an RGB image with a red left half and a blue right half."""
    width, height = 512, 256

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data))
        )

    row = b"\0" + bytes((255, 0, 0)) * (width // 2) + bytes((0, 0, 255)) * (width // 2)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def request_fixture(model: str, nonce: str, thinking: dict, fixture="text") -> dict:
    system = (
        f"Cache qualification document {nonce}.\n"
        "Read this catalog. Return only the mapped word for the requested item.\n"
        + "The reference catalog contains ordinary fictional item identifiers. "
        * 1800
        + "\nLookup table: item AX maps to COBALT; item BY maps to AMBER.\n"
        "Return only the mapped word, without an explanation."
    )
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Look up item AX"},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 20260918,
        "max_tokens": 2048,
        "cache_salt": nonce,
        "chat_template_kwargs": thinking,
    }
    if fixture == "vision":
        # The image precedes the long shared text: a hit beyond the minimum
        # prefix length includes image-dependent attention/recurrent state.
        request["messages"] = [
            {
                "role": "system",
                "content": "Answer questions about the image. Return only the color name.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_fixture()}},
                    {
                        "type": "text",
                        "text": f"Cache qualification document {nonce}.\n"
                        + "This reference document contains ordinary fictional item identifiers. "
                        * 1800
                        + "\nKeep the image in mind for the following question.",
                    },
                ],
            },
            {"role": "assistant", "content": "Ready."},
            {
                "role": "user",
                "content": "What color is the left half of the image? Return only the color name.",
            },
        ]
    elif fixture != "text":
        raise ValueError(f"Unknown request fixture {fixture!r}")
    return request


def fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Client:
    def __init__(self, base: str, cache_metrics: str, timeout: float):
        self.base, self.cache_metrics, self.timeout = (
            base.rstrip("/"),
            cache_metrics,
            timeout,
        )
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, url: str, payload: dict | None = None, *, post=False):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method="POST" if post else "GET",
            headers={"Content-Type": "application/json"},
        )
        with self.opener.open(req, timeout=self.timeout if post else 30) as response:
            return response.read().decode()

    def snapshot(self) -> dict:
        vllm = self.call(self.base + "/metrics")
        cache = self.call(self.cache_metrics)
        return {
            "vllm": parse_metrics(vllm),
            "cache": parse_metrics(cache),
            "raw_vllm": vllm,
            "raw_cache": cache,
            "epoch": time.time(),
        }

    def reset_gpu(self) -> dict:
        deadline = time.monotonic() + self.timeout
        attempts = 0
        while time.monotonic() < deadline:
            require_idle(self.snapshot()["vllm"])
            try:
                response = json.loads(
                    self.call(
                        self.base
                        + "/reset_prefix_cache?reset_external=false&reset_running_requests=false",
                        post=True,
                    )
                )
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    raise ValueError(
                        "Dedicated cache qualification requires "
                        "VLLM_SERVER_DEV_MODE=1 for the GPU prefix-reset API; "
                        "do not enable development endpoints on a public server."
                    ) from error
                raise
            attempts += 1
            if response.get("success") is True:
                return {"response": response, "attempts": attempts}
            time.sleep(1)
        raise TimeoutError("GPU prefix state is still held by requests/transfers")

    def completed_snapshot(self, before: dict) -> dict:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            after = self.snapshot()
            if delta(before["vllm"], after["vllm"], "completed", required=False) >= 1:
                require_idle(after["vllm"])
                return after
            time.sleep(1)
        raise TimeoutError(
            "The response completed but its serving metrics did not update"
        )


def run(args, client: Client) -> dict:
    out = args.output_dir
    if args.command == "prime":
        out.mkdir(parents=True, exist_ok=False)
        state = {
            "schema": SCHEMA,
            "model": args.model,
            "persistent": args.persistent,
            "fixture": args.fixture,
            "minimum_reused_tokens": args.minimum_reused_tokens,
            "request": request_fixture(
                args.model, uuid.uuid4().hex, args.thinking_kwargs, args.fixture
            ),
            "serving_container": args.container,
            "cache_container": args.cache_container or args.container,
        }
    else:
        state = json.loads((out / "prime.json").read_text())
        if state.get("schema") != SCHEMA or state.get("status") != "qualified":
            raise ValueError(
                "A successful prime receipt is required for persistent restore"
            )
        if not state["persistent"]:
            raise ValueError("The prime run did not qualify persistent publication")
        if fingerprint(state["request"]) != state["request_sha256"]:
            raise ValueError("The saved request was modified after the prime run")
    state["request_sha256"] = fingerprint(state["request"])
    identity = {
        role: container_identity(state[f"{role}_container"])
        for role in ("serving", "cache")
    }
    if args.command == "restore":
        require_restart(state["containers"], identity)
    report = dict(state, containers=identity, status="in_progress", stages=[])
    path = out / f"{args.command}.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        initial = client.snapshot()
        stages = [
            ("cold", "cold", "AX", "COBALT"),
            ("apc", "apc", "AX", "COBALT"),
            ("external_identical", "l1", "AX", "COBALT"),
            ("external_changed_suffix", "l1", "BY", "AMBER"),
        ]
        if args.command == "restore":
            stages = [("persistent_changed_suffix", "l2", "BY", "AMBER")]
        for label, stage, item, answer in stages:
            vision = state.get("fixture", "text") == "vision"
            if vision:
                answer = "RED" if item == "AX" else "BLUE"
            entry = {
                "name": label,
                "cache_stage": stage,
                "expected_answer": answer,
                "status": "in_progress",
            }
            report["stages"].append(entry)
            save()
            if stage != "apc":
                entry["gpu_reset"] = client.reset_gpu()
            before = client.snapshot()
            require_idle(before["vllm"])
            entry["before"] = before
            save()
            payload = json.loads(json.dumps(state["request"]))
            payload["messages"][-1]["content"] = (
                f"What color is the {'left' if item == 'AX' else 'right'} half of the image? Return only the color name."
                if vision
                else f"Look up item {item}"
            )
            started = time.monotonic()
            response = json.loads(
                client.call(client.base + "/v1/chat/completions", payload, post=True)
            )
            entry.update(
                response=response, seconds=time.monotonic() - started, before=before
            )
            save()
            after = client.completed_snapshot(before)
            entry["after"] = after
            save()
            require_answer(response, answer)
            entry["evidence"] = check_evidence(
                stage, before, after, state["minimum_reused_tokens"]
            )
            entry["status"] = "qualified"
            save()
            print(json.dumps({"stage": label, **entry["evidence"]}), flush=True)
        if args.command == "prime" and state["persistent"]:
            deadline = time.monotonic() + client.timeout
            while True:
                final = client.snapshot()
                if (
                    delta(initial["cache"], final["cache"], "l2_stored", required=False)
                    > 0
                ):
                    report["persistent_publication"] = {
                        "before": initial,
                        "after": final,
                    }
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("No durable L2 store completion was observed")
                time.sleep(1)
        report["status"] = "qualified"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prime", "restore"))
    parser.add_argument(
        "--dedicated-endpoint",
        action="store_true",
        required=True,
        help="Confirm exclusive use; the tool resets this endpoint's GPU prefix cache",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--cache-metrics-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--container", help="Serving container; required for prime")
    parser.add_argument(
        "--cache-container", help="CPU cache container if separate from serving"
    )
    parser.add_argument("--model", help="Advertised model ID; required for prime")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument(
        "--fixture",
        choices=("text", "vision"),
        default="text",
        help="Image-bearing reuse places the image before the long shared prefix",
    )
    parser.add_argument("--minimum-reused-tokens", type=int, default=4096)
    parser.add_argument("--thinking-kwargs", type=json.loads, default={})
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if args.command == "prime" and not (args.container and args.model):
        parser.error("prime requires --container and --model")
    if (
        args.minimum_reused_tokens <= 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("Token threshold and timeout must be positive")
    if not isinstance(args.thinking_kwargs, dict):
        parser.error("--thinking-kwargs must be a JSON object")
    report = run(args, Client(args.base_url, args.cache_metrics_url, args.timeout))
    print(
        json.dumps(
            {
                "status": report["status"],
                "receipt": str(args.output_dir / f"{args.command}.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
