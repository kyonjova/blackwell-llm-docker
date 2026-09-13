#!/usr/bin/env python3
"""Check Qwen text and image serving with temperature-one OpenAI requests."""

from __future__ import annotations

import argparse
import base64
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path


def color_image_url(rgb: tuple[int, int, int]) -> str:
    """Encode a solid-color PNG without an image-processing dependency."""
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack("!I", len(payload)) + kind + payload
            + struct.pack("!I", zlib.crc32(kind + payload))
        )

    width = height = 224
    pixels = (b"\0" + bytes(rgb) * width) * height
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen3.8-Flash-Next")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()
    cases = [("arithmetic", "What is 17 + 25? Answer only the number.", "42")]
    for color, rgb in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
        cases.append((f"vision_{color}", [
            {"type": "text", "text": "What is the dominant color? Answer one English color word."},
            {"type": "image_url", "image_url": {"url": color_image_url(rgb)}},
        ], color))
    results = []
    for pass_index in range(args.passes):
        for name, content, expected in cases:
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 256,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request = urllib.request.Request(
                args.base_url.rstrip("/") + "/v1/chat/completions",
                json.dumps(payload).encode(), {"Content-Type": "application/json"},
            )
            start = time.monotonic()
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.load(response)
            choice = body["choices"][0]
            answer = choice["message"].get("content") or ""
            normalized = answer.strip().lower().rstrip(".! ")
            passed = normalized == expected and choice["finish_reason"] == "stop"
            result = {
                "case": name, "pass": pass_index + 1, "passed": passed,
                "response": body, "elapsed_seconds": time.monotonic() - start,
            }
            results.append(result)
            print(json.dumps({key: value for key, value in result.items() if key != "response"}), flush=True)
    report = {
        "status": "qualified" if all(result["passed"] for result in results) else "failed",
        "scope": "Temperature-one arithmetic and solid-color image API checks; not a general accuracy evaluation",
        "base_url": args.base_url, "model": args.model, "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
