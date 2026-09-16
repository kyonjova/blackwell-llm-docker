"""Own cache/server process groups, readiness, SHM checks and bounded shutdown."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from runtime import ConfigError
from runtime.cache import CacheService


def read_json(url: str) -> dict:
    # Internal service readiness must not be redirected through an HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=1) as response:
        return json.load(response)


def validate_pool(status: dict, service: CacheService) -> None:
    pool = status.get("engine_driven_shm_pool") or {}
    if (
        pool.get("shm_name") != service.shm_name
        or int(pool.get("pool_size") or 0) < service.shm_bytes
    ):
        raise ConfigError(
            "LMCache must advertise the requested SHM arena and capacity; no pickle fallback. "
            f"Expected {service.shm_name!r}/{service.shm_bytes} bytes, "
            f"received {pool.get('shm_name')!r}/{pool.get('pool_size')} bytes"
        )


def preflight(service: CacheService, environment: dict) -> None:
    if any("UNRESOLVED-CHECKPOINT" in arg for arg in service.argv):
        raise ConfigError(
            "Persistent cache identity must be resolved before starting processes"
        )
    if service.shm_bytes:
        shm = Path("/dev/shm") / service.shm_name
        if shm.exists():
            raise ConfigError(f"Refusing to reuse an existing cache SHM arena: {shm}")
        stats = os.statvfs("/dev/shm")
        if stats.f_bavail * stats.f_frsize < service.shm_bytes:
            raise ConfigError(
                "Insufficient free /dev/shm for the complete engine-driven L1 arena"
            )
    interposer = Path("/opt/lmcache/lib/liblmcache_cumem_shareable.so")
    if (
        str(interposer) in environment.get("LD_PRELOAD", "").split(":")
        and not interposer.is_file()
    ):
        raise ConfigError(
            "LMCache-driven GLM transfer requires the packaged CUDA cuMem interposer"
        )
    broker = environment.get("LMCACHE_CUMEM_BROKER_DIR")
    for name in service.directories:
        path = Path(name)
        if path.is_symlink():
            raise ConfigError(f"Refusing a symlinked cache service directory: {path}")
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if name == broker and (
            path.stat().st_uid != os.geteuid() or path.stat().st_mode & 0o077
        ):
            raise ConfigError(
                "The cuMem broker directory must be owned by the serving user with mode 0700"
            )
    # A health response must not accidentally belong to another cache service.
    host = service.argv[service.argv.index("--host") + 1]
    http_host = service.argv[service.argv.index("--http-host") + 1]
    for flag, address in (
        ("--port", host),
        ("--http-port", http_host),
        ("--prometheus-port", http_host),
    ):
        port = int(service.argv[service.argv.index(flag) + 1])
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((address, port))
            except OSError as error:
                raise ConfigError(
                    f"Cache service address is unavailable: {address}:{port}"
                ) from error


def stop_groups(children, grace=10):
    for child in children:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    for child in children:
        try:
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # A group can still contain worker descendants after its leader exits.
    for child in children:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def supervise(
    service: CacheService,
    model_command: list[str],
    environment: dict,
    bootstrap: list[str],
) -> int:
    preflight(service, environment)
    children = []
    requested_signal = 0

    def request_shutdown(signum, _frame):
        nonlocal requested_signal
        requested_signal = signum

    previous = {
        sig: signal.signal(sig, request_shutdown)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    try:
        cache = subprocess.Popen(
            [*bootstrap, *service.argv],
            env={**environment, **service.environment},
            start_new_session=True,
        )
        children.append(cache)
        deadline = time.monotonic() + service.startup_timeout
        while not requested_signal:
            status = cache.poll()
            if status is not None:
                raise ConfigError(f"LMCache exited before readiness (status {status})")
            if time.monotonic() >= deadline:
                raise ConfigError("LMCache startup timed out")
            try:
                # /healthcheck can be non-JSON; /status is the transport contract.
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(service.health_url, timeout=1):
                    pass
                if service.shm_bytes:
                    validate_pool(
                        read_json(service.health_url.rsplit("/", 1)[0] + "/status"),
                        service,
                    )
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(0.1)
        if requested_signal:
            return 128 + requested_signal
        if cache.poll() is not None:
            raise ConfigError("LMCache exited after its readiness response")
        model = subprocess.Popen(model_command, env=environment, start_new_session=True)
        children.append(model)
        while not requested_signal:
            if cache.poll() is not None:
                raise ConfigError("LMCache exited while the model server was running")
            status = model.poll()
            if status is not None:
                return status if status >= 0 else 128 - status
            time.sleep(0.1)
        return 128 + requested_signal
    finally:
        stop_groups(children)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
