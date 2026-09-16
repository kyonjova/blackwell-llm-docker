"""Model/cache process ownership must hold on success and failure."""

import json
import subprocess
from dataclasses import replace
from io import BytesIO

import pytest

from runtime import ConfigError, supervisor
from runtime.launcher import resolve


@pytest.mark.parametrize(
    "cache_exit,model_exit,expected",
    [(None, 0, 0), (7, None, "before readiness"), (None, 3, 3)],
)
def test_supervisor_always_stops_its_children(
    monkeypatch, cache_exit, model_exit, expected
):
    service = replace(
        resolve("ds4-flash", env={"LMCACHE_MODE": "ram"}).cache_service, shm_bytes=0
    )
    children, stopped = [], []

    class Child:
        def __init__(self, command, **kwargs):
            self.command, self.environment = command, kwargs["env"]
            self.code = cache_exit if not children else model_exit
            children.append(self)

        def poll(self):
            return self.code

    class HTTP:
        def open(self, *_args, **_kwargs):
            return BytesIO(b"{}")

    monkeypatch.setattr(supervisor, "preflight", lambda *_: None)
    monkeypatch.setattr(subprocess, "Popen", Child)
    monkeypatch.setattr(
        supervisor, "stop_groups", lambda processes: stopped.extend(processes)
    )
    monkeypatch.setattr(supervisor.urllib.request, "build_opener", lambda *_: HTTP())
    if isinstance(expected, str):
        with pytest.raises(ConfigError, match=expected):
            supervisor.supervise(service, ["model"], {"CUDA_VISIBLE_DEVICES": "2"}, [])
    else:
        assert (
            supervisor.supervise(service, ["model"], {"CUDA_VISIBLE_DEVICES": "2"}, [])
            == expected
        )
        assert children[1].environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert children[0].environment["CUDA_VISIBLE_DEVICES"] == ""
    assert stopped == children


def test_wrong_shm_pool_cannot_start_a_model():
    service = resolve("ds4-flash", env={"LMCACHE_MODE": "ram"}).cache_service
    with pytest.raises(ConfigError, match="no pickle fallback"):
        supervisor.validate_pool(
            {
                "engine_driven_shm_pool": {
                    "shm_name": service.shm_name,
                    "pool_size": service.shm_bytes - 1,
                }
            },
            service,
        )


@pytest.mark.parametrize("valid_pool", [True, False])
def test_non_json_status_retries_but_invalid_pool_is_fatal(monkeypatch, valid_pool):
    service = resolve("ds4-flash", env={"LMCACHE_MODE": "ram"}).cache_service
    children, stopped, attempts = [], [], []

    class Child:
        def __init__(self, command, **kwargs):
            self.code = None if not children else 0
            children.append(self)

        def poll(self):
            return self.code

    class HTTP:
        def open(self, *_args, **_kwargs):
            return BytesIO(b"{}")

    def status(_url):
        attempts.append(1)
        if len(attempts) == 1:
            raise json.JSONDecodeError("not ready", "", 0)
        return {
            "engine_driven_shm_pool": {
                "shm_name": service.shm_name,
                "pool_size": service.shm_bytes if valid_pool else 0,
            }
        }

    monkeypatch.setattr(supervisor, "preflight", lambda *_: None)
    monkeypatch.setattr(subprocess, "Popen", Child)
    monkeypatch.setattr(supervisor, "stop_groups", lambda items: stopped.extend(items))
    monkeypatch.setattr(supervisor.urllib.request, "build_opener", lambda *_: HTTP())
    monkeypatch.setattr(supervisor, "read_json", status)
    monkeypatch.setattr(supervisor.time, "sleep", lambda *_: None)
    if valid_pool:
        assert supervisor.supervise(service, ["model"], {}, []) == 0
        assert len(children) == 2
    else:
        with pytest.raises(ConfigError, match="no pickle fallback"):
            supervisor.supervise(service, ["model"], {}, [])
        assert len(children) == 1
    assert len(attempts) == 2
    assert stopped == children
