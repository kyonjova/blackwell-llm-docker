from importlib.metadata import Distribution

from tools.jovian_wheel_runtime.verify_installed_requirements import dependency_errors


class FakeDistribution:
    def __init__(self, version: str, requirements: list[str] | None = None):
        self.version = version
        self.requires = requirements


def distributions(**packages: tuple[str, list[str] | None]) -> dict[str, Distribution]:
    return {
        name: FakeDistribution(version, requirements)  # type: ignore[dict-item]
        for name, (version, requirements) in packages.items()
    }


def test_accepts_visible_system_dependency() -> None:
    selected = distributions(
        vllm=("1.0", ["torch==2.14"]),
        torch=("2.14", None),
    )
    assert dependency_errors(selected) == []


def test_accepts_ngc_prerelease_with_compatible_lower_bound() -> None:
    selected = distributions(
        vllm=("1.0", ["torch>=2.9"]),
        torch=("2.14.0a0+4fdf77b940.nv26.8.63802676", None),
    )
    assert dependency_errors(selected) == []


def test_rejects_missing_dependency() -> None:
    selected = distributions(vllm=("1.0", ["torch==2.14"]))
    assert dependency_errors(selected) == [
        "vllm requires torch==2.14, but it is not installed"
    ]


def test_rejects_incompatible_dependency() -> None:
    selected = distributions(
        vllm=("1.0", ["torch==2.14"]),
        torch=("2.13", None),
    )
    assert dependency_errors(selected) == [
        "vllm requires torch==2.14, but torch 2.13 is selected"
    ]


def test_ignores_unselected_extra() -> None:
    selected = distributions(vllm=("1.0", ['torch==2.14; extra == "compile"']))
    assert dependency_errors(selected) == []


def test_ignores_unrelated_system_distribution() -> None:
    selected = distributions(
        vllm=("1.0", ["torch==2.14"]),
        torch=("2.14", None),
        telemetry=("1.0", ["unavailable>=2"]),
    )
    assert dependency_errors(selected, {"vllm"}) == []
