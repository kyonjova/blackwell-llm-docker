# syntax=docker/dockerfile:1.7

ARG SOURCE_IMAGE
FROM ${SOURCE_IMAGE} AS repack

ARG SOURCE_DATE_EPOCH
COPY tools/jovian_wheel_runtime/repack_installed_distribution.py /opt/local-inference/
COPY tools/jovian_wheel_runtime/build_torch_native_support.py /opt/local-inference/
COPY tools/jovian_wheel_runtime/torch_native_support.json /opt/local-inference/
RUN mkdir -p /artifacts/wheels \
 && python /opt/local-inference/repack_installed_distribution.py \
      --output-dir /artifacts/wheels \
      --source-date-epoch "${SOURCE_DATE_EPOCH}" \
      --allow-modified-record-entry \
      torch/lib/libtorch_global_deps.so=6991f14618ba04c4caa7d59bb5c520e930cfd6d40282e98eb83e2af169045894 \
      --rewrite-torch-runpaths \
      torch torchvision triton triton-kernels flash-attn nvidia-modelopt \
      > /artifacts/repack-provenance.json \
 && python /opt/local-inference/build_torch_native_support.py \
      --contract /opt/local-inference/torch_native_support.json \
      --output-dir /artifacts/wheels \
      --source-date-epoch "${SOURCE_DATE_EPOCH}" \
      > /artifacts/torch-native-support-provenance.json

FROM ${SOURCE_IMAGE} AS validate
COPY --from=repack /artifacts /artifacts
RUN mkdir -p /tmp/repack-validation \
 && python -m pip install --no-deps --target /tmp/repack-validation \
      /artifacts/wheels/torch-*.whl \
      /artifacts/wheels/torchvision-*.whl \
      /artifacts/wheels/triton-*.whl \
      /artifacts/wheels/triton_kernels-*.whl \
      /artifacts/wheels/flash_attn-*.whl \
      /artifacts/wheels/nvidia_modelopt-*.whl \
      /artifacts/wheels/local_inference_torch_native_support-*.whl \
 && python - <<'PY'
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

root = Path("/tmp/repack-validation").resolve()
sys.path.insert(0, str(root))
manifest = json.loads(Path("/artifacts/repack-provenance.json").read_text())
installed = {
    distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
    for distribution in importlib.metadata.distributions(path=[root])
}
for package in manifest["packages"]:
    name = package["name"].lower().replace("_", "-")
    assert name in installed, (name, installed)
    assert installed[name] == package["version"], (
        name,
        installed[name],
        package["version"],
    )
for module in (
    "torch",
    "torchvision",
    "triton",
    "triton_kernels",
    "flash_attn",
    "modelopt",
):
    imported = importlib.import_module(module)
    imported_path = Path(imported.__file__).resolve()
    assert imported_path.is_relative_to(root), (module, imported_path, root)
print("Installed-distribution wheel repack: PASS")
PY

FROM scratch AS artifacts
COPY --from=validate /artifacts /
