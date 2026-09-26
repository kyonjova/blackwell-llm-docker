"""Profiles may override a foundation variable only through platform-environment.json.

The model-neutral foundation drops every variable listed there from the image
config and the launcher re-applies it as the lowest layer, so hardware and
model profiles can override it. A profile-owned name left in the image config
fails the build audit (Image Config.Env contains profile-owned values).
"""

from runtime.launcher import platform_environment
from runtime.packaging import audit_image_metadata, owned_environment

# Config.Env names of the pinned NGC PyTorch foundation (source_image in
# runtime/platform-environment.json).
FOUNDATION_ENV = {
    "BASH_ENV", "CCCL_VERSION", "COCOAPI_VERSION", "CUBLASMP_VERSION", "CUBLAS_VERSION", "CUDA_ARCH_LIST",
    "CUDA_BINARY_LOADER_THREAD_COUNT", "CUDA_COMPONENT_LIST", "CUDA_DRIVER_VERSION", "CUDA_HOME",
    "CUDA_MODULE_LOADING", "CUDA_VERSION", "CUDLA_VERSION", "CUDNN_FRONTEND_VERSION", "CUDNN_VERSION",
    "CUFFT_VERSION", "CUFILE_VERSION", "CURAND_VERSION", "CUSOLVERMP_VERSION", "CUSOLVER_VERSION",
    "CUSPARSELT_VERSION", "CUSPARSE_VERSION", "CUTILE_PYTHON_VERSION", "CUTLASS_DSL_VERSION", "DALI_BUILD",
    "DALI_URL_SUFFIX", "DALI_VERSION", "DOCA_VERSION", "EFA_VERSION", "ENV", "GDRCOPY_VERSION", "HPCX_VERSION",
    "JUPYTER_PORT", "LC_ALL", "LD_LIBRARY_PATH", "LIBRARY_PATH", "MAXSMVER", "MODEL_OPT_VERSION",
    "MOFED_VERSION", "NCCL_NET_PLUGIN", "NCCL_VERSION", "NIXL_VERSION", "NPP_VERSION", "NSIGHT_COMPUTE_VERSION",
    "NSIGHT_SYSTEMS_VERSION", "NVFATBIN_VERSION", "NVFUSER_BUILD_VERSION", "NVFUSER_VERSION", "NVIDIA_BUILD_ID",
    "NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_PRODUCT_NAME", "NVIDIA_PYTORCH_VERSION", "NVIDIA_REQUIRE_CUDA",
    "NVIDIA_VISIBLE_DEVICES", "NVJITLINK_VERSION", "NVJPEG_VERSION", "NVPL_LAPACK_MATH_MODE",
    "NVPTXCOMPILER_VERSION", "NVRX_VERSION", "NVSHMEM_VERSION", "NVVM_VERSION", "OMPI_MCA_coll_hcoll_enable",
    "OPAL_PREFIX", "OPENMPI_VERSION", "OPENUCX_VERSION", "PATH", "PIP_BREAK_SYSTEM_PACKAGES", "PIP_CONSTRAINT",
    "PIP_DEFAULT_TIMEOUT", "POLYGRAPHY_VERSION", "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "PYTHONIOENCODING",
    "PYTORCH_BUILD_NUMBER", "PYTORCH_BUILD_VERSION", "PYTORCH_HOME", "PYTORCH_VERSION", "RDMACORE_VERSION",
    "SHELL", "TENSORBOARD_PORT", "TORCHAO_BUILD_VERSION", "TORCHINDUCTOR_CUTLASS_DIR",
    "TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION", "TORCHTITAN_BUILD_VERSION", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "TORCH_CUDA_ARCH_LIST", "TORCH_NCCL_USE_COMM_NONBLOCKING", "TRANSFORMER_ENGINE_VERSION",
    "TRITON_CUDACRT_PATH", "TRITON_CUDART_PATH", "TRITON_CUOBJDUMP_PATH", "TRITON_CUPTI_INCLUDE_PATH",
    "TRITON_CUPTI_LIB_PATH", "TRITON_NVDISASM_PATH", "TRITON_PTXAS_PATH", "TRTOSS_VERSION", "TRT_VERSION",
    "UCC_CL_BASIC_TLS", "UCC_EC_CUDA_EXEC_NUM_THREADS", "_CUDA_COMPAT_PATH",
}


def test_profile_owned_foundation_variables_are_platform_defaults():
    overlap = FOUNDATION_ENV & owned_environment()
    assert overlap == set(platform_environment()), (
        "Foundation variables that profiles set must be listed in runtime/platform-environment.json"
    )


def test_neutral_foundation_passes_the_image_audit():
    neutral = sorted(FOUNDATION_ENV - set(platform_environment()))
    audit_image_metadata({"Id": "sha256:" + "a" * 64, "Config": {"Env": [f"{name}=x" for name in neutral]}})


def test_rtx_pro_arch_list_overrides_the_foundation_default():
    assert platform_environment()["TORCH_CUDA_ARCH_LIST"] == "7.5 8.0 8.6 9.0 10.0 12.0+PTX"
