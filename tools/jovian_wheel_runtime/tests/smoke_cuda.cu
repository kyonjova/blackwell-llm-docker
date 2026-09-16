#include <cuda_runtime.h>

#include <cstdio>

int main() {
  int runtime_version = 0;
  cudaDeviceProp properties{};
  if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
    return 1;
  }
  std::printf("cuda_runtime=%d device=%s compute=%d.%d\n", runtime_version,
              properties.name, properties.major, properties.minor);
  return runtime_version == 13040 && properties.major == 12 ? 0 : 2;
}
