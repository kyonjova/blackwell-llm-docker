# Disk-backed Engram dependency

Status: implemented. The CUDA 13.4 multi-model image installs `liburing2` and
`liburing-dev` at Ubuntu package version `2.5-1build1`. The B12X loader requires
their library, headers and pkg-config metadata to compile its io_uring reader.
The application remains one filesystem layer over the 67-layer foundation.

The build verifier compiles and links a C program including `liburing.h`.
Publication additionally initializes an io_uring queue under the DS4.1 serving
permissions (`seccomp=unconfined`, unlimited memlock). Build-time validation
does not require syscalls blocked by the build sandbox.
The standalone builder's optional GPU smoke uses the same queue-check command.

## Validation

The runtime and wheel/container tool suites pass 282 CPU tests, including
missing-library failure, header/link probe, explicit syscall-policy gating,
publisher permissions and the one-application-layer invariant.

On Linux x86-64, the package-negative control used the published KK image
`sha256:79d8d57177e54435586a36f9e3ae5a609bf4e07d62f86ed7108f7cca192b7035`:
pkg-config failed to find liburing, and the dependency verifier rejected it.

In an isolated CPU-only container from that image, the recipe's dependency
installer added both packages. Header/link validation and
`io_uring_queue_init(2, ..., 0)` passed. The installed B12X loader compiled and
loaded successfully; its native extension resolved `io_uring_queue_init`.
No model weights or serving source files were modified.

A CPU container using Docker's default seccomp policy rejected
`io_uring_setup` with `EPERM`. Therefore NGC foundation selection alone must
not force queue creation during the Docker build; runtime qualification opts
in explicitly with the documented serving permissions.

This qualifies dependency discovery, compilation, dynamic linking and host
syscall access. DS4.1 end-to-end disk-table serving remains a separate pending
check; these CPU checks do not establish disk throughput or model correctness.
