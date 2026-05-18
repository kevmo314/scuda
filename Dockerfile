ARG CUDA_VERSION=13.1.0
ARG UBUNTU_VERSION=24.04
ARG CUDA_IMAGE_FLAVOR=devel
ARG CUDA_RUNTIME_IMAGE_FLAVOR=runtime

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG RUN_CODEGEN=0
ARG CMAKE_BUILD_TYPE=Release

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/scuda

COPY . /opt/scuda

RUN python3 -m venv /opt/scuda/.venv-codegen \
    && /opt/scuda/.venv-codegen/bin/pip install --no-cache-dir -r /opt/scuda/codegen/requirements.txt

# Generated sources are checked in because the current generator does not yet
# preserve every hand-maintained driver shim path. Set RUN_CODEGEN=1 only when
# updating generated sources intentionally for a CUDA header version.
RUN if [ "$RUN_CODEGEN" = "1" ]; then \
      cd /opt/scuda/codegen && /opt/scuda/.venv-codegen/bin/python ./codegen.py; \
    fi

RUN cmake -S /opt/scuda -B /opt/scuda/build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
      -DCMAKE_LIBRARY_PATH="${CUDA_HOME}/lib64/stubs"

FROM builder AS client-build

RUN cmake --build /opt/scuda/build --parallel --target scuda_driver scuda_nvml

RUN test -e /opt/scuda/build/libcuda.so.1 \
    && test -e /opt/scuda/build/libnvidia-ml.so.1 \
    && ln -sf libcuda.so.1 /opt/scuda/build/libcuda.so \
    && ln -sf libnvidia-ml.so.1 /opt/scuda/build/libnvidia-ml.so \
    && ! nm -D --defined-only /opt/scuda/build/libcuda.so.1 \
      | awk '{print $3}' \
      | grep -E '^cuda'

FROM builder AS server-build

RUN cmake --build /opt/scuda/build --parallel --target scuda_driver_server

RUN test -x /opt/scuda/build/scuda_driver_server

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_RUNTIME_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS client

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
ARG UBUNTU_VERSION
ARG NVIDIA_UTILS_PACKAGE=nvidia-utils-535
ARG NVIDIA_UTILS_VERSION=

LABEL org.opencontainers.image.title="scuda-client"
LABEL org.opencontainers.image.description="SCUDA client runtime with driver-only libcuda shim"
LABEL org.opencontainers.image.source="https://github.com/kevmo314/scuda"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    apt-get update; \
    mkdir -p /tmp/nvidia-utils; \
    cd /tmp/nvidia-utils; \
    if [ -n "$NVIDIA_UTILS_VERSION" ]; then \
      apt-get download "${NVIDIA_UTILS_PACKAGE}=${NVIDIA_UTILS_VERSION}"; \
    else \
      apt-get download "${NVIDIA_UTILS_PACKAGE}"; \
    fi; \
    dpkg-deb -x ./*.deb /tmp/nvidia-utils/root; \
    cp /tmp/nvidia-utils/root/usr/bin/nvidia-smi /usr/bin/nvidia-smi; \
    chmod +x /usr/bin/nvidia-smi; \
    rm -rf /var/lib/apt/lists/* /tmp/nvidia-utils

COPY --from=client-build /opt/scuda/build/libcuda.so.1 /opt/scuda/lib/libcuda.so.1
COPY --from=client-build /opt/scuda/build/libnvidia-ml.so.1 /opt/scuda/lib/libnvidia-ml.so.1

RUN ln -sf /opt/scuda/lib/libcuda.so.1 /opt/scuda/lib/libcuda.so \
    && ln -sf /opt/scuda/lib/libnvidia-ml.so.1 /opt/scuda/lib/libnvidia-ml.so

ENV SCUDA_LIB=/opt/scuda/lib/libcuda.so.1
ENV LD_LIBRARY_PATH=/opt/scuda/lib:${LD_LIBRARY_PATH}

ENTRYPOINT []
CMD ["bash"]

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_RUNTIME_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS server

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.title="scuda-server"
LABEL org.opencontainers.image.description="SCUDA server runtime"
LABEL org.opencontainers.image.source="https://github.com/kevmo314/scuda"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=server-build /opt/scuda/build/scuda_driver_server /opt/scuda/bin/scuda_driver_server

RUN chmod +x /opt/scuda/bin/scuda_driver_server

ENV SCUDA_PORT=14833
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 14833

ENTRYPOINT ["/opt/scuda/bin/scuda_driver_server"]
