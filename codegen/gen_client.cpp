#include <cuda.h>

#define SCUDA_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef SCUDA_CUDA_COMPAT_TYPES_ONLY

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "gen_api.h"

#include "rpc.h"

extern int rpc_size();
extern conn_t *rpc_client_get_connection(unsigned int index);
extern void rpc_close(conn_t *conn);
extern "C" CUresult scuda_cuInit_multi(unsigned int flags);
extern "C" CUresult scuda_cuDeviceGetCount_multi(int *count);
extern "C" CUresult scuda_cuDeviceGet_multi(CUdevice *device, int ordinal);
extern "C" CUresult scuda_cuDeviceCanAccessPeer_multi(int *canAccessPeer,
                                                       CUdevice dev,
                                                       CUdevice peerDev);
extern "C" conn_t *scuda_rpc_conn_for_device(CUdevice *device);
extern "C" CUdevice scuda_local_device_for_remote(conn_t *conn,
                                                  CUdevice remote_device);
extern "C" conn_t *scuda_rpc_conn_for_current_context();
extern "C" conn_t *scuda_rpc_conn_for_context(CUcontext ctx);
extern "C" conn_t *scuda_rpc_conn_for_module(CUmodule module);
extern "C" conn_t *scuda_rpc_conn_for_function(CUfunction function);
extern "C" conn_t *scuda_rpc_conn_for_stream(CUstream stream);
extern "C" conn_t *scuda_rpc_conn_for_event(CUevent event);
extern "C" conn_t *scuda_rpc_conn_for_deviceptr(CUdeviceptr ptr);
extern "C" void scuda_note_context_owner(CUcontext ctx, conn_t *conn);
extern "C" void scuda_note_module_owner(CUmodule module, conn_t *conn);
extern "C" void scuda_note_function_owner(CUfunction function, conn_t *conn);
extern "C" void scuda_note_stream_owner(CUstream stream, conn_t *conn);
extern "C" void scuda_note_event_owner(CUevent event, conn_t *conn);
extern "C" void scuda_note_deviceptr_owner(CUdeviceptr ptr, conn_t *conn);
extern "C" void scuda_prepare_host_range_write(void *host, size_t size);
extern "C" void scuda_mark_host_range_clean(void *host, size_t size);
extern "C" void scuda_remember_loaded_module_for_rpc(CUmodule module);
extern "C" CUresult
scuda_cuArrayCreate_v2_safe(CUarray *pHandle,
                            const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
extern "C" CUresult scuda_cuArray3DCreate_v2_safe(
    CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
extern "C" CUresult scuda_cuMemAllocManaged_safe(CUdeviceptr *dptr,
                                                 size_t bytesize,
                                                 unsigned int flags);
extern "C" CUresult scuda_cuMemFree_v2_safe(CUdeviceptr dptr);
extern "C" CUfunction scuda_translate_private_function_for_rpc(
    CUfunction function);
extern "C" CUresult scuda_cuCtxPushCurrent_virtual(CUcontext ctx);
extern "C" CUresult scuda_cuCtxPopCurrent_virtual(CUcontext *pctx);
extern "C" CUresult scuda_cuCtxSetCurrent_virtual(CUcontext ctx);
extern "C" CUresult scuda_cuCtxGetCurrent_virtual(CUcontext *pctx);
static constexpr int SCUDA_RPC_cuLinkAddData_v2 = 1000018;

struct scuda_client_link_state {
    CUlinkState remote_state = nullptr;
    std::vector<unsigned char> image;
    CUjitInputType image_type = CU_JIT_INPUT_PTX;
};

struct scuda_kernel_param_layout {
    uint32_t count = 0;
    size_t offsets[64] = {};
    size_t sizes[64] = {};
};

static CUresult scuda_get_kernel_param_layout_for_codegen(
    CUfunction f, scuda_kernel_param_layout *layout) {
    if (layout == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    constexpr int SCUDA_RPC_cuFuncGetParamLayout = 1000001;
    f = scuda_translate_private_function_for_rpc(f);
    conn_t *conn = scuda_rpc_conn_for_function(f);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, SCUDA_RPC_cuFuncGetParamLayout) < 0 ||
        rpc_write(conn, &f, sizeof(f)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &layout->count, sizeof(layout->count)) < 0 ||
        rpc_read(conn, layout->offsets, sizeof(layout->offsets)) < 0 ||
        rpc_read(conn, layout->sizes, sizeof(layout->sizes)) < 0 ||
        rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

static CUresult scuda_pack_kernel_params_for_codegen(
    const CUDA_KERNEL_NODE_PARAMS *nodeParams, scuda_kernel_param_layout *layout,
    std::vector<unsigned char> *packed) {
    if (nodeParams == nullptr || layout == nullptr || packed == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    if (nodeParams->extra != nullptr)
        return CUDA_ERROR_NOT_SUPPORTED;
    CUfunction func = nodeParams->func;
#if CUDA_VERSION >= 12000
    if (func == nullptr)
        func = reinterpret_cast<CUfunction>(nodeParams->kern);
#endif
    CUresult status = scuda_get_kernel_param_layout_for_codegen(func, layout);
    if (status != CUDA_SUCCESS)
        return status;
    if (layout->count > 64)
        return CUDA_ERROR_NOT_SUPPORTED;
    if (layout->count != 0 && nodeParams->kernelParams == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    size_t total_size = 0;
    for (uint32_t i = 0; i < layout->count; ++i) {
        if (nodeParams->kernelParams[i] == nullptr)
            return CUDA_ERROR_INVALID_VALUE;
        total_size = std::max(total_size, layout->offsets[i] + layout->sizes[i]);
    }
    packed->assign(total_size, 0);
    for (uint32_t i = 0; i < layout->count; ++i)
        memcpy(packed->data() + layout->offsets[i], nodeParams->kernelParams[i],
               layout->sizes[i]);
    return CUDA_SUCCESS;
}

CUresult cuInit(unsigned int Flags)
{
    return scuda_cuInit_multi(Flags);
}

CUresult cuDriverGetVersion(int* driverVersion)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuDriverGetVersion) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, driverVersion, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (driverVersion != nullptr) {
        const char *override_version = getenv("SCUDA_DRIVER_VERSION_OVERRIDE");
        if (override_version != nullptr) *driverVersion = atoi(override_version);
    }
    return return_value;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal)
{
    return scuda_cuDeviceGet_multi(device, ordinal);
}

CUresult cuDeviceGetCount(int* count)
{
    return scuda_cuDeviceGetCount_multi(count);
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceGetName) < 0 ||
        rpc_write(conn, &len, sizeof(int)) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, name, len * sizeof(char)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceGetUuid_v2) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, uuid, 16) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    std::size_t luid_len;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceGetLuid) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &luid_len, sizeof(std::size_t)) < 0 ||
        rpc_read(conn, luid, luid_len) < 0 ||
        rpc_read(conn, deviceNodeMask, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceTotalMem_v2) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, bytes, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceGetTexture1DLinearMaxWidth) < 0 ||
        rpc_write(conn, &format, sizeof(CUarray_format)) < 0 ||
        rpc_write(conn, &numChannels, sizeof(unsigned)) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, maxWidthInElements, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceGetAttribute) < 0 ||
        rpc_write(conn, &attrib, sizeof(CUdevice_attribute)) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pi, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceSetMemPool) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceGetMemPool) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceGetDefaultMemPool) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pool_out, sizeof(CUmemoryPool)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType type, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceGetExecAffinitySupport) < 0 ||
        rpc_write(conn, &type, sizeof(CUexecAffinityType)) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pi, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuFlushGPUDirectRDMAWrites) < 0 ||
        rpc_write(conn, &target, sizeof(CUflushGPUDirectRDMAWritesTarget)) < 0 ||
        rpc_write(conn, &scope, sizeof(CUflushGPUDirectRDMAWritesScope)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetProperties(CUdevprop* prop, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceGetProperties) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, prop, sizeof(CUdevprop)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceComputeCapability) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, major, sizeof(int)) < 0 ||
        rpc_read(conn, minor, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxRetain) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pctx, sizeof(CUcontext)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value == CUDA_SUCCESS && pctx != nullptr)
        scuda_note_context_owner(*pctx, conn);
    return return_value;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxRelease_v2) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxSetFlags_v2) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxGetState) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, flags, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, active, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxReset_v2) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxDestroy_v2(CUcontext ctx)
{
    conn_t *conn = scuda_rpc_conn_for_context(ctx);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuCtxDestroy_v2) < 0 ||
        rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx)
{
    return scuda_cuCtxPushCurrent_virtual(ctx);
}

CUresult cuCtxPopCurrent_v2(CUcontext* pctx)
{
    return scuda_cuCtxPopCurrent_virtual(pctx);
}

CUresult cuCtxSetCurrent(CUcontext ctx)
{
    return scuda_cuCtxSetCurrent_virtual(ctx);
}

CUresult cuCtxGetCurrent(CUcontext* pctx)
{
    return scuda_cuCtxGetCurrent_virtual(pctx);
}

CUresult cuCtxGetDevice(CUdevice* device)
{
    conn_t *conn = scuda_rpc_conn_for_current_context();
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuCtxGetDevice) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, device, sizeof(CUdevice)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value == CUDA_SUCCESS && device != nullptr)
        *device = scuda_local_device_for_remote(conn, *device);
    return return_value;
}

CUresult cuCtxGetFlags(unsigned int* flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxGetFlags) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, flags, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetId(CUcontext ctx, unsigned long long* ctxId)
{
    conn_t *conn = scuda_rpc_conn_for_context(ctx);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuCtxGetId) < 0 ||
        rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, ctxId, sizeof(unsigned long long)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSetLimit(CUlimit limit, size_t value)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxSetLimit) < 0 ||
        rpc_write(conn, &limit, sizeof(CUlimit)) < 0 ||
        rpc_write(conn, &value, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetLimit(size_t* pvalue, CUlimit limit)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxGetLimit) < 0 ||
        rpc_write(conn, &limit, sizeof(CUlimit)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pvalue, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetCacheConfig(CUfunc_cache* pconfig)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxGetCacheConfig) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pconfig, sizeof(CUfunc_cache)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSetCacheConfig(CUfunc_cache config)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxSetCacheConfig) < 0 ||
        rpc_write(conn, &config, sizeof(CUfunc_cache)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version)
{
    conn_t *conn = scuda_rpc_conn_for_context(ctx);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuCtxGetApiVersion) < 0 ||
        rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, version, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxGetStreamPriorityRange) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, leastPriority, sizeof(int)) < 0 ||
        rpc_read(conn, greatestPriority, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxResetPersistingL2Cache()
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxResetPersistingL2Cache) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType type)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxGetExecAffinity) < 0 ||
        rpc_write(conn, &type, sizeof(CUexecAffinityType)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pExecAffinity, sizeof(CUexecAffinityParam)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxAttach(CUcontext* pctx, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxAttach) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pctx, sizeof(CUcontext)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxDetach(CUcontext ctx)
{
    conn_t *conn = scuda_rpc_conn_for_context(ctx);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuCtxDetach) < 0 ||
        rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetSharedMemConfig(CUsharedconfig* pConfig)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxGetSharedMemConfig) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pConfig, sizeof(CUsharedconfig)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSetSharedMemConfig(CUsharedconfig config)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxSetSharedMemConfig) < 0 ||
        rpc_write(conn, &config, sizeof(CUsharedconfig)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleLoad(CUmodule* module, const char* fname)
{
    if (module == nullptr || fname == nullptr)
        return CUDA_ERROR_INVALID_VALUE;

    FILE *file = std::fopen(fname, "rb");
    if (file == nullptr)
        return CUDA_ERROR_FILE_NOT_FOUND;
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        return CUDA_ERROR_FILE_NOT_FOUND;
    }
    long file_size = std::ftell(file);
    if (file_size <= 0 || std::fseek(file, 0, SEEK_SET) != 0) {
        std::fclose(file);
        return CUDA_ERROR_INVALID_IMAGE;
    }
    std::vector<unsigned char> image(static_cast<size_t>(file_size));
    if (std::fread(image.data(), 1, image.size(), file) != image.size()) {
        std::fclose(file);
        return CUDA_ERROR_INVALID_IMAGE;
    }
    std::fclose(file);
    return cuModuleLoadData(module, image.data());
}

CUresult cuModuleUnload(CUmodule hmod)
{
    conn_t *conn = scuda_rpc_conn_for_module(hmod);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuModuleUnload) < 0 ||
        rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode* mode)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuModuleGetLoadingMode) < 0 ||
        rpc_write(conn, mode, sizeof(CUmoduleLoadingMode)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, mode, sizeof(CUmoduleLoadingMode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name)
{
    conn_t *conn = scuda_rpc_conn_for_module(hmod);
    CUresult return_value;
    std::size_t name_len = std::strlen(name) + 1;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuModuleGetFunction) < 0 ||
        rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(conn, name, name_len) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, hfunc, sizeof(CUfunction)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value == CUDA_SUCCESS && hfunc != nullptr)
        scuda_note_function_owner(*hfunc, conn);
    return return_value;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name)
{
    conn_t *conn = scuda_rpc_conn_for_module(hmod);
    CUresult return_value;
    size_t remote_bytes;
    std::size_t name_len = std::strlen(name) + 1;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuModuleGetGlobal_v2) < 0 ||
        rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(conn, name, name_len) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, &remote_bytes, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (bytes != nullptr)
        *bytes = remote_bytes;
    if (return_value == CUDA_SUCCESS && dptr != nullptr)
        scuda_note_deviceptr_owner(*dptr, conn);
    return return_value;
}

CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut)
{
    (void)numOptions;
    (void)options;
    (void)optionValues;
    if (stateOut == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *stateOut = reinterpret_cast<CUlinkState>(new scuda_client_link_state());
    return CUDA_SUCCESS;
}

static CUresult scuda_remote_cuLinkCreate(CUlinkState *stateOut)
{
    unsigned int numOptions = 0;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuLinkCreate_v2) < 0 ||
        rpc_write(conn, &numOptions, sizeof(numOptions)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, stateOut, sizeof(*stateOut)) < 0 ||
        rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

static CUresult scuda_remote_cuLinkAddData(CUlinkState state,
                                           CUjitInputType type,
                                           const void *data, size_t size)
{
    if (data == nullptr || size == 0)
        return CUDA_ERROR_INVALID_VALUE;
    size_t name_len = 0;
    unsigned int numOptions = 0;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, SCUDA_RPC_cuLinkAddData_v2) < 0 ||
        rpc_write(conn, &state, sizeof(state)) < 0 ||
        rpc_write(conn, &type, sizeof(type)) < 0 ||
        rpc_write(conn, &size, sizeof(size)) < 0 ||
        rpc_write(conn, data, size) < 0 ||
        rpc_write(conn, &name_len, sizeof(name_len)) < 0 ||
        rpc_write(conn, &numOptions, sizeof(numOptions)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

static CUresult scuda_ensure_remote_link_state(scuda_client_link_state *link)
{
    if (link == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    if (link->remote_state != nullptr)
        return CUDA_SUCCESS;
    CUresult result = scuda_remote_cuLinkCreate(&link->remote_state);
    if (result != CUDA_SUCCESS)
        return result;
    if (!link->image.empty()) {
        result = scuda_remote_cuLinkAddData(
            link->remote_state, link->image_type, link->image.data(),
            link->image.size());
        if (result != CUDA_SUCCESS)
            return result;
    }
    return CUDA_SUCCESS;
}

CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues)
{
    (void)name;
    (void)numOptions;
    (void)options;
    (void)optionValues;
    if (state == nullptr || data == nullptr || size == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (type != CU_JIT_INPUT_PTX && type != CU_JIT_INPUT_CUBIN && type != CU_JIT_INPUT_FATBINARY) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    auto *link = reinterpret_cast<scuda_client_link_state *>(state);
    if (link->remote_state != nullptr)
        return scuda_remote_cuLinkAddData(link->remote_state, type, data, size);
    const auto *begin = static_cast<const unsigned char *>(data);
    link->image.assign(begin, begin + size);
    link->image_type = type;
    if (link->image.empty() || link->image.back() != 0) {
        link->image.push_back(0);
    }
    return CUDA_SUCCESS;
}

CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues)
{
    if (path == nullptr || (numOptions != 0 && (options == nullptr || optionValues == nullptr)))
        return CUDA_ERROR_INVALID_VALUE;
    auto *link = reinterpret_cast<scuda_client_link_state *>(state);
    CUresult status = scuda_ensure_remote_link_state(link);
    if (status != CUDA_SUCCESS)
        return status;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    std::size_t path_len = std::strlen(path) + 1;
    if (rpc_write_start_request(conn, RPC_cuLinkAddFile_v2) < 0 ||
        rpc_write(conn, &link->remote_state, sizeof(CUlinkState)) < 0 ||
        rpc_write(conn, &type, sizeof(CUjitInputType)) < 0 ||
        rpc_write(conn, &path_len, sizeof(std::size_t)) < 0 ||
        rpc_write(conn, path, path_len) < 0 ||
        rpc_write(conn, &numOptions, sizeof(unsigned int)) < 0 ||
        (numOptions != 0 && rpc_write(conn, options, numOptions * sizeof(CUjit_option)) < 0) ||
        (numOptions != 0 && rpc_write(conn, optionValues, numOptions * sizeof(void*)) < 0) ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut)
{
    if (state == nullptr || cubinOut == nullptr || sizeOut == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto *link = reinterpret_cast<scuda_client_link_state *>(state);
    if (link->remote_state != nullptr) {
        conn_t *conn = rpc_client_get_connection(0);
        CUresult return_value;
        size_t remote_size = 0;
        if (rpc_write_start_request(conn, RPC_cuLinkComplete) < 0 ||
            rpc_write(conn, &link->remote_state, sizeof(link->remote_state)) < 0 ||
            rpc_wait_for_response(conn) < 0 ||
            rpc_read(conn, &remote_size, sizeof(remote_size)) < 0)
            return CUDA_ERROR_DEVICE_UNAVAILABLE;
        void *copy = remote_size == 0 ? nullptr : malloc(remote_size);
        if (remote_size != 0 && copy == nullptr)
            return CUDA_ERROR_OUT_OF_MEMORY;
        if ((remote_size != 0 && rpc_read(conn, copy, remote_size) < 0) ||
            rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
            rpc_read_end(conn) < 0) {
            free(copy);
            return CUDA_ERROR_DEVICE_UNAVAILABLE;
        }
        if (return_value != CUDA_SUCCESS) {
            free(copy);
            return return_value;
        }
        *cubinOut = copy;
        *sizeOut = remote_size;
        return CUDA_SUCCESS;
    }
    if (link->image.empty()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    void *copy = malloc(link->image.size());
    if (copy == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    memcpy(copy, link->image.data(), link->image.size());
    *cubinOut = copy;
    *sizeOut = link->image.size();
    return CUDA_SUCCESS;
}

CUresult cuLinkDestroy(CUlinkState state)
{
    auto *link = reinterpret_cast<scuda_client_link_state *>(state);
    if (link != nullptr && link->remote_state != nullptr) {
        conn_t *conn = rpc_client_get_connection(0);
        CUresult return_value;
        if (rpc_write_start_request(conn, RPC_cuLinkDestroy) < 0 ||
            rpc_write(conn, &link->remote_state, sizeof(link->remote_state)) < 0 ||
            rpc_wait_for_response(conn) < 0 ||
            rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
            rpc_read_end(conn) < 0)
            return CUDA_ERROR_DEVICE_UNAVAILABLE;
        delete link;
        return return_value;
    }
    delete link;
    return CUDA_SUCCESS;
}

#ifdef cuLinkCreate
#undef cuLinkCreate
#endif
CUresult cuLinkCreate(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut)
{
    return cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
}

#ifdef cuLinkAddData
#undef cuLinkAddData
#endif
CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues)
{
    return cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    std::size_t name_len = std::strlen(name) + 1;
    if (rpc_write_start_request(conn, RPC_cuModuleGetTexRef) < 0 ||
        rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(conn, name, name_len) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pTexRef, sizeof(CUtexref)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    std::size_t name_len = std::strlen(name) + 1;
    if (rpc_write_start_request(conn, RPC_cuModuleGetSurfRef) < 0 ||
        rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(conn, name, name_len) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pSurfRef, sizeof(CUsurfref)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryLoadFromFile(CUlibrary* library, const char* fileName, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    std::size_t fileName_len = std::strlen(fileName) + 1;
    if (rpc_write_start_request(conn, RPC_cuLibraryLoadFromFile) < 0 ||
        rpc_write(conn, &fileName_len, sizeof(std::size_t)) < 0 ||
        rpc_write(conn, fileName, fileName_len) < 0 ||
        rpc_write(conn, &numJitOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, jitOptions, numJitOptions * sizeof(CUjit_option)) < 0 ||
        rpc_write(conn, jitOptionsValues, numJitOptions * sizeof(void*)) < 0 ||
        rpc_write(conn, &numLibraryOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, libraryOptions, numLibraryOptions * sizeof(CUlibraryOption)) < 0 ||
        rpc_write(conn, libraryOptionValues, numLibraryOptions * sizeof(void*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, library, sizeof(CUlibrary)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryUnload(CUlibrary library)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuLibraryUnload) < 0 ||
        rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    std::size_t name_len = std::strlen(name) + 1;
    if (rpc_write_start_request(conn, RPC_cuLibraryGetKernel) < 0 ||
        rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
        rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(conn, name, name_len) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pKernel, sizeof(CUkernel)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetModule(CUmodule* pMod, CUlibrary library)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuLibraryGetModule) < 0 ||
        rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pMod, sizeof(CUmodule)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value == CUDA_SUCCESS)
        scuda_remember_loaded_module_for_rpc(*pMod);
    return return_value;
}

CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuKernelGetFunction) < 0 ||
        rpc_write(conn, &kernel, sizeof(CUkernel)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pFunc, sizeof(CUfunction)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    size_t remote_bytes;
    std::size_t name_len = std::strlen(name) + 1;
    if (rpc_write_start_request(conn, RPC_cuLibraryGetGlobal) < 0 ||
        rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
        rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(conn, name, name_len) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, &remote_bytes, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (bytes != nullptr)
        *bytes = remote_bytes;
    return return_value;
}

CUresult cuLibraryGetManaged(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    size_t remote_bytes;
    std::size_t name_len = std::strlen(name) + 1;
    if (rpc_write_start_request(conn, RPC_cuLibraryGetManaged) < 0 ||
        rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
        rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(conn, name, name_len) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, &remote_bytes, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (bytes != nullptr)
        *bytes = remote_bytes;
    return return_value;
}

CUresult cuLibraryGetUnifiedFunction(void** fptr, CUlibrary library, const char* symbol)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    std::size_t symbol_len = std::strlen(symbol) + 1;
    if (rpc_write_start_request(conn, RPC_cuLibraryGetUnifiedFunction) < 0 ||
        rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
        rpc_write(conn, &symbol_len, sizeof(std::size_t)) < 0 ||
        rpc_write(conn, symbol, symbol_len) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, fptr, sizeof(void*)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuKernelGetAttribute(int* pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuKernelGetAttribute) < 0 ||
        rpc_write(conn, pi, sizeof(int)) < 0 ||
        rpc_write(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(conn, &kernel, sizeof(CUkernel)) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pi, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuKernelSetAttribute) < 0 ||
        rpc_write(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(conn, &val, sizeof(int)) < 0 ||
        rpc_write(conn, &kernel, sizeof(CUkernel)) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuKernelSetCacheConfig) < 0 ||
        rpc_write(conn, &kernel, sizeof(CUkernel)) < 0 ||
        rpc_write(conn, &config, sizeof(CUfunc_cache)) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetInfo_v2(size_t* free, size_t* total)
{
    conn_t *conn = scuda_rpc_conn_for_current_context();
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemGetInfo_v2) < 0 ||
        rpc_write(conn, free, sizeof(size_t)) < 0 ||
        rpc_write(conn, total, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, free, sizeof(size_t)) < 0 ||
        rpc_read(conn, total, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize)
{
    conn_t *conn = scuda_rpc_conn_for_current_context();
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemAlloc_v2) < 0 ||
        rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &bytesize, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value == CUDA_SUCCESS && dptr != nullptr)
        scuda_note_deviceptr_owner(*dptr, conn);
    return return_value;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)
{
    conn_t *conn = scuda_rpc_conn_for_current_context();
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemAllocPitch_v2) < 0 ||
        rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, pPitch, sizeof(size_t)) < 0 ||
        rpc_write(conn, &WidthInBytes, sizeof(size_t)) < 0 ||
        rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
        rpc_write(conn, &ElementSizeBytes, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, pPitch, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value == CUDA_SUCCESS && dptr != nullptr)
        scuda_note_deviceptr_owner(*dptr, conn);
    return return_value;
}

CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    return scuda_cuMemFree_v2_safe(dptr);
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dptr);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemGetAddressRange_v2) < 0 ||
        rpc_write(conn, pbase, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, psize, sizeof(size_t)) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pbase, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, psize, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags)
{
    return scuda_cuMemAllocManaged_safe(dptr, bytesize, flags);
}

CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    std::size_t pciBusId_len = std::strlen(pciBusId) + 1;
    if (rpc_write_start_request(conn, RPC_cuDeviceGetByPCIBusId) < 0 ||
        rpc_write(conn, dev, sizeof(CUdevice)) < 0 ||
        rpc_write(conn, &pciBusId_len, sizeof(std::size_t)) < 0 ||
        rpc_write(conn, pciBusId, pciBusId_len) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, dev, sizeof(CUdevice)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev)
{
    conn_t *conn = scuda_rpc_conn_for_device(&dev);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuDeviceGetPCIBusId) < 0 ||
        rpc_write(conn, &len, sizeof(int)) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pciBusId, len * sizeof(char)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuIpcGetEventHandle) < 0 ||
        rpc_write(conn, pHandle, sizeof(CUipcEventHandle)) < 0 ||
        rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pHandle, sizeof(CUipcEventHandle)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuIpcOpenEventHandle) < 0 ||
        rpc_write(conn, phEvent, sizeof(CUevent)) < 0 ||
        rpc_write(conn, &handle, sizeof(CUipcEventHandle)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phEvent, sizeof(CUevent)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuIpcGetMemHandle) < 0 ||
        rpc_write(conn, pHandle, sizeof(CUipcMemHandle)) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pHandle, sizeof(CUipcMemHandle)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcOpenMemHandle_v2(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuIpcOpenMemHandle_v2) < 0 ||
        rpc_write(conn, pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &handle, sizeof(CUipcMemHandle)) < 0 ||
        rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcCloseMemHandle(CUdeviceptr dptr)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuIpcCloseMemHandle) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dst);
    conn_t *src_conn = scuda_rpc_conn_for_deviceptr(src);
    if (conn != src_conn)
        return CUDA_ERROR_NOT_SUPPORTED;
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemcpy) < 0 ||
        rpc_write(conn, &dst, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &src, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemcpyPeer) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &dstContext, sizeof(CUcontext)) < 0 ||
        rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &srcContext, sizeof(CUcontext)) < 0 ||
        rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dstDevice);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemcpyHtoD_v2) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(conn, srcHost, ByteCount) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(srcDevice);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemcpyDtoH_v2) < 0 ||
        rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    scuda_prepare_host_range_write(dstHost, ByteCount);
    if (rpc_read(conn, dstHost, ByteCount) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value == CUDA_SUCCESS)
        scuda_mark_host_range_clean(dstHost, ByteCount);
    return return_value;
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dstDevice);
    conn_t *src_conn = scuda_rpc_conn_for_deviceptr(srcDevice);
    if (conn != src_conn)
        return CUDA_ERROR_NOT_SUPPORTED;
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemcpyDtoD_v2) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(srcDevice);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemcpyDtoA_v2) < 0 ||
        rpc_write(conn, &dstArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &dstOffset, sizeof(size_t)) < 0 ||
        rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dstDevice);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemcpyAtoD_v2) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &srcArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &srcOffset, sizeof(size_t)) < 0 ||
        rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyAtoH_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemcpyAtoH_v2) < 0 ||
        rpc_write(conn, &srcArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &srcOffset, sizeof(size_t)) < 0 ||
        rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, dstHost, ByteCount) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemcpyAtoA_v2) < 0 ||
        rpc_write(conn, &dstArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &dstOffset, sizeof(size_t)) < 0 ||
        rpc_write(conn, &srcArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &srcOffset, sizeof(size_t)) < 0 ||
        rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemcpyPeerAsync) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &dstContext, sizeof(CUcontext)) < 0 ||
        rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &srcContext, sizeof(CUcontext)) < 0 ||
        rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dstDevice);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemcpyHtoDAsync_v2) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(conn, srcHost, ByteCount) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dstDevice);
    conn_t *src_conn = scuda_rpc_conn_for_deviceptr(srcDevice);
    if (conn != src_conn)
        return CUDA_ERROR_NOT_SUPPORTED;
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemcpyDtoDAsync_v2) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dstDevice);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemsetD8_v2) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &uc, sizeof(unsigned char)) < 0 ||
        rpc_write(conn, &N, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dstDevice);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemsetD16_v2) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &us, sizeof(unsigned short)) < 0 ||
        rpc_write(conn, &N, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dstDevice);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemsetD32_v2) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &ui, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &N, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dstDevice);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemsetD2D8_v2) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(conn, &uc, sizeof(unsigned char)) < 0 ||
        rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
        rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dstDevice);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemsetD2D16_v2) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(conn, &us, sizeof(unsigned short)) < 0 ||
        rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
        rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height)
{
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dstDevice);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemsetD2D32_v2) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(conn, &ui, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
        rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

#ifdef cuMemsetD8
#undef cuMemsetD8
#endif
extern "C" CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    return cuMemsetD8_v2(dstDevice, uc, N);
}

#ifdef cuMemsetD16
#undef cuMemsetD16
#endif
extern "C" CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    return cuMemsetD16_v2(dstDevice, us, N);
}

#ifdef cuMemsetD32
#undef cuMemsetD32
#endif
extern "C" CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    return cuMemsetD32_v2(dstDevice, ui, N);
}

#ifdef cuMemsetD2D8
#undef cuMemsetD2D8
#endif
extern "C" CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height)
{
    return cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height);
}

#ifdef cuMemsetD2D16
#undef cuMemsetD2D16
#endif
extern "C" CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
{
    return cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height);
}

#ifdef cuMemsetD2D32
#undef cuMemsetD2D32
#endif
extern "C" CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height)
{
    return cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height);
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemsetD8Async) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &uc, sizeof(unsigned char)) < 0 ||
        rpc_write(conn, &N, sizeof(size_t)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemsetD16Async) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &us, sizeof(unsigned short)) < 0 ||
        rpc_write(conn, &N, sizeof(size_t)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemsetD32Async) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &ui, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &N, sizeof(size_t)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemsetD2D8Async) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(conn, &uc, sizeof(unsigned char)) < 0 ||
        rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
        rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemsetD2D16Async) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(conn, &us, sizeof(unsigned short)) < 0 ||
        rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
        rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemsetD2D32Async) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(conn, &ui, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
        rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayCreate_v2(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray)
{
    return scuda_cuArrayCreate_v2_safe(pHandle, pAllocateArray);
}

CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuArrayGetDescriptor_v2) < 0 ||
        rpc_write(conn, pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0 ||
        rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuArrayGetSparseProperties) < 0 ||
        rpc_write(conn, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_write(conn, &array, sizeof(CUarray)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMipmappedArrayGetSparseProperties) < 0 ||
        rpc_write(conn, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_write(conn, &mipmap, sizeof(CUmipmappedArray)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuArrayGetMemoryRequirements) < 0 ||
        rpc_write(conn, memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_write(conn, &array, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMipmappedArrayGetMemoryRequirements) < 0 ||
        rpc_write(conn, memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_write(conn, &mipmap, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuArrayGetPlane) < 0 ||
        rpc_write(conn, pPlaneArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &planeIdx, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pPlaneArray, sizeof(CUarray)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayDestroy(CUarray hArray)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuArrayDestroy) < 0 ||
        rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArray3DCreate_v2(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray)
{
    return scuda_cuArray3DCreate_v2_safe(pHandle, pAllocateArray);
}

CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuArray3DGetDescriptor_v2) < 0 ||
        rpc_write(conn, pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
        rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMipmappedArrayCreate) < 0 ||
        rpc_write(conn, pHandle, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(conn, &pMipmappedArrayDesc, sizeof(const CUDA_ARRAY3D_DESCRIPTOR*)) < 0 ||
        rpc_write(conn, &numMipmapLevels, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pHandle, sizeof(CUmipmappedArray)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMipmappedArrayGetLevel) < 0 ||
        rpc_write(conn, pLevelArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(conn, &level, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pLevelArray, sizeof(CUarray)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMipmappedArrayDestroy) < 0 ||
        rpc_write(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemAddressReserve) < 0 ||
        rpc_write(conn, ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &size, sizeof(size_t)) < 0 ||
        rpc_write(conn, &alignment, sizeof(size_t)) < 0 ||
        rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemAddressFree) < 0 ||
        rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &size, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags)
{
    if (handle == nullptr || prop == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemCreate) < 0 ||
        rpc_write(conn, &size, sizeof(size_t)) < 0 ||
        rpc_write(conn, prop, sizeof(CUmemAllocationProp)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemRelease) < 0 ||
        rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemMap) < 0 ||
        rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &size, sizeof(size_t)) < 0 ||
        rpc_write(conn, &offset, sizeof(size_t)) < 0 ||
        rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemMapArrayAsync) < 0 ||
        rpc_write(conn, mapInfoList, sizeof(CUarrayMapInfo)) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, mapInfoList, sizeof(CUarrayMapInfo)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemUnmap) < 0 ||
        rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &size, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count)
{
    if (count != 0 && desc == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemSetAccess) < 0 ||
        rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &size, sizeof(size_t)) < 0 ||
        rpc_write(conn, &count, sizeof(size_t)) < 0 ||
        (count != 0 && rpc_write(conn, desc, count * sizeof(CUmemAccessDesc)) < 0) ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr)
{
    if (flags == nullptr || location == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemGetAccess) < 0 ||
        rpc_write(conn, location, sizeof(CUmemLocation)) < 0 ||
        rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, flags, sizeof(unsigned long long)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option)
{
    if (granularity == nullptr || prop == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemGetAllocationGranularity) < 0 ||
        rpc_write(conn, prop, sizeof(CUmemAllocationProp)) < 0 ||
        rpc_write(conn, &option, sizeof(CUmemAllocationGranularity_flags)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, granularity, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemGetAllocationPropertiesFromHandle) < 0 ||
        rpc_write(conn, prop, sizeof(CUmemAllocationProp)) < 0 ||
        rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, prop, sizeof(CUmemAllocationProp)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemFreeAsync) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemAllocAsync) < 0 ||
        rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &bytesize, sizeof(size_t)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemPoolTrimTo) < 0 ||
        rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(conn, &minBytesToKeep, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemPoolSetAccess) < 0 ||
        rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(conn, &map, sizeof(const CUmemAccessDesc*)) < 0 ||
        rpc_write(conn, &count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemPoolGetAccess) < 0 ||
        rpc_write(conn, flags, sizeof(CUmemAccess_flags)) < 0 ||
        rpc_write(conn, &memPool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(conn, location, sizeof(CUmemLocation)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, flags, sizeof(CUmemAccess_flags)) < 0 ||
        rpc_read(conn, location, sizeof(CUmemLocation)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemPoolCreate) < 0 ||
        rpc_write(conn, pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(conn, &poolProps, sizeof(const CUmemPoolProps*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolDestroy(CUmemoryPool pool)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemPoolDestroy) < 0 ||
        rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemAllocFromPoolAsync) < 0 ||
        rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &bytesize, sizeof(size_t)) < 0 ||
        rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemPoolExportPointer) < 0 ||
        rpc_write(conn, shareData_out, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, shareData_out, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemPoolImportPointer) < 0 ||
        rpc_write(conn, ptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(conn, shareData, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, ptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, shareData, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuMemRangeGetAttributes) < 0 ||
        rpc_write(conn, data, sizeof(void*)) < 0 ||
        rpc_write(conn, dataSizes, sizeof(size_t)) < 0 ||
        rpc_write(conn, attributes, sizeof(CUmem_range_attribute)) < 0 ||
        rpc_write(conn, &numAttributes, sizeof(size_t)) < 0 ||
        rpc_write(conn, &devPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, data, sizeof(void*)) < 0 ||
        rpc_read(conn, dataSizes, sizeof(size_t)) < 0 ||
        rpc_read(conn, attributes, sizeof(CUmem_range_attribute)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuPointerSetAttribute) < 0 ||
        rpc_write(conn, &value, sizeof(const void*)) < 0 ||
        rpc_write(conn, &attribute, sizeof(CUpointer_attribute)) < 0 ||
        rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

extern "C" CUresult scuda_cuPointerGetAttributes_safe(
    unsigned int numAttributes, CUpointer_attribute *attributes, void **data,
    CUdeviceptr ptr);

CUresult cuPointerGetAttributes(unsigned int numAttributes,
                                CUpointer_attribute *attributes, void **data,
                                CUdeviceptr ptr) {
    return scuda_cuPointerGetAttributes_safe(numAttributes, attributes, data,
                                             ptr);
}

CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags)
{
    conn_t *conn = scuda_rpc_conn_for_current_context();
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuStreamCreate) < 0 ||
        rpc_write(conn, phStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phStream, sizeof(CUstream)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value == CUDA_SUCCESS && phStream != nullptr)
        scuda_note_stream_owner(*phStream, conn);
    return return_value;
}

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority)
{
    conn_t *conn = scuda_rpc_conn_for_current_context();
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuStreamCreateWithPriority) < 0 ||
        rpc_write(conn, phStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &priority, sizeof(int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phStream, sizeof(CUstream)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value == CUDA_SUCCESS && phStream != nullptr)
        scuda_note_stream_owner(*phStream, conn);
    return return_value;
}

CUresult cuStreamGetPriority(CUstream hStream, int* priority)
{
    conn_t *conn = scuda_rpc_conn_for_stream(hStream);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuStreamGetPriority) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, priority, sizeof(int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, priority, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags)
{
    conn_t *conn = scuda_rpc_conn_for_stream(hStream);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuStreamGetFlags) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, flags, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetId(CUstream hStream, unsigned long long* streamId)
{
    conn_t *conn = scuda_rpc_conn_for_stream(hStream);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuStreamGetId) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, streamId, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, streamId, sizeof(unsigned long long)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx)
{
    conn_t *conn = scuda_rpc_conn_for_stream(hStream);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuStreamGetCtx) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, pctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pctx, sizeof(CUcontext)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value == CUDA_SUCCESS && pctx != nullptr)
        scuda_note_context_owner(*pctx, conn);
    return return_value;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags)
{
    conn_t *conn = hStream != nullptr ? scuda_rpc_conn_for_stream(hStream) : scuda_rpc_conn_for_event(hEvent);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuStreamWaitEvent) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
        rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuStreamBeginCapture_v2) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &mode, sizeof(CUstreamCaptureMode)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuThreadExchangeStreamCaptureMode) < 0 ||
        rpc_write(conn, mode, sizeof(CUstreamCaptureMode)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, mode, sizeof(CUstreamCaptureMode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    CUgraph graph_tmp = nullptr;
    CUgraph *graph_out = phGraph == nullptr ? &graph_tmp : phGraph;
    if (rpc_write_start_request(conn, RPC_cuStreamEndCapture) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, graph_out, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, graph_out, sizeof(CUgraph)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuStreamIsCapturing) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuStreamAttachMemAsync) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &length, sizeof(size_t)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamQuery(CUstream hStream)
{
    conn_t *conn = scuda_rpc_conn_for_stream(hStream);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuStreamQuery) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamDestroy_v2(CUstream hStream)
{
    conn_t *conn = scuda_rpc_conn_for_stream(hStream);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuStreamDestroy_v2) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamCopyAttributes(CUstream dst, CUstream src)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuStreamCopyAttributes) < 0 ||
        rpc_write(conn, &dst, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &src, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out)
{
    if (value_out == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuStreamGetAttribute) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &attr, sizeof(CUstreamAttrID)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, value_out, sizeof(CUstreamAttrValue)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value)
{
    if (value == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuStreamSetAttribute) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &attr, sizeof(CUstreamAttrID)) < 0 ||
        rpc_write(conn, value, sizeof(CUstreamAttrValue)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags)
{
    conn_t *conn = scuda_rpc_conn_for_current_context();
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuEventCreate) < 0 ||
        rpc_write(conn, phEvent, sizeof(CUevent)) < 0 ||
        rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phEvent, sizeof(CUevent)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value == CUDA_SUCCESS && phEvent != nullptr)
        scuda_note_event_owner(*phEvent, conn);
    return return_value;
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream)
{
    conn_t *conn = hStream != nullptr ? scuda_rpc_conn_for_stream(hStream) : scuda_rpc_conn_for_event(hEvent);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuEventRecord) < 0 ||
        rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags)
{
    conn_t *conn = hStream != nullptr ? scuda_rpc_conn_for_stream(hStream) : scuda_rpc_conn_for_event(hEvent);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuEventRecordWithFlags) < 0 ||
        rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventQuery(CUevent hEvent)
{
    conn_t *conn = scuda_rpc_conn_for_event(hEvent);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuEventQuery) < 0 ||
        rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventDestroy_v2(CUevent hEvent)
{
    conn_t *conn = scuda_rpc_conn_for_event(hEvent);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuEventDestroy_v2) < 0 ||
        rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventElapsedTime_v2(float* pMilliseconds, CUevent hStart, CUevent hEnd)
{
    conn_t *conn = scuda_rpc_conn_for_event(hStart);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuEventElapsedTime_v2) < 0 ||
        rpc_write(conn, pMilliseconds, sizeof(float)) < 0 ||
        rpc_write(conn, &hStart, sizeof(CUevent)) < 0 ||
        rpc_write(conn, &hEnd, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pMilliseconds, sizeof(float)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuImportExternalMemory) < 0 ||
        rpc_write(conn, extMem_out, sizeof(CUexternalMemory)) < 0 ||
        rpc_write(conn, &memHandleDesc, sizeof(const CUDA_EXTERNAL_MEMORY_HANDLE_DESC*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, extMem_out, sizeof(CUexternalMemory)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuExternalMemoryGetMappedBuffer) < 0 ||
        rpc_write(conn, devPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &extMem, sizeof(CUexternalMemory)) < 0 ||
        rpc_write(conn, &bufferDesc, sizeof(const CUDA_EXTERNAL_MEMORY_BUFFER_DESC*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, devPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuExternalMemoryGetMappedMipmappedArray) < 0 ||
        rpc_write(conn, mipmap, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(conn, &extMem, sizeof(CUexternalMemory)) < 0 ||
        rpc_write(conn, &mipmapDesc, sizeof(const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, mipmap, sizeof(CUmipmappedArray)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDestroyExternalMemory(CUexternalMemory extMem)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuDestroyExternalMemory) < 0 ||
        rpc_write(conn, &extMem, sizeof(CUexternalMemory)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuImportExternalSemaphore) < 0 ||
        rpc_write(conn, extSem_out, sizeof(CUexternalSemaphore)) < 0 ||
        rpc_write(conn, &semHandleDesc, sizeof(const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, extSem_out, sizeof(CUexternalSemaphore)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuSignalExternalSemaphoresAsync) < 0 ||
        rpc_write(conn, &extSemArray, sizeof(const CUexternalSemaphore*)) < 0 ||
        rpc_write(conn, &paramsArray, sizeof(const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS*)) < 0 ||
        rpc_write(conn, &numExtSems, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuWaitExternalSemaphoresAsync) < 0 ||
        rpc_write(conn, &extSemArray, sizeof(const CUexternalSemaphore*)) < 0 ||
        rpc_write(conn, &paramsArray, sizeof(const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS*)) < 0 ||
        rpc_write(conn, &numExtSems, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuDestroyExternalSemaphore) < 0 ||
        rpc_write(conn, &extSem, sizeof(CUexternalSemaphore)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuStreamWaitValue32_v2) < 0 ||
        rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &value, sizeof(cuuint32_t)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuStreamWaitValue64_v2) < 0 ||
        rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &value, sizeof(cuuint64_t)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuStreamWriteValue32_v2) < 0 ||
        rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &value, sizeof(cuuint32_t)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuStreamWriteValue64_v2) < 0 ||
        rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &value, sizeof(cuuint64_t)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuStreamBatchMemOp_v2) < 0 ||
        rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, paramArray, sizeof(CUstreamBatchMemOpParams)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, paramArray, sizeof(CUstreamBatchMemOpParams)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc)
{
    conn_t *conn = scuda_rpc_conn_for_function(hfunc);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuFuncGetAttribute) < 0 ||
        rpc_write(conn, pi, sizeof(int)) < 0 ||
        rpc_write(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pi, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value)
{
    hfunc = scuda_translate_private_function_for_rpc(hfunc);
    conn_t *conn = scuda_rpc_conn_for_function(hfunc);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuFuncSetAttribute) < 0 ||
        rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(conn, &value, sizeof(int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    conn_t *conn = scuda_rpc_conn_for_function(hfunc);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuFuncSetCacheConfig) < 0 ||
        rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &config, sizeof(CUfunc_cache)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncGetModule(CUmodule* hmod, CUfunction hfunc)
{
    conn_t *conn = scuda_rpc_conn_for_function(hfunc);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuFuncGetModule) < 0 ||
        rpc_write(conn, hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, hmod, sizeof(CUmodule)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value == CUDA_SUCCESS && hmod != nullptr)
        scuda_note_module_owner(*hmod, conn);
    return return_value;
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams)
{
    struct scuda_kernel_param_layout {
        uint32_t count = 0;
        size_t offsets[64] = {};
        size_t sizes[64] = {};
    };
    constexpr int SCUDA_RPC_cuFuncGetParamLayout = 1000001;
    extern CUresult scuda_sync_mapped_host_to_device_for_launch(
        unsigned char *packed, const size_t *offsets, const size_t *sizes,
        uint32_t count);

    conn_t *conn = rpc_client_get_connection(0);
    scuda_kernel_param_layout layout;
    CUresult return_value;
    if (rpc_write_start_request(conn, SCUDA_RPC_cuFuncGetParamLayout) < 0 ||
        rpc_write(conn, &f, sizeof(f)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &layout.count, sizeof(layout.count)) < 0 ||
        rpc_read(conn, layout.offsets, sizeof(layout.offsets)) < 0 ||
        rpc_read(conn, layout.sizes, sizeof(layout.sizes)) < 0 ||
        rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (return_value != CUDA_SUCCESS)
        return return_value;
    if (layout.count > 64)
        return CUDA_ERROR_NOT_SUPPORTED;

    size_t total_size = 0;
    for (uint32_t i = 0; i < layout.count; ++i) {
        if (kernelParams == nullptr || kernelParams[i] == nullptr)
            return CUDA_ERROR_INVALID_VALUE;
        total_size = std::max(total_size, layout.offsets[i] + layout.sizes[i]);
    }
    std::vector<unsigned char> packed(total_size);
    for (uint32_t i = 0; i < layout.count; ++i)
        memcpy(packed.data() + layout.offsets[i], kernelParams[i], layout.sizes[i]);

    return_value = scuda_sync_mapped_host_to_device_for_launch(
        packed.data(), layout.offsets, layout.sizes, layout.count);
    if (return_value != CUDA_SUCCESS)
        return return_value;

    if (rpc_write_start_request(conn, RPC_cuLaunchCooperativeKernel) < 0 ||
        rpc_write(conn, &f, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &gridDimX, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &gridDimY, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &gridDimZ, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &blockDimX, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &blockDimY, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &blockDimZ, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &sharedMemBytes, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(conn, &layout.count, sizeof(layout.count)) < 0 ||
        rpc_write(conn, &total_size, sizeof(total_size)) < 0 ||
        rpc_write(conn, packed.data(), packed.size()) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuLaunchCooperativeKernelMultiDevice) < 0 ||
        rpc_write(conn, launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0 ||
        rpc_write(conn, &numDevices, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuFuncSetBlockShape) < 0 ||
        rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &x, sizeof(int)) < 0 ||
        rpc_write(conn, &y, sizeof(int)) < 0 ||
        rpc_write(conn, &z, sizeof(int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuFuncSetSharedSize) < 0 ||
        rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &bytes, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuParamSetSize) < 0 ||
        rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &numbytes, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuParamSeti) < 0 ||
        rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &offset, sizeof(int)) < 0 ||
        rpc_write(conn, &value, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSetf(CUfunction hfunc, int offset, float value)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuParamSetf) < 0 ||
        rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &offset, sizeof(int)) < 0 ||
        rpc_write(conn, &value, sizeof(float)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunch(CUfunction f)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuLaunch) < 0 ||
        rpc_write(conn, &f, sizeof(CUfunction)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuLaunchGrid) < 0 ||
        rpc_write(conn, &f, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &grid_width, sizeof(int)) < 0 ||
        rpc_write(conn, &grid_height, sizeof(int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuLaunchGridAsync) < 0 ||
        rpc_write(conn, &f, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &grid_width, sizeof(int)) < 0 ||
        rpc_write(conn, &grid_height, sizeof(int)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuParamSetTexRef) < 0 ||
        rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &texunit, sizeof(int)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuFuncSetSharedMemConfig) < 0 ||
        rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &config, sizeof(CUsharedconfig)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphCreate) < 0 ||
        rpc_write(conn, phGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraph, sizeof(CUgraph)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphKernelNodeGetParams_v2) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, nodeParams, sizeof(CUDA_KERNEL_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, nodeParams, sizeof(CUDA_KERNEL_NODE_PARAMS)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphKernelNodeSetParams_v2) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_KERNEL_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphMemcpyNodeGetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, nodeParams, sizeof(CUDA_MEMCPY3D)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, nodeParams, sizeof(CUDA_MEMCPY3D)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphMemcpyNodeSetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_MEMCPY3D*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphMemsetNodeGetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphMemsetNodeSetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_MEMSET_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphHostNodeGetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, nodeParams, sizeof(CUDA_HOST_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, nodeParams, sizeof(CUDA_HOST_NODE_PARAMS)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphHostNodeSetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_HOST_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphAddChildGraphNode) < 0 ||
        rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(conn, &childGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphChildGraphNodeGetGraph) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, phGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraph, sizeof(CUgraph)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphAddEmptyNode) < 0 ||
        rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphAddEventRecordNode) < 0 ||
        rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphEventRecordNodeGetEvent) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, event_out, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, event_out, sizeof(CUevent)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphEventRecordNodeSetEvent) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphAddEventWaitNode) < 0 ||
        rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphEventWaitNodeGetEvent) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, event_out, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, event_out, sizeof(CUevent)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphEventWaitNodeSetEvent) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphAddExternalSemaphoresSignalNode) < 0 ||
        rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExternalSemaphoresSignalNodeGetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, params_out, sizeof(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, params_out, sizeof(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExternalSemaphoresSignalNodeSetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphAddExternalSemaphoresWaitNode) < 0 ||
        rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_EXT_SEM_WAIT_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExternalSemaphoresWaitNodeGetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, params_out, sizeof(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, params_out, sizeof(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExternalSemaphoresWaitNodeSetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_EXT_SEM_WAIT_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddBatchMemOpNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphAddBatchMemOpNode) < 0 ||
        rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_BATCH_MEM_OP_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphBatchMemOpNodeGetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, nodeParams_out, sizeof(CUDA_BATCH_MEM_OP_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, nodeParams_out, sizeof(CUDA_BATCH_MEM_OP_NODE_PARAMS)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphBatchMemOpNodeSetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_BATCH_MEM_OP_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExecBatchMemOpNodeSetParams) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_BATCH_MEM_OP_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams)
{
    if (phGraphNode == nullptr || nodeParams == nullptr || (numDependencies != 0 && dependencies == nullptr))
        return CUDA_ERROR_INVALID_VALUE;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphAddMemAllocNode) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
        (numDependencies != 0 && rpc_write(conn, dependencies, numDependencies * sizeof(CUgraphNode)) < 0) ||
        rpc_write(conn, nodeParams, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_read(conn, nodeParams, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphMemAllocNodeGetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUdeviceptr dptr)
{
    if (phGraphNode == nullptr || (numDependencies != 0 && dependencies == nullptr))
        return CUDA_ERROR_INVALID_VALUE;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphAddMemFreeNode) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
        (numDependencies != 0 && rpc_write(conn, dependencies, numDependencies * sizeof(CUgraphNode)) < 0) ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphMemFreeNodeGetParams) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, dptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, dptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGraphMemTrim(CUdevice device)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuDeviceGraphMemTrim) < 0 ||
        rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphClone) < 0 ||
        rpc_write(conn, phGraphClone, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &originalGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraphClone, sizeof(CUgraph)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphNodeFindInClone) < 0 ||
        rpc_write(conn, phNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &hOriginalNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &hClonedGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phNode, sizeof(CUgraphNode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* type)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphNodeGetType) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, type, sizeof(CUgraphNodeType)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, type, sizeof(CUgraphNodeType)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphGetRootNodes) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, rootNodes, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, numRootNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, rootNodes, sizeof(CUgraphNode)) < 0 ||
        rpc_read(conn, numRootNodes, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphDestroyNode(CUgraphNode hNode)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphDestroyNode) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphInstantiateWithFlags) < 0 ||
        rpc_write(conn, phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphInstantiateWithParams(CUgraphExec* phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS* instantiateParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphInstantiateWithParams) < 0 ||
        rpc_write(conn, phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, instantiateParams, sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_read(conn, instantiateParams, sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t* flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExecGetFlags) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, flags, sizeof(cuuint64_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, flags, sizeof(cuuint64_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams)
{
    if (nodeParams == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    scuda_kernel_param_layout layout;
    std::vector<unsigned char> packed;
    CUresult status =
        scuda_pack_kernel_params_for_codegen(nodeParams, &layout, &packed);
    if (status != CUDA_SUCCESS)
        return status;
    CUDA_KERNEL_NODE_PARAMS serial_params = *nodeParams;
    serial_params.kernelParams = nullptr;
    serial_params.extra = nullptr;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    size_t packed_size = packed.size();
    if (rpc_write_start_request(conn, RPC_cuGraphExecKernelNodeSetParams_v2) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &serial_params, sizeof(serial_params)) < 0 ||
        rpc_write(conn, &layout.count, sizeof(layout.count)) < 0 ||
        rpc_write(conn, &packed_size, sizeof(packed_size)) < 0 ||
        (packed_size != 0 && rpc_write(conn, packed.data(), packed_size) < 0) ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExecMemcpyNodeSetParams) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &copyParams, sizeof(const CUDA_MEMCPY3D*)) < 0 ||
        rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExecMemsetNodeSetParams) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &memsetParams, sizeof(const CUDA_MEMSET_NODE_PARAMS*)) < 0 ||
        rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExecHostNodeSetParams) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_HOST_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExecChildGraphNodeSetParams) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &childGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExecEventRecordNodeSetEvent) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExecEventWaitNodeSetEvent) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExecExternalSemaphoresSignalNodeSetParams) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExecExternalSemaphoresWaitNodeSetParams) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(const CUDA_EXT_SEM_WAIT_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphNodeSetEnabled) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphNodeGetEnabled) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphUpload) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphLaunch) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecDestroy(CUgraphExec hGraphExec)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExecDestroy) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphDestroy(CUgraph hGraph)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphDestroy) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo* resultInfo)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphExecUpdate_v2) < 0 ||
        rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphKernelNodeCopyAttributes) < 0 ||
        rpc_write(conn, &dst, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &src, sizeof(CUgraphNode)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphKernelNodeGetAttribute) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &attr, sizeof(CUkernelNodeAttrID)) < 0 ||
        rpc_write(conn, value_out, sizeof(CUkernelNodeAttrValue)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, value_out, sizeof(CUkernelNodeAttrValue)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphKernelNodeSetAttribute) < 0 ||
        rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &attr, sizeof(CUkernelNodeAttrID)) < 0 ||
        rpc_write(conn, &value, sizeof(const CUkernelNodeAttrValue*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphDebugDotPrint) < 0 ||
        rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &path, sizeof(const char*)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuUserObjectRetain(CUuserObject object, unsigned int count)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuUserObjectRetain) < 0 ||
        rpc_write(conn, &object, sizeof(CUuserObject)) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuUserObjectRelease(CUuserObject object, unsigned int count)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuUserObjectRelease) < 0 ||
        rpc_write(conn, &object, sizeof(CUuserObject)) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphRetainUserObject) < 0 ||
        rpc_write(conn, &graph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &object, sizeof(CUuserObject)) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphReleaseUserObject) < 0 ||
        rpc_write(conn, &graph, sizeof(CUgraph)) < 0 ||
        rpc_write(conn, &object, sizeof(CUuserObject)) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor) < 0 ||
        rpc_write(conn, numBlocks, sizeof(int)) < 0 ||
        rpc_write(conn, &func, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &blockSize, sizeof(int)) < 0 ||
        rpc_write(conn, &dynamicSMemSize, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, numBlocks, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) < 0 ||
        rpc_write(conn, numBlocks, sizeof(int)) < 0 ||
        rpc_write(conn, &func, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &blockSize, sizeof(int)) < 0 ||
        rpc_write(conn, &dynamicSMemSize, sizeof(size_t)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, numBlocks, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuOccupancyAvailableDynamicSMemPerBlock) < 0 ||
        rpc_write(conn, dynamicSmemSize, sizeof(size_t)) < 0 ||
        rpc_write(conn, &func, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &numBlocks, sizeof(int)) < 0 ||
        rpc_write(conn, &blockSize, sizeof(int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, dynamicSmemSize, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyMaxPotentialClusterSize(int* clusterSize, CUfunction func, const CUlaunchConfig* config)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuOccupancyMaxPotentialClusterSize) < 0 ||
        rpc_write(conn, clusterSize, sizeof(int)) < 0 ||
        rpc_write(conn, &func, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &config, sizeof(const CUlaunchConfig*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, clusterSize, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyMaxActiveClusters(int* numClusters, CUfunction func, const CUlaunchConfig* config)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuOccupancyMaxActiveClusters) < 0 ||
        rpc_write(conn, numClusters, sizeof(int)) < 0 ||
        rpc_write(conn, &func, sizeof(CUfunction)) < 0 ||
        rpc_write(conn, &config, sizeof(const CUlaunchConfig*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, numClusters, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetArray) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetMipmappedArray) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetAddress_v2(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetAddress_v2) < 0 ||
        rpc_write(conn, ByteOffset, sizeof(size_t)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &bytes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, ByteOffset, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetAddress2D_v3) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &desc, sizeof(const CUDA_ARRAY_DESCRIPTOR*)) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &Pitch, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetFormat) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &fmt, sizeof(CUarray_format)) < 0 ||
        rpc_write(conn, &NumPackedComponents, sizeof(int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetAddressMode) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &dim, sizeof(int)) < 0 ||
        rpc_write(conn, &am, sizeof(CUaddress_mode)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetFilterMode) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &fm, sizeof(CUfilter_mode)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetMipmapFilterMode) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &fm, sizeof(CUfilter_mode)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetMipmapLevelBias) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &bias, sizeof(float)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetMipmapLevelClamp) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &minMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_write(conn, &maxMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetMaxAnisotropy) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &maxAniso, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetBorderColor) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, pBorderColor, sizeof(float)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pBorderColor, sizeof(float)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefSetFlags) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetAddress_v2(CUdeviceptr* pdptr, CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefGetAddress_v2) < 0 ||
        rpc_write(conn, pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefGetArray) < 0 ||
        rpc_write(conn, phArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phArray, sizeof(CUarray)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefGetMipmappedArray) < 0 ||
        rpc_write(conn, phMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefGetAddressMode) < 0 ||
        rpc_write(conn, pam, sizeof(CUaddress_mode)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(conn, &dim, sizeof(int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pam, sizeof(CUaddress_mode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefGetFilterMode) < 0 ||
        rpc_write(conn, pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefGetFormat) < 0 ||
        rpc_write(conn, pFormat, sizeof(CUarray_format)) < 0 ||
        rpc_write(conn, pNumChannels, sizeof(int)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pFormat, sizeof(CUarray_format)) < 0 ||
        rpc_read(conn, pNumChannels, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefGetMipmapFilterMode) < 0 ||
        rpc_write(conn, pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefGetMipmapLevelBias) < 0 ||
        rpc_write(conn, pbias, sizeof(float)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pbias, sizeof(float)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefGetMipmapLevelClamp) < 0 ||
        rpc_write(conn, pminMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_write(conn, pmaxMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pminMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_read(conn, pmaxMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefGetMaxAnisotropy) < 0 ||
        rpc_write(conn, pmaxAniso, sizeof(int)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pmaxAniso, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefGetBorderColor) < 0 ||
        rpc_write(conn, pBorderColor, sizeof(float)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pBorderColor, sizeof(float)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefGetFlags) < 0 ||
        rpc_write(conn, pFlags, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pFlags, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefCreate(CUtexref* pTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefCreate) < 0 ||
        rpc_write(conn, pTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pTexRef, sizeof(CUtexref)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefDestroy(CUtexref hTexRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexRefDestroy) < 0 ||
        rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuSurfRefSetArray) < 0 ||
        rpc_write(conn, &hSurfRef, sizeof(CUsurfref)) < 0 ||
        rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuSurfRefGetArray) < 0 ||
        rpc_write(conn, phArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &hSurfRef, sizeof(CUsurfref)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, phArray, sizeof(CUarray)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc)
{
    if (pTexObject == nullptr || pResDesc == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    uint32_t has_tex_desc = pTexDesc != nullptr ? 1 : 0;
    uint32_t has_res_view_desc = pResViewDesc != nullptr ? 1 : 0;
    CUDA_TEXTURE_DESC tex_desc = {};
    CUDA_RESOURCE_VIEW_DESC res_view_desc = {};
    if (pTexDesc != nullptr)
        tex_desc = *pTexDesc;
    if (pResViewDesc != nullptr)
        res_view_desc = *pResViewDesc;
    if (rpc_write_start_request(conn, RPC_cuTexObjectCreate) < 0 ||
        rpc_write(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_write(conn, &has_tex_desc, sizeof(uint32_t)) < 0 ||
        (has_tex_desc != 0 && rpc_write(conn, &tex_desc, sizeof(CUDA_TEXTURE_DESC)) < 0) ||
        rpc_write(conn, &has_res_view_desc, sizeof(uint32_t)) < 0 ||
        (has_res_view_desc != 0 && rpc_write(conn, &res_view_desc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0) ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pTexObject, sizeof(CUtexObject)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectDestroy(CUtexObject texObject)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexObjectDestroy) < 0 ||
        rpc_write(conn, &texObject, sizeof(CUtexObject)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexObjectGetResourceDesc) < 0 ||
        rpc_write(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_write(conn, &texObject, sizeof(CUtexObject)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexObjectGetTextureDesc) < 0 ||
        rpc_write(conn, pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0 ||
        rpc_write(conn, &texObject, sizeof(CUtexObject)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuTexObjectGetResourceViewDesc) < 0 ||
        rpc_write(conn, pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0 ||
        rpc_write(conn, &texObject, sizeof(CUtexObject)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc)
{
    if (pSurfObject == nullptr || pResDesc == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuSurfObjectCreate) < 0 ||
        rpc_write(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pSurfObject, sizeof(CUsurfObject)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfObjectDestroy(CUsurfObject surfObject)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuSurfObjectDestroy) < 0 ||
        rpc_write(conn, &surfObject, sizeof(CUsurfObject)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuSurfObjectGetResourceDesc) < 0 ||
        rpc_write(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_write(conn, &surfObject, sizeof(CUsurfObject)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev)
{
    return scuda_cuDeviceCanAccessPeer_multi(canAccessPeer, dev, peerDev);
}

CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxEnablePeerAccess) < 0 ||
        rpc_write(conn, &peerContext, sizeof(CUcontext)) < 0 ||
        rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuCtxDisablePeerAccess) < 0 ||
        rpc_write(conn, &peerContext, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuDeviceGetP2PAttribute) < 0 ||
        rpc_write(conn, value, sizeof(int)) < 0 ||
        rpc_write(conn, &attrib, sizeof(CUdevice_P2PAttribute)) < 0 ||
        rpc_write(conn, &srcDevice, sizeof(CUdevice)) < 0 ||
        rpc_write(conn, &dstDevice, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, value, sizeof(int)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphicsUnregisterResource) < 0 ||
        rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphicsSubResourceGetMappedArray) < 0 ||
        rpc_write(conn, pArray, sizeof(CUarray)) < 0 ||
        rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_write(conn, &arrayIndex, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &mipLevel, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pArray, sizeof(CUarray)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphicsResourceGetMappedMipmappedArray) < 0 ||
        rpc_write(conn, pMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphicsResourceGetMappedPointer_v2) < 0 ||
        rpc_write(conn, pDevPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, pSize, sizeof(size_t)) < 0 ||
        rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pDevPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(conn, pSize, sizeof(size_t)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphicsResourceSetMapFlags_v2) < 0 ||
        rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphicsMapResources) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
    conn_t *conn = rpc_client_get_connection(0);
    CUresult return_value;
    if (rpc_write_start_request(conn, RPC_cuGraphicsUnmapResources) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

#ifdef cuDeviceTotalMem
#undef cuDeviceTotalMem
#endif
extern "C" CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev)
{
    return cuDeviceTotalMem_v2(bytes, dev);
}

#ifdef cuDeviceGetUuid
#undef cuDeviceGetUuid
#endif
extern "C" CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev)
{
    return cuDeviceGetUuid_v2(uuid, dev);
}

#ifdef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxRelease
#endif
extern "C" CUresult cuDevicePrimaryCtxRelease(CUdevice dev)
{
    return cuDevicePrimaryCtxRelease_v2(dev);
}

#ifdef cuDevicePrimaryCtxSetFlags
#undef cuDevicePrimaryCtxSetFlags
#endif
extern "C" CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags)
{
    return cuDevicePrimaryCtxSetFlags_v2(dev, flags);
}

#ifdef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxReset
#endif
extern "C" CUresult cuDevicePrimaryCtxReset(CUdevice dev)
{
    return cuDevicePrimaryCtxReset_v2(dev);
}

#ifdef cuCtxDestroy
#undef cuCtxDestroy
#endif
extern "C" CUresult cuCtxDestroy(CUcontext ctx)
{
    return cuCtxDestroy_v2(ctx);
}

#ifdef cuCtxPopCurrent
#undef cuCtxPopCurrent
#endif
extern "C" CUresult cuCtxPopCurrent(CUcontext* pctx)
{
    return cuCtxPopCurrent_v2(pctx);
}

#ifdef cuCtxPushCurrent
#undef cuCtxPushCurrent
#endif
extern "C" CUresult cuCtxPushCurrent(CUcontext ctx)
{
    return cuCtxPushCurrent_v2(ctx);
}

#ifdef cuModuleGetGlobal
#undef cuModuleGetGlobal
#endif
extern "C" CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name)
{
    return cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

#ifdef cuMemAlloc
#undef cuMemAlloc
#endif
extern "C" CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize)
{
    return cuMemAlloc_v2(dptr, bytesize);
}

#ifdef cuMemAllocPitch
#undef cuMemAllocPitch
#endif
extern "C" CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)
{
    return cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

#ifdef cuMemFree
#undef cuMemFree
#endif
extern "C" CUresult cuMemFree(CUdeviceptr dptr)
{
    return cuMemFree_v2(dptr);
}

#ifdef cuMemcpyHtoD
#undef cuMemcpyHtoD
#endif
extern "C" CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount)
{
    return cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

#ifdef cuMemcpyHtoDAsync
#undef cuMemcpyHtoDAsync
#endif
extern "C" CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream)
{
    return cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
}

#ifdef cuMemcpyDtoH
#undef cuMemcpyDtoH
#endif
extern "C" CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    return cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

#ifdef cuMemcpyDtoD
#undef cuMemcpyDtoD
#endif
extern "C" CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
}

#ifdef cuMemcpyDtoDAsync
#undef cuMemcpyDtoDAsync
#endif
extern "C" CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    return cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
}

#ifdef cuStreamBeginCapture
#undef cuStreamBeginCapture
#endif
extern "C" CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode)
{
    return cuStreamBeginCapture_v2(hStream, mode);
}

#ifdef cuGraphExecUpdate
#undef cuGraphExecUpdate
#endif
#if CUDA_VERSION >= 12000
extern "C" CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo* resultInfo)
{
    return cuGraphExecUpdate_v2(hGraphExec, hGraph, resultInfo);
}
#else
extern "C" CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode* hErrorNode_out, CUgraphExecUpdateResult* updateResult_out)
{
    CUgraphExecUpdateResultInfo resultInfo = {};
    CUresult result = cuGraphExecUpdate_v2(hGraphExec, hGraph, &resultInfo);
    if (hErrorNode_out != nullptr)
        *hErrorNode_out = resultInfo.errorNode;
    if (updateResult_out != nullptr)
        *updateResult_out = resultInfo.result;
    return result;
}
#endif

#ifdef cuMemcpy_ptds
#undef cuMemcpy_ptds
#endif
extern "C" CUresult cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    return cuMemcpy(dst, src, ByteCount);
}

#ifdef cuMemcpyPeer_ptds
#undef cuMemcpyPeer_ptds
#endif
extern "C" CUresult cuMemcpyPeer_ptds(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount)
{
    return cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
}

#ifdef cuMemcpyPeerAsync_ptsz
#undef cuMemcpyPeerAsync_ptsz
#endif
extern "C" CUresult cuMemcpyPeerAsync_ptsz(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream)
{
    return cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
}

#ifdef cuMemsetD8Async_ptsz
#undef cuMemsetD8Async_ptsz
#endif
extern "C" CUresult cuMemsetD8Async_ptsz(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    return cuMemsetD8Async(dstDevice, uc, N, hStream);
}

#ifdef cuMemsetD16Async_ptsz
#undef cuMemsetD16Async_ptsz
#endif
extern "C" CUresult cuMemsetD16Async_ptsz(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    return cuMemsetD16Async(dstDevice, us, N, hStream);
}

#ifdef cuMemsetD32Async_ptsz
#undef cuMemsetD32Async_ptsz
#endif
extern "C" CUresult cuMemsetD32Async_ptsz(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    return cuMemsetD32Async(dstDevice, ui, N, hStream);
}

#ifdef cuMemsetD2D8Async_ptsz
#undef cuMemsetD2D8Async_ptsz
#endif
extern "C" CUresult cuMemsetD2D8Async_ptsz(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream)
{
    return cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);
}

#ifdef cuMemsetD2D16Async_ptsz
#undef cuMemsetD2D16Async_ptsz
#endif
extern "C" CUresult cuMemsetD2D16Async_ptsz(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream)
{
    return cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);
}

#ifdef cuMemsetD2D32Async_ptsz
#undef cuMemsetD2D32Async_ptsz
#endif
extern "C" CUresult cuMemsetD2D32Async_ptsz(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream)
{
    return cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);
}

#ifdef cuStreamGetPriority_ptsz
#undef cuStreamGetPriority_ptsz
#endif
extern "C" CUresult cuStreamGetPriority_ptsz(CUstream hStream, int* priority)
{
    return cuStreamGetPriority(hStream, priority);
}

#ifdef cuStreamGetId_ptsz
#undef cuStreamGetId_ptsz
#endif
extern "C" CUresult cuStreamGetId_ptsz(CUstream hStream, unsigned long long* streamId)
{
    return cuStreamGetId(hStream, streamId);
}

#ifdef cuStreamGetFlags_ptsz
#undef cuStreamGetFlags_ptsz
#endif
extern "C" CUresult cuStreamGetFlags_ptsz(CUstream hStream, unsigned int* flags)
{
    return cuStreamGetFlags(hStream, flags);
}

#ifdef cuStreamGetCtx_ptsz
#undef cuStreamGetCtx_ptsz
#endif
extern "C" CUresult cuStreamGetCtx_ptsz(CUstream hStream, CUcontext* pctx)
{
    return cuStreamGetCtx(hStream, pctx);
}

#ifdef cuStreamWaitEvent_ptsz
#undef cuStreamWaitEvent_ptsz
#endif
extern "C" CUresult cuStreamWaitEvent_ptsz(CUstream hStream, CUevent hEvent, unsigned int Flags)
{
    return cuStreamWaitEvent(hStream, hEvent, Flags);
}

#ifdef cuStreamEndCapture_ptsz
#undef cuStreamEndCapture_ptsz
#endif
extern "C" CUresult cuStreamEndCapture_ptsz(CUstream hStream, CUgraph* phGraph)
{
    return cuStreamEndCapture(hStream, phGraph);
}

#ifdef cuStreamIsCapturing_ptsz
#undef cuStreamIsCapturing_ptsz
#endif
extern "C" CUresult cuStreamIsCapturing_ptsz(CUstream hStream, CUstreamCaptureStatus* captureStatus)
{
    return cuStreamIsCapturing(hStream, captureStatus);
}

#ifdef cuStreamAttachMemAsync_ptsz
#undef cuStreamAttachMemAsync_ptsz
#endif
extern "C" CUresult cuStreamAttachMemAsync_ptsz(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags)
{
    return cuStreamAttachMemAsync(hStream, dptr, length, flags);
}

#ifdef cuStreamQuery_ptsz
#undef cuStreamQuery_ptsz
#endif
extern "C" CUresult cuStreamQuery_ptsz(CUstream hStream)
{
    return cuStreamQuery(hStream);
}

#ifdef cuEventRecord_ptsz
#undef cuEventRecord_ptsz
#endif
extern "C" CUresult cuEventRecord_ptsz(CUevent hEvent, CUstream hStream)
{
    return cuEventRecord(hEvent, hStream);
}

#ifdef cuEventRecordWithFlags_ptsz
#undef cuEventRecordWithFlags_ptsz
#endif
extern "C" CUresult cuEventRecordWithFlags_ptsz(CUevent hEvent, CUstream hStream, unsigned int flags)
{
    return cuEventRecordWithFlags(hEvent, hStream, flags);
}

#ifdef cuGraphicsMapResources_ptsz
#undef cuGraphicsMapResources_ptsz
#endif
extern "C" CUresult cuGraphicsMapResources_ptsz(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
    return cuGraphicsMapResources(count, resources, hStream);
}

#ifdef cuGraphicsUnmapResources_ptsz
#undef cuGraphicsUnmapResources_ptsz
#endif
extern "C" CUresult cuGraphicsUnmapResources_ptsz(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
    return cuGraphicsUnmapResources(count, resources, hStream);
}

#ifdef cuLaunchCooperativeKernel_ptsz
#undef cuLaunchCooperativeKernel_ptsz
#endif
extern "C" CUresult cuLaunchCooperativeKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams)
{
    return cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
}

#ifdef cuSignalExternalSemaphoresAsync_ptsz
#undef cuSignalExternalSemaphoresAsync_ptsz
#endif
extern "C" CUresult cuSignalExternalSemaphoresAsync_ptsz(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream)
{
    return cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
}

#ifdef cuWaitExternalSemaphoresAsync_ptsz
#undef cuWaitExternalSemaphoresAsync_ptsz
#endif
extern "C" CUresult cuWaitExternalSemaphoresAsync_ptsz(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream)
{
    return cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
}

#ifdef cuGraphInstantiateWithParams_ptsz
#undef cuGraphInstantiateWithParams_ptsz
#endif
extern "C" CUresult cuGraphInstantiateWithParams_ptsz(CUgraphExec* phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS* instantiateParams)
{
    return cuGraphInstantiateWithParams(phGraphExec, hGraph, instantiateParams);
}

#ifdef cuGraphUpload_ptsz
#undef cuGraphUpload_ptsz
#endif
extern "C" CUresult cuGraphUpload_ptsz(CUgraphExec hGraphExec, CUstream hStream)
{
    return cuGraphUpload(hGraphExec, hStream);
}

#ifdef cuGraphLaunch_ptsz
#undef cuGraphLaunch_ptsz
#endif
extern "C" CUresult cuGraphLaunch_ptsz(CUgraphExec hGraphExec, CUstream hStream)
{
    return cuGraphLaunch(hGraphExec, hStream);
}

#ifdef cuStreamCopyAttributes_ptsz
#undef cuStreamCopyAttributes_ptsz
#endif
extern "C" CUresult cuStreamCopyAttributes_ptsz(CUstream dst, CUstream src)
{
    return cuStreamCopyAttributes(dst, src);
}

#ifdef cuStreamGetAttribute_ptsz
#undef cuStreamGetAttribute_ptsz
#endif
extern "C" CUresult cuStreamGetAttribute_ptsz(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out)
{
    return cuStreamGetAttribute(hStream, attr, value_out);
}

#ifdef cuStreamSetAttribute_ptsz
#undef cuStreamSetAttribute_ptsz
#endif
extern "C" CUresult cuStreamSetAttribute_ptsz(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value)
{
    return cuStreamSetAttribute(hStream, attr, value);
}

#ifdef cuMemMapArrayAsync_ptsz
#undef cuMemMapArrayAsync_ptsz
#endif
extern "C" CUresult cuMemMapArrayAsync_ptsz(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream)
{
    return cuMemMapArrayAsync(mapInfoList, count, hStream);
}

#ifdef cuMemFreeAsync_ptsz
#undef cuMemFreeAsync_ptsz
#endif
extern "C" CUresult cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream)
{
    return cuMemFreeAsync(dptr, hStream);
}

#ifdef cuMemAllocAsync_ptsz
#undef cuMemAllocAsync_ptsz
#endif
extern "C" CUresult cuMemAllocAsync_ptsz(CUdeviceptr* dptr, size_t bytesize, CUstream hStream)
{
    return cuMemAllocAsync(dptr, bytesize, hStream);
}

#ifdef cuMemAllocFromPoolAsync_ptsz
#undef cuMemAllocFromPoolAsync_ptsz
#endif
extern "C" CUresult cuMemAllocFromPoolAsync_ptsz(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream)
{
    return cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream);
}

std::unordered_map<std::string, void *> functionMap = {
    {"cuInit", (void *)cuInit},
    {"cuDriverGetVersion", (void *)cuDriverGetVersion},
    {"cuDeviceGet", (void *)cuDeviceGet},
    {"cuDeviceGetCount", (void *)cuDeviceGetCount},
    {"cuDeviceGetName", (void *)cuDeviceGetName},
    {"cuDeviceGetUuid_v2", (void *)cuDeviceGetUuid_v2},
    {"cuDeviceGetLuid", (void *)cuDeviceGetLuid},
    {"cuDeviceTotalMem_v2", (void *)cuDeviceTotalMem_v2},
    {"cuDeviceGetTexture1DLinearMaxWidth", (void *)cuDeviceGetTexture1DLinearMaxWidth},
    {"cuDeviceGetAttribute", (void *)cuDeviceGetAttribute},
    {"cuDeviceSetMemPool", (void *)cuDeviceSetMemPool},
    {"cuDeviceGetMemPool", (void *)cuDeviceGetMemPool},
    {"cuDeviceGetDefaultMemPool", (void *)cuDeviceGetDefaultMemPool},
    {"cuDeviceGetExecAffinitySupport", (void *)cuDeviceGetExecAffinitySupport},
    {"cuFlushGPUDirectRDMAWrites", (void *)cuFlushGPUDirectRDMAWrites},
    {"cuDeviceGetProperties", (void *)cuDeviceGetProperties},
    {"cuDeviceComputeCapability", (void *)cuDeviceComputeCapability},
    {"cuDevicePrimaryCtxRetain", (void *)cuDevicePrimaryCtxRetain},
    {"cuDevicePrimaryCtxRelease_v2", (void *)cuDevicePrimaryCtxRelease_v2},
    {"cuDevicePrimaryCtxSetFlags_v2", (void *)cuDevicePrimaryCtxSetFlags_v2},
    {"cuDevicePrimaryCtxGetState", (void *)cuDevicePrimaryCtxGetState},
    {"cuDevicePrimaryCtxReset_v2", (void *)cuDevicePrimaryCtxReset_v2},
    {"cuCtxDestroy_v2", (void *)cuCtxDestroy_v2},
    {"cuCtxPushCurrent_v2", (void *)cuCtxPushCurrent_v2},
    {"cuCtxPopCurrent_v2", (void *)cuCtxPopCurrent_v2},
    {"cuCtxSetCurrent", (void *)cuCtxSetCurrent},
    {"cuCtxGetCurrent", (void *)cuCtxGetCurrent},
    {"cuCtxGetDevice", (void *)cuCtxGetDevice},
    {"cuCtxGetFlags", (void *)cuCtxGetFlags},
    {"cuCtxGetId", (void *)cuCtxGetId},
    {"cuCtxSetLimit", (void *)cuCtxSetLimit},
    {"cuCtxGetLimit", (void *)cuCtxGetLimit},
    {"cuCtxGetCacheConfig", (void *)cuCtxGetCacheConfig},
    {"cuCtxSetCacheConfig", (void *)cuCtxSetCacheConfig},
    {"cuCtxGetApiVersion", (void *)cuCtxGetApiVersion},
    {"cuCtxGetStreamPriorityRange", (void *)cuCtxGetStreamPriorityRange},
    {"cuCtxResetPersistingL2Cache", (void *)cuCtxResetPersistingL2Cache},
    {"cuCtxGetExecAffinity", (void *)cuCtxGetExecAffinity},
    {"cuCtxAttach", (void *)cuCtxAttach},
    {"cuCtxDetach", (void *)cuCtxDetach},
    {"cuCtxGetSharedMemConfig", (void *)cuCtxGetSharedMemConfig},
    {"cuCtxSetSharedMemConfig", (void *)cuCtxSetSharedMemConfig},
    {"cuModuleLoad", (void *)cuModuleLoad},
    {"cuModuleUnload", (void *)cuModuleUnload},
    {"cuModuleGetLoadingMode", (void *)cuModuleGetLoadingMode},
    {"cuModuleGetFunction", (void *)cuModuleGetFunction},
    {"cuModuleGetGlobal_v2", (void *)cuModuleGetGlobal_v2},
    {"cuLinkCreate", (void *)cuLinkCreate_v2},
    {"cuLinkCreate_v2", (void *)cuLinkCreate_v2},
    {"cuLinkAddData", (void *)cuLinkAddData_v2},
    {"cuLinkAddData_v2", (void *)cuLinkAddData_v2},
    {"cuLinkAddFile_v2", (void *)cuLinkAddFile_v2},
    {"cuLinkComplete", (void *)cuLinkComplete},
    {"cuLinkDestroy", (void *)cuLinkDestroy},
    {"cuModuleGetTexRef", (void *)cuModuleGetTexRef},
    {"cuModuleGetSurfRef", (void *)cuModuleGetSurfRef},
    {"cuLibraryLoadFromFile", (void *)cuLibraryLoadFromFile},
    {"cuLibraryUnload", (void *)cuLibraryUnload},
    {"cuLibraryGetKernel", (void *)cuLibraryGetKernel},
    {"cuLibraryGetModule", (void *)cuLibraryGetModule},
    {"cuKernelGetFunction", (void *)cuKernelGetFunction},
    {"cuLibraryGetGlobal", (void *)cuLibraryGetGlobal},
    {"cuLibraryGetManaged", (void *)cuLibraryGetManaged},
    {"cuLibraryGetUnifiedFunction", (void *)cuLibraryGetUnifiedFunction},
    {"cuKernelGetAttribute", (void *)cuKernelGetAttribute},
    {"cuKernelSetAttribute", (void *)cuKernelSetAttribute},
    {"cuKernelSetCacheConfig", (void *)cuKernelSetCacheConfig},
    {"cuMemGetInfo_v2", (void *)cuMemGetInfo_v2},
    {"cuMemAlloc_v2", (void *)cuMemAlloc_v2},
    {"cuMemAllocPitch_v2", (void *)cuMemAllocPitch_v2},
    {"cuMemFree_v2", (void *)cuMemFree_v2},
    {"cuMemGetAddressRange_v2", (void *)cuMemGetAddressRange_v2},
    {"cuMemAllocManaged", (void *)cuMemAllocManaged},
    {"cuDeviceGetByPCIBusId", (void *)cuDeviceGetByPCIBusId},
    {"cuDeviceGetPCIBusId", (void *)cuDeviceGetPCIBusId},
    {"cuIpcGetEventHandle", (void *)cuIpcGetEventHandle},
    {"cuIpcOpenEventHandle", (void *)cuIpcOpenEventHandle},
    {"cuIpcGetMemHandle", (void *)cuIpcGetMemHandle},
    {"cuIpcOpenMemHandle_v2", (void *)cuIpcOpenMemHandle_v2},
    {"cuIpcCloseMemHandle", (void *)cuIpcCloseMemHandle},
    {"cuMemcpy", (void *)cuMemcpy},
    {"cuMemcpyPeer", (void *)cuMemcpyPeer},
    {"cuMemcpyHtoD_v2", (void *)cuMemcpyHtoD_v2},
    {"cuMemcpyDtoH_v2", (void *)cuMemcpyDtoH_v2},
    {"cuMemcpyDtoD_v2", (void *)cuMemcpyDtoD_v2},
    {"cuMemcpyDtoA_v2", (void *)cuMemcpyDtoA_v2},
    {"cuMemcpyAtoD_v2", (void *)cuMemcpyAtoD_v2},
    {"cuMemcpyAtoH_v2", (void *)cuMemcpyAtoH_v2},
    {"cuMemcpyAtoA_v2", (void *)cuMemcpyAtoA_v2},
    {"cuMemcpyPeerAsync", (void *)cuMemcpyPeerAsync},
    {"cuMemcpyHtoDAsync_v2", (void *)cuMemcpyHtoDAsync_v2},
    {"cuMemcpyDtoDAsync_v2", (void *)cuMemcpyDtoDAsync_v2},
    {"cuMemsetD8_v2", (void *)cuMemsetD8_v2},
    {"cuMemsetD16_v2", (void *)cuMemsetD16_v2},
    {"cuMemsetD32_v2", (void *)cuMemsetD32_v2},
    {"cuMemsetD8", (void *)cuMemsetD8_v2},
    {"cuMemsetD16", (void *)cuMemsetD16_v2},
    {"cuMemsetD32", (void *)cuMemsetD32_v2},
    {"cuMemsetD2D8_v2", (void *)cuMemsetD2D8_v2},
    {"cuMemsetD2D16_v2", (void *)cuMemsetD2D16_v2},
    {"cuMemsetD2D32_v2", (void *)cuMemsetD2D32_v2},
    {"cuMemsetD2D8", (void *)cuMemsetD2D8_v2},
    {"cuMemsetD2D16", (void *)cuMemsetD2D16_v2},
    {"cuMemsetD2D32", (void *)cuMemsetD2D32_v2},
    {"cuMemsetD8Async", (void *)cuMemsetD8Async},
    {"cuMemsetD16Async", (void *)cuMemsetD16Async},
    {"cuMemsetD32Async", (void *)cuMemsetD32Async},
    {"cuMemsetD2D8Async", (void *)cuMemsetD2D8Async},
    {"cuMemsetD2D16Async", (void *)cuMemsetD2D16Async},
    {"cuMemsetD2D32Async", (void *)cuMemsetD2D32Async},
    {"cuArrayCreate_v2", (void *)cuArrayCreate_v2},
    {"cuArrayGetDescriptor_v2", (void *)cuArrayGetDescriptor_v2},
    {"cuArrayGetSparseProperties", (void *)cuArrayGetSparseProperties},
    {"cuMipmappedArrayGetSparseProperties", (void *)cuMipmappedArrayGetSparseProperties},
    {"cuArrayGetMemoryRequirements", (void *)cuArrayGetMemoryRequirements},
    {"cuMipmappedArrayGetMemoryRequirements", (void *)cuMipmappedArrayGetMemoryRequirements},
    {"cuArrayGetPlane", (void *)cuArrayGetPlane},
    {"cuArrayDestroy", (void *)cuArrayDestroy},
    {"cuArray3DCreate_v2", (void *)cuArray3DCreate_v2},
    {"cuArray3DGetDescriptor_v2", (void *)cuArray3DGetDescriptor_v2},
    {"cuMipmappedArrayCreate", (void *)cuMipmappedArrayCreate},
    {"cuMipmappedArrayGetLevel", (void *)cuMipmappedArrayGetLevel},
    {"cuMipmappedArrayDestroy", (void *)cuMipmappedArrayDestroy},
    {"cuMemAddressReserve", (void *)cuMemAddressReserve},
    {"cuMemAddressFree", (void *)cuMemAddressFree},
    {"cuMemCreate", (void *)cuMemCreate},
    {"cuMemRelease", (void *)cuMemRelease},
    {"cuMemMap", (void *)cuMemMap},
    {"cuMemMapArrayAsync", (void *)cuMemMapArrayAsync},
    {"cuMemUnmap", (void *)cuMemUnmap},
    {"cuMemSetAccess", (void *)cuMemSetAccess},
    {"cuMemGetAccess", (void *)cuMemGetAccess},
    {"cuMemGetAllocationGranularity", (void *)cuMemGetAllocationGranularity},
    {"cuMemGetAllocationPropertiesFromHandle", (void *)cuMemGetAllocationPropertiesFromHandle},
    {"cuMemFreeAsync", (void *)cuMemFreeAsync},
    {"cuMemAllocAsync", (void *)cuMemAllocAsync},
    {"cuMemPoolTrimTo", (void *)cuMemPoolTrimTo},
    {"cuMemPoolSetAccess", (void *)cuMemPoolSetAccess},
    {"cuMemPoolGetAccess", (void *)cuMemPoolGetAccess},
    {"cuMemPoolCreate", (void *)cuMemPoolCreate},
    {"cuMemPoolDestroy", (void *)cuMemPoolDestroy},
    {"cuMemAllocFromPoolAsync", (void *)cuMemAllocFromPoolAsync},
    {"cuMemPoolExportPointer", (void *)cuMemPoolExportPointer},
    {"cuMemPoolImportPointer", (void *)cuMemPoolImportPointer},
    {"cuMemRangeGetAttributes", (void *)cuMemRangeGetAttributes},
    {"cuPointerSetAttribute", (void *)cuPointerSetAttribute},
    {"cuPointerGetAttributes", (void *)cuPointerGetAttributes},
    {"cuStreamCreate", (void *)cuStreamCreate},
    {"cuStreamCreateWithPriority", (void *)cuStreamCreateWithPriority},
    {"cuStreamGetPriority", (void *)cuStreamGetPriority},
    {"cuStreamGetFlags", (void *)cuStreamGetFlags},
    {"cuStreamGetId", (void *)cuStreamGetId},
    {"cuStreamGetCtx", (void *)cuStreamGetCtx},
    {"cuStreamWaitEvent", (void *)cuStreamWaitEvent},
    {"cuStreamBeginCapture_v2", (void *)cuStreamBeginCapture_v2},
    {"cuThreadExchangeStreamCaptureMode", (void *)cuThreadExchangeStreamCaptureMode},
    {"cuStreamEndCapture", (void *)cuStreamEndCapture},
    {"cuStreamIsCapturing", (void *)cuStreamIsCapturing},
    {"cuStreamAttachMemAsync", (void *)cuStreamAttachMemAsync},
    {"cuStreamQuery", (void *)cuStreamQuery},
    {"cuStreamDestroy_v2", (void *)cuStreamDestroy_v2},
    {"cuStreamCopyAttributes", (void *)cuStreamCopyAttributes},
    {"cuStreamGetAttribute", (void *)cuStreamGetAttribute},
    {"cuStreamSetAttribute", (void *)cuStreamSetAttribute},
    {"cuEventCreate", (void *)cuEventCreate},
    {"cuEventRecord", (void *)cuEventRecord},
    {"cuEventRecordWithFlags", (void *)cuEventRecordWithFlags},
    {"cuEventQuery", (void *)cuEventQuery},
    {"cuEventDestroy_v2", (void *)cuEventDestroy_v2},
    {"cuEventElapsedTime_v2", (void *)cuEventElapsedTime_v2},
    {"cuImportExternalMemory", (void *)cuImportExternalMemory},
    {"cuExternalMemoryGetMappedBuffer", (void *)cuExternalMemoryGetMappedBuffer},
    {"cuExternalMemoryGetMappedMipmappedArray", (void *)cuExternalMemoryGetMappedMipmappedArray},
    {"cuDestroyExternalMemory", (void *)cuDestroyExternalMemory},
    {"cuImportExternalSemaphore", (void *)cuImportExternalSemaphore},
    {"cuSignalExternalSemaphoresAsync", (void *)cuSignalExternalSemaphoresAsync},
    {"cuWaitExternalSemaphoresAsync", (void *)cuWaitExternalSemaphoresAsync},
    {"cuDestroyExternalSemaphore", (void *)cuDestroyExternalSemaphore},
    {"cuStreamWaitValue32_v2", (void *)cuStreamWaitValue32_v2},
    {"cuStreamWaitValue64_v2", (void *)cuStreamWaitValue64_v2},
    {"cuStreamWriteValue32_v2", (void *)cuStreamWriteValue32_v2},
    {"cuStreamWriteValue64_v2", (void *)cuStreamWriteValue64_v2},
    {"cuStreamBatchMemOp_v2", (void *)cuStreamBatchMemOp_v2},
    {"cuFuncGetAttribute", (void *)cuFuncGetAttribute},
    {"cuFuncSetAttribute", (void *)cuFuncSetAttribute},
    {"cuFuncSetCacheConfig", (void *)cuFuncSetCacheConfig},
    {"cuFuncGetModule", (void *)cuFuncGetModule},
    {"cuLaunchCooperativeKernel", (void *)cuLaunchCooperativeKernel},
    {"cuLaunchCooperativeKernelMultiDevice", (void *)cuLaunchCooperativeKernelMultiDevice},
    {"cuFuncSetBlockShape", (void *)cuFuncSetBlockShape},
    {"cuFuncSetSharedSize", (void *)cuFuncSetSharedSize},
    {"cuParamSetSize", (void *)cuParamSetSize},
    {"cuParamSeti", (void *)cuParamSeti},
    {"cuParamSetf", (void *)cuParamSetf},
    {"cuLaunch", (void *)cuLaunch},
    {"cuLaunchGrid", (void *)cuLaunchGrid},
    {"cuLaunchGridAsync", (void *)cuLaunchGridAsync},
    {"cuParamSetTexRef", (void *)cuParamSetTexRef},
    {"cuFuncSetSharedMemConfig", (void *)cuFuncSetSharedMemConfig},
    {"cuGraphCreate", (void *)cuGraphCreate},
    {"cuGraphKernelNodeGetParams_v2", (void *)cuGraphKernelNodeGetParams_v2},
    {"cuGraphKernelNodeSetParams_v2", (void *)cuGraphKernelNodeSetParams_v2},
    {"cuGraphMemcpyNodeGetParams", (void *)cuGraphMemcpyNodeGetParams},
    {"cuGraphMemcpyNodeSetParams", (void *)cuGraphMemcpyNodeSetParams},
    {"cuGraphMemsetNodeGetParams", (void *)cuGraphMemsetNodeGetParams},
    {"cuGraphMemsetNodeSetParams", (void *)cuGraphMemsetNodeSetParams},
    {"cuGraphHostNodeGetParams", (void *)cuGraphHostNodeGetParams},
    {"cuGraphHostNodeSetParams", (void *)cuGraphHostNodeSetParams},
    {"cuGraphAddChildGraphNode", (void *)cuGraphAddChildGraphNode},
    {"cuGraphChildGraphNodeGetGraph", (void *)cuGraphChildGraphNodeGetGraph},
    {"cuGraphAddEmptyNode", (void *)cuGraphAddEmptyNode},
    {"cuGraphAddEventRecordNode", (void *)cuGraphAddEventRecordNode},
    {"cuGraphEventRecordNodeGetEvent", (void *)cuGraphEventRecordNodeGetEvent},
    {"cuGraphEventRecordNodeSetEvent", (void *)cuGraphEventRecordNodeSetEvent},
    {"cuGraphAddEventWaitNode", (void *)cuGraphAddEventWaitNode},
    {"cuGraphEventWaitNodeGetEvent", (void *)cuGraphEventWaitNodeGetEvent},
    {"cuGraphEventWaitNodeSetEvent", (void *)cuGraphEventWaitNodeSetEvent},
    {"cuGraphAddExternalSemaphoresSignalNode", (void *)cuGraphAddExternalSemaphoresSignalNode},
    {"cuGraphExternalSemaphoresSignalNodeGetParams", (void *)cuGraphExternalSemaphoresSignalNodeGetParams},
    {"cuGraphExternalSemaphoresSignalNodeSetParams", (void *)cuGraphExternalSemaphoresSignalNodeSetParams},
    {"cuGraphAddExternalSemaphoresWaitNode", (void *)cuGraphAddExternalSemaphoresWaitNode},
    {"cuGraphExternalSemaphoresWaitNodeGetParams", (void *)cuGraphExternalSemaphoresWaitNodeGetParams},
    {"cuGraphExternalSemaphoresWaitNodeSetParams", (void *)cuGraphExternalSemaphoresWaitNodeSetParams},
    {"cuGraphAddBatchMemOpNode", (void *)cuGraphAddBatchMemOpNode},
    {"cuGraphBatchMemOpNodeGetParams", (void *)cuGraphBatchMemOpNodeGetParams},
    {"cuGraphBatchMemOpNodeSetParams", (void *)cuGraphBatchMemOpNodeSetParams},
    {"cuGraphExecBatchMemOpNodeSetParams", (void *)cuGraphExecBatchMemOpNodeSetParams},
    {"cuGraphAddMemAllocNode", (void *)cuGraphAddMemAllocNode},
    {"cuGraphMemAllocNodeGetParams", (void *)cuGraphMemAllocNodeGetParams},
    {"cuGraphAddMemFreeNode", (void *)cuGraphAddMemFreeNode},
    {"cuGraphMemFreeNodeGetParams", (void *)cuGraphMemFreeNodeGetParams},
    {"cuDeviceGraphMemTrim", (void *)cuDeviceGraphMemTrim},
    {"cuGraphClone", (void *)cuGraphClone},
    {"cuGraphNodeFindInClone", (void *)cuGraphNodeFindInClone},
    {"cuGraphNodeGetType", (void *)cuGraphNodeGetType},
    {"cuGraphGetRootNodes", (void *)cuGraphGetRootNodes},
    {"cuGraphDestroyNode", (void *)cuGraphDestroyNode},
    {"cuGraphInstantiateWithFlags", (void *)cuGraphInstantiateWithFlags},
    {"cuGraphInstantiateWithParams", (void *)cuGraphInstantiateWithParams},
    {"cuGraphExecGetFlags", (void *)cuGraphExecGetFlags},
    {"cuGraphExecKernelNodeSetParams_v2", (void *)cuGraphExecKernelNodeSetParams_v2},
    {"cuGraphExecMemcpyNodeSetParams", (void *)cuGraphExecMemcpyNodeSetParams},
    {"cuGraphExecMemsetNodeSetParams", (void *)cuGraphExecMemsetNodeSetParams},
    {"cuGraphExecHostNodeSetParams", (void *)cuGraphExecHostNodeSetParams},
    {"cuGraphExecChildGraphNodeSetParams", (void *)cuGraphExecChildGraphNodeSetParams},
    {"cuGraphExecEventRecordNodeSetEvent", (void *)cuGraphExecEventRecordNodeSetEvent},
    {"cuGraphExecEventWaitNodeSetEvent", (void *)cuGraphExecEventWaitNodeSetEvent},
    {"cuGraphExecExternalSemaphoresSignalNodeSetParams", (void *)cuGraphExecExternalSemaphoresSignalNodeSetParams},
    {"cuGraphExecExternalSemaphoresWaitNodeSetParams", (void *)cuGraphExecExternalSemaphoresWaitNodeSetParams},
    {"cuGraphNodeSetEnabled", (void *)cuGraphNodeSetEnabled},
    {"cuGraphNodeGetEnabled", (void *)cuGraphNodeGetEnabled},
    {"cuGraphUpload", (void *)cuGraphUpload},
    {"cuGraphLaunch", (void *)cuGraphLaunch},
    {"cuGraphExecDestroy", (void *)cuGraphExecDestroy},
    {"cuGraphDestroy", (void *)cuGraphDestroy},
    {"cuGraphExecUpdate_v2", (void *)cuGraphExecUpdate_v2},
    {"cuGraphKernelNodeCopyAttributes", (void *)cuGraphKernelNodeCopyAttributes},
    {"cuGraphKernelNodeGetAttribute", (void *)cuGraphKernelNodeGetAttribute},
    {"cuGraphKernelNodeSetAttribute", (void *)cuGraphKernelNodeSetAttribute},
    {"cuGraphDebugDotPrint", (void *)cuGraphDebugDotPrint},
    {"cuUserObjectRetain", (void *)cuUserObjectRetain},
    {"cuUserObjectRelease", (void *)cuUserObjectRelease},
    {"cuGraphRetainUserObject", (void *)cuGraphRetainUserObject},
    {"cuGraphReleaseUserObject", (void *)cuGraphReleaseUserObject},
    {"cuOccupancyMaxActiveBlocksPerMultiprocessor", (void *)cuOccupancyMaxActiveBlocksPerMultiprocessor},
    {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", (void *)cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags},
    {"cuOccupancyAvailableDynamicSMemPerBlock", (void *)cuOccupancyAvailableDynamicSMemPerBlock},
    {"cuOccupancyMaxPotentialClusterSize", (void *)cuOccupancyMaxPotentialClusterSize},
    {"cuOccupancyMaxActiveClusters", (void *)cuOccupancyMaxActiveClusters},
    {"cuTexRefSetArray", (void *)cuTexRefSetArray},
    {"cuTexRefSetMipmappedArray", (void *)cuTexRefSetMipmappedArray},
    {"cuTexRefSetAddress_v2", (void *)cuTexRefSetAddress_v2},
    {"cuTexRefSetAddress2D_v3", (void *)cuTexRefSetAddress2D_v3},
    {"cuTexRefSetFormat", (void *)cuTexRefSetFormat},
    {"cuTexRefSetAddressMode", (void *)cuTexRefSetAddressMode},
    {"cuTexRefSetFilterMode", (void *)cuTexRefSetFilterMode},
    {"cuTexRefSetMipmapFilterMode", (void *)cuTexRefSetMipmapFilterMode},
    {"cuTexRefSetMipmapLevelBias", (void *)cuTexRefSetMipmapLevelBias},
    {"cuTexRefSetMipmapLevelClamp", (void *)cuTexRefSetMipmapLevelClamp},
    {"cuTexRefSetMaxAnisotropy", (void *)cuTexRefSetMaxAnisotropy},
    {"cuTexRefSetBorderColor", (void *)cuTexRefSetBorderColor},
    {"cuTexRefSetFlags", (void *)cuTexRefSetFlags},
    {"cuTexRefGetAddress_v2", (void *)cuTexRefGetAddress_v2},
    {"cuTexRefGetArray", (void *)cuTexRefGetArray},
    {"cuTexRefGetMipmappedArray", (void *)cuTexRefGetMipmappedArray},
    {"cuTexRefGetAddressMode", (void *)cuTexRefGetAddressMode},
    {"cuTexRefGetFilterMode", (void *)cuTexRefGetFilterMode},
    {"cuTexRefGetFormat", (void *)cuTexRefGetFormat},
    {"cuTexRefGetMipmapFilterMode", (void *)cuTexRefGetMipmapFilterMode},
    {"cuTexRefGetMipmapLevelBias", (void *)cuTexRefGetMipmapLevelBias},
    {"cuTexRefGetMipmapLevelClamp", (void *)cuTexRefGetMipmapLevelClamp},
    {"cuTexRefGetMaxAnisotropy", (void *)cuTexRefGetMaxAnisotropy},
    {"cuTexRefGetBorderColor", (void *)cuTexRefGetBorderColor},
    {"cuTexRefGetFlags", (void *)cuTexRefGetFlags},
    {"cuTexRefCreate", (void *)cuTexRefCreate},
    {"cuTexRefDestroy", (void *)cuTexRefDestroy},
    {"cuSurfRefSetArray", (void *)cuSurfRefSetArray},
    {"cuSurfRefGetArray", (void *)cuSurfRefGetArray},
    {"cuTexObjectCreate", (void *)cuTexObjectCreate},
    {"cuTexObjectDestroy", (void *)cuTexObjectDestroy},
    {"cuTexObjectGetResourceDesc", (void *)cuTexObjectGetResourceDesc},
    {"cuTexObjectGetTextureDesc", (void *)cuTexObjectGetTextureDesc},
    {"cuTexObjectGetResourceViewDesc", (void *)cuTexObjectGetResourceViewDesc},
    {"cuSurfObjectCreate", (void *)cuSurfObjectCreate},
    {"cuSurfObjectDestroy", (void *)cuSurfObjectDestroy},
    {"cuSurfObjectGetResourceDesc", (void *)cuSurfObjectGetResourceDesc},
    {"cuDeviceCanAccessPeer", (void *)cuDeviceCanAccessPeer},
    {"cuCtxEnablePeerAccess", (void *)cuCtxEnablePeerAccess},
    {"cuCtxDisablePeerAccess", (void *)cuCtxDisablePeerAccess},
    {"cuDeviceGetP2PAttribute", (void *)cuDeviceGetP2PAttribute},
    {"cuGraphicsUnregisterResource", (void *)cuGraphicsUnregisterResource},
    {"cuGraphicsSubResourceGetMappedArray", (void *)cuGraphicsSubResourceGetMappedArray},
    {"cuGraphicsResourceGetMappedMipmappedArray", (void *)cuGraphicsResourceGetMappedMipmappedArray},
    {"cuGraphicsResourceGetMappedPointer_v2", (void *)cuGraphicsResourceGetMappedPointer_v2},
    {"cuGraphicsResourceSetMapFlags_v2", (void *)cuGraphicsResourceSetMapFlags_v2},
    {"cuGraphicsMapResources", (void *)cuGraphicsMapResources},
    {"cuGraphicsUnmapResources", (void *)cuGraphicsUnmapResources},
    {"cuDeviceTotalMem", (void *)cuDeviceTotalMem_v2},
    {"cuDeviceGetUuid", (void *)cuDeviceGetUuid_v2},
    {"cuDevicePrimaryCtxRelease", (void *)cuDevicePrimaryCtxRelease_v2},
    {"cuDevicePrimaryCtxSetFlags", (void *)cuDevicePrimaryCtxSetFlags_v2},
    {"cuDevicePrimaryCtxReset", (void *)cuDevicePrimaryCtxReset_v2},
    {"cuCtxDestroy", (void *)cuCtxDestroy_v2},
    {"cuCtxPopCurrent", (void *)cuCtxPopCurrent_v2},
    {"cuCtxPushCurrent", (void *)cuCtxPushCurrent_v2},
    {"cuModuleGetGlobal", (void *)cuModuleGetGlobal_v2},
    {"cuMemAlloc", (void *)cuMemAlloc_v2},
    {"cuMemAllocPitch", (void *)cuMemAllocPitch_v2},
    {"cuMemFree", (void *)cuMemFree_v2},
    {"cuMemcpyHtoD", (void *)cuMemcpyHtoD_v2},
    {"cuMemcpyHtoDAsync", (void *)cuMemcpyHtoDAsync_v2},
    {"cuMemcpyDtoH", (void *)cuMemcpyDtoH_v2},
    {"cuMemcpyDtoD", (void *)cuMemcpyDtoD_v2},
    {"cuMemcpyDtoDAsync", (void *)cuMemcpyDtoDAsync_v2},
    {"cuMemsetD8", (void *)cuMemsetD8_v2},
    {"cuStreamBeginCapture", (void *)cuStreamBeginCapture_v2},
    {"cuGraphExecUpdate", (void *)cuGraphExecUpdate_v2},
    {"cuMemcpy_ptds", (void *)cuMemcpy},
    {"cuMemcpyPeer_ptds", (void *)cuMemcpyPeer},
    {"cuMemcpyPeerAsync_ptsz", (void *)cuMemcpyPeerAsync},
    {"cuMemsetD8Async_ptsz", (void *)cuMemsetD8Async},
    {"cuMemsetD16Async_ptsz", (void *)cuMemsetD16Async},
    {"cuMemsetD32Async_ptsz", (void *)cuMemsetD32Async},
    {"cuMemsetD2D8Async_ptsz", (void *)cuMemsetD2D8Async},
    {"cuMemsetD2D16Async_ptsz", (void *)cuMemsetD2D16Async},
    {"cuMemsetD2D32Async_ptsz", (void *)cuMemsetD2D32Async},
    {"cuStreamGetPriority_ptsz", (void *)cuStreamGetPriority},
    {"cuStreamGetId_ptsz", (void *)cuStreamGetId},
    {"cuStreamGetFlags_ptsz", (void *)cuStreamGetFlags},
    {"cuStreamGetCtx_ptsz", (void *)cuStreamGetCtx},
    {"cuStreamWaitEvent_ptsz", (void *)cuStreamWaitEvent},
    {"cuStreamEndCapture_ptsz", (void *)cuStreamEndCapture},
    {"cuStreamIsCapturing_ptsz", (void *)cuStreamIsCapturing},
    {"cuStreamAttachMemAsync_ptsz", (void *)cuStreamAttachMemAsync},
    {"cuStreamQuery_ptsz", (void *)cuStreamQuery},
    {"cuEventRecord_ptsz", (void *)cuEventRecord},
    {"cuEventRecordWithFlags_ptsz", (void *)cuEventRecordWithFlags},
    {"cuGraphicsMapResources_ptsz", (void *)cuGraphicsMapResources},
    {"cuGraphicsUnmapResources_ptsz", (void *)cuGraphicsUnmapResources},
    {"cuLaunchCooperativeKernel_ptsz", (void *)cuLaunchCooperativeKernel},
    {"cuSignalExternalSemaphoresAsync_ptsz", (void *)cuSignalExternalSemaphoresAsync},
    {"cuWaitExternalSemaphoresAsync_ptsz", (void *)cuWaitExternalSemaphoresAsync},
    {"cuGraphInstantiateWithParams_ptsz", (void *)cuGraphInstantiateWithParams},
    {"cuGraphUpload_ptsz", (void *)cuGraphUpload},
    {"cuGraphLaunch_ptsz", (void *)cuGraphLaunch},
    {"cuStreamCopyAttributes_ptsz", (void *)cuStreamCopyAttributes},
    {"cuStreamGetAttribute_ptsz", (void *)cuStreamGetAttribute},
    {"cuStreamSetAttribute_ptsz", (void *)cuStreamSetAttribute},
    {"cuMemMapArrayAsync_ptsz", (void *)cuMemMapArrayAsync},
    {"cuMemFreeAsync_ptsz", (void *)cuMemFreeAsync},
    {"cuMemAllocAsync_ptsz", (void *)cuMemAllocAsync},
    {"cuMemAllocFromPoolAsync_ptsz", (void *)cuMemAllocFromPoolAsync},
};

void *get_function_pointer(const char *name)
{
    auto it = functionMap.find(name);
    if (it == functionMap.end())
        return nullptr;
    return it->second;
}
