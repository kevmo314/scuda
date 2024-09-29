from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import (
    Function,
    Parameter,
    Pointer,
    Type,
)
from typing import Optional
import dataclasses

IGNORE_FUNCTIONS = {
    "nvmlDeviceCcuGetStreamState",
    "nvmlDeviceCcuSetStreamState",
    "cuOccupancyMaxPotentialBlockSize",
    "cuOccupancyMaxPotentialBlockSizeWithFlags",
    "cuGetProcAddress",
    "cuGetProcAddress_v2",
    # these need void* size annotations
    "cuModuleLoadData",
    "cuModuleLoadDataEx",
    "cuModuleLoadFatBinary",
    "cuLibraryLoadData",
    "cuPointerSetAttribute",
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cudaLaunchCooperativeKernel",
    "cudaMemcpy",  # this has a manual implementation, param dependent
    "cudaMemcpyPeer",  # this has a manual implementation, param dependent
    "cudaMemcpy2D",  # this has a manual implementation, param dependent
    "cudaMemcpy2DToArray",  # this has a manual implementation, param dependent
    "cudaMemcpyToSymbol",  # this has a manual implementation, param dependent
    "cudaMemcpyFromSymbol",  # this has a manual implementation, param dependent
    "cudaMemcpyAsync",  # this has a manual implementation, param dependent
    "cudaMemcpyPeerAsync",  # this has a manual implementation, param dependent
    "cudaMemcpy2DAsync",  # this has a manual implementation, param dependent
    "cudaMemcpy2DToArrayAsync",  # this has a manual implementation, param dependent
    "cudaMemcpyToSymbolAsync",  # this has a manual implementation, param dependent
    "cudaMemcpyFromSymbolAsync",  # this has a manual implementation, param dependent
}


UPGRADES = [
    ("nvmlInit", "nvmlInit_v2"),
    ("nvmlDeviceGetPciInfo", "nvmlDeviceGetPciInfo_v3"),
    ("nvmlDeviceGetCount", "nvmlDeviceGetCount_v2"),
    ("nvmlDeviceGetHandleByIndex", "nvmlDeviceGetHandleByIndex_v2"),
    ("nvmlDeviceGetHandleByPciBusId", "nvmlDeviceGetHandleByPciBusId_v2"),
    ("nvmlDeviceGetNvLinkRemotePciInfo", "nvmlDeviceGetNvLinkRemotePciInfo_v2"),
    ("nvmlDeviceRemoveGpu", "nvmlDeviceRemoveGpu_v2"),
    ("nvmlDeviceGetGridLicensableFeatures", "nvmlDeviceGetGridLicensableFeatures_v4"),
    ("nvmlEventSetWait", "nvmlEventSetWait_v2"),
    ("nvmlDeviceGetAttributes", "nvmlDeviceGetAttributes_v2"),
    ("nvmlComputeInstanceGetInfo", "nvmlComputeInstanceGetInfo_v2"),
    ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetComputeRunningProcesses_v3"),
    (
        "nvmlDeviceGetGraphicsRunningProcesses",
        "nvmlDeviceGetGraphicsRunningProcesses_v3",
    ),
    (
        "nvmlDeviceGetMPSComputeRunningProcesses",
        "nvmlDeviceGetMPSComputeRunningProcesses_v3",
    ),
    ("nvmlGetBlacklistDeviceCount", "nvmlGetExcludedDeviceCount"),
    ("nvmlGetBlacklistDeviceInfoByIndex", "nvmlGetExcludedDeviceInfoByIndex"),
    (
        "nvmlDeviceGetGpuInstancePossiblePlacements",
        "nvmlDeviceGetGpuInstancePossiblePlacements_v2",
    ),
    ("nvmlVgpuInstanceGetLicenseInfo", "nvmlVgpuInstanceGetLicenseInfo_v2"),
    ("cuDeviceTotalMem", "cuDeviceTotalMem_v2"),
    ("cuCtxCreate", "cuCtxCreate_v2"),
    ("cuCtxCreate_v3", "cuCtxCreate_v3"),
    ("cuModuleGetGlobal", "cuModuleGetGlobal_v2"),
    ("cuMemGetInfo", "cuMemGetInfo_v2"),
    ("cuMemAlloc", "cuMemAlloc_v2"),
    ("cuMemAllocPitch", "cuMemAllocPitch_v2"),
    ("cuMemFree", "cuMemFree_v2"),
    ("cuMemGetAddressRange", "cuMemGetAddressRange_v2"),
    ("cuMemAllocHost", "cuMemAllocHost_v2"),
    ("cuMemHostGetDevicePointer", "cuMemHostGetDevicePointer_v2"),
    ("cuMemcpyHtoD", "cuMemcpyHtoD_v2"),
    ("cuMemcpyDtoH", "cuMemcpyDtoH_v2"),
    ("cuMemcpyDtoD", "cuMemcpyDtoD_v2"),
    ("cuMemcpyDtoA", "cuMemcpyDtoA_v2"),
    ("cuMemcpyAtoD", "cuMemcpyAtoD_v2"),
    ("cuMemcpyHtoA", "cuMemcpyHtoA_v2"),
    ("cuMemcpyAtoH", "cuMemcpyAtoH_v2"),
    ("cuMemcpyAtoA", "cuMemcpyAtoA_v2"),
    ("cuMemcpyHtoAAsync", "cuMemcpyHtoAAsync_v2"),
    ("cuMemcpyAtoHAsync", "cuMemcpyAtoHAsync_v2"),
    ("cuMemcpy2D", "cuMemcpy2D_v2"),
    ("cuMemcpy2DUnaligned", "cuMemcpy2DUnaligned_v2"),
    ("cuMemcpy3D", "cuMemcpy3D_v2"),
    ("cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2"),
    ("cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2"),
    ("cuMemcpyDtoDAsync", "cuMemcpyDtoDAsync_v2"),
    ("cuMemcpy2DAsync", "cuMemcpy2DAsync_v2"),
    ("cuMemcpy3DAsync", "cuMemcpy3DAsync_v2"),
    ("cuMemsetD8", "cuMemsetD8_v2"),
    ("cuMemsetD16", "cuMemsetD16_v2"),
    ("cuMemsetD32", "cuMemsetD32_v2"),
    ("cuMemsetD2D8", "cuMemsetD2D8_v2"),
    ("cuMemsetD2D16", "cuMemsetD2D16_v2"),
    ("cuMemsetD2D32", "cuMemsetD2D32_v2"),
    ("cuArrayCreate", "cuArrayCreate_v2"),
    ("cuArrayGetDescriptor", "cuArrayGetDescriptor_v2"),
    ("cuArray3DCreate", "cuArray3DCreate_v2"),
    ("cuArray3DGetDescriptor", "cuArray3DGetDescriptor_v2"),
    ("cuTexRefSetAddress", "cuTexRefSetAddress_v2"),
    ("cuTexRefGetAddress", "cuTexRefGetAddress_v2"),
    ("cuGraphicsResourceGetMappedPointer", "cuGraphicsResourceGetMappedPointer_v2"),
    ("cuCtxDestroy", "cuCtxDestroy_v2"),
    ("cuCtxPopCurrent", "cuCtxPopCurrent_v2"),
    ("cuCtxPushCurrent", "cuCtxPushCurrent_v2"),
    ("cuStreamDestroy", "cuStreamDestroy_v2"),
    ("cuEventDestroy", "cuEventDestroy_v2"),
    ("cuTexRefSetAddress2D", "cuTexRefSetAddress2D_v3"),
    ("cuLinkCreate", "cuLinkCreate_v2"),
    ("cuLinkAddData", "cuLinkAddData_v2"),
    ("cuLinkAddFile", "cuLinkAddFile_v2"),
    ("cuMemHostRegister", "cuMemHostRegister_v2"),
    ("cuGraphicsResourceSetMapFlags", "cuGraphicsResourceSetMapFlags_v2"),
    ("cuStreamBeginCapture", "cuStreamBeginCapture_v2"),
    ("cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease_v2"),
    ("cuDevicePrimaryCtxReset", "cuDevicePrimaryCtxReset_v2"),
    ("cuDevicePrimaryCtxSetFlags", "cuDevicePrimaryCtxSetFlags_v2"),
    ("cuDeviceGetUuid_v2", "cuDeviceGetUuid_v2"),
    ("cuIpcOpenMemHandle", "cuIpcOpenMemHandle_v2"),
    ("cuGraphInstantiate", "cuGraphInstantiateWithFlags"),
    ("cuGraphExecUpdate", "cuGraphExecUpdate_v2"),
    ("cuGetProcAddress", "cuGetProcAddress_v2"),
    ("cuGraphAddKernelNode", "cuGraphAddKernelNode_v2"),
    ("cuGraphKernelNodeGetParams", "cuGraphKernelNodeGetParams_v2"),
    ("cuGraphKernelNodeSetParams", "cuGraphKernelNodeSetParams_v2"),
    ("cuGraphExecKernelNodeSetParams", "cuGraphExecKernelNodeSetParams_v2"),
    ("cuStreamWriteValue32", "cuStreamWriteValue32_v2"),
    ("cuStreamWaitValue32", "cuStreamWaitValue32_v2"),
    ("cuStreamWriteValue64", "cuStreamWriteValue64_v2"),
    ("cuStreamWaitValue64", "cuStreamWaitValue64_v2"),
    ("cuStreamBatchMemOp", "cuStreamBatchMemOp_v2"),
    ("cuStreamGetCaptureInfo", "cuStreamGetCaptureInfo_v2"),
    ("cuStreamGetCaptureInfo_v2", "cuStreamGetCaptureInfo_v2"),
    ("cudaSignalExternalSemaphoresAsync", "cudaSignalExternalSemaphoresAsync_v2"),
    ("cudaWaitExternalSemaphoresAsync", "cudaWaitExternalSemaphoresAsync_v2"),
    ("cudaStreamGetCaptureInfo", "cudaStreamGetCaptureInfo_v2"),
    ("cudaGetDeviceProperties", "cudaGetDeviceProperties_v2"),
]


SIZE_PARAMS = {
    "nvmlSystemGetDriverVersion": {"version": "length"},
    "nvmlSystemGetNVMLVersion": {"version": "length"},
    "nvmlSystemGetProcessName": {"name": "length"},
    "nvmlSystemGetHicVersion": {"hwbcEntries": "hwbcCount"},
    "nvmlUnitGetDevices": {"devices": "deviceCount"},
    "nvmlDeviceGetName": {"name": "length"},
    "nvmlDeviceGetSerial": {"serial": "length"},
    "nvmlDeviceGetMemoryAffinity": {"nodeSet": "nodeSetSize"},
    "nvmlDeviceGetCpuAffinityWithinScope": {"cpuSet": "cpuSetSize"},
    "nvmlDeviceGetCpuAffinity": {"cpuSet": "cpuSetSize"},
    "nvmlDeviceGetTopologyNearestGpus": {"deviceArray": "count"},
    "nvmlDeviceGetUUID": {"uuid": "length"},
    "nvmlVgpuInstanceGetMdevUUID": {"mdevUuid": "size"},
    "nvmlDeviceGetBoardPartNumber": {"partNumber": "length"},
    "nvmlDeviceGetInforomVersion": {"version": "length"},
    "nvmlDeviceGetInforomImageVersion": {"version": "length"},
    "nvmlDeviceGetEncoderSessions": {"sessionInfos": "sessionCount"},
    "nvmlDeviceGetVbiosVersion": {"version": "length"},
    "nvmlDeviceGetComputeRunningProcesses_v3": {"infos": "infoCount"},
    "nvmlDeviceGetGraphicsRunningProcesses_v3": {"infos": "infoCount"},
    "nvmlDeviceGetMPSComputeRunningProcesses_v3": {"infos": "infoCount"},
    "nvmlDeviceGetSamples": {"samples": "sampleCount"},
    "nvmlDeviceGetAccountingPids": {"pids": "count"},
    "nvmlDeviceGetRetiredPages": {"addresses": "pageCount"},
    "nvmlDeviceGetRetiredPages_v2": {"addresses": "pageCount"},
    "nvmlDeviceGetFieldValues": {"values": "valuesCount"},
    "nvmlDeviceClearFieldValues": {"values": "valuesCount"},
    "nvmlDeviceGetSupportedVgpus": {"vgpuTypeIds": "vgpuCount"},
    "nvmlDeviceGetCreatableVgpus": {"vgpuTypeIds": "vgpuCount"},
    "nvmlVgpuTypeGetName": {"vgpuTypeName": "size"},
    "nvmlVgpuTypeGetLicense": {"vgpuTypeLicenseString": "size"},
    "nvmlVgpuTypeGetMaxInstances": {"vgpuTypeId": "vgpuInstanceCount"},
    "nvmlDeviceGetActiveVgpus": {"vgpuInstances": "vgpuCount"},
    "nvmlVgpuInstanceGetVmID": {"vmId": "size"},
    "nvmlVgpuInstanceGetUUID": {"uuid": "size"},
    "nvmlVgpuInstanceGetVmDriverVersion": {"version": "length"},
    "nvmlVgpuInstanceGetGpuPciId": {"vgpuPciId": "length"},
    "nvmlDeviceGetPgpuMetadataString": {"pgpuMetadata": "bufferSize"},
    "nvmlDeviceGetVgpuUtilization": {"utilizationSamples": "vgpuInstanceSamplesCount"},
    "nvmlDeviceGetVgpuProcessUtilization": {
        "utilizationSamples": "vgpuProcessSamplesCount"
    },
    "nvmlVgpuInstanceGetAccountingPids": {"pids": "count"},
    "nvmlDeviceGetGpuInstancePossiblePlacements_v2": {"placements": "count"},
    "nvmlDeviceGetGpuInstances": {"gpuInstances": "count"},
    "nvmlGpuInstanceGetComputeInstancePossiblePlacements": {"gpuInstances": "count"},
    "nvmlGpuInstanceGetComputeInstances": {"computeInstances": "count"},
    "cuMemcpyDtoH_v2": {"dstHost": "ByteCount"},
    "cuMemcpyHtoD_v2": {"srcHost": "ByteCount"},
    "cuMemcpyHtoDAsync_v2": {"srcHost": "ByteCount"},
    "cuMemcpyDtoHAsync_v2": {"dstHost": "ByteCount"},
    "cuMemcpyHtoD": {"srcHost": "ByteCount"},
    "cuMemcpyDtoH": {"dstHost": "ByteCount"},
    "cuMemcpyHtoDAsync": {"srcHost": "ByteCount"},
    "cuMemcpyDtoHAsync": {"dstHost": "ByteCount"},
    "cuMemcpyHtoA_v2": {"srcHost": "ByteCount"},
    "cuMemcpyAtoH_v2": {"dstHost": "ByteCount"},
    "cuMemcpyHtoAAsync_v2": {"srcHost": "ByteCount"},
    "cuMemcpyAtoHAsync_v2": {"dstHost": "ByteCount"},
    "cuMemcpyHtoA": {"srcHost": "ByteCount"},
    "cuMemcpyAtoH": {"dstHost": "ByteCount"},
    "cuMemcpyHtoAAsync": {"srcHost": "ByteCount"},
    "cuMemcpyAtoHAsync": {"dstHost": "ByteCount"},
}


def error_const(return_type: str) -> str:
    if return_type == "nvmlReturn_t":
        return "NVML_ERROR_GPU_IS_LOST"
    if return_type == "CUresult":
        return "CUDA_ERROR_DEVICE_UNAVAILABLE"
    if return_type == "cudaError_t":
        return "cudaErrorDevicesUnavailable"
    raise NotImplementedError("Unknown return type: %s" % return_type)


@dataclasses.dataclass
class ClientRpcParameter:
    write: tuple[str, str]
    read: tuple[str, str, str]
    write_decl: Optional[str] = None


@dataclasses.dataclass
class ServerRpcParameter:
    write: tuple[str, str]
    read: tuple[str, str]
    write_decl: Optional[str] = None


def deconstify(param: Parameter) -> Parameter:
    """
    Remove const from the parameter type for reading on the server.
    """
    if isinstance(param.type, Type):
        return dataclasses.replace(
            param, type=dataclasses.replace(param.type, const=False)
        )
    if isinstance(param.type, Pointer):
        return dataclasses.replace(
            param,
            type=dataclasses.replace(
                param.type, ptr_to=dataclasses.replace(param.type.ptr_to, const=False)
            ),
        )
    return param


def is_void_pointer(param: Parameter) -> bool:
    return isinstance(param.type, Pointer) and (
        param.type.ptr_to.format() == "void"
        or param.type.ptr_to.format() == "const void"
    )


def promote_size_parameters(function_name: str, parameters: list[Parameter]):
    for param in parameters:
        if param.name is None:
            continue
        if param.name in SIZE_PARAMS.get(function_name, {}).values():
            yield param
    for param in parameters:
        if param.name is None:
            continue
        if param.name not in SIZE_PARAMS.get(function_name, {}).values():
            yield param


def client_write_pass(function: Function) -> list[ClientRpcParameter]:
    """
    Parameters in write-order for the client-side pass for this function.
    """
    parameters = list(
        promote_size_parameters(function.name.format(), function.parameters)
    )
    offset = 0
    output = []
    for i, param in enumerate(parameters):
        if isinstance(param.type, Type):
            # type args are read-only parameters
            output.append(
                ClientRpcParameter(
                    write=("&%s" % param.name, "sizeof(%s)" % param.type.format()),
                    read=(
                        "%s;" % param.type.format_decl(param.name),
                        "&%s" % param.name,
                        "sizeof(%s)" % param.type.format(),
                    ),
                )
            )
        elif isinstance(param.type, Pointer):
            if is_void_pointer(param):
                # handle const void* and void* params
                # if it is const void* it is a read-only parameter and if it is void* it is a write-only parameter.
                if not param.type.ptr_to.const:
                    # write-only parameter, ignore
                    continue

                if param.name == "func":
                    # this is a special case, we write this thing as an opaque pointer
                    output.append(
                        ClientRpcParameter(
                            write=(
                                param.name,
                                "sizeof(%s)" % param.type.ptr_to.format(),
                            ),
                            read=(
                                "%s = (%s)malloc(sizeof(%s));"
                                % (
                                    param.type.format_decl(param.name),
                                    param.type.format(),
                                    param.type.ptr_to.format(),
                                ),
                                "&%s" % param.name,
                                "sizeof(%s)" % param.type.ptr_to.format(),
                            ),
                        )
                    )
                    continue

                size_param_name = SIZE_PARAMS.get(function.name.format(), {}).get(
                    param.name
                )

                try:
                    size_param = next(
                        p for p in function.parameters if p.name == size_param_name
                    )
                except StopIteration:
                    raise NotImplementedError(
                        "Void pointer parameters must have a size parameter. fn: %s, param: %s"
                        % (function.name.format(), param.name)
                    )

                if isinstance(size_param.type, Type):
                    # the size parameter is a value type, so we can just pass the size directly.
                    output.append(
                        ClientRpcParameter(
                            write=(
                                param.name,
                                "%s * sizeof(%s)"
                                % (size_param.name, param.type.ptr_to.format()),
                            ),
                            read=(
                                "%s = (%s)malloc(%s * sizeof(%s));"
                                % (
                                    param.type.format_decl(param.name),
                                    param.type.format(),
                                    size_param.name,
                                    param.type.ptr_to.format(),
                                ),
                                "&%s" % param.name,
                                "%s * sizeof(%s)"
                                % (size_param.name, param.type.ptr_to.format()),
                            ),
                        )
                    )
                elif isinstance(size_param.type, Pointer):
                    # the size parameter is a pointer, so we need to dereference it.
                    output.append(
                        ClientRpcParameter(
                            write=(
                                param.name,
                                "*%s * sizeof(%s)"
                                % (size_param.name, param.type.ptr_to.format()),
                            ),
                            read=(
                                "%s = (%s)malloc(%s * sizeof(%s));"
                                % (
                                    param.type.format_decl(param.name),
                                    param.type.format(),
                                    size_param.name,
                                    param.type.ptr_to.format(),
                                ),
                                "&%s" % param.name,
                                "%s * sizeof(%s)"
                                % (size_param.name, param.type.ptr_to.format()),
                            ),
                        )
                    )
                else:
                    raise NotImplementedError(
                        "Unknown size parameter type: %s" % size_param.type
                    )

            elif not param.type.ptr_to.const:
                # handle T* params
                # these differ from const void*/void* because they may not have a size parameter, in which
                # case this is a read/write parameter. generally, it is too ambiguous to determine if these are
                # read-only or write-only, so we will treat them as read/write.
                size_param_name = SIZE_PARAMS.get(function.name.format(), {}).get(
                    param.name
                )
                try:
                    size_param = next(
                        p for p in function.parameters if p.name == size_param_name
                    )
                    # treat this as an array parameter
                    if isinstance(size_param.type, Type):
                        # the size parameter is a value type, so we can just pass the size directly.
                        output.append(
                            ClientRpcParameter(
                                write=(
                                    param.name,
                                    "%s * sizeof(%s)"
                                    % (size_param.name, param.type.ptr_to.format()),
                                ),
                                read=(
                                    "%s = (%s)malloc(%s * sizeof(%s));"
                                    % (
                                        param.type.format_decl(param.name),
                                        param.type.format(),
                                        size_param.name,
                                        param.type.ptr_to.format(),
                                    ),
                                    "&%s" % param.name,
                                    "%s * sizeof(%s)"
                                    % (size_param.name, param.type.ptr_to.format()),
                                ),
                            )
                        )
                    elif isinstance(size_param.type, Pointer):
                        # the size parameter is a pointer, so we need to dereference it.
                        output.append(
                            ClientRpcParameter(
                                write=(
                                    param.name,
                                    "*%s * sizeof(%s)"
                                    % (size_param.name, param.type.ptr_to.format()),
                                ),
                                read=(
                                    "%s = (%s)malloc(%s * sizeof(%s));"
                                    % (
                                        param.type.format_decl(param.name),
                                        param.type.format(),
                                        size_param.name,
                                        param.type.ptr_to.format(),
                                    ),
                                    "%s" % param.name,
                                    "%s * sizeof(%s)"
                                    % (size_param.name, param.type.ptr_to.format()),
                                ),
                            )
                        )
                    else:
                        raise NotImplementedError(
                            "Unknown size parameter type: %s" % size_param.type
                        )
                except StopIteration:
                    # treat this as a value parameter
                    output.append(
                        ClientRpcParameter(
                            write=(
                                "%s" % param.name,
                                "sizeof(%s)" % param.type.ptr_to.format(),
                            ),
                            read=(
                                "%s;" % param.type.ptr_to.format_decl(param.name),
                                "&%s" % param.name,
                                "sizeof(%s)" % param.type.ptr_to.format(),
                            ),
                        )
                    )

            elif param.type.ptr_to.format() == "const char":
                # handle const char* params as read-only but we need to inject a length prefix because this thing is probably null-terminated.
                output.append(
                    ClientRpcParameter(
                        write_decl="int %s_size = std::strlen(%s) + 1;"
                        % (param.name, param.name),
                        write=(
                            "&%s_size" % param.name,
                            "sizeof(int)",
                        ),
                        read=(
                            "int %s_size;" % param.name,
                            "&%s_size" % param.name,
                            "sizeof(int)",
                        ),
                    )
                )
                output.append(
                    ClientRpcParameter(
                        write=(
                            "%s" % param.name,
                            "%s_size * sizeof(char)" % param.name,
                        ),
                        read=(
                            "%s = (%s)malloc(%s_size * sizeof(char));"
                            % (
                                param.type.format_decl(param.name),
                                param.type.format(),
                                param.name,
                            ),
                            "&%s" % param.name,
                            "sizeof(%s_size * sizeof(char))" % param.name,
                        ),
                    )
                )

            else:
                # handle const T* params
                # this is a const pointer, we will dereference it and write it as a value as this is a read-only parameter.
                size_param_name = SIZE_PARAMS.get(function.name.format(), {}).get(
                    param.name
                )
                try:
                    size_param = next(
                        p for p in function.parameters if p.name == size_param_name
                    )
                    # treat this as an array parameter
                    output.append(
                        ClientRpcParameter(
                            write=(
                                "&%s" % param.name,
                                "%s * sizeof(%s)"
                                % (size_param.name, param.type.ptr_to.format()),
                            ),
                            read=(
                                "%s = (%s)malloc(%s * sizeof(%s));"
                                % (
                                    param.type.format_decl(param.name),
                                    param.type.format(),
                                    size_param.name,
                                    param.type.ptr_to.format(),
                                ),
                                "&%s" % param.name,
                                "sizeof(%s)" % param.type.ptr_to.format(),
                            ),
                        )
                    )

                except StopIteration:
                    output.append(
                        ClientRpcParameter(
                            write=(
                                "&%s" % param.name,
                                "sizeof(%s)" % param.type.ptr_to.format(),
                            ),
                            read=(
                                "%s;"
                                % deconstify(param).type.ptr_to.format_decl(param.name),
                                "&%s" % param.name,
                                "sizeof(%s)" % param.type.ptr_to.format(),
                            ),
                        )
                    )
        else:
            raise NotImplementedError("Unknown parameter type: %s" % param.type)
    return output


def server_write_pass(function: Function) -> list[ServerRpcParameter]:
    """
    Parameters in write-order for the server-side pass for this function.
    """
    parameters = list(
        promote_size_parameters(function.name.format(), function.parameters)
    )
    for i, param in enumerate(parameters):
        if isinstance(param.type, Type):
            # type args are read-only parameters
            parameters[i] = None
        elif isinstance(param.type, Pointer):
            if is_void_pointer(param):
                # handle const void* and void* params
                # if it is const void* it is a read-only parameter and if it is void* it is a write-only parameter.
                if param.type.ptr_to.const:
                    # read-only parameter, ignore
                    parameters[i] = None
                    continue

                size_param_name = SIZE_PARAMS.get(function.name.format(), {}).get(
                    param.name
                )
                try:
                    size_param = next(
                        p for p in function.parameters if p.name == size_param_name
                    )
                except StopIteration:
                    print(function.parameters)
                    raise NotImplementedError(
                        "Void pointer parameters must have a size parameter."
                    )

                if isinstance(size_param.type, Type):
                    # the size parameter is a value type, so we can just pass the size directly.
                    parameters[i] = ServerRpcParameter(
                        write_decl="void *%s = malloc(%s);"
                        % (param.name, size_param.name),
                        write=(param.name, size_param.name),
                        read=(param.name, size_param.name),
                    )
                elif isinstance(size_param.type, Pointer):
                    # the size parameter is a pointer, so we need to dereference it.
                    parameters[i] = ServerRpcParameter(
                        write_decl="void *%s = malloc(*%s);"
                        % (param.name, size_param.name),
                        write=(param.name, "*%s" % size_param.name),
                        read=(param.name, "*%s" % size_param.name),
                    )
                else:
                    raise NotImplementedError(
                        "Unknown size parameter type: %s" % size_param.type
                    )

            elif not param.type.ptr_to.const:
                # handle T* params
                # these differ from const void*/void* because they may not have a size parameter, in which
                # case this is a read/write parameter. generally, it is too ambiguous to determine if these are
                # read-only or write-only, so we will treat them as read/write.
                size_param_name = SIZE_PARAMS.get(function.name.format(), {}).get(
                    param.name
                )
                try:
                    size_param = next(
                        p for p in function.parameters if p.name == size_param_name
                    )
                    # treat this as an array parameter
                    if isinstance(size_param.type, Type):
                        # the size parameter is a value type, so we can just pass the size directly.
                        parameters[i] = ServerRpcParameter(
                            # no decl necessary because this is read/write so we already have a malloc from client
                            write=(
                                "%s" % param.name,
                                "%s * sizeof(%s)"
                                % (size_param.name, param.type.ptr_to.format()),
                            ),
                            read=(
                                "%s" % param.name,
                                "%s * sizeof(%s)"
                                % (size_param.name, param.type.ptr_to.format()),
                            ),
                        )
                    elif isinstance(size_param.type, Pointer):
                        # the size parameter is a pointer, so we need to dereference it.
                        parameters[i] = ServerRpcParameter(
                            # no decl necessary because this is read/write so we already have a malloc from client
                            write=(
                                "%s" % param.name,
                                "%s * sizeof(%s)"
                                % (size_param.name, param.type.ptr_to.format()),
                            ),
                            read=(
                                "%s" % param.name,
                                "*%s * sizeof(%s)"
                                % (size_param.name, param.type.ptr_to.format()),
                            ),
                        )
                    else:
                        raise NotImplementedError(
                            "Unknown size parameter type: %s" % size_param.type
                        )
                except StopIteration:
                    # treat this as a value parameter
                    parameters[i] = ServerRpcParameter(
                        # no decl necessary because this is read/write so we already have a malloc from client
                        write=(
                            "&%s" % param.name,
                            "sizeof(%s)" % param.type.ptr_to.format(),
                        ),
                        read=(
                            "%s" % param.name,
                            "sizeof(%s)" % param.type.ptr_to.format(),
                        ),
                    )
            else:
                # handle const T* params
                # these are read-only parameters, ignore
                parameters[i] = None
    return [p for p in parameters if p]


def main():
    options = ParserOptions(preprocessor=make_gcc_preprocessor())

    nvml_ast: ParsedData = parse_file("/usr/include/nvml.h", options=options)
    cuda_ast: ParsedData = parse_file("/usr/include/cuda.h", options=options)
    cudart_ast: ParsedData = parse_file(
        "/usr/include/cuda_runtime_api.h", options=options
    )

    functions = (
        nvml_ast.namespace.functions
        + cuda_ast.namespace.functions
        + cudart_ast.namespace.functions
    )
    for f in functions:
        if f.return_type.format() not in ["nvmlReturn_t", "CUresult", "cudaError_t"]:
            print("Skipping function: %s" % f.name.format())
    functions = [
        f
        for f in functions
        if f.return_type.format() in ["nvmlReturn_t", "CUresult", "cudaError_t"]
    ]
    functions = [f for f in functions if f.name.format() not in IGNORE_FUNCTIONS]

    read_params = [client_write_pass(f) for f in functions]
    write_params = [server_write_pass(f) for f in functions]

    with open("gen_api.h", "w") as f:
        for i, function in enumerate(functions):
            f.write(
                "#define RPC_{name} {index}\n".format(
                    name=function.name.format(), index=i
                )
            )

    with open("gen_client.cpp", "w") as f:
        f.write(
            """
// Generated code.

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
                
#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

extern int rpc_start_request(const unsigned int request);
extern int rpc_write(const void *data, const size_t size);
extern int rpc_read(void *data, const size_t size);
extern int rpc_wait_for_response(const unsigned int request_id);
extern int rpc_end_request(void *return_value, const unsigned int request_id);

""".lstrip()
        )

        for i, function in enumerate(functions):
            # construct the function signature
            f.write(
                """
{doxygen}{return_type} {name}({params}) {{
    {return_type} return_value;
    {decls}
    int request_id = rpc_start_request(RPC_{name});
    if (request_id < 0 ||\n""".format(
                    doxygen=function.doxygen + "\n" if function.doxygen else "",
                    return_type=function.return_type.format(),
                    name=function.name.format(),
                    decls="\n    ".join(
                        p.write_decl for p in read_params[i] if p.write_decl
                    ),
                    params=", ".join(
                        p.type.format_decl(p.name) for p in function.parameters
                    ),
                ).lstrip()
            )
            # write the entire parameter list as a block of memory
            for param in read_params[i]:
                f.write("        rpc_write(%s, %s) < 0 ||\n" % param.write)
            # wait for response
            f.write("        rpc_wait_for_response(request_id) < 0 ||\n")
            # read responses back
            for param in write_params[i]:
                f.write("        rpc_read(%s, %s) < 0 ||\n" % param.read)
            f.write(
                """        rpc_end_request(&return_value, request_id) < 0)
        return {error_const};
    return return_value;
}}

""".format(
                    error_const=error_const(function.return_type.format())
                )
            )
        # write the trailer
        f.write(
            """
std::unordered_map<std::string, void *> functionMap = {
"""
        )
        for function in functions:
            f.write(
                '    {"%s", (void *)%s},\n'
                % (function.name.format(), function.name.format())
            )
        for a, b in UPGRADES:
            f.write('    {"%s", (void *)%s},\n' % (a, b))
        f.write(
            """
};

void *get_function_pointer(const char *name)
{
    auto it = functionMap.find(name);
    if (it != functionMap.end())
        return it->second;
    return nullptr;
}
"""
        )

    with open("gen_server.cpp", "w") as f:
        f.write(
            "// Generated code.\n\n"
            "#include <cuda.h>\n"
            "#include <cuda_runtime_api.h>\n"
            "#include <nvml.h>\n"
            "#include <string>\n"
            "#include <unordered_map>\n\n"
            '#include "gen_api.h"\n'
            '#include "gen_server.h"\n\n'
            "extern int rpc_read(const void *conn, void *data, const size_t size);"
            "extern int rpc_write(const void *conn, const void *data, const size_t size);"
            "extern int rpc_end_request(const void *conn);"
            "extern int rpc_start_response(const void *conn, const int request_id);"
        )
        for i, function in enumerate(functions):
            # construct the function signature
            f.write(
                "{doxygen}int handle_{name}(void *conn) {{\n".format(
                    doxygen=function.doxygen + "\n" if function.doxygen else "",
                    name=function.name.format(),
                )
            )

            # read parameters from the client
            for param in read_params[i]:
                f.write("    %s\n" % param.read[0])
                f.write(
                    "    if (rpc_read(conn, %s, %s) < 0)\n        return -1;\n"
                    % param.read[1:]
                )

            f.write("    int request_id = rpc_end_request(conn);\n")
            f.write("    if (request_id < 0)\n")
            f.write("        return -1;\n\n")

            # call the function
            f.write(
                "    %s = %s("
                % (
                    function.return_type.format_decl(name="result"),
                    function.name.format(),
                )
            )
            for j, param in enumerate(function.parameters):
                if param.name is None:
                    f.write("nullptr")
                else:
                    if (
                        isinstance(param.type, Pointer)
                        and param.type.ptr_to.format() != "const char"
                    ):
                        size_param_name = SIZE_PARAMS.get(
                            function.name.format(), {}
                        ).get(param.name)
                        if size_param_name:
                            f.write(param.name)
                        else:
                            f.write("&%s" % param.name)
                    else:
                        f.write(param.name)
                if j < len(function.parameters) - 1:
                    f.write(", ")
            f.write(");\n\n")

            f.write("    if (rpc_start_response(conn, request_id) < 0)\n")
            f.write("        return -1;\n\n")

            # write pointer vars
            for param in write_params[i]:
                if param.write_decl:
                    f.write("    %s\n" % param.write_decl)
            for param in write_params[i]:
                f.write(
                    "    if (rpc_write(conn, %s, %s) < 0)\n        return -1;\n"
                    % param.write
                )
            f.write("    return result;\n")
            f.write("}\n\n")

        f.write("static RequestHandler opHandlers[] = {\n")
        for function in functions:
            f.write("    handle_%s,\n" % (function.name.format()))
        f.write("};\n\n")

        f.write("RequestHandler get_handler(const int op)\n")
        f.write("{\n")
        f.write("    return opHandlers[op];\n")
        f.write("}\n")


if __name__ == "__main__":
    main()
