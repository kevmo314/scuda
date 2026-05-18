from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Type, Pointer, Parameter, Function, Array
from typing import Optional, Union
from dataclasses import dataclass
import os
import glob

# this table is manually generated from the cuda.h headers
MANUAL_REMAPPINGS = [
    ("cuDeviceTotalMem", "cuDeviceTotalMem_v2"),
    ("cuDeviceGetUuid", "cuDeviceGetUuid_v2"),
    ("cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease_v2"),
    ("cuDevicePrimaryCtxSetFlags", "cuDevicePrimaryCtxSetFlags_v2"),
    ("cuDevicePrimaryCtxReset", "cuDevicePrimaryCtxReset_v2"),
    ("cuCtxDestroy", "cuCtxDestroy_v2"),
    ("cuCtxPopCurrent", "cuCtxPopCurrent_v2"),
    ("cuCtxPushCurrent", "cuCtxPushCurrent_v2"),
    ("cuModuleGetGlobal", "cuModuleGetGlobal_v2"),
    ("cuMemAlloc", "cuMemAlloc_v2"),
    ("cuMemAllocPitch", "cuMemAllocPitch_v2"),
    ("cuMemFree", "cuMemFree_v2"),
    ("cuMemcpyHtoD", "cuMemcpyHtoD_v2"),
    ("cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2"),
    ("cuMemcpyDtoH", "cuMemcpyDtoH_v2"),
    ("cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2"),
    ("cuMemcpyDtoD", "cuMemcpyDtoD_v2"),
    ("cuMemcpyDtoDAsync", "cuMemcpyDtoDAsync_v2"),
    ("cuMemsetD8", "cuMemsetD8_v2"),
    ("cuStreamBeginCapture", "cuStreamBeginCapture_v2"),
    ("cuGraphAddKernelNode", "cuGraphAddKernelNode_v2"),
    ("cuGraphExecUpdate", "cuGraphExecUpdate_v2"),
    ("cuMemcpy_ptds", "cuMemcpy"),
    ("cuMemcpyAsync_ptsz", "cuMemcpyAsync"),
    ("cuMemcpyPeer_ptds", "cuMemcpyPeer"),
    ("cuMemcpyPeerAsync_ptsz", "cuMemcpyPeerAsync"),
    ("cuMemcpy3DPeer_ptds", "cuMemcpy3DPeer"),
    ("cuMemcpy3DPeerAsync_ptsz", "cuMemcpy3DPeerAsync"),
    ("cuMemPrefetchAsync_ptsz", "cuMemPrefetchAsync"),
    ("cuMemsetD8Async_ptsz", "cuMemsetD8Async"),
    ("cuMemsetD16Async_ptsz", "cuMemsetD16Async"),
    ("cuMemsetD32Async_ptsz", "cuMemsetD32Async"),
    ("cuMemsetD2D8Async_ptsz", "cuMemsetD2D8Async"),
    ("cuMemsetD2D16Async_ptsz", "cuMemsetD2D16Async"),
    ("cuMemsetD2D32Async_ptsz", "cuMemsetD2D32Async"),
    ("cuStreamGetPriority_ptsz", "cuStreamGetPriority"),
    ("cuStreamGetId_ptsz", "cuStreamGetId"),
    ("cuStreamGetFlags_ptsz", "cuStreamGetFlags"),
    ("cuStreamGetCtx_ptsz", "cuStreamGetCtx"),
    ("cuStreamWaitEvent_ptsz", "cuStreamWaitEvent"),
    ("cuStreamEndCapture_ptsz", "cuStreamEndCapture"),
    ("cuStreamIsCapturing_ptsz", "cuStreamIsCapturing"),
    ("cuStreamUpdateCaptureDependencies_ptsz", "cuStreamUpdateCaptureDependencies"),
    ("cuStreamAddCallback_ptsz", "cuStreamAddCallback"),
    ("cuStreamAttachMemAsync_ptsz", "cuStreamAttachMemAsync"),
    ("cuStreamQuery_ptsz", "cuStreamQuery"),
    ("cuStreamSynchronize_ptsz", "cuStreamSynchronize"),
    ("cuEventRecord_ptsz", "cuEventRecord"),
    ("cuEventRecordWithFlags_ptsz", "cuEventRecordWithFlags"),
    ("cuLaunchKernel_ptsz", "cuLaunchKernel"),
    ("cuLaunchKernelEx_ptsz", "cuLaunchKernelEx"),
    ("cuLaunchHostFunc_ptsz", "cuLaunchHostFunc"),
    ("cuGraphicsMapResources_ptsz", "cuGraphicsMapResources"),
    ("cuGraphicsUnmapResources_ptsz", "cuGraphicsUnmapResources"),
    ("cuLaunchCooperativeKernel_ptsz", "cuLaunchCooperativeKernel"),
    ("cuSignalExternalSemaphoresAsync_ptsz", "cuSignalExternalSemaphoresAsync"),
    ("cuWaitExternalSemaphoresAsync_ptsz", "cuWaitExternalSemaphoresAsync"),
    ("cuGraphInstantiateWithParams_ptsz", "cuGraphInstantiateWithParams"),
    ("cuGraphUpload_ptsz", "cuGraphUpload"),
    ("cuGraphLaunch_ptsz", "cuGraphLaunch"),
    ("cuStreamCopyAttributes_ptsz", "cuStreamCopyAttributes"),
    ("cuStreamGetAttribute_ptsz", "cuStreamGetAttribute"),
    ("cuStreamSetAttribute_ptsz", "cuStreamSetAttribute"),
    ("cuMemMapArrayAsync_ptsz", "cuMemMapArrayAsync"),
    ("cuMemFreeAsync_ptsz", "cuMemFreeAsync"),
    ("cuMemAllocAsync_ptsz", "cuMemAllocAsync"),
    ("cuMemAllocFromPoolAsync_ptsz", "cuMemAllocFromPoolAsync"),
]

NVML_RPC_FUNCTIONS = [
    "nvmlInit_v2",
    "nvmlInitWithFlags",
    "nvmlShutdown",
    "nvmlSystemGetDriverVersion",
    "nvmlSystemGetNVMLVersion",
    "nvmlSystemGetCudaDriverVersion",
    "nvmlSystemGetCudaDriverVersion_v2",
    "nvmlDeviceGetCount_v2",
    "nvmlDeviceGetHandleByIndex_v2",
    "nvmlDeviceGetHandleByUUID",
    "nvmlDeviceGetHandleByPciBusId_v2",
    "nvmlDeviceGetName",
    "nvmlDeviceGetUUID",
    "nvmlDeviceGetIndex",
    "nvmlDeviceGetMinorNumber",
    "nvmlDeviceGetPciInfo_v3",
    "nvmlDeviceGetMemoryInfo",
    "nvmlDeviceGetUtilizationRates",
    "nvmlDeviceGetTemperature",
    "nvmlDeviceGetPowerUsage",
    "nvmlDeviceGetPowerManagementLimit",
    "nvmlDeviceGetClockInfo",
    "nvmlDeviceGetMaxClockInfo",
    "nvmlDeviceGetPerformanceState",
    "nvmlDeviceGetComputeMode",
    "nvmlDeviceGetPersistenceMode",
    "nvmlDeviceGetFanSpeed",
    "nvmlDeviceGetBrand",
    "nvmlDeviceGetVbiosVersion",
    "nvmlDeviceGetSerial",
    "nvmlDeviceGetBoardPartNumber",
    "nvmlDeviceGetDisplayMode",
    "nvmlDeviceGetDisplayActive",
    "nvmlDeviceGetCurrPcieLinkGeneration",
    "nvmlDeviceGetCurrPcieLinkWidth",
    "nvmlDeviceGetMaxPcieLinkGeneration",
    "nvmlDeviceGetMaxPcieLinkWidth",
    "nvmlDeviceGetPcieThroughput",
    "nvmlDeviceGetPcieReplayCounter",
    "nvmlDeviceGetComputeRunningProcesses",
    "nvmlDeviceGetComputeRunningProcesses_v2",
    "nvmlDeviceGetGraphicsRunningProcesses",
    "nvmlDeviceGetGraphicsRunningProcesses_v2",
    "nvmlDeviceGetMPSComputeRunningProcesses",
    "nvmlDeviceGetMPSComputeRunningProcesses_v2",
    "nvmlEventSetCreate",
    "nvmlEventSetFree",
    "nvmlEventSetWait_v2",
    "nvmlDeviceRegisterEvents",
    "nvmlDeviceGetMaxMigDeviceCount",
    "nvmlDeviceGetEccMode",
    "nvmlDeviceGetTemperatureV",
    "nvmlDeviceGetEnforcedPowerLimit",
    "nvmlDeviceGetMemoryInfo_v2",
    "nvmlDeviceGetMigMode",
    "nvmlDeviceGetVirtualizationMode",
    "nvmlDeviceIsMigDeviceHandle",
]

SKIP_FUNCTIONS = {
    "cuStreamUpdateCaptureDependencies_v2",
    "cuGraphGetEdges_v2",
    "cuGraphNodeGetDependencies_v2",
    "cuGraphNodeGetDependentNodes_v2",
    "cuGraphAddDependencies_v2",
    "cuGraphRemoveDependencies_v2",
}


@dataclass
class NullableOperation:
    """
    Nullable operations are operations that are passed as a pointer that can be null.
    """

    send: bool
    recv: bool
    parameter: Parameter
    ptr: Pointer

    def client_rpc_write(self, f):
        if not self.send:
            return
        f.write(
            "        rpc_write(conn, &{param_name}, sizeof({server_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )

        f.write(
            "        ({param_name} != nullptr && rpc_write(conn, {param_name}, sizeof({base_type})) < 0) ||\n".format(
                param_name=self.parameter.name,
                # void is treated differently from non void pointer types
                base_type=(
                    self.ptr.format()
                    if self.ptr.ptr_to.format() == "const void"
                    else self.ptr.ptr_to.format()
                ),
            )
        )

    def client_unified_copy(self, f, direction, error):
        f.write(
            "    if (maybe_copy_unified_arg(conn, (void*){name}, cudaMemcpyDeviceToHost) < 0)\n".format(
                name=self.parameter.name
            )
        )
        f.write("      return {error};\n".format(error=error))

    @property
    def server_declaration(self) -> str:
        c = self.ptr.ptr_to.const
        self.ptr.ptr_to.const = False
        # void is treated differently from non void pointer types
        s = (
            f"    {self.ptr.format()} {self.parameter.name}_null_check;\n"
            + f"""    {self.ptr.format() if self.ptr.ptr_to.format() == "void" else self.ptr.ptr_to.format()} {self.parameter.name};\n"""
        )
        self.ptr.ptr_to.const = c
        return s

    def server_rpc_read(self, f):
        if not self.send:
            return
        f.write(
            "        rpc_read(conn, &{param_name}_null_check, sizeof({server_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )
        f.write(
            "        ({param_name}_null_check && rpc_read(conn, &{param_name}, sizeof({base_type})) < 0) ||\n".format(
                param_name=self.parameter.name,
                # void is treated differently from non void pointer types
                base_type=(
                    self.ptr.format()
                    if self.ptr.ptr_to.format() == "const void"
                    else self.ptr.ptr_to.format()
                ),
            )
        )

    @property
    def server_reference(self) -> str:
        return f"&{self.parameter.name}"

    def server_rpc_write(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_write(conn, &{param_name}_null_check, sizeof({server_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )
        f.write(
            "        ({param_name}_null_check && rpc_write(conn, {param_name}, sizeof({base_type})) < 0) ||\n".format(
                param_name=self.parameter.name,
                base_type=self.ptr.ptr_to.format(),
            )
        )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_read(conn, &{param_name}_null_check, sizeof({server_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )
        f.write(
            "        ({param_name}_null_check && rpc_read(conn, {param_name}, sizeof({base_type})) < 0) ||\n".format(
                param_name=self.parameter.name,
                base_type=self.ptr.ptr_to.format(),
            )
        )


@dataclass
class ArrayOperation:
    """
    Array operations are operations that are passed as a pointer to an array.
    """

    send: bool
    recv: bool
    iter: bool
    parameter: Parameter
    ptr: Pointer
    # if int, it's a constant length, if Parameter, it's a variable length.
    length: Union[int, Parameter]

    @property
    def is_void_bytes(self) -> bool:
        return self.ptr.ptr_to.format() in ("void", "const void")

    def byte_count_expr(self) -> str:
        if isinstance(self.length, int):
            return str(self.length)
        if isinstance(self.length.type, Pointer):
            return f"*{self.length.name}"
        return self.length.name

    def element_count_expr(self) -> str:
        if isinstance(self.length, int):
            return str(self.length)
        if isinstance(self.length.type, Pointer):
            return f"*{self.length.name}"
        return self.length.name

    def transfer_size_expr(self) -> str:
        if self.is_void_bytes:
            return self.byte_count_expr()
        return f"{self.element_count_expr()} * sizeof({self.ptr.ptr_to.format()})"

    def mutable_ptr_format(self) -> str:
        c = self.ptr.ptr_to.const
        self.ptr.ptr_to.const = False
        result = self.ptr.format()
        self.ptr.ptr_to.const = c
        return result

    def client_rpc_write(self, f):
        if self.iter:
            loop_template = """
                [=]() -> bool {{
                    for (size_t i = 0; i < {length}; ++i) {{
                        if (rpc_write(0, &{param_name}[i], sizeof({param_type})) < 0) {{
                            printf("Failed to write Dependency[%zu]\\n", i);
                            return false;
                        }}
                    }}
                    return true;
                }}() == false ||
                """.strip()

            f.write(loop_template.format(
                length=self.length.name,
                param_type=self.ptr.ptr_to.format(),
                param_name=self.parameter.name,
            ))
            return

        if not self.send:
            return
        elif isinstance(self.length, int):
            f.write(
                "        rpc_write(conn, {param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.length,
                )
            )
        # array length operations are handled differently than char
        elif isinstance(self.ptr, Array):
            f.write(
                "        rpc_write(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    param_type=self.ptr.array_of.format(),
                )
            )
        else:
            f.write(
                "        rpc_write(conn, {param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.transfer_size_expr(),
                )
            )

    def client_unified_copy(self, f, direction, error):
        f.write(
            "    if (maybe_copy_unified_arg(conn, (void*){name}, {direction}) < 0)\n".format(
                name=self.parameter.name, direction=direction
            )
        )
        f.write("      return {error};\n".format(error=error))

        if isinstance(self.length, int):
            f.write(
                "    for (int i = 0; i < {name} && is_unified_pointer(conn, (void*){param}); i++)\n".format(
                    param=self.parameter.name, name=self.length
                )
            )
            f.write(
                "      if (maybe_copy_unified_arg(conn, (void*)&{name}[i], {direction}) < 0 )\n".format(
                    name=self.parameter.name, direction=direction
                )
            )
            f.write("        return {error};\n".format(error=error))

            return

        if hasattr(self.length.type, "ptr_to"):
            # need to cast the int a bit differently here
            f.write(
                "    for (int i = 0; i < static_cast<int>(*{name}) && is_unified_pointer(conn, (void*){param}); i++)\n".format(
                    param=self.parameter.name, name=self.length.name
                )
            )
            f.write(
                "      if (maybe_copy_unified_arg(conn, (void*)&{name}[i], {direction}) < 0)\n".format(
                    name=self.parameter.name, direction=direction
                )
            )
            f.write("        return {error};\n".format(error=error))
        else:
            if hasattr(self.parameter.type, "ptr_to"):
                f.write(
                    "    for (int i = 0; i < static_cast<int>({name}) && is_unified_pointer(conn, (void*){param}); i++)\n".format(
                        param=self.parameter.name, name=self.length.name
                    )
                )
                f.write(
                    "      if (maybe_copy_unified_arg(conn, (void*)&{name}[i], {direction}) < 0)\n".format(
                        name=self.parameter.name, direction=direction
                    )
                )
                f.write("        return {error};\n".format(error=error))
            else:
                f.write(
                    "    for (int i = 0; i < static_cast<int>({name}) && is_unified_pointer(conn, (void*){param}); i++)\n".format(
                        param=self.parameter.name, name=self.length.name
                    )
                )
                f.write(
                    "      if (maybe_copy_unified_arg(conn, (void*){name}[i], {direction}) < 0)\n".format(
                        name=self.parameter.name, direction=direction
                    )
                )
                f.write("        return {error};\n".format(error=error))

    @property
    def server_declaration(self) -> str:
        if isinstance(self.ptr, Array):
            c = self.ptr.array_of.const
            self.ptr.array_of.const = False
            s = f"    {self.ptr.array_of.format()}* {self.parameter.name} = nullptr;\n"
            self.ptr.array_of.const = c
        elif self.iter:
            c = self.ptr.ptr_to.const
            self.ptr.ptr_to.const = False
            s = f"    std::vector<{self.ptr.ptr_to.format()}> {self.parameter.name};\n"
            self.ptr.ptr_to.const = c
        else:
            c = self.ptr.ptr_to.const
            self.ptr.ptr_to.const = False
            s = f"    {self.ptr.format()} {self.parameter.name};\n"
            self.ptr.ptr_to.const = c
        return s

    def server_rpc_read(self, f, index) -> Optional[str]:
        if self.iter:
            lambda_template = """
            [=, &{param_name}]() -> bool {{
                {param_name}.resize({length});  // Resize the dependencies vector
                for (size_t i = 0; i < {length}; ++i) {{
                    if (rpc_read(conn, &{param_name}[i], sizeof({param_type})) < 0) {{
                        return false;
                    }}
                }}
                return true;
            }}() == false ||
            """.strip()

            f.write(lambda_template.format(
                param_name=self.parameter.name,
                length=self.length.name,
                param_type=self.ptr.ptr_to.format(),
            ))
            return

        if not self.send:
            # if this parameter is recv only and it's a type pointer, it needs to be malloc'd.
            if isinstance(self.ptr, Pointer):
                f.write("        false)\n")
                f.write("        goto ERROR_{index};\n".format(index=index))
                f.write(
                    "    {param_name} = ({server_type})malloc({size});\n".format(
                        param_name=self.parameter.name,
                        server_type=self.ptr.format(),
                        size=self.transfer_size_expr(),
                    )
                )
                f.write("    if(")
                return self.parameter.name
            return
        elif isinstance(self.ptr, Pointer):
            f.write("        false)\n")
            f.write("        goto ERROR_{index};\n".format(index=index))
            f.write(
                "    {param_name} = ({server_type})malloc({size});\n".format(
                    param_name=self.parameter.name,
                    server_type=self.mutable_ptr_format(),
                    size=self.transfer_size_expr(),
                )
            )
            f.write("    if ({param_name} == nullptr)\n".format(
                param_name=self.parameter.name
            ))
            f.write("        goto ERROR_{index};\n".format(index=index))
            f.write("    if(\n")
            f.write(
                "        rpc_read(conn, {param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.transfer_size_expr(),
                )
            )
            defer = self.parameter.name
        elif isinstance(self.length, int):
            f.write(
                "        rpc_read(conn, &{param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.length,
                )
            )
        elif isinstance(self.ptr, Array):
            f.write(
                "        rpc_read(conn, &{param_name}, sizeof({param_type}*)) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    param_type=self.ptr.array_of.format(),
                )
            )
        else:
            f.write(
                "        rpc_read(conn, {param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.transfer_size_expr(),
                )
            )
        if 'defer' in locals():
            return defer

    @property
    def server_reference(self) -> str:
        if self.iter:
            return f"{self.parameter.name}.data()"

        return self.parameter.name

    def server_rpc_write(self, f):
        if not self.recv:
            return
        if isinstance(self.length, int):
            f.write(
                "        rpc_write(conn, {param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.length,
                )
            )
        else:
            f.write(
                "        rpc_write(conn, {param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.transfer_size_expr(),
                )
            )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        if isinstance(self.length, int):
            f.write(
                "        rpc_read(conn, {param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.length,
                )
            )
        else:
            f.write(
                "        rpc_read(conn, {param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.transfer_size_expr(),
                )
            )


@dataclass
class NullTerminatedOperation:
    """
    Null terminated operations are operations that are passed as a null terminated string.
    """

    send: bool
    recv: bool
    parameter: Parameter
    ptr: Pointer

    def client_rpc_write(self, f):
        if not self.send:
            return
        f.write(
            "        rpc_write(conn, &{param_name}_len, sizeof(std::size_t)) < 0 ||\n".format(
                param_name=self.parameter.name,
            )
        )
        f.write(
            "        rpc_write(conn, {param_name}, {param_name}_len) < 0 ||\n".format(
                param_name=self.parameter.name,
            )
        )

    @property
    def server_declaration(self) -> str:
        return (
            f"    {self.ptr.format()} {self.parameter.name};\n"
            + f"    std::size_t {self.parameter.name}_len;\n"
        )

    def client_unified_copy(self, f, direction, error):
        f.write(
            "    if (maybe_copy_unified_arg(conn, (void*){name}, {direction}) < 0)\n".format(
                name=self.parameter.name, direction=direction
            )
        )
        f.write("      return {error};\n".format(error=error))

    def server_rpc_read(self, f, index) -> Optional[str]:
        if not self.send:
            return
        f.write(
            "        rpc_read(conn, &{param_name}_len, sizeof(std::size_t)) < 0)\n".format(
                param_name=self.parameter.name
            )
        )
        f.write("        goto ERROR_{index};\n".format(index=index))
        f.write(
            "    {param_name} = ({server_type})malloc({param_name}_len);\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )
        f.write(
            "    if (rpc_read(conn, (void *){param_name}, {param_name}_len) < 0 ||\n".format(
                param_name=self.parameter.name
            )
        )
        return self.parameter.name

    @property
    def server_reference(self) -> str:
        return self.parameter.name

    def server_rpc_write(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_write(conn, &{param_name}_len, sizeof(std::size_t)) < 0 ||\n".format(
                param_name=self.parameter.name,
            )
        )
        f.write(
            "        rpc_write(conn, {param_name}, {param_name}_len) < 0 ||\n".format(
                param_name=self.parameter.name,
            )
        )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_read(conn, &{param_name}_len, sizeof(std::size_t)) < 0 ||\n".format(
                param_name=self.parameter.name
            )
        )
        f.write(
            "        rpc_read(conn, {param_name}, {param_name}_len) < 0 ||\n".format(
                param_name=self.parameter.name
            )
        )


@dataclass
class OpaqueTypeOperation:
    """
    Opaque type operations are operations that are passed as an opaque type. That is, the
    data is written directly without any additional dereferencing.
    """

    send: bool
    recv: bool
    parameter: Parameter
    type_: Union[Type, Pointer]

    def client_rpc_write(self, f):
        if not self.send:
            return
        else:
            f.write(
                "        rpc_write(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    param_type=self.type_.format(),
                )
            )

    @property
    def server_declaration(self) -> str:
        if isinstance(self.type_, Pointer) and self.recv:
            return f"    {self.type_.ptr_to.format()} {self.parameter.name};\n"
        # ensure we don't have a const struct, otherwise we can't initialise it properly; ex: "const cudnnTensorDescriptor_t xDesc;" is invalid...
        # but "const cudnnTensorDescriptor_t *xDesc" IS valid. This subtle change carries reprecussions.
        elif (
            "const " in self.type_.format()
            and "void" not in self.type_.format()
            and "*" not in self.type_.format()
        ):
            return f"   {self.type_.format().replace('const', '')} {self.parameter.name};\n"
        else:
            return f"    {self.type_.format()} {self.parameter.name};\n"

    def client_unified_copy(self, f, direction, error):
        if isinstance(self.type_, Pointer):
            f.write(
                "    if (maybe_copy_unified_arg(conn, (void*){name}, {direction}) < 0)\n".format(
                    name=self.parameter.name, direction=direction
                )
            )
            f.write("      return {error};\n".format(error=error))
        else:
            f.write(
                "    if (maybe_copy_unified_arg(conn, (void*)&{name}, {direction}) < 0)\n".format(
                    name=self.parameter.name, direction=direction
                )
            )
            f.write("      return {error};\n".format(error=error))

    def server_rpc_read(self, f):
        if not self.send:
            return
        f.write(
            "        rpc_read(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.format(),
            )
        )

    @property
    def server_reference(self) -> str:
        if self.recv:
            return f"&{self.parameter.name}"
        return self.parameter.name

    def server_rpc_write(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_write(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.format(),
            )
        )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_read(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.format(),
            )
        )


@dataclass
class DereferenceOperation:
    """
    Opaque type operations are operations that are passed as an opaque type. That is, the
    data is written directly without any additional dereferencing.
    """

    send: bool
    recv: bool
    parameter: Parameter
    type_: Pointer

    def client_rpc_write(self, f):
        if not self.send:
            return
        f.write(
            "        rpc_write(conn, {param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.ptr_to.format(),
            )
        )

    @property
    def server_declaration(self) -> str:
        c = self.type_.ptr_to.const
        self.type_.ptr_to.const = False
        result = f"    {self.type_.ptr_to.format()} {self.parameter.name};\n"
        self.type_.ptr_to.const = c
        return result

    def server_rpc_read(self, f):
        if not self.send:
            return
        f.write(
            "        rpc_read(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.ptr_to.format(),
            )
        )

    def client_unified_copy(self, f, direction, error):
        f.write(
            "    if (maybe_copy_unified_arg(conn, (void*){name}, {direction}) < 0)\n".format(
                name=self.parameter.name, direction=direction
            )
        )
        f.write("      return {error};\n".format(error=error))

    @property
    def server_reference(self) -> str:
        return f"&{self.parameter.name}"

    def server_rpc_write(self, f):
        if not self.recv:
            return
        f.write(
            "        rpc_write(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.ptr_to.format(),
            )
        )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        # if this parameter is recv only then dereference it.
        f.write(
            "        rpc_read(conn, {param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.ptr_to.format(),
            )
        )


Operation = Union[
    NullableOperation,
    ArrayOperation,
    NullTerminatedOperation,
    OpaqueTypeOperation,
    DereferenceOperation,
]


# parses a function annotation. if disabled is encountered, returns True for short circuiting.
def parse_annotation(
    annotation: str, params: list[Parameter]
) -> list[tuple[Operation, bool]]:
    operations: list[Operation] = []

    if not annotation:
        return operations
    for line in annotation.split("\n"):
        # we treat disabled functions a bit differently.
        # we should record that this function is disabled so that we can create the RPC method but not the actual handler.
        # this is so that we don't break API compatability by commenting/uncommenting functions from our annotations file.
        if "@disabled" in line or "@DISABLED" in line:
            # we can return instantly; the function is disabled and we don't care about operations at this time.
            return operations, True
        if line.startswith("/**"):
            continue
        if line.startswith("*/"):
            continue
        if line.startswith("*"):
            line = line[2:]
        if line.startswith("@param"):
            parts = line.split()

            if len(parts) < 3:
                continue
            try:
                param = next(p for p in params if p.name == parts[1])
            except StopIteration:
                raise NotImplementedError(f"Parameter {parts[1]} not found")
            args = parts[3:]
            send = parts[2] == "SEND_ONLY" or parts[2] == "SEND_RECV"
            recv = parts[2] == "RECV_ONLY" or parts[2] == "SEND_RECV"

            # if there's a length or size arg, use the type, otherwise use the ptr_to type
            length_arg = next((arg for arg in args if arg.startswith("LENGTH:")), None)

            if isinstance(param.type, Pointer):
                if param.type.ptr_to.const:
                    recv = False

                size_arg = next((arg for arg in args if arg.startswith("SIZE:")), None)
                iter_arg = next((arg for arg in args if arg.startswith("ITER:")), None)
                null_terminated = "NULL_TERMINATED" in args
                nullable = "NULLABLE" in args
                deref = "DEREF" in args

                # validate that only one of the arguments is present
                if (
                    sum([bool(length_arg), bool(size_arg), null_terminated, nullable])
                    > 1
                ):
                    raise NotImplementedError(
                        "Only one of LENGTH, SIZE, NULL_TERMINATED, or NULLABLE can be specified"
                    )

                if deref:
                    operations.append(
                        DereferenceOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            type_=param.type,
                        )
                    )
                elif length_arg:
                    # if it has a length, it's an array operation with variable length
                    length_param = next(
                        p for p in params if p.name == length_arg.split(":")[1]
                    )
                    operations.append(
                        ArrayOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            ptr=param.type,
                            length=length_param,
                            iter=False
                        )
                    )
                elif size_arg:
                    # if it has a size, it's an array operation with constant length
                    operations.append(
                        ArrayOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            ptr=param.type,
                            length=int(size_arg.split(":")[1]),
                            iter=False
                        )
                    )
                elif iter_arg:
                    print(f"ITER FOUND!! {param}")
                    length_param = next(
                        p for p in params if p.name == iter_arg.split(":")[1]
                    )
                    operations.append(
                        ArrayOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            ptr=param.type,
                            length=length_param,
                            iter=True
                        )
                    )
                elif null_terminated:
                    # if it's null terminated, it's a null terminated operation
                    operations.append(
                        NullTerminatedOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            ptr=param.type,
                        )
                    )
                elif nullable:
                    # if it's nullable, it's a nullable operation
                    operations.append(
                        NullableOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            ptr=param.type,
                        )
                    )
                else:
                    # otherwise, it's a pointer to a single value or another pointer
                    if recv:
                        if param.type.ptr_to.format() == "void":
                            raise NotImplementedError(
                                "Cannot dereference a void pointer"
                            )
                        # this is an out parameter so use the base type as the server declaration
                        operations.append(
                            DereferenceOperation(
                                send=send,
                                recv=recv,
                                parameter=param,
                                type_=param.type,
                            )
                        )
                    else:
                        # otherwise, treat it as an opaque type
                        operations.append(
                            OpaqueTypeOperation(
                                send=send,
                                recv=recv,
                                parameter=param,
                                type_=param.type,
                            )
                        )
            elif isinstance(param.type, Type):
                if param.type.const:
                    recv = False
                operations.append(
                    OpaqueTypeOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        type_=param.type,
                    )
                )
            elif isinstance(param.type, Array):
                length_param = next(
                    p for p in params if p.name == length_arg.split(":")[1]
                )
                if param.type.const:
                    recv = False
                operations.append(
                    ArrayOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        ptr=param.type,
                        length=length_param,
                    )
                )
            elif size_arg:
                # if it has a size, it's an array operation with constant length
                operations.append(
                    ArrayOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        ptr=param.type,
                        length=int(size_arg.split(":")[1]),
                    )
                )
            elif null_terminated:
                # if it's null terminated, it's a null terminated operation
                operations.append(
                    NullTerminatedOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        ptr=param.type,
                    )
                )
            elif nullable:
                # if it's nullable, it's a nullable operation
                operations.append(
                    NullableOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        ptr=param.type,
                    )
                )
            else:
                # otherwise, it's a pointer to a single value or another pointer
                if recv:
                    if param.type.ptr_to.format() == "void":
                        raise NotImplementedError("Cannot dereference a void pointer")
                    # this is an out parameter so use the base type as the server declaration
                    operations.append(
                        DereferenceOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            type_=param.type,
                        )
                    )
                else:
                    # otherwise, treat it as an opaque type
                    operations.append(
                        OpaqueTypeOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            type_=param.type,
                        )
                    )
        elif isinstance(param.type, Type):
            if param.type.const:
                recv = False
            operations.append(
                OpaqueTypeOperation(
                    send=send,
                    recv=recv,
                    parameter=param,
                    type_=param.type,
                )
            )
        elif isinstance(param.type, Array):
            length_param = next(p for p in params if p.name == length_arg.split(":")[1])
            if param.type.array_of.const:
                recv = False
            operations.append(
                ArrayOperation(
                    send=send,
                    recv=recv,
                    parameter=param,
                    ptr=param.type,
                    length=length_param,
                )
            )
        else:
            raise NotImplementedError("Unknown type")
    return operations, False


def error_const(return_type: str) -> str:
    if return_type == "nvmlReturn_t":
        return "NVML_ERROR_GPU_IS_LOST"
    if return_type == "CUresult":
        return "CUDA_ERROR_DEVICE_UNAVAILABLE"
    if return_type == "cudaError_t":
        return "cudaErrorDevicesUnavailable"
    if return_type == "cublasStatus_t":
        return "CUBLAS_STATUS_NOT_INITIALIZED"
    if return_type == "cudnnStatus_t":
        return "CUDNN_STATUS_NOT_INITIALIZED"
    if return_type == "size_t":
        return "size_t"
    if return_type == "const char*":
        return "const char*"
    if return_type == "void":
        return "void"
    if return_type == "struct cudaChannelFormatDesc":
        return "struct cudaChannelFormatDesc"
    raise NotImplementedError("Unknown return type: %s" % return_type)


def prefix_std(type: str) -> str:
    # if type in ["size_t", "std::size_t"]:
    #     return "std::size_t"
    return type


def format_function_params(function: Function) -> list[str]:
    params = []
    for param in function.parameters:
        if param.name and "[]" in param.type.format():
            params.append(
                "{type} {name}".format(
                    type=param.type.format().replace("[]", ""),
                    name=param.name + "[]",
                )
            )
        elif param.name:
            params.append(
                "{type} {name}".format(
                    type=param.type.format(),
                    name=param.name,
                )
            )
        else:
            params.append(param.type.format())
    return params


def format_call_args(function: Function) -> list[str]:
    return [param.name for param in function.parameters if param.name]


# List of possible directories to search for header files
COMMON_INCLUDE_DIRS = [
    "./",
    "/usr/local/cuda/include/",
    "/opt/cuda/include/",
    "/usr/local/include/",
    "/usr/include/",
    "/usr/include/nvidia/",
]


# Function to locate a file in common include directories
def find_header_file(filename):
    for include_dir in COMMON_INCLUDE_DIRS:
        matches = glob.glob(os.path.join(include_dir, "**", filename), recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Header file '{filename}' not found in common include directories."
    )


def main():
    options = ParserOptions(
        preprocessor=make_gcc_preprocessor(
            include_paths=["/usr/local/cuda/include"],
        ),
    )

    try:
        cuda_header = find_header_file("cuda.h")
        annotations_header = find_header_file("annotations.h")
    except FileNotFoundError as e:
        print(e)
        return

    # Parse the files
    cuda_ast: ParsedData = parse_file(cuda_header, options=options)
    annotations: ParsedData = parse_file(annotations_header, options=options)
    functions = [
        function
        for function in cuda_ast.namespace.functions
        if function.name.format().startswith("cu")
        and function.name.format() not in SKIP_FUNCTIONS
    ]

    functions_with_annotations: list[tuple[Function, Function, list[Operation]]] = []

    dupes = {}

    for function in functions:
        # ensure duplicate functions can't be written
        if dupes.get(function.name.format()):
            continue

        dupes[function.name.format()] = True

        try:
            annotation = next(
                f for f in annotations.namespace.functions if f.name == function.name
            )
        except StopIteration:
            print(f"Annotation for {function.name} not found")
            continue
        try:
            operations, is_func_disabled = parse_annotation(
                annotation.doxygen, function.parameters
            )
        except Exception as e:
            print(f"Error parsing annotation for {function.name}: {e}")
            continue
        functions_with_annotations.append(
            (function, annotation, operations, is_func_disabled)
        )

    with open("gen_api.h", "w") as f:
        for i, (function, _, _, _) in enumerate(functions_with_annotations):
            f.write(
                "#define RPC_{name} {value}\n".format(
                    name=function.name.format(),
                    value=i,
                )
            )
        for i, name in enumerate(NVML_RPC_FUNCTIONS, len(functions_with_annotations)):
            f.write(
                "#define RPC_{name} {value}\n".format(
                    name=name,
                    value=i,
                )
            )

    with open("gen_client.cpp", "w") as f:
        f.write(
            "#include <cuda.h>\n"
            "\n"
            "#include <cstring>\n"
            "#include <string>\n"
            "#include <unordered_map>\n\n"
            '#include "gen_api.h"\n\n'
            '#include "rpc.h"\n\n'
            "extern int rpc_size();\n"
            "extern conn_t *rpc_client_get_connection(unsigned int index);\n"
            "extern void rpc_close(conn_t *conn);\n\n"
        )
        for function, annotation, operations, disabled in functions_with_annotations:
            # we don't generate client function definitions for disabled functions; only the RPC definitions.
            if disabled:
                continue

            joined_params = ", ".join(format_function_params(function))

            f.write(
                "{return_type} {name}({params})\n".format(
                    return_type=function.return_type.format(),
                    name=function.name.format(),
                    params=joined_params,
                )
            )
            f.write("{\n")

            f.write("    conn_t *conn = rpc_client_get_connection(0);\n")

            f.write(
                "    {return_type} return_value;\n".format(
                    return_type=function.return_type.format()
                )
            )

            # compute the strlen's for null-terminated operations.
            for operation in operations:
                if isinstance(operation, NullTerminatedOperation):
                    if operation.send:
                        f.write(
                            "    std::size_t {param_name}_len = std::strlen({param_name}) + 1;\n".format(
                                param_name=operation.parameter.name
                            )
                        )
                    else:
                        f.write(
                            "    std::size_t {param_name}_len;\n".format(
                                param_name=operation.parameter.name
                            )
                        )

            f.write(
                "    if (rpc_write_start_request(conn, RPC_{name}) < 0 ||\n".format(
                    name=function.name.format()
                )
            )

            for operation in operations:
                operation.client_rpc_write(f)

            f.write("        rpc_wait_for_response(conn) < 0 ||\n")

            for operation in operations:
                operation.client_rpc_read(f)

            f.write(
                "        rpc_read(conn, &return_value, sizeof({return_type})) < 0 ||\n".format(
                    return_type=function.return_type.format()
                )
            )
            f.write("        rpc_read_end(conn) < 0)\n")
            f.write(
                "        return {error_return};\n".format(
                    error_return=error_const(function.return_type.format())
                )
            )

            if function.name.format() == "cuDriverGetVersion":
                f.write("    if (driverVersion != nullptr) {\n")
                f.write("        const char *override_version = getenv(\"SCUDA_DRIVER_VERSION_OVERRIDE\");\n")
                f.write("        if (override_version != nullptr) *driverVersion = atoi(override_version);\n")
                f.write("    }\n")

            f.write("    return return_value;\n")
            f.write("}\n\n")

        function_by_name = {
            function.name.format(): function
            for function, _, _, disabled in functions_with_annotations
            if not disabled
        }
        for alias, target in MANUAL_REMAPPINGS:
            if alias in function_by_name or target not in function_by_name:
                continue
            target_function = function_by_name[target]
            f.write("#ifdef {name}\n#undef {name}\n#endif\n".format(name=alias))
            f.write(
                'extern "C" {return_type} {name}({params})\n'.format(
                    return_type=target_function.return_type.format(),
                    name=alias,
                    params=", ".join(format_function_params(target_function)),
                )
            )
            f.write("{\n")
            call = "{target}({args})".format(
                target=target,
                args=", ".join(format_call_args(target_function)),
            )
            if target_function.return_type.format() == "void":
                f.write("    {call};\n".format(call=call))
                f.write("}\n\n")
            else:
                f.write("    return {call};\n".format(call=call))
                f.write("}\n\n")

        f.write("std::unordered_map<std::string, void *> functionMap = {\n")
        for function, _, _, disabled in functions_with_annotations:
            if disabled:
                continue

            f.write(
                '    {{"{name}", (void *){name}}},\n'.format(
                    name=function.name.format()
                )
            )
        # write manual overrides
        function_names = set(
            f.name.format()
            for f, _, _, disabled in functions_with_annotations
            if not disabled
        )
        for x, y in MANUAL_REMAPPINGS:
            # ensure y exists in the function list
            if y not in function_names:
                print(f"Skipping manual remapping {x} -> {y}")
                continue
            f.write(
                '    {{"{x}", (void *){y}}},\n'.format(
                    x=x,
                    y=y,
                )
            )
        f.write("};\n\n")

        f.write("void *get_function_pointer(const char *name)\n")
        f.write("{\n")
        f.write("    auto it = functionMap.find(name);\n")
        f.write("    if (it == functionMap.end())\n")
        f.write("        return nullptr;\n")
        f.write("    return it->second;\n")
        f.write("}\n")

    with open("gen_server.cpp", "w") as f:
        f.write(
            "#include <iostream>\n"
            "#include <cuda.h>\n"
            "\n"
            "#include <cstring>\n"
            "#include <string>\n"
            "#include <unordered_map>\n\n"
            '#include "gen_api.h"\n\n'
            '#include <vector>\n\n'
            '#include <cstdio>\n\n'
            '#include "gen_server.h"\n\n'
            '#include <cstdio>\n\n'
            '#include "rpc.h"\n\n'
            '#include "nvml_server.h"\n\n'
        )
        for function, annotation, operations, disabled in functions_with_annotations:
            if disabled:
                continue

            # parse the annotation doxygen
            f.write(
                "int handle_{name}(conn_t *conn)\n".format(
                    name=function.name.format(),
                )
            )
            f.write("{\n")

            defers = []

            for operation in operations:
                f.write(operation.server_declaration)

            f.write("    int request_id;\n")

            # we only generate return from non-void types
            if function.return_type.format() != "void":
                f.write(
                    "    {return_type} scuda_intercept_result;\n".format(
                        return_type=function.return_type.format()
                    )
                )
            else:
                f.write("    void* scuda_intercept_result;\n")

            f.write("    if (\n")
            for operation in operations:
                if isinstance(operation, NullTerminatedOperation) or isinstance(
                    operation, ArrayOperation
                ):
                    if error := operation.server_rpc_read(f, len(defers)):
                        defers.append(error)
                else:
                    operation.server_rpc_read(f)
            f.write("        false)\n")
            f.write("        goto ERROR_{index};\n".format(index=len(defers)))

            f.write("\n")

            f.write("    request_id = rpc_read_end(conn);\n")
            f.write("    if (request_id < 0)\n")
            f.write("        goto ERROR_{index};\n".format(index=len(defers)))

            params: list[str] = []
            # these need to be in function param order, not operation order.
            for param in function.parameters:
                for op in operations:
                    if op.parameter.name == param.name:
                        params.append(op.server_reference)

            if function.return_type.format() != "void":
                f.write(
                    "    scuda_intercept_result = {name}({params});\n\n".format(
                        name=function.name.format(),
                        params=", ".join(params),
                    )
                )
            else:
                f.write(
                    "    {name}({params});\n\n".format(
                        name=function.name.format(),
                        params=", ".join(params),
                    )
                )

            f.write("    if (rpc_write_start_response(conn, request_id) < 0 ||\n")

            for operation in operations:
                operation.server_rpc_write(f)

            f.write(
                "        rpc_write(conn, &scuda_intercept_result, sizeof({return_type})) < 0 ||\n".format(
                    return_type=function.return_type.format()
                )
            )
            f.write("        rpc_write_end(conn) < 0)\n")
            f.write("        goto ERROR_{index};\n".format(index=len(defers)))
            f.write("\n")
            f.write("    return 0;\n")

            for i, defer in enumerate(defers):
                f.write("ERROR_{index}:\n".format(index=len(defers) - i))
                f.write("    free((void *) {param_name});\n".format(param_name=defer))
            f.write("ERROR_0:\n")
            f.write("    return -1;\n")
            f.write("}\n\n")

        f.write("static RequestHandler opHandlers[] = {\n")
        for function, _, _, disabled in functions_with_annotations:
            if disabled:
                f.write("    nullptr,\n")
            else:
                f.write("    handle_{name},\n".format(name=function.name.format()))
        for name in NVML_RPC_FUNCTIONS:
            f.write("    handle_{name},\n".format(name=name))
        f.write("};\n\n")

        f.write("RequestHandler get_handler(const int op)\n")
        f.write("{\n")
        f.write("    if (op < 0 || op >= static_cast<int>(sizeof(opHandlers) / sizeof(opHandlers[0])))\n")
        f.write("        return nullptr;\n")
        f.write("    return opHandlers[op];\n")
        f.write("}\n")


if __name__ == "__main__":
    main()
