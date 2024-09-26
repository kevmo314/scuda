from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import (
    Function,
    Parameter,
    Pointer,
    Type,
)
from typing import Optional

IGNORE_FUNCTIONS = {
    "nvmlDeviceCcuGetStreamState",
    "nvmlDeviceCcuSetStreamState",
    "cuOccupancyMaxPotentialBlockSize",
    "cuOccupancyMaxPotentialBlockSizeWithFlags",
    "cuGetProcAddress",
    "cuGetProcAddress_v2",
}


def heap_allocation_size(function: Function, param: Parameter) -> Optional[str]:
    """
    Returns the size of heap-allocated parameters. Because there is no convention in
    Nvidia's API for how these parameters are defined, we need to manually specify
    which ones are heap allocated and how much memory they require.
    """
    function_name = function.name.format()
    parameter_name = param.name.format()
    if function_name == "nvmlSystemGetDriverVersion":
        if parameter_name == "version":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if function_name == "nvmlSystemGetNVMLVersion":
        if parameter_name == "version":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if function_name == "nvmlSystemGetProcessName":
        if parameter_name == "name":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if function_name == "nvmlSystemGetHicVersion":
        if parameter_name == "hwbcEntries":
            return (
                "*hwbcCount * sizeof(nvmlHwbcEntry_t)",
                "hwbcCount * sizeof(nvmlHwbcEntry_t)",
            )
    if function_name == "nvmlUnitGetDevices":
        if parameter_name == "devices":
            return (
                "*deviceCount * sizeof(nvmlDevice_t)",
                "deviceCount * sizeof(nvmlDevice_t)",
            )
    if function_name == "nvmlDeviceGetName":
        if parameter_name == "name":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if function_name == "nvmlDeviceGetSerial":
        if parameter_name == "serial":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if function_name == "nvmlDeviceGetMemoryAffinity":
        if parameter_name == "nodeSet":
            return (
                "nodeSetSize * sizeof(unsigned long)",
                "nodeSetSize * sizeof(unsigned long)",
            )
    if function_name == "nvmlDeviceGetCpuAffinityWithinScope":
        if parameter_name == "cpuSet":
            return (
                "cpuSetSize * sizeof(unsigned long)",
                "cpuSetSize * sizeof(unsigned long)",
            )
    if function_name == "nvmlDeviceGetCpuAffinity":
        if parameter_name == "cpuSet":
            return (
                "cpuSetSize * sizeof(unsigned long)",
                "cpuSetSize * sizeof(unsigned long)",
            )
    if function_name == "nvmlDeviceGetTopologyNearestGpus":
        if parameter_name == "deviceArray":
            return ("*count * sizeof(nvmlDevice_t)", "count * sizeof(nvmlDevice_t)")
    if function_name == "nvmlDeviceGetUUID":
        if parameter_name == "uuid":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if function_name == "nvmlVgpuInstanceGetMdevUUID":
        if parameter_name == "mdevUuid":
            return ("size * sizeof(char)", "size * sizeof(char)")
    if function_name == "nvmlDeviceGetBoardPartNumber":
        if parameter_name == "partNumber":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if function_name == "nvmlDeviceGetInforomVersion":
        if parameter_name == "version":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if function_name == "nvmlDeviceGetInforomImageVersion":
        if parameter_name == "version":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if function_name == "nvmlDeviceGetEncoderSessions":
        if parameter_name == "sessionInfos":
            return (
                "*sessionCount * sizeof(nvmlEncoderSessionInfo_t)",
                "sessionCount * sizeof(nvmlEncoderSessionInfo_t)",
            )
    if function_name == "nvmlDeviceGetVbiosVersion":
        if parameter_name == "version":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if function_name == "nvmlDeviceGetComputeRunningProcesses_v3":
        if parameter_name == "infos":
            return (
                "*infoCount * sizeof(nvmlProcessInfo_t)",
                "infoCount * sizeof(nvmlProcessInfo_t)",
            )
    if function_name == "nvmlDeviceGetGraphicsRunningProcesses_v3":
        if parameter_name == "infos":
            return (
                "*infoCount * sizeof(nvmlProcessInfo_t)",
                "infoCount * sizeof(nvmlProcessInfo_t)",
            )
    if function_name == "nvmlDeviceGetMPSComputeRunningProcesses_v3":
        if parameter_name == "infos":
            return (
                "*infoCount * sizeof(nvmlProcessInfo_t)",
                "infoCount * sizeof(nvmlProcessInfo_t)",
            )
    if function_name == "nvmlDeviceGetSamples":
        if parameter_name == "samples":
            return (
                "*sampleCount * sizeof(nvmlSample_t)",
                "sampleCount * sizeof(nvmlSample_t)",
            )
    if function_name == "nvmlDeviceGetAccountingPids":
        if parameter_name == "pids":
            return ("*count * sizeof(unsigned int)", "count * sizeof(unsigned int)")
    if function_name == "nvmlDeviceGetRetiredPages":
        if parameter_name == "addresses":
            return (
                "*pageCount * sizeof(unsigned long long)",
                "pageCount * sizeof(unsigned long long)",
            )
    if function_name == "nvmlDeviceGetRetiredPages_v2":
        if parameter_name == "addresses":
            return (
                "*pageCount * sizeof(unsigned long long)",
                "pageCount * sizeof(unsigned long long)",
            )
    if function_name == "nvmlDeviceGetFieldValues":
        if parameter_name == "values":
            return (
                "valuesCount * sizeof(nvmlFieldValue_t)",
                "valuesCount * sizeof(nvmlFieldValue_t)",
            )
    if function_name == "nvmlDeviceClearFieldValues":
        if parameter_name == "values":
            return (
                "valuesCount * sizeof(nvmlFieldValue_t)",
                "valuesCount * sizeof(nvmlFieldValue_t)",
            )
    if function_name == "nvmlDeviceGetSupportedVgpus":
        if parameter_name == "vgpuTypeIds":
            return (
                "*vgpuCount * sizeof(nvmlVgpuTypeId_t)",
                "vgpuCount * sizeof(nvmlVgpuTypeId_t)",
            )
    if function_name == "nvmlDeviceGetCreatableVgpus":
        if parameter_name == "vgpuTypeIds":
            return (
                "*vgpuCount * sizeof(nvmlVgpuTypeId_t)",
                "vgpuCount * sizeof(nvmlVgpuTypeId_t)",
            )
    if function_name == "nvmlVgpuTypeGetName":
        if parameter_name == "vgpuTypeName":
            return ("*size * sizeof(char)", "size * sizeof(char)")
    if function_name == "nvmlVgpuTypeGetLicense":
        if parameter_name == "vgpuTypeLicenseString":
            return ("size * sizeof(char)", "size * sizeof(char)")
    if function_name == "nvmlVgpuTypeGetMaxInstances":
        if parameter_name == "vgpuTypeId":
            return (
                "*vgpuInstanceCount * sizeof(unsigned int)",
                "vgpuInstanceCount * sizeof(unsigned int)",
            )
    if function_name == "nvmlDeviceGetActiveVgpus":
        if parameter_name == "vgpuInstances":
            return (
                "*vgpuCount * sizeof(nvmlVgpuInstance_t)",
                "vgpuCount * sizeof(nvmlVgpuInstance_t)",
            )
    if function_name == "nvmlVgpuInstanceGetVmID":
        if parameter_name == "vmId":
            return ("size * sizeof(char)", "size * sizeof(char)")
    if function_name == "nvmlVgpuInstanceGetUUID":
        if parameter_name == "uuid":
            return ("size * sizeof(char)", "size * sizeof(char)")
    if function_name == "nvmlVgpuInstanceGetVmDriverVersion":
        if parameter_name == "version":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if function_name == "nvmlVgpuInstanceGetGpuPciId":
        if parameter_name == "vgpuPciId":
            return ("*length * sizeof(char)", "length * sizeof(char)")
    if function_name == "nvmlDeviceGetPgpuMetadataString":
        if parameter_name == "pgpuMetadata":
            return ("*bufferSize * sizeof(char)", "bufferSize * sizeof(char)")
    if function_name == "nvmlDeviceGetVgpuUtilization":
        if parameter_name == "utilizationSamples":
            return (
                "*vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)",
                "vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)",
            )
    if function_name == "nvmlDeviceGetVgpuProcessUtilization":
        if parameter_name == "utilizationSamples":
            return (
                "*vgpuProcessSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)",
                "vgpuProcessSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)",
            )
    if function_name == "nvmlVgpuInstanceGetAccountingPids":
        if parameter_name == "pids":
            return ("*count * sizeof(unsigned int)", "count * sizeof(unsigned int)")
    if function_name == "nvmlDeviceGetGpuInstancePossiblePlacements_v2":
        if parameter_name == "placements":
            return (
                "*count * sizeof(nvmlGpuInstancePlacement_t)",
                "count * sizeof(nvmlGpuInstancePlacement_t)",
            )
    if function_name == "nvmlDeviceGetGpuInstances":
        if parameter_name == "gpuInstances":
            return (
                "*count * sizeof(nvmlGpuInstance_t)",
                "count * sizeof(nvmlGpuInstance_t)",
            )
    if function_name == "nvmlGpuInstanceGetComputeInstancePossiblePlacements":
        if parameter_name == "gpuInstances":
            return (
                "*count * sizeof(nvmlComputeInstancePlacement_t)",
                "count * sizeof(nvmlComputeInstancePlacement_t)",
            )
    if function_name == "nvmlGpuInstanceGetComputeInstances":
        if parameter_name == "computeInstances":
            return (
                "*count * sizeof(nvmlComputeInstance_t)",
                "count * sizeof(nvmlComputeInstance_t)",
            )


def error_const(return_type: str) -> str:
    if return_type == "nvmlReturn_t":
        return "NVML_ERROR_GPU_IS_LOST"
    if return_type == "CUresult":
        return "CUDA_ERROR_DEVICE_UNAVAILABLE"
    if return_type == "cudaError_t":
        return "cudaErrorDevicesUnavailable"
    raise NotImplementedError("Unknown return type: %s" % return_type)


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

    int request_id = rpc_start_request(RPC_{name});
    if (request_id < 0 ||\n""".format(
                    doxygen=function.doxygen + "\n" if function.doxygen else "",
                    return_type=function.return_type.format(),
                    name=function.name.format(),
                    params=", ".join(
                        p.type.format_decl(p.name) for p in function.parameters
                    ),
                ).lstrip()
            )
            # write the entire parameter list as a block of memory
            for param in function.parameters:
                if not param.name:
                    continue
                if isinstance(param.type, Type):
                    f.write(
                        "        rpc_write(&%s, sizeof(%s)) < 0 ||\n"
                        % (param.name, param.type.format())
                    )
                if isinstance(param.type, Pointer) and not heap_allocation_size(
                    function, param
                ):
                    # only write pointers if they don't have a heap allocation size (and thus are not server return params)
                    const = param.type.ptr_to.const
                    param.type.ptr_to.const = False
                    if param.type.ptr_to.format() == "void":
                        f.write(
                            "        rpc_write(%s, sizeof(%s)) < 0 ||\n"
                            % (param.name, param.type.format())
                        )
                    else:
                        f.write(
                            "        rpc_write(%s, sizeof(%s)) < 0 ||\n"
                            % (param.name, param.type.ptr_to.format())
                        )
                    param.type.ptr_to.const = const
            # wait for response
            f.write("        rpc_wait_for_response(request_id) < 0 ||\n")
            # read responses back
            for param in function.parameters:
                if (
                    param.name is None
                    or not isinstance(param.type, Pointer)
                    or param.type.ptr_to.const
                ):
                    # only consider non-const pointers for return values
                    continue
                size = heap_allocation_size(function, param)
                if size:
                    f.write("        rpc_read(%s, %s) < 0 ||\n" % (param.name, size[0]))
                else:
                    if param.type.ptr_to.format() == "void":
                        f.write(
                            "        rpc_read(%s, sizeof(%s)) < 0 ||\n"
                            % (param.name.format(), param.type.format())
                        )
                    else:
                        f.write(
                            "        rpc_read(%s, sizeof(%s)) < 0 ||\n"
                            % (param.name.format(), param.type.ptr_to.format())
                        )
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
            """
// Generated code.

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
                
#include <string>
#include <unordered_map>

#include "gen_api.h"
#include "gen_server.h"

extern int rpc_read(const void *conn, void *data, const size_t size);
extern int rpc_write(const void *conn, const void *data, const size_t size);
extern int rpc_end_request(const void *conn);
extern int rpc_start_response(const void *conn, const int request_id);

""".lstrip()
        )
        for i, function in enumerate(functions):
            # construct the function signature
            f.write(
                "{doxygen}int handle_{name}(void *conn) {{\n".format(
                    doxygen=function.doxygen + "\n" if function.doxygen else "",
                    name=function.name.format(),
                )
            )

            # read non-heap-allocated variables
            stack_vars: list[Parameter] = []
            heap_vars: list[Parameter] = []
            for param in function.parameters:
                if param.name is None:
                    continue
                if isinstance(param.type, Pointer) and heap_allocation_size(
                    function, param
                ):
                    heap_vars.append(param)
                else:
                    stack_vars.append(param)
            if len(stack_vars) > 0:
                for param in stack_vars:
                    if isinstance(param.type, Pointer):
                        const = param.type.ptr_to.const
                        param.type.ptr_to.const = False
                        if param.type.ptr_to.format() == "void":
                            f.write("    %s %s;\n" % (param.type.format(), param.name))
                        else:
                            f.write(
                                "    %s %s;\n"
                                % (param.type.ptr_to.format(), param.name)
                            )
                        param.type.ptr_to.const = const
                    else:
                        f.write("    %s;\n" % param.format())
                f.write("\n")
                for i, param in enumerate(stack_vars):
                    leading = "        " if i > 0 else "    if ("
                    trailing = " ||\n" if i < len(stack_vars) - 1 else ")\n"
                    if (
                        isinstance(param.type, Pointer)
                        and param.type.ptr_to.format() != "void"
                    ):
                        const = param.type.ptr_to.const
                        param.type.ptr_to.const = False
                        if param.type.ptr_to.format() == "void":
                            f.write(
                                "%srpc_read(conn, &%s, sizeof(%s)) < 0%s"
                                % (leading, param.name, param.type.format(), trailing)
                            )
                        else:
                            f.write(
                                "%srpc_read(conn, &%s, sizeof(%s)) < 0%s"
                                % (
                                    leading,
                                    param.name,
                                    param.type.ptr_to.format(),
                                    trailing,
                                )
                            )
                        param.type.ptr_to.const = const
                    else:
                        f.write(
                            "%srpc_read(conn, &%s, sizeof(%s)) < 0%s"
                            % (leading, param.name, param.type.format(), trailing)
                        )
                f.write("        return -1;\n\n")

            f.write("    int request_id = rpc_end_request(conn);\n")
            f.write("    if (request_id < 0)\n")
            f.write("        return -1;\n\n")

            # malloc heap-allocated variables
            if len(heap_vars) > 0:
                for param in heap_vars:
                    size = heap_allocation_size(function, param)
                    if not size:
                        continue
                    f.write(
                        "    %s = (%s)malloc(%s);\n"
                        % (param.format(), param.type.format(), size[1])
                    )
                f.write("\n")

            # call the function
            f.write(
                "    %s = %s("
                % (
                    function.return_type.format_decl(name="result"),
                    function.name.format(),
                )
            )
            for i, param in enumerate(function.parameters):
                if param.name is None:
                    f.write("nullptr")
                elif isinstance(param.type, Pointer) and not heap_allocation_size(
                    function, param
                ):
                    f.write("&%s" % param.name)
                else:
                    f.write("%s" % param.name)
                if i < len(function.parameters) - 1:
                    f.write(", ")
            f.write(");\n\n")

            f.write("    if (rpc_start_response(conn, request_id) < 0)\n")
            f.write("        return -1;\n\n")

            # write pointer vars
            ptr_vars = [
                p
                for p in function.parameters
                if p.name is not None
                and isinstance(p.type, Pointer)
                and not p.type.ptr_to.const
            ]
            if len(ptr_vars) > 0:
                for i, param in enumerate(ptr_vars):
                    size = heap_allocation_size(function, param)
                    if size:
                        f.write(
                            "%srpc_write(conn, %s, %s) < 0%s"
                            % (
                                "        " if i > 0 else "    if (",
                                param.name,
                                size[1],
                                " ||\n" if i < len(ptr_vars) - 1 else ")\n",
                            )
                        )
                    else:
                        f.write(
                            "%srpc_write(conn, &%s, sizeof(%s)) < 0%s"
                            % (
                                "        " if i > 0 else "    if (",
                                param.name,
                                param.type.format(),
                                " ||\n" if i < len(ptr_vars) - 1 else ")\n",
                            )
                        )
                f.write("        return -1;\n\n")
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
