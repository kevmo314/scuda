from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Type, Pointer, Parameter, Function
from typing import Optional
from dataclasses import dataclass
import copy

# this table is manually generated from the cuda.h headers
MANUAL_REMAPPINGS = [
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

# a list of manually implemented cuda/nvml functions.
# these are automatically appended to each file; operation order is maintained as well.
MANUAL_IMPLEMENTATIONS = ["cudaMemcpy", "cudaMemcpyAsync", "cudaLaunchKernel", "__cudaRegisterFunction", "__cudaRegisterFatBinary"]


@dataclass
class Operation:
    send: bool
    recv: bool
    args: list[str]

    server_type: Type | Pointer
    parameter: Parameter

    length_parameter: Optional[Parameter]

    @property
    def nullable(self) -> bool:
        return (
            isinstance(self.server_type, Pointer) and self.parameter.type.ptr_to.const
        )

    @property
    def null_terminated(self) -> bool:
        return len(self.args) > 0 and self.args[0] == "NULL_TERMINATED"

    @property
    def array_size(self) -> Optional[str]:
        for arg in self.args:
            if arg.startswith("SIZE:"):
                return arg.split(":")[1]

    @property
    def is_opaque_pointer(self) -> bool:
        return (
            isinstance(self.server_type, Pointer)
            and isinstance(self.server_type.ptr_to, Type)
            and self.server_type.ptr_to.format() == "void"
        )

    @property
    def server_reference(self) -> str:
        if (
            isinstance(self.server_type, Type)
            and isinstance(self.parameter.type, Pointer)
        ) or self.is_opaque_pointer:
            return "&" + self.parameter.name
        return self.parameter.name


def parse_annotation(annotation: str, params: list[Parameter]) -> list[Operation]:
    operations: list[Operation] = []
    if not annotation:
        return operations
    for line in annotation.split("\n"):
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
            length_parameter = None
            if isinstance(param.type, Pointer):
                # if there's a length or size arg, use the type, otherwise use the ptr_to type
                length_arg = next(
                    (arg for arg in args if arg.startswith("LENGTH:")), None
                )
                size_arg = next((arg for arg in args if arg.startswith("SIZE:")), None)
                null_terminated = "NULL_TERMINATED" in args
                if param.type.ptr_to.const and parts[2] != "SEND_ONLY":
                    # if the parameter is const, then it's a sending parameter only.
                    # validate it as such.
                    raise NotImplementedError(
                        f"const pointer parameter {param.name} must be SEND_ONLY"
                    )
                if length_arg or size_arg:
                    # if it has a length or size, it's an array parameter
                    if length_arg:
                        length_parameter = next(
                            p for p in params if p.name == length_arg.split(":")[1]
                        )
                    server_type = copy.deepcopy(param.type)
                    server_type.ptr_to.const = False
                elif param.type.ptr_to.format() == "void":
                    # treat void pointers as opaque
                    server_type = param.type
                elif null_terminated:
                    # treat null-terminated strings as a special case
                    server_type = param.type
                else:
                    # otherwise, this is a pointer to a single value
                    server_type = param.type.ptr_to
                    server_type.const = False
            elif isinstance(param.type, Type):
                server_type = param.type
            else:
                raise NotImplementedError("Unknown type")
            operation = Operation(
                send=send,
                recv=parts[2] == "RECV_ONLY" or parts[2] == "SEND_RECV",
                args=args,
                server_type=server_type,
                parameter=param,
                length_parameter=length_parameter,
            )
            operations.append(operation)
    return operations


def error_const(return_type: str) -> str:
    if return_type == "nvmlReturn_t":
        return "NVML_ERROR_GPU_IS_LOST"
    if return_type == "CUresult":
        return "CUDA_ERROR_DEVICE_UNAVAILABLE"
    if return_type == "cudaError_t":
        return "cudaErrorDevicesUnavailable"
    raise NotImplementedError("Unknown return type: %s" % return_type)


def prefix_std(type: str) -> str:
    # if type in ["size_t", "std::size_t"]:
    #     return "std::size_t"
    return type


def main():
    options = ParserOptions(preprocessor=make_gcc_preprocessor())

    nvml_ast: ParsedData = parse_file("/usr/include/nvml.h", options=options)
    cuda_ast: ParsedData = parse_file("/usr/include/cuda.h", options=options)
    cudart_ast: ParsedData = parse_file(
        "/usr/include/cuda_runtime_api.h", options=options
    )
    annotations: ParsedData = parse_file("annotations.h", options=options)

    functions = (
        nvml_ast.namespace.functions
        + cuda_ast.namespace.functions
        + cudart_ast.namespace.functions
    )

    functions_with_annotations: list[tuple[Function, Function, list[Operation]]] = []
    for function in functions:
        try:
            annotation = next(
                f for f in annotations.namespace.functions if f.name == function.name
            )
        except StopIteration:
            print(f"Annotation for {function.name} not found")
            continue
        try:
            operations = parse_annotation(annotation.doxygen, function.parameters)
        except Exception as e:
            print(f"Error parsing annotation for {function.name}: {e}")
            continue
        functions_with_annotations.append((function, annotation, operations))

    with open("gen_api.h", "w") as f:
        lastIndex = 0
        # visited = set()
        for i, (function, _, _) in enumerate(functions_with_annotations):
            # value = hash(function.name.format()) % (2**32)
            # if value in visited:
            #     print(f"Hash collision for {function.name.format()}")
            #     continue
            f.write(
                "#define RPC_{name} {value}\n".format(
                    name=function.name.format(),
                    value=i,
                )
            )
            lastIndex += 1

        # Append any manually written implementations to our rpc list so that we maintain proper operation order
        for j, (handler) in enumerate(MANUAL_IMPLEMENTATIONS):
            f.write(
                "#define RPC_{name} {value}\n".format(
                    name=handler.format(),
                    value=j + lastIndex,
                )
            )

    with open("gen_client.cpp", "w") as f:
        f.write(
            "#include <nvml.h>\n"
            "#include <cuda.h>\n"
            "#include <cuda_runtime_api.h>\n\n"
            "#include <cstring>\n"
            "#include <string>\n"
            "#include <unordered_map>\n\n"
            '#include "gen_api.h"\n\n'
            '#include "manual_client.h"\n\n'
            "extern int rpc_start_request(const unsigned int request);\n"
            "extern int rpc_write(const void *data, const std::size_t size);\n"
            "extern int rpc_read(void *data, const std::size_t size);\n"
            "extern int rpc_wait_for_response(const unsigned int request_id);\n"
            "extern int rpc_end_request(void *return_value, const unsigned int request_id);\n"
        )
        for function, annotation, operations in functions_with_annotations:
            f.write(
                "{return_type} {name}({params})\n".format(
                    return_type=function.return_type.format(),
                    name=function.name.format(),
                    params=", ".join(
                        (
                            "{type} {name}".format(
                                type=param.type.format(),
                                name=param.name,
                            )
                            if param.name
                            else param.type.format()
                        )
                        for param in function.parameters
                    ),
                )
            )
            f.write("{\n")

            f.write(
                "    {return_type} return_value;\n\n".format(
                    return_type=function.return_type.format()
                )
            )

            f.write(
                "    int request_id = rpc_start_request(RPC_{name});\n".format(
                    name=function.name.format()
                )
            )

            for operation in operations:
                if operation.null_terminated:
                    f.write(
                        "    std::size_t {param_name}_len = std::strlen({param_name}) + 1;\n".format(
                            param_name=operation.parameter.name
                        )
                    )

            f.write("    if (request_id < 0 ||\n")

            for operation in operations:
                if operation.send:
                    if operation.null_terminated:
                        f.write(
                            "        rpc_write(&{param_name}_len, sizeof(std::size_t)) < 0 ||\n".format(
                                param_name=operation.parameter.name,
                            )
                        )
                        f.write(
                            "        rpc_write({param_name}, {param_name}_len) < 0 ||\n".format(
                                param_name=operation.parameter.name,
                            )
                        )
                    elif length := operation.length_parameter:
                        if isinstance(length.type, Pointer):
                            f.write(
                                "        rpc_write({param_name}, *{length} * sizeof({param_type})) < 0 ||\n".format(
                                    param_name=operation.parameter.name,
                                    param_type=operation.server_type.ptr_to.format(),
                                    length=length.name,
                                )
                            )
                        else:
                            f.write(
                                "        rpc_write({param_name}, {length} * sizeof({param_type})) < 0 ||\n".format(
                                    param_name=operation.parameter.name,
                                    param_type=operation.server_type.ptr_to.format(),
                                    length=length.name,
                                )
                            )
                    elif size := operation.array_size:
                        f.write(
                            "        rpc_write({param_name}, {size}) < 0 ||\n".format(
                                param_name=operation.parameter.name,
                                size=size,
                            )
                        )
                    elif operation.nullable:
                        # write the pointer since it is nonzero if not-null
                        f.write(
                            "        rpc_write(&{param_name}, sizeof({param_type})) < 0 ||\n".format(
                                param_name=operation.parameter.name,
                                param_type=operation.server_type.format(),
                            )
                        )
                        f.write(
                            "        ({param_name} != nullptr && rpc_write({param_name}, sizeof({base_type})) < 0) ||\n".format(
                                param_name=operation.parameter.name,
                                base_type=operation.server_type.ptr_to.format(),
                            )
                        )
                    else:
                        f.write(
                            "        rpc_write(&{param_name}, sizeof({param_type})) < 0 ||\n".format(
                                param_name=operation.parameter.name,
                                param_type=operation.server_type.format(),
                            )
                        )
            f.write("        rpc_wait_for_response(request_id) < 0 ||\n")
            for operation in operations:
                if operation.recv:
                    if operation.null_terminated:
                        f.write(
                            "        rpc_read(&{param_name}_len, sizeof({param_name}_len)) < 0 ||\n".format(
                                param_name=operation.parameter.name
                            )
                        )
                        f.write(
                            "        rpc_read({param_name}, {param_name}_len) < 0 ||\n".format(
                                param_name=operation.parameter.name
                            )
                        )
                    elif length := operation.length_parameter:
                        if isinstance(length.type, Pointer):
                            f.write(
                                "        rpc_read({param_name}, *{length} * sizeof({param_type})) < 0 ||\n".format(
                                    param_name=operation.parameter.name,
                                    param_type=operation.server_type.ptr_to.format(),
                                    length=length.name,
                                )
                            )
                        else:
                            f.write(
                                "        rpc_read({param_name}, {length} * sizeof({param_type})) < 0 ||\n".format(
                                    param_name=operation.parameter.name,
                                    param_type=operation.server_type.ptr_to.format(),
                                    length=length.name,
                                )
                            )
                    elif size := operation.array_size:
                        f.write(
                            "        rpc_read({param_name}, {size}) < 0 ||\n".format(
                                param_name=operation.parameter.name,
                                size=size,
                            )
                        )
                    else:
                        f.write(
                            "        rpc_read({param_name}, sizeof({param_type})) < 0 ||\n".format(
                                param_name=operation.parameter.name,
                                param_type=operation.server_type.format(),
                            )
                        )
            f.write("        rpc_end_request(&return_value, request_id) < 0)\n")
            f.write(
                "        return {error_return};\n".format(
                    error_return=error_const(function.return_type.format())
                )
            )
            f.write("    return return_value;\n")

            f.write("}\n\n")

        f.write("std::unordered_map<std::string, void *> functionMap = {\n")

        # we need the base nvmlInit, this is important and should be kept here in the codegen.
        f.write('    {"nvmlInit", (void *)nvmlInit_v2},\n')
        for function, _, _ in functions_with_annotations:
            f.write(
                '    {{"{name}", (void *){name}}},\n'.format(
                    name=function.name.format()
                )
            )
        # write manual overrides
        function_names = set(f.name.format() for f, _, _ in functions_with_annotations)
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
        for x in MANUAL_IMPLEMENTATIONS:
            f.write(
                '    {{"{x}", (void *){x}}},\n'.format(
                    x=x,
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
            "#include <nvml.h>\n"
            "#include <cuda.h>\n"
            "#include <cuda_runtime_api.h>\n\n"
            "#include <cstring>\n"
            "#include <string>\n"
            "#include <unordered_map>\n\n"
            '#include "gen_api.h"\n\n'
            '#include "gen_server.h"\n\n'
            '#include "manual_server.h"\n\n'
            "extern int rpc_read(const void *conn, void *data, const std::size_t size);\n"
            "extern int rpc_write(const void *conn, const void *data, const std::size_t size);\n"
            "extern int rpc_end_request(const void *conn);\n"
            "extern int rpc_start_response(const void *conn, const int request_id);\n\n"
        )
        for function, annotation, operations in functions_with_annotations:
            # parse the annotation doxygen
            f.write(
                "int handle_{name}(void *conn)\n".format(
                    name=function.name.format(),
                )
            )
            f.write("{\n")

            for operation in operations:
                if operation.null_terminated:
                    # write the length declaration and the parameter declaration
                    f.write(
                        "    std::size_t {param_name}_len;\n".format(
                            param_name=operation.parameter.name
                        )
                    )
                    f.write(
                        "    if (rpc_read(conn, &{param_name}_len, sizeof({param_name}_len)) < 0)\n".format(
                            param_name=operation.parameter.name
                        )
                    )
                    f.write("        return -1;\n")
                    if isinstance(operation.server_type, Pointer):
                        # write the malloc declaration
                        f.write(
                            "    {server_type} {param_name} = ({server_type})malloc({param_name}_len);\n".format(
                                param_name=operation.parameter.name,
                                server_type=prefix_std(operation.server_type.format()),
                            )
                        )
                    else:
                        # write just the parameter declaration
                        f.write(
                            "    {server_type} {param_name};\n".format(
                                server_type=prefix_std(operation.server_type.format()),
                                param_name=operation.parameter.name,
                            )
                        )
                elif isinstance(operation.server_type, Pointer):
                    # write the malloc declaration
                    if length := operation.length_parameter:
                        f.write(
                            "    {server_type} {param_name} = ({server_type})malloc({length} * sizeof({base_type}));\n".format(
                                param_name=operation.parameter.name,
                                server_type=prefix_std(operation.server_type.format()),
                                length=length.name,
                                base_type=operation.server_type.ptr_to.format(),
                            )
                        )
                    elif size := operation.array_size:
                        f.write(
                            "    {server_type} {param_name} = ({server_type})malloc({size});\n".format(
                                param_name=operation.parameter.name,
                                server_type=prefix_std(operation.server_type.format()),
                                size=size,
                            )
                        )
                    elif operation.is_opaque_pointer:
                        # write just the parameter declaration
                        f.write(
                            "    {server_type} {param_name};\n".format(
                                server_type=prefix_std(operation.server_type.format()),
                                param_name=operation.parameter.name,
                            )
                        )
                    elif operation.nullable:
                        # write a parameter declaration and read the first byte to determine if it's null
                        f.write(
                            "    bool {param_name}_null_check;\n".format(
                                param_name=operation.parameter.name
                            )
                        )
                        f.write(
                            "    if (rpc_read(conn, &{param_name}_null_check, 1) < 0)\n".format(
                                param_name=operation.parameter.name
                            )
                        )
                        f.write("        return -1;\n")
                        # write param declaration
                        f.write(
                            "    {server_type} {param_name} = nullptr;\n".format(
                                server_type=prefix_std(operation.server_type.format()),
                                param_name=operation.parameter.name,
                            )
                        )
                        f.write(
                            "    if ({param_name}_null_check) {{\n".format(
                                param_name=operation.parameter.name
                            )
                        )
                        f.write(
                            "        {param_name} = ({server_type})malloc(sizeof({base_type}));\n".format(
                                param_name=operation.parameter.name,
                                server_type=operation.server_type.format(),
                                base_type=operation.server_type.ptr_to.format(),
                            )
                        )
                        f.write(
                            "        if (rpc_read(conn, {param_name}, sizeof({base_type})) < 0)\n".format(
                                param_name=operation.parameter.name,
                                base_type=operation.server_type.ptr_to.format(),
                            )
                        )
                        f.write("            return -1;\n")
                        f.write("    }\n")
                    else:
                        print(function, operation)
                        raise NotImplementedError(
                            "Could not determine length, this parameter is a pointer but neither length nor size is specified"
                        )
                else:
                    # write just the parameter declaration
                    f.write(
                        "    {server_type} {param_name};\n".format(
                            server_type=prefix_std(operation.server_type.format()),
                            param_name=operation.parameter.name,
                        )
                    )
                if operation.send:
                    f.write(
                        "    if (rpc_read(conn, &{param_name}, sizeof({param_type})) < 0)\n".format(
                            param_name=operation.parameter.name,
                            param_type=operation.server_type.format(),
                        )
                    )
                    f.write("        return -1;\n")

            f.write("\n")
            f.write(
                "    int request_id = rpc_end_request(conn);\n".format(
                    name=function.name.format()
                )
            )
            f.write("    if (request_id < 0)\n")
            f.write("        return -1;\n\n")

            params: list[str] = []
            # these need to be in function param order, not operation order.
            for param in function.parameters:
                operation = next(
                    op for op in operations if op.parameter.name == param.name
                )
                params.append(operation.server_reference)

            f.write(
                "    {return_type} result = {name}({params});\n\n".format(
                    return_type=function.return_type.format(),
                    name=function.name.format(),
                    params=", ".join(params),
                )
            )

            f.write("    if (rpc_start_response(conn, request_id) < 0)\n")
            f.write("        return -1;\n")

            for operation in operations:
                if operation.recv:
                    if operation.null_terminated:
                        f.write(
                            "    if (rpc_write(conn, &{param_name}_len, sizeof({param_name}_len)) < 0)\n".format(
                                param_name=operation.server_reference
                            )
                        )
                        f.write("        return -1;\n")
                        f.write(
                            "    if (rpc_write(conn, {param_name}, {param_name}_len) < 0)\n".format(
                                param_name=operation.parameter.name
                            )
                        )
                        f.write("        return -1;\n")
                    elif length := operation.length_parameter:
                        f.write(
                            "    if (rpc_write(conn, {param_name}, {length} * sizeof({param_type})) < 0)\n".format(
                                param_name=operation.server_reference,
                                param_type=operation.server_type.ptr_to.format(),
                                length=length.name,
                            )
                        )
                        f.write("        return -1;\n")
                    elif size := operation.array_size:
                        f.write(
                            "    if (rpc_write(conn, {param_name}, {size}) < 0)\n".format(
                                param_name=operation.server_reference,
                                size=size,
                            )
                        )
                        f.write("        return -1;\n")
                    else:
                        f.write(
                            "    if (rpc_write(conn, {param_name}, sizeof({param_type})) < 0)\n".format(
                                param_name=operation.server_reference,
                                param_type=operation.server_type.format(),
                            )
                        )
                        f.write("        return -1;\n")

            f.write("\n")
            f.write("    return result;\n")
            f.write("}\n\n")

        f.write("static RequestHandler opHandlers[] = {\n")
        for function, _, _ in functions_with_annotations:
            f.write("    handle_{name},\n".format(name=function.name.format()))
        for handler in MANUAL_IMPLEMENTATIONS:
            f.write("    handle_{name},\n".format(name=handler.format()))
        f.write("};\n\n")

        f.write("RequestHandler get_handler(const int op)\n")
        f.write("{\n")
        f.write("    return opHandlers[op];\n")
        f.write("}\n")


if __name__ == "__main__":
    main()
