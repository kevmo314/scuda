from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Type, Pointer, Parameter, Function, Array
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

# These functions are not exposed in header files, but we need to make sure they are...
# properly added to our client/server definitions.
# These, ideally, should never be added or removed. 
INTERNAL_FUNCTIONS = [
    "__cudaRegisterVar",
    "__cudaRegisterFunction",
    "__cudaRegisterFatBinary",
    "__cudaRegisterFatBinaryEnd",
    "__cudaPushCallConfiguration",
    "__cudaPopCallConfiguration"
]

# a list of manually implemented cuda/nvml functions.
# these are automatically appended to each file; operation order is maintained as well.
MANUAL_IMPLEMENTATIONS = [
    "cudaFree",
    "cudaMemcpy",
    "cudaMemcpyAsync",
    "cudaLaunchKernel",
    "cudaMallocManaged",
]

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
            "        rpc_write(0, &{param_name}, sizeof({server_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )

        f.write(
            "        ({param_name} != nullptr && rpc_write(0, {param_name}, sizeof({base_type})) < 0) ||\n".format(
                param_name=self.parameter.name,
                # void is treated differently from non void pointer types
                base_type=(self.ptr.format() if self.ptr.ptr_to.format() == "const void" else self.ptr.ptr_to.format()),
            )
        )

    @property
    def server_declaration(self) -> str:
        c = self.ptr.ptr_to.const
        self.ptr.ptr_to.const = False
        # void is treated differently from non void pointer types
        s = f"    {self.ptr.format()} {self.parameter.name}_null_check;\n" + \
            f"    {self.ptr.format() if self.ptr.ptr_to.format() == "void" else self.ptr.ptr_to.format()} {self.parameter.name};\n"
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
                base_type=(self.ptr.format() if self.ptr.ptr_to.format() == "const void" else self.ptr.ptr_to.format()),
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
            "        rpc_read(0, &{param_name}_null_check, sizeof({server_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                server_type=self.ptr.format(),
            )
        )
        f.write(
            "        ({param_name}_null_check && rpc_read(0, {param_name}, sizeof({base_type})) < 0) ||\n".format(
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
    parameter: Parameter
    ptr: Pointer
    length: int | Parameter  # if int, it's a constant length, if Parameter, it's a variable length.

    def client_rpc_write(self, f):
        if not self.send:
            return
        elif isinstance(self.length, int):
            f.write(
                "        rpc_write(0, {param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.length,
                )
            )
        # array length operations are handled differently than char
        elif isinstance(self.ptr, Array):
            f.write(
                "        rpc_write(0, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    param_type=self.parameter.name,
                )
            )
        else:
            if isinstance(self.length.type, Pointer):
                length = "*" + self.length.name
            else:
                length = self.length.name
            f.write(
                "        rpc_write(0, {param_name}, {length} * sizeof({param_type})) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    param_type=self.ptr.ptr_to.format(),
                    length=length,
                )
            )
            

    @property
    def server_declaration(self) -> str:
        if isinstance(self.ptr, Array):
            c = self.ptr.const
            self.ptr.const = False
            # const[] isn't a valid part of a variable declaration
            s = f"    {self.ptr.format().replace("const[]", "")}* {self.parameter.name} = nullptr;\n"
            self.ptr.const = c
        else:
            c = self.ptr.ptr_to.const
            self.ptr.ptr_to.const = False
            s = f"    {self.ptr.format()} {self.parameter.name};\n"
            self.ptr.ptr_to.const = c
        return s
        
    def server_rpc_read(self, f):
        if not self.send:
            return
        elif isinstance(self.length, int):
            f.write(
                "        rpc_read(conn, &{param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.length,
                )
            )
        elif isinstance(self.ptr, Array):
            f.write(
                "        rpc_read(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    param_type=self.ptr.format().replace("[]", ""),
                )
            )
        else:
            if isinstance(self.length.type, Pointer):
                length = "*" + self.length.name
            else:
                length = self.length.name
            f.write(
                "        rpc_read(conn, {param_name}, {length} * sizeof({param_type})) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    param_type=self.ptr.ptr_to.format(),
                    length=length,
                )
            )

    @property
    def server_reference(self) -> str:
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
                "        rpc_write(conn, {param_name}, {length} * sizeof({param_type})) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    param_type=self.ptr.ptr_to.format(),
                    length=self.length.name,
                )
            )

    def client_rpc_read(self, f):
        if not self.recv:
            return
        if isinstance(self.length, int):
            f.write(
                "        rpc_read(0, {param_name}, {size}) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    size=self.length,
                )
            )
        else:
            if isinstance(self.length.type, Pointer):
                length = "*" + self.length.name
            else:
                length = self.length.name
            f.write(
                "        rpc_read(0, {param_name}, {length} * sizeof({param_type})) < 0 ||\n".format(
                    param_name=self.parameter.name,
                    param_type=self.ptr.ptr_to.format(),
                    length=length,
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
            "        rpc_write(0, &{param_name}_len, sizeof(std::size_t)) < 0 ||\n".format(
                param_name=self.parameter.name,
            )
        )
        f.write(
            "        rpc_write(0, {param_name}, {param_name}_len) < 0 ||\n".format(
                param_name=self.parameter.name,
            )
        )

    @property
    def server_declaration(self) -> str:
        return f"    {self.ptr.format()} {self.parameter.name};\n" + \
                f"    std::size_t {self.parameter.name}_len;\n"

    def server_rpc_read(self, f, index) -> Optional[str]:
        if not self.send:
            return
        f.write(
            "        rpc_read(conn, &{param_name}_len, sizeof(std::size_t)) < 0)\n".format(
                param_name=self.parameter.name
            )
        )
        f.write("        goto ERROR_{index};\n".format(index=index))
        f.write("    {param_name} = ({server_type})malloc({param_name}_len);\n".format(
            param_name=self.parameter.name,
            server_type=self.ptr.format(),
        ))
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
            "        rpc_read(0, &{param_name}_len, sizeof(std::size_t)) < 0 ||\n".format(
                param_name=self.parameter.name
            )
        )
        f.write(
            "        rpc_read(0, {param_name}, {param_name}_len) < 0 ||\n".format(
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
    type_: Type | Pointer

    def client_rpc_write(self, f):
        if not self.send:
            return
        else:
            f.write(
                "        rpc_write(0, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
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
        elif "const " in self.type_.format() and not "void" in self.type_.format() and not "*" in self.type_.format():
            return f"   {self.type_.format().replace("const", "")} {self.parameter.name};\n"
        else:
            return f"    {self.type_.format()} {self.parameter.name};\n"

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
            "        rpc_read(0, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
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
            "        rpc_write(0, {param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.ptr_to.format(),
            )
        )

    @property
    def server_declaration(self) -> str:
        return f"    {self.type_.ptr_to.format()} {self.parameter.name};\n"

    def server_rpc_read(self, f):
        if not self.send:
            return
        f.write(
            "        rpc_read(conn, &{param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.ptr_to.format(),
            )
        )

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
            "        rpc_read(0, {param_name}, sizeof({param_type})) < 0 ||\n".format(
                param_name=self.parameter.name,
                param_type=self.type_.ptr_to.format(),
            )
        )

Operation = NullableOperation | ArrayOperation | NullTerminatedOperation | OpaqueTypeOperation | DereferenceOperation

# parses a function annotation. if disabled is encountered, returns True for short circuiting.
def parse_annotation(annotation: str, params: list[Parameter]) -> list[tuple[Operation, bool]]:
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
            recv = (parts[2] == "RECV_ONLY" or parts[2] == "SEND_RECV")

            # if there's a length or size arg, use the type, otherwise use the ptr_to type
            length_arg = next(
                (arg for arg in args if arg.startswith("LENGTH:")), None
            )

            if isinstance(param.type, Pointer):
                if param.type.ptr_to.const:
                    recv = False
                
                size_arg = next((arg for arg in args if arg.startswith("SIZE:")), None)
                null_terminated = "NULL_TERMINATED" in args
                nullable = "NULLABLE" in args

                # validate that only one of the arguments is present
                if sum([bool(length_arg), bool(size_arg), null_terminated, nullable]) > 1:
                    raise NotImplementedError(
                        "Only one of LENGTH, SIZE, NULL_TERMINATED, or NULLABLE can be specified"
                    )
            
                if length_arg:
                    # if it has a length, it's an array operation with variable length
                    length_param = next(p for p in params if p.name == length_arg.split(":")[1])
                    operations.append(ArrayOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        ptr=param.type,
                        length=length_param,
                    ))
                elif size_arg:
                    # if it has a size, it's an array operation with constant length
                    operations.append(ArrayOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        ptr=param.type,
                        length=int(size_arg.split(":")[1]),
                    ))
                elif null_terminated:
                    # if it's null terminated, it's a null terminated operation
                    operations.append(NullTerminatedOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        ptr=param.type,
                    ))
                elif nullable:
                    # if it's nullable, it's a nullable operation
                    operations.append(NullableOperation(
                        send=send,
                        recv=recv,
                        parameter=param,
                        ptr=param.type,
                    ))
                else:
                    # otherwise, it's a pointer to a single value or another pointer
                    if recv:
                        if param.type.ptr_to.format() == "void":
                            raise NotImplementedError("Cannot dereference a void pointer")
                        # this is an out parameter so use the base type as the server declaration
                        operations.append(DereferenceOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            type_=param.type,
                        ))
                    else:
                        # otherwise, treat it as an opaque type
                        operations.append(OpaqueTypeOperation(
                            send=send,
                            recv=recv,
                            parameter=param,
                            type_=param.type,
                        ))
            elif isinstance(param.type, Type):
                if param.type.const:
                    recv = False
                operations.append(OpaqueTypeOperation(
                    send=send,
                    recv=recv,
                    parameter=param,
                    type_=param.type,
                ))
            elif isinstance(param.type, Array):
                length_param = next(p for p in params if p.name == length_arg.split(":")[1])
                if param.type.const:
                    recv = False
                operations.append(ArrayOperation(
                    send=send,
                    recv=recv,
                    parameter=param,
                    ptr=param.type,
                    length=length_param,
                ))
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


def main():
    options = ParserOptions(preprocessor=make_gcc_preprocessor(defines=["CUBLASAPI="]))

    nvml_ast: ParsedData = parse_file("/usr/include/nvml.h", options=options)
    cudnn_graph_ast: ParsedData = parse_file("/usr/include/cudnn_graph.h", options=options)
    cudnn_ops_ast: ParsedData = parse_file("/usr/include/cudnn_ops.h", options=options)
    cuda_ast: ParsedData = parse_file("/usr/include/cuda.h", options=options)
    cublas_ast: ParsedData = parse_file("/usr/include/cublas_api.h", options=options)
    cudart_ast: ParsedData = parse_file(
        "/usr/include/cuda_runtime_api.h", options=options
    )
    annotations: ParsedData = parse_file("annotations.h", options=options)

    # any new parsed libaries should be appended to the END of this list.
    # this is to maintain proper ordering of our RPC calls.
    functions = (
        nvml_ast.namespace.functions
        + cuda_ast.namespace.functions
        + cudart_ast.namespace.functions
        + cublas_ast.namespace.functions
        + cudnn_graph_ast.namespace.functions
        + cudnn_ops_ast.namespace.functions
    )

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
            operations, is_func_disabled = parse_annotation(annotation.doxygen, function.parameters)
        except Exception as e:
            print(f"Error parsing annotation for {function.name}: {e}")
            continue
        functions_with_annotations.append((function, annotation, operations, is_func_disabled))

    with open("gen_api.h", "w") as f:
        lastIndex = 0

        for i, function in enumerate(INTERNAL_FUNCTIONS):
            f.write(
                "#define RPC_{name} {value}\n".format(
                    name=function.format(),
                    value=i,
                )
            )
            lastIndex += 1

        for i, (function, _, _, _) in enumerate(functions_with_annotations):
            f.write(
                "#define RPC_{name} {value}\n".format(
                    name=function.name.format(),
                    value=i + lastIndex,
                )
            )

    with open("gen_client.cpp", "w") as f:
        f.write(
            "#include <nvml.h>\n"
            "#include <cuda.h>\n"
            "#include <cudnn.h>\n"
            "#include <cublas_v2.h>\n"
            "#include <cuda_runtime_api.h>\n\n"
            "#include <cstring>\n"
            "#include <string>\n"
            "#include <unordered_map>\n\n"
            '#include "gen_api.h"\n\n'
            '#include "manual_client.h"\n\n'
            "extern int rpc_size();\n"
            "extern int rpc_start_request(const int index, const unsigned int request);\n"
            "extern int rpc_write(const int index, const void *data, const std::size_t size);\n"
            "extern int rpc_end_request(const int index);\n"
            "extern int rpc_wait_for_response(const int index);\n"
            "extern int rpc_read(const int index, void *data, const std::size_t size);\n"
            "extern int rpc_end_response(const int index, void *return_value);\n"
            "extern int rpc_close();\n\n"
        )
        for function, annotation, operations, disabled in functions_with_annotations:
            # we don't generate client function definitions for disabled functions; only the RPC definitions.
            if disabled: continue

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

            joined_params = ", ".join(params)

            f.write(
                "{return_type} {name}({params})\n".format(
                    return_type=function.return_type.format(),
                    name=function.name.format(),
                    params=joined_params
                )
            )
            f.write("{\n")

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
                "    if (rpc_start_request(0, RPC_{name}) < 0 ||\n".format(
                    name=function.name.format()
                )
            )

            for operation in operations:
                operation.client_rpc_write(f)

            f.write("        rpc_wait_for_response(0) < 0 ||\n")

            for operation in operations:
                operation.client_rpc_read(f)

            f.write("        rpc_end_response(0, &return_value) < 0)\n")
            f.write(
                "        return {error_return};\n".format(
                    error_return=error_const(function.return_type.format())
                )
            )

            if function.name.format() == "nvmlShutdown":
                f.write("    if (rpc_close() < 0)\n")
                f.write("        return {error_return};\n".format(error_return=error_const(function.return_type.format())))

            f.write("    return return_value;\n")

            f.write("}\n\n")

        f.write("std::unordered_map<std::string, void *> functionMap = {\n")

        # we need the base nvmlInit, this is important and should be kept here in the codegen.
        for function in INTERNAL_FUNCTIONS:
            f.write(
                '    {{"{name}", (void *){name}}},\n'.format(
                    name=function.format()
                )
            )
        f.write('    {"nvmlInit", (void *)nvmlInit_v2},\n')
        for function, _, _, disabled in functions_with_annotations:
            if disabled: continue

            f.write(
                '    {{"{name}", (void *){name}}},\n'.format(
                    name=function.name.format()
                )
            )
        # write manual overrides
        function_names = set(f.name.format() for f, _, _, disabled in functions_with_annotations if not disabled)
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
            "#include <iostream>\n"
            "#include <cuda.h>\n"
            "#include <cudnn.h>\n"
            "#include <cublas_v2.h>\n"
            "#include <cuda_runtime_api.h>\n\n"
            "#include <cstring>\n"
            "#include <string>\n"
            "#include <unordered_map>\n\n"
            '#include "gen_api.h"\n\n'
            '#include "gen_server.h"\n\n'
            '#include "manual_server.h"\n\n'
            "extern int rpc_read(const void *conn, void *data, const std::size_t size);\n"
            "extern int rpc_end_request(const void *conn);\n"
            "extern int rpc_start_response(const void *conn, const int request_id);\n"
            "extern int rpc_write(const void *conn, const void *data, const std::size_t size);\n"
            "extern int rpc_end_response(const void *conn, void *return_value);\n\n"
        )
        for function, annotation, operations, disabled in functions_with_annotations:
            if function.name.format() in MANUAL_IMPLEMENTATIONS or disabled: continue

            # parse the annotation doxygen
            f.write(
                "int handle_{name}(void *conn)\n".format(
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
                f.write("    {return_type} scuda_intercept_result;\n".format(return_type=function.return_type.format()))
            else:
                f.write("    void* scuda_intercept_result;\n".format(return_type=function.return_type.format()))

            f.write("    if (\n")
            for operation in operations:
                if isinstance(operation, NullTerminatedOperation):
                    if error := operation.server_rpc_read(f, len(defers)):
                        defers.append(error)
                else:
                    operation.server_rpc_read(f)
            f.write("        false)\n")
            f.write("        goto ERROR_{index};\n".format(index=len(defers)))

            f.write("\n")

            f.write(
                "    request_id = rpc_end_request(conn);\n".format(
                    name=function.name.format()
                )
            )
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

            f.write("    if (rpc_start_response(conn, request_id) < 0 ||\n")

            for operation in operations:
                operation.server_rpc_write(f)
            
            f.write("        rpc_end_response(conn, &scuda_intercept_result) < 0)\n")
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
        for function in INTERNAL_FUNCTIONS:
            f.write("    handle_{name},\n".format(name=function.format()))
        for function, _, _, disabled in functions_with_annotations:
            if disabled and function.name.format() not in MANUAL_IMPLEMENTATIONS:
                f.write("    nullptr,\n")
            else:
                f.write("    handle_{name},\n".format(name=function.name.format()))
        f.write("};\n\n")

        f.write("RequestHandler get_handler(const int op)\n")
        f.write("{\n")
        f.write("    if (op > (sizeof(opHandlers) / sizeof(opHandlers[0])))\n")
        f.write("        return nullptr;\n")
        f.write("    return opHandlers[op];\n")
        f.write("}\n")


if __name__ == "__main__":
    main()
