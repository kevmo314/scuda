from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Type, Pointer
from typing import Optional

class Operation:
    param_name: str
    send: bool
    recv: bool
    args: list[str]

    @property
    def null_terminated(self) -> bool:
        return len(self.args) > 0 and self.args[0] == "NULL_TERMINATED"
    
    @property
    def array_length(self) -> Optional[str]:
        for arg in self.args:
            if arg.startswith("LENGTH:"):
                return arg.split(":")[1]

    @property
    def array_size(self) -> Optional[str]:
        for arg in self.args:
            if arg.startswith("SIZE:"):
                return arg.split(":")[1]
    

def parse_annotation(annotation: str) -> list[Operation]:
    operations = []
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
            operation = Operation()
            operation.param_name = parts[1]
            if parts[2] == "SEND_ONLY":
                operation.send = True
                operation.recv = False
            elif parts[2] == "SEND_RECV":
                operation.send = True
                operation.recv = True
            elif parts[2] == "RECV_ONLY":
                operation.send = False
                operation.recv = True
            operation.args = parts[3:]
            if length := operation.array_length:
                found = False
                for op in operations:
                    if op.param_name == length.removeprefix("*"):
                        found = True
                        break
                if not found:
                    print(f"Array length {length} not found on {annotation}")
            operations.append(operation)
    return operations

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

    functions_with_annotations = []
    for function in functions:
        try:
            annotation = next(
                f for f in annotations.namespace.functions if f.name == function.name
            )
        except StopIteration:
            print(f"Annotation for {function.name} not found")
            continue
        operations = parse_annotation(annotation.doxygen)
        functions_with_annotations.append((function, annotation, operations))

    with open("gen_api.h", "w") as f:
        visited = set()
        for function in functions:
            value = hash(function.name.format()) % (2**32)
            if value in visited:
                print(f"Hash collision for {function.name.format()}")
                continue
            f.write("#define RPC_{name} {value}\n".format(
                name=function.name.format(),
                value=value,
            ))

    with open("gen_client2.cpp", "w") as f:
        f.write(
            "#include <nvml.h>\n"
            "#include <cuda.h>\n"
            "#include <cuda_runtime_api.h>\n\n"
            "#include <cstring>\n"
            "#include <string>\n"
            "#include <unordered_map>\n\n"
            "#include \"gen_api.h\"\n\n"
            "extern int rpc_start_request(const unsigned int request);\n"
            "extern int rpc_write(const void *data, const size_t size);\n"
            "extern int rpc_read(void *data, const size_t size);\n"
            "extern int rpc_wait_for_response(const unsigned int request_id);\n"
            "extern int rpc_end_request(void *return_value, const unsigned int request_id);\n"
        )
        for function, annotation, operations in functions_with_annotations:
            f.write(
                "{return_type} {name}({params}) {{\n".format(
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

            f.write("    {return_type} return_value;\n\n".format(
                return_type=function.return_type.format()
            ))

            f.write("    int request_id = rpc_start_request(RPC_{name});\n".format(
                name=function.name.format()
            ))

            for operation in operations:
                if operation.null_terminated:
                    f.write("    size_t {param_name}_len = strlen({param_name}) + 1;\n".format(
                        param_name=operation.param_name
                    ))
    
            f.write("    if (request_id < 0 ||\n")

            for operation in operations:
                if operation.send:
                    if operation.null_terminated:
                        f.write("        rpc_write(&{param_name}_len, sizeof({param_name}_len)) < 0 ||\n".format(
                            param_name=operation.param_name
                        ))
                        f.write("        rpc_write({param_name}, {param_name}_len) < 0 ||\n".format(
                            param_name=operation.param_name
                        ))
                    else:
                        f.write("        rpc_write(&{param_name}, sizeof({param_name})) < 0 ||\n".format(
                            param_name=operation.param_name
                        ))
            f.write("        rpc_wait_for_response(request_id) < 0 ||\n")
            for operation in operations:
                if operation.recv:
                    if operation.null_terminated:
                        f.write("        rpc_read(&{param_name}_len, sizeof({param_name}_len)) < 0 ||\n".format(
                            param_name=operation.param_name
                        ))
                        f.write("        rpc_read({param_name}, {param_name}_len) < 0 ||\n".format(
                            param_name=operation.param_name
                        ))
                    elif length := operation.array_length:
                        f.write("        rpc_read({param_name}, {length} * sizeof({param_name})) < 0 ||\n".format(
                            param_name=operation.param_name,
                            length=length,
                        ))
                    elif size := operation.array_size:
                        f.write("        rpc_read({param_name}, {size}) < 0 ||\n".format(
                            param_name=operation.param_name,
                            size=size,
                        ))
                    else:
                        f.write("        rpc_read(&{param_name}, sizeof({param_name})) < 0 ||\n".format(
                            param_name=operation.param_name
                        ))
            f.write("        rpc_end_request(&return_value, request_id) < 0)\n")
            f.write("        return NVML_ERROR_GPU_IS_LOST;\n")
            f.write("    return return_value;\n")
            
            f.write("}\n\n")

        f.write("std::unordered_map<std::string, void *> functionMap = {\n")
        for function, _, _ in functions_with_annotations:
            f.write("    {{\"{name}\", (void *){name}}},\n".format(
                name=function.name.format()
            ))
        f.write("};\n\n")

        f.write("void *get_function_pointer(const char *name)\n")
        f.write("{\n")
        f.write("    auto it = functionMap.find(name);\n")
        f.write("    if (it == functionMap.end())\n")
        f.write("        return nullptr;\n")
        f.write("    return it->second;\n")
        f.write("}\n")

    with open("gen_server2.cpp", "w") as f:
        f.write(
            "#include <nvml.h>\n"
            "#include <cuda.h>\n"
            "#include <cuda_runtime_api.h>\n\n"
            "#include <cstring>\n"
            "#include <string>\n"
            "#include <unordered_map>\n\n"
            "#include \"gen_api.h\"\n\n"
            "#include \"gen_server.h\"\n\n"
            "extern int rpc_read(const void *conn, void *data, const size_t size);\n"
            "extern int rpc_write(const void *conn, const void *data, const size_t size);\n"
            "extern int rpc_end_request(const void *conn);\n"
            "extern int rpc_start_response(const void *conn, const int request_id);\n\n"
        )
        for function, annotation, operations in functions_with_annotations:
            # parse the annotation doxygen
            operations = [(operation, next(
                    p for p in function.parameters if p.name == operation.param_name
                )) for operation in operations]

            f.write(
                "int handle_{name}(void *conn)\n".format(
                    name=function.name.format(),
                )
            )
            f.write("{\n")

            for operation, param in operations:
                if operation.null_terminated:
                    f.write("    size_t {param_name}_len;\n".format(
                        param_name=operation.param_name
                    ))
                f.write("    {param_type} {param_name};\n".format(
                    param_type=param.type.format(),
                    param_name=operation.param_name,
                ))

            f.write("\n")

            for operation, param in operations:
                if operation.send:
                    if operation.null_terminated:
                        f.write("    if (rpc_read(conn, &{param_name}_len, sizeof({param_name}_len)) < 0)\n".format(
                            param_name=operation.param_name
                        ))
                        f.write("        return -1;\n")
                        f.write("    if (rpc_read(conn, &{param_name}, {param_name}_len) < 0)\n".format(
                            param_name=operation.param_name
                        ))
                        f.write("        return -1;\n")
                    else:
                        f.write("    if (rpc_read(conn, &{param_name}, sizeof({param_name})) < 0)\n".format(
                            param_name=operation.param_name
                        ))
                        f.write("        return -1;\n")

            f.write("\n")
            f.write("    int request_id = rpc_end_request(conn);\n".format(
                name=function.name.format()
            ))
            f.write("    if (request_id < 0)\n")
            f.write("        return -1;\n\n")

            f.write("    {return_type} return_value = {name}({params});\n\n".format(
                return_type=function.return_type.format(),
                name=function.name.format(),
                params=", ".join(param.name for param in function.parameters),
            ))

            f.write("    if (rpc_start_response(conn, request_id) < 0)\n")
            f.write("        return -1;\n")

            for operation, param in operations:
                if operation.recv:
                    if operation.null_terminated:
                        f.write("    if (rpc_write(conn, &{param_name}_len, sizeof({param_name}_len)) < 0)\n".format(
                            param_name=operation.param_name
                        ))
                        f.write("        return -1;\n")
                        f.write("    if (rpc_write(conn, &{param_name}, {param_name}_len) < 0)\n".format(
                            param_name=operation.param_name
                        ))
                        f.write("        return -1;\n")
                    elif length := operation.array_length:
                        f.write("    if (rpc_write(conn, {param_name}, {length} * sizeof({param_name})) < 0)\n".format(
                            param_name=operation.param_name,
                            length=length,
                        ))
                        f.write("        return -1;\n")
                    elif size := operation.array_size:
                        f.write("    if (rpc_write(conn, {param_name}, {size}) < 0)\n".format(
                            param_name=operation.param_name,
                            size=size,
                        ))
                        f.write("        return -1;\n")
                    else:
                        f.write("    if (rpc_write(conn, &{param_name}, sizeof({param_name})) < 0)\n".format(
                            param_name=operation.param_name
                        ))
                        f.write("        return -1;\n")
            
            f.write("}\n\n")

            
        f.write("static RequestHandler opHandlers[] = {\n")
        for function, _, _ in functions_with_annotations:
            f.write("    handle_{name},\n".format(
                name=function.name.format()
            ))
        f.write("};\n\n")

        f.write("RequestHandler get_handler(const int op)\n")
        f.write("{\n")
        f.write("    return opHandlers[op];\n")
        f.write("}\n")


if __name__ == "__main__":
    main()
