from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Type, Pointer


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

    with open("annotations.h", "a") as f:
        for function in functions:
            if any(f.name == function.name for f in annotations.namespace.functions):
                continue
            # produce some best-guess annotations
            f.write("/**\n")
            for param in function.parameters:
                if isinstance(param.type, Type):
                    f.write(
                        " * @param {name} SEND_ONLY\n".format(
                            name=param.name, type=param.type.format()
                        )
                    )
                elif isinstance(param.type, Pointer):
                    f.write(
                        " * @param {name} SEND_RECV\n".format(
                            name=param.name, type=param.type.format()
                        )
                    )
            f.write(" */\n")
            f.write(
                "{return_type} {name}({params});\n".format(
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


if __name__ == "__main__":
    main()
