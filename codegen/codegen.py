import sys
from pycparser import parse_file, c_ast

def format_param(param):
    if isinstance(param, c_ast.TypeDecl):
        return '%s %s' % (' '.join(param.type.names), param.declname)
    if isinstance(param, c_ast.PtrDecl):
        return '%s *%s' % (' '.join(param.type.type.names), param.type.declname)
    if isinstance(param, c_ast.Typename):
        return '%s' % (' '.join(param.type.type.names))
    if isinstance(param, c_ast.Decl):
        return format_param(param.type)
    print(param)
    raise NotImplementedError('Unsupported param type: %s' % type(param))

def format_type_name(param):
    if isinstance(param, c_ast.TypeDecl):
        return ' '.join(param.type.names)
    if isinstance(param, c_ast.PtrDecl):
        return ' '.join(param.type.type.names)
    if isinstance(param, c_ast.Typename):
        return ' '.join(param.type.type.names)
    if isinstance(param, c_ast.Decl):
        return format_type_name(param.type)
    print(param)
    raise NotImplementedError('Unsupported param type: %s' % type(param))

class FuncDefVisitor(c_ast.NodeVisitor):
    def visit_Decl(self, node):
        if not isinstance(node.type, c_ast.FuncDecl):
            return
        return_type = node.type.type.type
        if not isinstance(return_type, c_ast.IdentifierType):
            return
        params = node.type.args.params if node.type.args else []

        # construct the function signature
        print('{return_type} {name}({params})'.format(
            return_type=return_type.names[0],
            name=node.name,
            params=', '.join(format_param(p) for p in params)
        ))
        print('{')
        print('    int request_id = rpc_start_request(RPC_%s);' % node.name)
        print('    if (request_id < 0 ||')
        # write the entire parameter list as a block of memory
        for p in params:
            if isinstance(p.type, c_ast.TypeDecl):
                print('        rpc_write(&%s, sizeof(%s)) < 0 ||' % (p.name, format_type_name(p.type)))
            if isinstance(p.type, c_ast.PtrDecl):
                print('        rpc_write(%s, sizeof(%s)) < 0 ||' % (p.name, format_type_name(p.type)))
        print('        rpc_wait_for_response(request_id) < 0 ||')
        for p in params:
            if isinstance(p.type, c_ast.PtrDecl):
                print('        rpc_read(%s, sizeof(%s)) < 0 ||' % (p.name, format_type_name(p.type)))
        print('        return NVML_ERROR_GPU_IS_LOST;')
        print('    return rpc_get_return(request_id);')
        print('}')
        print()


def show_func_defs(filename):
    # Note that cpp is used. Provide a path to your own cpp or
    # make sure one exists in PATH.
    ast = parse_file(filename, use_cpp=True)

    v = FuncDefVisitor()
    v.visit(ast)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        show_func_defs(sys.argv[1])
    else:
        print("Please provide a filename as argument")