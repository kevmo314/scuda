import io
import sys
from pycparser import parse_file, c_ast
from typing import Optional

IGNORE_FUNCTIONS = {
    "nvmlDeviceCcuSetStreamState",
    "nvmlDeviceCcuGetStreamState",
}

def heap_allocation_size(node: c_ast.FuncDecl, p: c_ast.PtrDecl) -> Optional[str]:
    """
    Returns the size of heap-allocated parameters. Because there is no convention in
    Nvidia's API for how these parameters are defined, we need to manually specify
    which ones are heap allocated and how much memory they require.
    """
    if node.name == "nvmlSystemGetDriverVersion":
        if p.name == "version":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if node.name == "nvmlSystemGetNVMLVersion":
        if p.name == "version":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if node.name == "nvmlSystemGetProcessName":
        if p.name == "name":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if node.name == "nvmlSystemGetHicVersion":
        if p.name == "hwbcEntries":
            return ("*hwbcCount * sizeof(nvmlHwbcEntry_t)", "hwbcCount * sizeof(nvmlHwbcEntry_t)")
    if node.name == "nvmlUnitGetDevices":
        if p.name == "devices":
            return ("*deviceCount * sizeof(nvmlDevice_t)", "deviceCount * sizeof(nvmlDevice_t)")
    if node.name == "nvmlDeviceGetName":
        if p.name == "name":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if node.name == "nvmlDeviceGetSerial":
        if p.name == "serial":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if node.name == "nvmlDeviceGetMemoryAffinity":
        if p.name == "nodeSet":
            return ("nodeSetSize * sizeof(unsigned long)", "nodeSetSize * sizeof(unsigned long)")
    if node.name == "nvmlDeviceGetCpuAffinityWithinScope":
        if p.name == "cpuSet":
            return ("cpuSetSize * sizeof(unsigned long)", "cpuSetSize * sizeof(unsigned long)")
    if node.name == "nvmlDeviceGetCpuAffinity":
        if p.name == "cpuSet":
            return ("cpuSetSize * sizeof(unsigned long)", "cpuSetSize * sizeof(unsigned long)")
    if node.name == "nvmlDeviceGetTopologyNearestGpus":
        if p.name == "deviceArray":
            return ("*count * sizeof(nvmlDevice_t)", "count * sizeof(nvmlDevice_t)")
    if node.name == "nvmlDeviceGetUUID":
        if p.name == "uuid":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if node.name == "nvmlVgpuInstanceGetMdevUUID":
        if p.name == "mdevUuid":
            return ("size * sizeof(char)", "size * sizeof(char)")
    if node.name == "nvmlDeviceGetBoardPartNumber":
        if p.name == "partNumber":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if node.name == "nvmlDeviceGetInforomVersion":
        if p.name == "version":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if node.name == "nvmlDeviceGetInforomImageVersion":
        if p.name == "version":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if node.name == "nvmlDeviceGetEncoderSessions":
        if p.name == "sessionInfos":
            return ("*sessionCount * sizeof(nvmlEncoderSessionInfo_t)", "sessionCount * sizeof(nvmlEncoderSessionInfo_t)")
    if node.name == "nvmlDeviceGetVbiosVersion":
        if p.name == "version":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if node.name == "nvmlDeviceGetComputeRunningProcesses_v3":
        if p.name == "infos":
            return ("*infoCount * sizeof(nvmlProcessInfo_t)", "infoCount * sizeof(nvmlProcessInfo_t)")
    if node.name == "nvmlDeviceGetGraphicsRunningProcesses_v3":
        if p.name == "infos":
            return ("*infoCount * sizeof(nvmlProcessInfo_t)", "infoCount * sizeof(nvmlProcessInfo_t)")
    if node.name == "nvmlDeviceGetMPSComputeRunningProcesses_v3":
        if p.name == "infos":
            return ("*infoCount * sizeof(nvmlProcessInfo_t)", "infoCount * sizeof(nvmlProcessInfo_t)")
    if node.name == "nvmlDeviceGetSamples":
        if p.name == "samples":
            return ("*sampleCount * sizeof(nvmlSample_t)", "sampleCount * sizeof(nvmlSample_t)")
    if node.name == "nvmlDeviceGetAccountingPids":
        if p.name == "pids":
            return ("*count * sizeof(unsigned int)", "count * sizeof(unsigned int)")
    if node.name == "nvmlDeviceGetRetiredPages":
        if p.name == "addresses":
            return ("*pageCount * sizeof(unsigned long long)", "pageCount * sizeof(unsigned long long)")
    if node.name == "nvmlDeviceGetRetiredPages_v2":
        if p.name == "addresses":
            return ("*pageCount * sizeof(unsigned long long)", "pageCount * sizeof(unsigned long long)")
    if node.name == "nvmlDeviceGetFieldValues":
        if p.name == "values":
            return ("valuesCount * sizeof(nvmlFieldValue_t)", "valuesCount * sizeof(nvmlFieldValue_t)")
    if node.name == "nvmlDeviceClearFieldValues":
        if p.name == "values":
            return ("valuesCount * sizeof(nvmlFieldValue_t)", "valuesCount * sizeof(nvmlFieldValue_t)")
    if node.name == "nvmlDeviceGetSupportedVgpus":
        if p.name == "vgpuTypeIds":
            return ("*vgpuCount * sizeof(nvmlVgpuTypeId_t)", "vgpuCount * sizeof(nvmlVgpuTypeId_t)")
    if node.name == "nvmlDeviceGetCreatableVgpus":
        if p.name == "vgpuTypeIds":
            return ("*vgpuCount * sizeof(nvmlVgpuTypeId_t)", "vgpuCount * sizeof(nvmlVgpuTypeId_t)")
    if node.name == "nvmlVgpuTypeGetName":
        if p.name == "vgpuTypeName":
            return ("*size * sizeof(char)", "size * sizeof(char)")
    if node.name == "nvmlVgpuTypeGetLicense":
        if p.name == "vgpuTypeLicenseString":
            return ("size * sizeof(char)", "size * sizeof(char)")
    if node.name == "nvmlVgpuTypeGetMaxInstances":
        if p.name == "vgpuTypeId":
            return ("*vgpuInstanceCount * sizeof(unsigned int)", "vgpuInstanceCount * sizeof(unsigned int)")
    if node.name == "nvmlDeviceGetActiveVgpus":
        if p.name == "vgpuInstances":
            return ("*vgpuCount * sizeof(nvmlVgpuInstance_t)", "vgpuCount * sizeof(nvmlVgpuInstance_t)")
    if node.name == "nvmlVgpuInstanceGetVmID":
        if p.name == "vmId":
            return ("size * sizeof(char)", "size * sizeof(char)")
    if node.name == "nvmlVgpuInstanceGetUUID":
        if p.name == "uuid":
            return ("size * sizeof(char)", "size * sizeof(char)")
    if node.name == "nvmlVgpuInstanceGetVmDriverVersion":
        if p.name == "version":
            return ("length * sizeof(char)", "length * sizeof(char)")
    if node.name == "nvmlVgpuInstanceGetGpuPciId":
        if p.name == "vgpuPciId":
            return ("*length * sizeof(char)", "length * sizeof(char)")
    if node.name == "nvmlDeviceGetPgpuMetadataString":
        if p.name == "pgpuMetadata":
            return ("*bufferSize * sizeof(char)", "bufferSize * sizeof(char)")
    if node.name == "nvmlDeviceGetVgpuUtilization":
        if p.name == "utilizationSamples":
            return ("*vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)", "vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)")
    if node.name == "nvmlDeviceGetVgpuProcessUtilization":
        if p.name == "utilizationSamples":
            return ("*vgpuProcessSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)", "vgpuProcessSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)")
    if node.name == "nvmlVgpuInstanceGetAccountingPids":
        if p.name == "pids":
            return ("*count * sizeof(unsigned int)", "count * sizeof(unsigned int)")
    if node.name == "nvmlDeviceGetGpuInstancePossiblePlacements_v2":
        if p.name == "placements":
            return ("*count * sizeof(nvmlGpuInstancePlacement_t)", "count * sizeof(nvmlGpuInstancePlacement_t)")
    if node.name == "nvmlDeviceGetGpuInstances":
        if p.name == "gpuInstances":
            return ("*count * sizeof(nvmlGpuInstance_t)", "count * sizeof(nvmlGpuInstance_t)")
    if node.name == "nvmlGpuInstanceGetComputeInstancePossiblePlacements":
        if p.name == "gpuInstances":
            return ("*count * sizeof(nvmlComputeInstancePlacement_t)", "count * sizeof(nvmlComputeInstancePlacement_t)")
    if node.name == "nvmlGpuInstanceGetComputeInstances":
        if p.name == "computeInstances":
            return ("*count * sizeof(nvmlComputeInstance_t)", "count * sizeof(nvmlComputeInstance_t)")

def format_param(param):
    if isinstance(param, c_ast.TypeDecl):
        return '%s %s' % (' '.join(param.type.names), param.declname)
    if isinstance(param, c_ast.PtrDecl):
        return '%s *%s' % (' '.join(param.type.quals + param.type.type.names), param.type.declname)
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

class ClientCodegenVisitor(c_ast.NodeVisitor):
    def __init__(self, sink: io.IOBase):
        self.sink = sink
        self.sink.write("#include <nvml.h>\n")
        self.sink.write("#include <string>\n");
        self.sink.write("#include <unordered_map>\n\n")
        self.sink.write("#include \"gen_api.h\"\n\n")

        self.sink.write("extern int rpc_start_request(const unsigned int request);\n");
        self.sink.write("extern int rpc_write(const void *data, const size_t size);\n");
        self.sink.write("extern int rpc_read(void *data, const size_t size);\n");
        self.sink.write("extern int rpc_wait_for_response(const unsigned int request_id);\n");
        self.sink.write("extern int rpc_end_request(void *return_value, const unsigned int request_id);\n\n");
    
        self.functions = []

    def close(self):
        self.sink.write('std::unordered_map<std::string, void *> functionMap = {\n')
        for f in self.functions:
            self.sink.write('    {\"%s\", (void *)%s},\n' % (f, f))
        self.sink.write('};\n\n')
        
        self.sink.write('void *get_function_pointer(const char *name)\n')
        self.sink.write('{\n')
        self.sink.write('    auto it = functionMap.find(name);\n')
        self.sink.write('    if (it != functionMap.end())\n')
        self.sink.write('        return it->second;\n')
        self.sink.write('    return nullptr;\n')
        self.sink.write('}\n')
        
    def visit_Decl(self, node):
        if not isinstance(node.type, c_ast.FuncDecl):
            return
        return_type = node.type.type.type
        if not isinstance(return_type, c_ast.IdentifierType):
            return
        params = node.type.args.params if node.type.args else []

        if node.name in IGNORE_FUNCTIONS:
            return

        # construct the function signature
        self.sink.write('{return_type} {name}({params})\n'.format(
            return_type=return_type.names[0],
            name=node.name,
            params=', '.join(format_param(p) for p in params)
        ))
        self.functions.append(node.name)
        self.sink.write('{\n')
        self.sink.write('    int request_id = rpc_start_request(RPC_%s);\n' % node.name)
        self.sink.write('    %s return_value;\n' % return_type.names[0])
        self.sink.write('    if (request_id < 0 ||\n')
        # write the entire parameter list as a block of memory
        for p in params:
            if p.name is None:
                continue
            if isinstance(p.type, c_ast.TypeDecl):
                self.sink.write('        rpc_write(&%s, sizeof(%s)) < 0 ||\n' % (p.name, format_type_name(p.type)))
            if isinstance(p.type, c_ast.PtrDecl) and not heap_allocation_size(node, p):
                self.sink.write('        rpc_write(%s, sizeof(%s)) < 0 ||\n' % (p.name, format_type_name(p.type)))
        ptrs = [p for p in params if isinstance(p.type, c_ast.PtrDecl) and not heap_allocation_size(node, p)]
        self.sink.write('        rpc_wait_for_response(request_id) < 0 ||\n')
        for p in params:
            if p.name is None or not isinstance(p.type, c_ast.PtrDecl) or 'const' in p.type.type.quals:
                continue
            size = heap_allocation_size(node, p)
            if size:
                self.sink.write('        rpc_read(%s, %s) < 0 ||\n' % (p.name, size[0]))
            else:
                self.sink.write('        rpc_read(%s, sizeof(%s)) < 0 ||\n' % (p.name, format_type_name(p.type)))
        self.sink.write('        rpc_end_request(&return_value, request_id) < 0)\n')
        self.sink.write('        return NVML_ERROR_GPU_IS_LOST;\n')
        self.sink.write('    return return_value;\n')
        self.sink.write('}\n\n')

class ServerCodegenVisitor(c_ast.NodeVisitor):
    def __init__(self, sink: io.IOBase):
        self.sink = sink
        self.sink.write("#include <nvml.h>\n\n")
        self.sink.write("#include \"gen_api.h\"\n")
        self.sink.write("#include \"gen_server.h\"\n\n")

        self.sink.write("extern int rpc_read(const void *conn, void *data, const size_t size);\n");
        self.sink.write("extern int rpc_write(const void *conn, const void *data, const size_t size);\n");
        self.sink.write("extern int rpc_end_request(const void *conn);\n");
        self.sink.write("extern int rpc_start_response(const void *conn, const int request_id);\n");
    
        self.functions = []

    def close(self):
        self.sink.write('static RequestHandler opHandlers[] = {\n')
        for f in self.functions:
            self.sink.write('    handle_%s,\n' % (f))
        self.sink.write('};\n\n')
        
        self.sink.write('RequestHandler get_handler(const int op)\n')
        self.sink.write('{\n')
        self.sink.write('    return opHandlers[op];\n')
        self.sink.write('}\n')
        
    def visit_Decl(self, node):
        if not isinstance(node.type, c_ast.FuncDecl):
            return
        return_type = node.type.type.type
        if not isinstance(return_type, c_ast.IdentifierType):
            return
        params = node.type.args.params if node.type.args else []

        if node.name in IGNORE_FUNCTIONS:
            return

        # construct the function signature
        self.sink.write('int handle_{name}(void *conn)\n'.format(name=node.name))
        self.sink.write('{\n')
        self.functions.append(node.name)

        # read non-heap-allocated variables
        stack_vars = [p for p in params if p.name is not None and isinstance(p.type, c_ast.TypeDecl) or (isinstance(p.type, c_ast.PtrDecl) and not heap_allocation_size(node, p))]
        heap_vars = [p for p in params if p.name is not None and isinstance(p.type, c_ast.PtrDecl) and heap_allocation_size(node, p)]
        if len(stack_vars) > 0:
            for p in stack_vars:
                self.sink.write('    %s %s;\n' % (format_type_name(p.type), p.name))
            self.sink.write('\n')
            for i, p in enumerate(stack_vars):
                self.sink.write('%srpc_read(conn, &%s, sizeof(%s)) < 0%s' % (
                    '        ' if i > 0 else '    if (', p.name, format_type_name(p.type), ' ||\n' if i < len(stack_vars) - 1 else ')\n'))
            self.sink.write('        return -1;\n\n')

        self.sink.write('    int request_id = rpc_end_request(conn);\n')
        self.sink.write('    if (request_id < 0)\n')
        self.sink.write('        return -1;\n\n')

        # malloc heap-allocated variables
        if len(heap_vars) > 0:
            for p in heap_vars:
                size = heap_allocation_size(node, p)
                if not size:
                    continue
                self.sink.write('    %s *%s = (%s *)malloc(%s);\n' % (format_type_name(p.type), p.name, format_type_name(p.type), size[1]))
            self.sink.write('\n')
        return_type = node.type.type.type

        # call the function
        self.sink.write('    %s result = %s(' % (return_type.names[0], node.name))
        for i, p in enumerate(params):
            if p.name is None:
                continue
            if isinstance(p.type, c_ast.PtrDecl) and not heap_allocation_size(node, p):
                self.sink.write('&%s' % p.name)
            else:
                self.sink.write('%s' % p.name)
            if i < len(params) - 1:
                self.sink.write(', ')
        self.sink.write(');\n\n')


        self.sink.write('    if (rpc_start_response(conn, request_id) < 0)\n')
        self.sink.write('        return -1;\n\n')

        # write pointer vars
        ptr_vars = [p for p in params if p.name is not None and isinstance(p.type, c_ast.PtrDecl) and 'const' not in p.type.type.quals]
        if len(ptr_vars) > 0:
            for i, p in enumerate(ptr_vars):
                size = heap_allocation_size(node, p)
                if size:
                    self.sink.write('%srpc_write(conn, %s, %s) < 0%s' % (
                        '        ' if i > 0 else '    if (', p.name, size[1], ' ||\n' if i < len(ptr_vars) - 1 else ')\n'))
                else:
                    self.sink.write('%srpc_write(conn, &%s, sizeof(%s)) < 0%s' % (
                        '        ' if i > 0 else '    if (', p.name, format_type_name(p.type), ' ||\n' if i < len(ptr_vars) - 1 else ')\n'))
            self.sink.write('        return -1;\n\n')
        self.sink.write('    return result;\n')
        self.sink.write('}\n\n')

class HeaderCodegenVisitor(c_ast.NodeVisitor):
    def __init__(self, sink: io.IOBase):
        self.sink = sink
        self.index = 0
        
    def visit_Decl(self, node):
        if not isinstance(node.type, c_ast.FuncDecl):
            return
        return_type = node.type.type.type
        if not isinstance(return_type, c_ast.IdentifierType):
            return

        if node.name in IGNORE_FUNCTIONS:
            return

        # construct the function signature
        self.sink.write('#define RPC_{name} {index}\n'.format(name=node.name, index=self.index))
        self.index += 1


def show_func_defs(filename):
    # Note that cpp is used. Provide a path to your own cpp or
    # make sure one exists in PATH.
    ast = parse_file(filename, use_cpp=True)

    with open('gen_api.h', 'w') as f:
        v = HeaderCodegenVisitor(f)
        v.visit(ast)

    with open('gen_client.cu', 'w') as f:
        v = ClientCodegenVisitor(f)
        v.visit(ast)
        v.close()

    with open('gen_server.cu', 'w') as f:
        v = ServerCodegenVisitor(f)
        v.visit(ast)
        v.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        show_func_defs(sys.argv[1])
    else:
        print("Please provide a filename as argument")