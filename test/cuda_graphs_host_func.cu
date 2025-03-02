/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3

typedef struct callBackData {
  const char *fn_name;
  double *data;
} callBackData_t;

void CUDART_CB myHostNodeCallback(void *data) {
  // Check status of GPU after stream operations are done
  callBackData_t *tmp = (callBackData_t *)(data);
  // checkCudaErrors(tmp->status);

  double *result = (double *)(tmp->data);
  char *function = (char *)(tmp->fn_name);
  printf("[%s] Host callback final reduced sum = %lf\n", function, *result);
}

void cudaGraphsManual() {
  cudaStream_t streamForGraph;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> nodeDependencies;
  double result_h = 1.0;

  checkCudaErrors(cudaStreamCreate(&streamForGraph));
  checkCudaErrors(cudaGraphCreate(&graph, 0));

  cudaGraphNode_t hostNode;
  cudaHostNodeParams hostParams = {0};
  hostParams.fn = myHostNodeCallback;
  callBackData_t hostFnData;
  hostFnData.data = &result_h;
  hostFnData.fn_name = "cudaGraphsManual";
  hostParams.userData = &hostFnData;

  checkCudaErrors(cudaGraphAddHostNode(&hostNode, graph,
                                       nodeDependencies.data(),
                                       nodeDependencies.size(), &hostParams));

  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
  printf("\nNum of nodes in the graph created manually = %zu\n", numNodes);

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  cudaGraph_t clonedGraph;
  cudaGraphExec_t clonedGraphExec;
  checkCudaErrors(cudaGraphClone(&clonedGraph, graph));
  checkCudaErrors(
      cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
  }

  checkCudaErrors(cudaStreamSynchronize(streamForGraph));

  printf("Cloned Graph Output.. \n");
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    checkCudaErrors(cudaGraphLaunch(clonedGraphExec, streamForGraph));
  }
  checkCudaErrors(cudaStreamSynchronize(streamForGraph));

  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaGraphExecDestroy(clonedGraphExec));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaGraphDestroy(clonedGraph));
  checkCudaErrors(cudaStreamDestroy(streamForGraph));
}

int main(int argc, char **argv) {
  printf("threads per block  = %d\n", THREADS_PER_BLOCK);
  printf("Graph Launch iterations = %d\n", GRAPH_LAUNCH_ITERATIONS);

  cudaGraphsManual();

  return EXIT_SUCCESS;
}
