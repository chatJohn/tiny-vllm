#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <cuda_bf16.h>

using bfloat16 = __nv_bfloat16;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                        \
    if(T.options().dtype() != th_type) {                            \
        std::cout << "Tensor info:" << T.options() << std::endl;    \
        throw std::runtime_error("value must be " #th_type);        \
    }


// one row for one block
__global__ void embedding_bf16_kernel(bfloat16* output, bfloat16* weight, int64_t* token_ids, const int embedding_dim){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int64_t token_id = token_ids[bid];
    output = output + bid * embedding_dim; // get the start addr for WRITE
    weight = weight + token_id * embedding_dim; // get the start addr for READ
    for(int i = tid; i < embedding_dim; i += blockDim.x){
        output[i] = weight[i];
    }
}


void embedding_bf16(torch::Tensor& output, torch::Tensor& weight, torch::Tensor& token_ids){
    CHECK_TORCH_TENSOR_DTYPE(output, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(weight, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(token_ids, torch::kInt64);
    int64_t num_tokens = token_ids.numel();
    int64_t embedding_dim = weight.size(-1);
    dim3 block(256);
    dim3 grid(num_tokens);
    embedding_bf16_kernel<<<grid, block>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr()),
        reinterpret_cast<bfloat16*>(weight.data_ptr()),
        reinterpret_cast<int64_t*>(token_ids.data_ptr()),
        embedding_dim
    );
}