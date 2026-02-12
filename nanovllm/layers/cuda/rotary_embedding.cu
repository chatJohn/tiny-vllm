#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
#include <cuda_bf16.h>


using bfloat16 = __nv_bfloat16;

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }


// one wrap process one head_dim
// block(32, 8), grid(row / 8)
__global__ void rotary_embedding_inplace_bf16_kernel(
    int64_t* positions, 
    bfloat16* query, 
    bfloat16* key, 
    float* sin_cos_cache, 
    int head_dim_half, 
    int num_batched_tokens, 
    int num_heads_query, 
    int num_heads_key
){
    const int lane_id = threadIdx.x;
    const int wrap_id = threadIdx.y;
    const int row = blockDim.y * blockIdx.x + wrap_id;
    if(row < num_batched_tokens * num_heads_query){
        query = query + head_dim_half * 2 * row;
        const int64_t pos = positions[row / num_heads_query];
        float* cache = sin_cos_cache + pos * head_dim_half * 2;
        for(int i = lane_id; i < head_dim_half; i += 32){
            float x = static_cast<float>(query[i]);
            float y = static_cast<float>(query[i + head_dim_half]);
            float cos = cache[i];
            float sin = cache[i + head_dim_half];
            query[i] = (bfloat16)(x * cos - y * sin);
            query[i + head_dim_half] = (bfloat16)(x * sin + y * cos);
        }
    }
    if(row < num_batched_tokens * num_heads_key){
        key = key + head_dim_half * 2 * row;
        const int64_t pos = positions[row / num_heads_key];
        float* cache = sin_cos_cache + pos * head_dim_half * 2;
        for(int i = lane_id; i < head_dim_half; i += 32){
            float x = static_cast<float>(key[i]);
            float y = static_cast<float>(key[i + head_dim_half]);
            float cos = cache[i];
            float sin= cache[i + head_dim_half];
            key[i] = (bfloat16)(x * cos - y * sin);
            key[i + head_dim_half] = (bfloat16)(x * sin + y * cos);
        }
    } 
}


void rotary_embedding_bf16(
    torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& sin_cos_cache
){
    CHECK_TORCH_TENSOR_DTYPE(positions, torch::kInt64);
    CHECK_TORCH_TENSOR_DTYPE(query, torch::kBFloat16);  // [num_batched_tokens, num_heads_query, head_dim], unseqeeze [num_batched_tokens * num_heads_query, head_dim]
    CHECK_TORCH_TENSOR_DTYPE(key, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(sin_cos_cache, torch::kFloat32);
    int num_batched_tokens = 1;
    const int ndim = query.dim();
    for(int i = 0; i < ndim - 2; i ++) {
        num_batched_tokens *= query.size(i);
    }
    int num_heads_query = query.size(-2);
    int num_heads_key = key.size(-2);
    int head_dim = query.size(-1);
    dim3 block(32, 8);
    dim3 grid((num_batched_tokens * max(num_heads_query, num_heads_key) + 7) / 8);
    rotary_embedding_inplace_bf16_kernel<<<grid, block>>>(
        reinterpret_cast<int64_t*>(positions.data_ptr()),
        reinterpret_cast<bfloat16*>(query.data_ptr()),
        reinterpret_cast<bfloat16*>(key.data_ptr()),
        reinterpret_cast<float*>(sin_cos_cache.data_ptr()),
        head_dim >> 1, num_batched_tokens, num_heads_query, num_heads_key
    );
}