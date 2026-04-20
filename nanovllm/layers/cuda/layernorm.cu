#include <iostream>
#include <torch/extension.h>
#include <cuda_bf16.h>

using bfloat16 = __nv_bfloat16;
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                        \
    if(T.options().dtype() != th_type) {                            \
        std::cout << "Tensor info:" << T.options() << std::endl;    \
        throw std::runtime_error("value must be " #th_type);        \
    }


__device__ __forceinline__ float wrap_reduce(float val){
    for(int i = 16; i >= 1; i >>= 1){
        val += __shfl_down_sync(0xffffffff, val, i);
    }
    return val;
}

// y = gamma[weight] * x[input] / sqrt(mean(x^2) + eps)
__global__ void rmsnorm_bf16_kernel(bfloat16* output, bfloat16* input, bfloat16* weight, float eps, int hidden_size){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row = bid;
    // one block for one row
    // warp to warps
    output = output + row * hidden_size;
    input = input + row * hidden_size;
    float sum = 0.f;
    for(int i = tid; i < hidden_size; i += blockDim.x){
        float x = static_cast<float>(input[i]);  // 修复：使用static_cast而不是reinterpret_cast
        sum += x * x;
    }
    sum = wrap_reduce(sum); // partial wrap for one block, so there are many wraps
    // idx 0 warp to sum all wraps' partial sum
    const int NUM_WRAPS = (blockDim.x + 31) / 32;
    const int wid = tid >> 5;
    const int lid = tid & 31; // same as tid % 32
    __shared__ float sdata[32];
    if(lid == 0){
        sdata[wid] = sum; // store all wraps' partial sum
    }
    __syncthreads();
    if(wid == 0){
        sum = lid < NUM_WRAPS ? sdata[lid] : 0.f;
        sum = wrap_reduce(sum);
        if(lid == 0){
            sum = rsqrtf(sum / hidden_size + eps);
            sdata[0] = sum;
        }
    }
    __syncthreads();
    sum = sdata[0];
    for(int i = tid; i < hidden_size; i += blockDim.x){
        float x = static_cast<float>(input[i]);  // 修复：使用static_cast而不是reinterpret_cast
        output[i] = (bfloat16)(x * sum) * weight[i];
    }
}


__global__ void rmsnorm_fused_add_implace_bf16_kernel(bfloat16* input, bfloat16* residual, bfloat16* weight, float eps, int hidden_size){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row = bid; 
    input = input + row * hidden_size;
    residual = residual + row * hidden_size;
    float var = 0.f;
    for(int i = tid; i < hidden_size; i += blockDim.x){
        float x = static_cast<float>(input[i]);  // 修复：使用static_cast而不是reinterpret_cast
        float r = static_cast<float>(residual[i]);  // 修复：使用static_cast而不是reinterpret_cast
        x = x + r;
        residual[i] = (bfloat16)x;
        var += x * x;
    }
    var = wrap_reduce(var);
    // same as before
    const int NUM_WRAPS = (blockDim.x + 31) / 32;
    const int wid = tid >> 5;
    const int lid = tid & 31; // same as tid % 32
    __shared__ float sdata[32];
    if(lid == 0){
        sdata[wid] = var; // store all wraps' partial sum
    }
    __syncthreads();
    if(wid == 0){
        var = lid < NUM_WRAPS ? sdata[lid] : 0.f;
        var = wrap_reduce(var);
        if(lid == 0){
            var = rsqrtf(var / hidden_size + eps);
            sdata[0] = var;
        }
    }
    __syncthreads();
    var = sdata[0];
    for(int i = tid; i < hidden_size; i += blockDim.x){
        float x = static_cast<float>(residual[i]);  // 修复：使用static_cast而不是reinterpret_cast
        input[i] = (bfloat16)(x * var) * weight[i];
    } 
}


void rmsnrom_bf16(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, float eps){
    CHECK_TORCH_TENSOR_DTYPE(output, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(weight, torch::kBFloat16);
    int64_t hidden_size = input.size(-1);
    int64_t rows = input.numel() / hidden_size;
    dim3 block(256);
    dim3 grid(rows);
    rmsnorm_bf16_kernel<<<grid, block>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr()),
        reinterpret_cast<bfloat16*>(input.data_ptr()),
        reinterpret_cast<bfloat16*>(weight.data_ptr()),
        eps, 
        hidden_size
    );   
}

void rmsnorm_fused_add_bf16(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, float eps){
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(residual, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(weight, torch::kBFloat16);
    int64_t hidden_size = input.size(-1);
    int64_t rows = input.numel() / hidden_size;
    dim3 block(256);
    dim3 grid(rows);
    rmsnorm_fused_add_implace_bf16_kernel<<<grid, block>>>(
        reinterpret_cast<bfloat16*>(input.data_ptr()),
        reinterpret_cast<bfloat16*>(residual.data_ptr()),
        reinterpret_cast<bfloat16*>(weight.data_ptr()),
        eps,
        hidden_size
    );
}