#include <iostream>
#include <torch/extension.h>
#include <cuda_bf16.h>

using bfloat16 = __nv_bfloat16;
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                        \
    if(T.options().dtype() != th_type) {                            \
        std::cout << "Tensor info:" << T.options() << std::endl;    \
        throw std::runtime_error("value must be " #th_type);        \
    }

// silu(x) = x / (1.f + exp(-x))

template<typename T>
__device__ __forceinline__ T silu(const T& x){
    return (T)(((float)x )/ (1.0f + expf((float)(-x))));
}

__global__ void silu_and_mul_bf16_kernel(bfloat16* output, bfloat16* x, bfloat16* y, const int N){  // 修复：bfloat*改为bfloat16*
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int idx = tx + bx * blockDim.x;
    if(idx < N){
        output[idx] = silu<bfloat16>(x[idx]) * y[idx];
    }
}


#define BF168(x) (reinterpret_cast<float4*>(&(x))[0])

__global__ void silu_and_mul_bf16x8_kernel(
    bfloat16* output, 
    bfloat16* x, 
    bfloat16* y, 
    int N
) {
    const int idx_thread = threadIdx.x + blockIdx.x * blockDim.x;
    const int idx = 8 * idx_thread;

    if (idx + 7 < N) {
        bfloat16 reg_x[8];
        bfloat16 reg_y[8];
        BF168(reg_x) = BF168(x[idx]);
        BF168(reg_y) = BF168(y[idx]);
        bfloat16 reg_out[8];
        reg_out[0] = silu<bfloat16>(reg_x[0]) * reg_y[0];
        reg_out[1] = silu<bfloat16>(reg_x[1]) * reg_y[1];
        reg_out[2] = silu<bfloat16>(reg_x[2]) * reg_y[2];
        reg_out[3] = silu<bfloat16>(reg_x[3]) * reg_y[3];
        reg_out[4] = silu<bfloat16>(reg_x[4]) * reg_y[4];
        reg_out[5] = silu<bfloat16>(reg_x[5]) * reg_y[5];
        reg_out[6] = silu<bfloat16>(reg_x[6]) * reg_y[6];
        reg_out[7] = silu<bfloat16>(reg_x[7]) * reg_y[7];
        BF168(output[idx]) = BF168(reg_out);
    }
}

void silu_and_mul_bf16(torch::Tensor &output, torch::Tensor &x, torch::Tensor &y){
    CHECK_TORCH_TENSOR_DTYPE(output, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kBFloat16);  // 修复：torch::BFloat16改为torch::kBFloat16
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kBFloat16);  // 修复：torch::BFloat16改为torch::kBFloat16
    int64_t N = x.numel();
    dim3 block(256);
    dim3 grid((N + 256 * 8 - 1) / (256 * 8));
    silu_and_mul_bf16x8_kernel<<<grid, block>>>(
        reinterpret_cast<bfloat16*>(output.data_ptr()),
        reinterpret_cast<bfloat16*>(x.data_ptr()),
        reinterpret_cast<bfloat16*>(y.data_ptr()),
        N
    );
}