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


#define BHALF4(x) (reinterpret_cast<float2*>(&(x))[0])
#define BHALF8(x) (reinterpret_cast<float4*>(&(x))[0])
#define BHALF4_CONST(x) (reinterpret_cast<const float2*>(&(x))[0])
// In Matrix, TN is col, TM is row, so change the order if you read
// each thread compute the size of (TN, TM),
// each block computes the size of (BN, BM),
// So, dim3 block(BN / TN, BM / TM)
// and, dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM)
// tiled and shared mem mm
// First, A is (M, K), B is (K, N), C is (M, N)
template<int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
__global__ void bfloat16_matrix_multiplication(const bfloat16* A, const bfloat16* B, bfloat16* C, int M, int N, int K){
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tid = ty * blockDim.x + tx;
  A = A + blockDim.z * M * K; // continuous batching
  B = B + blockDim.z * K * N;
  C = C + blockDim.z * M * N;
  __shared__ bfloat16 sa[BM][BK];
  __shared__ bfloat16 sb[BN][BK];
  bfloat16 ra[TM]; // register data
  bfloat16 rb[TN];
  float rc[TM][TN] = {0.f};
  
  // BM * BN / (TM * TN) is the number of thread
  const int load_a_elements_per_thread = BM * BK / ((BM * BN) / (TM * TN));
  const int load_b_elements_per_thread = BK * BN / ((BM * BN) / (TM * TN));
  // how many threads needed when loading one row data of A(B), also for block
  const int load_a_threads_per_row = BK / load_a_elements_per_thread;
  const int load_b_thread_per_row = BK / load_b_elements_per_thread;
  
  const int load_a_smem_m = tid / load_a_threads_per_row;
  const int load_a_gmem_m = by * BM + load_a_smem_m;
  const int load_b_smem_n = tid / load_b_thread_per_row;
  const int load_b_gmem_n = bx * BN + load_b_smem_n;

  for(int bk = 0; bk < (K + BK - 1) / BK; bk ++) {
        for(int i = 0; i < load_a_elements_per_thread; i ++) {
            const int load_a_smem_k = (tid % load_a_threads_per_row) * load_a_elements_per_thread + i;
            const int load_a_gmem_k = bk * BK + load_a_smem_k;
            if(load_a_gmem_m < M && load_a_gmem_k < K) sa[load_a_smem_m][load_a_smem_k] = A[load_a_gmem_m * K + load_a_gmem_k];
            else sa[load_a_smem_m][load_a_smem_k] = (bfloat16)(0.f);
        }

        
        for(int i = 0; i < load_b_elements_per_thread; i ++) {
            const int load_b_smem_k = (tid % load_b_thread_per_row) * load_b_elements_per_thread + i;
            const int load_b_gmem_k = bk * BK + load_b_smem_k;
            if(load_b_gmem_n < N && load_b_gmem_k < K) sb[load_b_smem_n][load_b_smem_k] = B[load_b_gmem_n * K + load_b_gmem_k];
            else sb[load_b_smem_n][load_b_smem_k] = (bfloat16)(0.f);
        }

        __syncthreads();

        for(int k = 0; k < BK; k ++) {
            for(int m = 0; m < TM; m ++) {
                ra[m] = sa[ty + m * blockDim.y][k];
            }
            for(int n = 0; n < TN; n ++) {
                rb[n] = sb[tx + n * blockDim.x][k];
            }
            for(int m = 0; m < TM; m ++) {
                for(int n = 0; n < TN; n ++) {
                    rc[m][n] = ((float)(ra[m]) * (float)(rb[n])) + rc[m][n];
                }
            }
        }

        __syncthreads();
    }

    for(int m = 0; m < TM; m ++) {
        for(int n = 0; n < TN; n ++) {
            const int store_c_gmem_m = by * BM + ty + m * blockDim.y;
            const int store_c_gmem_n = bx * BN + tx + n * blockDim.x;
            if(store_c_gmem_m < M && store_c_gmem_n < N) {
                const int store_c_gmem_ptr = store_c_gmem_m * N + store_c_gmem_n;
                C[store_c_gmem_ptr] = (bfloat16)rc[m][n];
            }
        }
    }
}

void linear_bf16(torch::Tensor& A, torch::Tensor& B, torch::Tensor& C) {
    // C = A * B
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(B, torch::kBFloat16);
    CHECK_TORCH_TENSOR_DTYPE(C, torch::kBFloat16);
    int ndim = A.dim();
    int BATCH = 1;
    for(int i = 0; i < ndim-2; i ++) BATCH *= A.size(i);
    int M = A.size(-2);
    int K = A.size(-1);
    int N = B.size(-2);
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 blockDim(BN/TN, BM/TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
    bfloat16_matrix_multiplication<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
        reinterpret_cast<bfloat16*>(A.data_ptr()), 
        reinterpret_cast<bfloat16*>(B.data_ptr()), 
        reinterpret_cast<bfloat16*>(C.data_ptr()), 
        M, N, K
    );
}