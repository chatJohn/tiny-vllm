#include <torch/extension.h>

void silu_and_mul_bf16(torch::Tensor& output, torch::Tensor& x, torch::Tensor& y);
void embedding_bf16(torch::Tensor& output, torch::Tensor& weight, torch::Tensor& token_ids);
void rmsnrom_bf16(torch::Tensor& output, torch::Tensor& input, torch::Tensor& weight, float eps);
void rmsnorm_fused_add_bf16(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, float eps);
void linear_bf16(torch::Tensor& A, torch::Tensor& B, torch::Tensor& C);
void rotary_embedding_bf16(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, torch::Tensor& sin_cos_cache);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("silu_and_mul_bf16", &silu_and_mul_bf16, "silu_and_mul_bf16");
    m.def("embedding_bf16", &embedding_bf16, "embedding_bf16");
    m.def("rmsnorm_bf16", &rmsnrom_bf16, "rmsnorm_bf16");
    m.def("rmsnorm_fused_add_bf16", &rmsnorm_fused_add_bf16, "rmsnorm_fused_add_bf16");
    m.def("linear_bf16", &linear_bf16, "linear_bf16");
    m.def("rotary_embedding_bf16", &rotary_embedding_bf16, "rotary_embedding_bf16"); 
}