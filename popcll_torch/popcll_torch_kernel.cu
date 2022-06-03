#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace
{
    template <typename scalar_t>
    __global__ void popcll_torch_cuda_forward_kernel(
        const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> input,
        torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> counts)
    {
        // index
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < input.size(0))
        {
            // Count operation
            counts[i] = __popcll(input[i]);
        }
    }
} // namespace

torch::Tensor popcll_torch_cuda_forward(torch::Tensor input)
{
    const auto batch_size = input.size(0);

    auto counts = torch::zeros_like(input);

    const int threads = 1024;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_INTEGRAL_TYPES(input.type(), "popcll_torch_forward_cuda", ([&] {
        popcll_torch_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            counts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>());
    }));

    return counts;
}
