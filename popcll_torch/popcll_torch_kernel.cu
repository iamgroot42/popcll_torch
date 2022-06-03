#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace
{
    template <typename scalar_t>
    __global__ void popcll_torch_cuda_kernel(
        const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> input,
        torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> counts)
    {
        // Opt for a grid-strided loop
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        const int n = input.size(0);
        for (int i = index; i < n; i += stride)
        {
            // Count operation
            counts[i] = __popcll(input[i]);
        }
    }
} // namespace

torch::Tensor popcll_torch_cuda(torch::Tensor input)
{
    const auto batch_size = input.size(0);

    auto counts = torch::zeros_like(input);

    const int blockSize = 1024;
    const int numBlocks = (batch_size + blockSize - 1) / blockSize;

    AT_DISPATCH_INTEGRAL_TYPES(input.type(), "popcll_torch_forward_cuda", ([&] {
        popcll_torch_cuda_kernel<scalar_t><<<numBlocks, blockSize>>>(
            input.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            counts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>());
    }));

    return counts;
}
