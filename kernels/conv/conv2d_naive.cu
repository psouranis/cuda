#include <stdio.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define IN_TILE_DIM 5
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))
#define FILTER_RADIUS 1

__global__ void convolution_2D_basic_kernel(float* N, float* F, float* P, int r, int width, int height) {
	int outCol = blockIdx.x * blockDim.x + threadIdx.x;
	int outRow = blockIdx.y * blockDim.y + threadIdx.y;
	float Pvalue = 0.0f;
	for (int fRow = 0; fRow < 2*r + 1; fRow++) {
		for (int fCol = 0; fCol < 2*r + 1; fCol++) {
			inRow = outRow -r + fRow;
			inCol = outCol - r + fCol;
			if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
				Pvalue += F[fRow][fCol] * N[inRow * width + inCol];
			}
		}
	}
	P[outRow][outCol] = Pvalue;
}

torch::Tensor conv2d(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(IN_TILE_DIM, IN_TILE_DIM); // launches thread blocks whose dimension matches that of the input tiles
    dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                          (height + threads_per_block.y - 1) / threads_per_block.y);

    convolution_tiled_2D_const_mem_kernel<<<number_of_blocks, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);

    return result;
    }