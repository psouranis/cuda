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

__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1] {{1,2,3},
                                                              {4,5,6},
                                                              {7,8,9}};
                                                              
__global__ void convolution_tiled_2D_const_mem_kernel(float* N, float* P, 
													  int width, int height) {
	int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
	int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
	// Loading input tile
	__shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
	if(row>=0 && row<height && col>=0 && col<height){
		N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
	} else {
		N_s[threadIdx.y][threadIdx.x] = 0.0f;
	}
	__syncthreads();
	// Caclulating output elements
	int tileCol = threadIdx.x - FILTER_RADIUS;
	int tileRow = threadIdx.y - FILTER_RADIUS;
	// Turning off the threads at the edges of the block
	if (col >= 0 && col < width && row >=0 && row < width) {
		if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 
								&& tileRow < OUT_TILE_DIM) { 
			float Pvalue = 0.0f;
			for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
				for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
					Pvalue += F_c[fRow][fCol] * N_s[tileRow+fRow][tileCol+fCol];
				}
			}
			P[row*width+col] = Pvalue;
		}
	}
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