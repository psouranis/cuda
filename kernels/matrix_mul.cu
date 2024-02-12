#include <stdio.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void matrix_mul_kernel_row(const float* matrix_a, const float* matrix_b, float* result, 
                                  int width){
	
	int row = threadIdx.x;
 
	// Check for thread out of bound access
	if (row < width){
        for (int col = 0; col < width; ++col) {
            float sum = 0;	
            for (int k = 0; k < width; ++k) {
    			sum += matrix_a[row * width + k] * matrix_b[k * width + col];
            }
            result[row * width + col] = sum;
        }
    }
}

torch::Tensor matrix_mul_row(torch::Tensor matrix_a, torch::Tensor matrix_b){

    const auto width = matrix_a.size(0);
    auto result =  torch::empty_like(matrix_a);

    matrix_mul_kernel_row<<<1, width>>>(
        matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), result.data_ptr<float>(), width); // assume square matrix
    return result;
}

__global__ void matrix_mul_kernel_col(const float* matrix_a, const float* matrix_b, float* result, 
                                  int width){
	
	int col = threadIdx.x;
 
	// Check for thread out of bound access
	if (col < width){
        for (int row = 0; row < width; ++row) {
            float sum = 0;	
            for (int k = 0; k < width; ++k) {
    			sum += matrix_a[row * width + k] * matrix_b[k * width + col];
            }
    		result[row * width + col] = sum;
        }
    }
}

torch::Tensor matrix_mul_col(torch::Tensor matrix_a, torch::Tensor matrix_b){

    const auto width = matrix_a.size(0);
    auto result =  torch::empty_like(matrix_a);

    matrix_mul_kernel_col<<<1, width>>>(
        matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), result.data_ptr<float>(), width); // assume square matrix
    return result;
}


// defintion with memory management
void matrix_mul(float* matrix_A, float* matrix_B, float* matrix_C, int width){

	float *A_d, *B_d, *C_d;
	int size = width * width * sizeof(float);

	cudaMalloc((void **) &A_d, size);
	cudaMalloc((void **) &B_d, size);
	cudaMalloc((void **) &C_d, size);

	cudaMemcpy(A_d, matrix_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, matrix_B, size, cudaMemcpyHostToDevice);

	// We need only one block sincec each thread will calculate one row / one col
	matrix_mul_kernel_row<<<1, width>>>(A_d, B_d, C_d, width);
    // matrix_mul_kernel_col<<<1, width>>>(A_d, B_d, C_d, width)

	cudaMemcpy(matrix_C, C_d, size, cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

/// Matrix vector multiplication

__global__ void matrix_vec_mul_kernel(const float* matrix_b, const float* vec_c, float* vec_a, int width){
	
	int idx = threadIdx.x;
 
	// Check for thread out of bound access
	if (idx < width){
		float sum = 0;
		for (int col = 0; col < width; ++col) {
			sum += matrix_b[idx * width + col] * vec_c[col];
		}
        vec_a[idx] = sum;
    }
}

torch::Tensor matrix_vec_mul(torch::Tensor matrix_b, torch::Tensor vec_c){

    const auto width = matrix_b.size(0);
    auto vec_a =  torch::empty_like(vec_c);

    matrix_vec_mul_kernel<<<1, width>>>(
        matrix_b.data_ptr<float>(), vec_c.data_ptr<float>(), vec_a.data_ptr<float>(), width); // assume square matrix
    return vec_a;
}

// definition with memory management
void matrix_vec_mul(float* vec_A, float* matrix_b, float* vec_C, int width) {

    float *A_d, *B_d, *C_d;
    int size_A = width * sizeof(float);
    int size_B = width * width * sizeof(float);

    cudaMalloc((void **)&A_d, size_A);
    cudaMalloc((void **)&B_d, size_B);
    cudaMalloc((void **)&C_d, size_B);

    cudaMemcpy(C_d, vec_C, size_B, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, matrix_b, size_B, cudaMemcpyHostToDevice);

    matrix_vec_mul<<<1, width>>>(B_d, C_d, A_d, width);

	cudaMemcpy(vec_A, A_d, size_A, cudaMemcpyDeviceToHost);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(A_d);
}
