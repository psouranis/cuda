#define IN_TILE_DIM 6
#define OUT_TILE_DIM 4

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
	int iStart = blockIdx.z * OUT_TILE_DIM;
	int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
	int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
	__shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
	__shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
	__shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
	if(iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
		inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart -1)*N*N + j*N + k];
	}
	if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
		inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j*N + k];
	}
	for(int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
		if(i+1>=0 && i+1<N && j>= 0 && j< N && k>=0 && k < N) {
			inNext_s[threadIdx.y][threadIdx.x] = in[(i+1)*N*N + j*N +k];
		}
		__syncthreads();
		if(i>=1 && i<N-1 && j>=1 && j<N-1 && k>=1 && k<N-1) {
			if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
				out[i*N*N + j*N + k] = c0 * inCurr_s[threadIdx.y][threadIdx.x]
									 + c1 * inCurr_s[threadIdx.y][threadIdx.x-1]
									 + c2 * inCurr_s[threadIdx.y][threadIdx.x+1]
									 + c3 * inCurr_s[threadIdx.y+1][threadIdx.x]
									 + c4 * inCurr_s[threadIdx.y-1][threadIdx.x]
									 + c5 * inPrev_s[threadIdx.y][threadIdx.x]
									 + c6 * inNext_s[threadIdx.y][threadIdx.x];
			}
		}
		__syncthreads();
		inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
		inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
	}
}