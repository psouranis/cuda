# define NUM_BINS 7

__global__ void histo_private_kernel(char* data, unsigned int length, unsigned int* histo) {
	// Initialize private bins
	__shared__ unsigned int histo_s[NUM_BINS];
	for(unsigned int bin=threadIdx.x; bin<NUM_BINS; bin+=blockDim.x) {
		histo_s[bin] = 0u;
	}
	__syncthreads();
	// Histogram
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	for(unsigned int i=tid; i<length; i+=blockDim.x*gridDim.x) {
		int alphabet_position = data[i] - 'a';
		if(alphabet_position>=0 && alphabet_position<26) {
			atomicAdd(&(histo_s[alphabet_position/4]), 1);
		}
	}
	__syncthreads();
	// Commit to global memory
	for(unsigned int bin=threadIdx.x; bin<NUM_BINS; bin+=blockDim.x) {
		unsigned int binValue = histo_s[bin];
		if(binValue>0) {
			atomicAdd(&(histo[bin]), binValue);
		}
	}
}