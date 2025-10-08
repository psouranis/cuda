# define NUM_BINS 7

__global__ void histo_private_kernel(char* data, unsigned int length,
									 unsigned int* histo) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i < length) {
		int alphabet_position = data[i] - 'a';
		if (alphabet_position>=0 && alphabet_position<26) {
			atomicAdd(&(histo[blockIdx.x*NUM_BINS + alphabet_position/4]), 1);
		}
	}
	
	if(blockIdx.x > 0) {
		__syncthreads();
		for(unsigned int bin=threadIdx.x; bin<NUM_BINS; bin+=blockDim.x) {
			unsigned int binValue = histo[blockIdx.x*NUM_BINS + bin];
			if (binValue > 0) {
				atomicAdd(&(histo[bin]), binValue);
			}
		}
	}									 
}