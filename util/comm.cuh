#ifndef COMM_CUDA
#define COMM_CUDA

#define TID (threadIdx.x+blockIdx.x*blockDim.x)
#define LANE (threadIdx.x&31)
#define WARPID ((threadIdx.x+blockIdx.x*blockDim.x)>>5)

#endif

