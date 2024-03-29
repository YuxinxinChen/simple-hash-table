#ifndef COMM_CUDA
#define COMM_CUDA

#define TID (threadIdx.x+blockIdx.x*blockDim.x)
#define LANE (threadIdx.x&31)
#define WARPID ((threadIdx.x+blockIdx.x*blockDim.x)>>5)
__device__ uint32_t flagMask0 = 3;
__device__ uint32_t flagMask1 = 768;
__device__ uint32_t flagMask2 = 196608;
__device__ uint32_t flagMask3 = 50331648;
__device__ uint32_t flagMask4 = 1;
__device__ uint32_t flagMask5 = 65536;

#endif

#ifndef align_up
#define align_up(num, align) \
            (((num) + ((align) - 1)) & ~((align) - 1))
#endif

#ifndef align_down
#define align_down(num, align) \
            ((num) & ~((align) - 1))
#endif

