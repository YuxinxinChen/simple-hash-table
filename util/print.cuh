#ifndef PRINT_ARRAY_H
#define PRINT_ARRAY_H
__global__ void printArray(float *array, uint32_t size)
{
    if(threadIdx.x+blockDim.x*blockIdx.x == 0)
    {
        for(int i=0; i<size; i++)
            printf("%f ", array[i]);
        printf("\n");
    }
}
__global__ void printArray(int *array, uint32_t size)
{
    if(threadIdx.x+blockDim.x*blockIdx.x == 0)
    {
        for(int i=0; i<size; i++)
            printf("%d ", array[i]);
        printf("\n");
    }
}
#endif
