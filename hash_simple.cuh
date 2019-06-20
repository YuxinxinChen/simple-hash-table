#include<iostream>

#include "util/error_util.cuh"
#include "util/murmur.cuh"
#include "util/comm.cuh"

using namespace std;

template<typename T, typename F>
struct SimpleHash;

template<typename T, typename F>
__global__ void SetEmpty(SimpleHash<T, F> hash_table)
{
    for(uint32_t i=TID; i< hash_table.table_size; i+=gridDim.x*blockDim.x)
        hash_table.table[i] = hash_table.empty_slot;
}

template<typename T, typename F>
struct SimpleHash
{
    T * table; // key table
    T empty_slot;
    uint32_t table_size;
    uint32_t seed;
    F hashFunc;

    SimpleHash(uint32_t _table_size, T _empty_slot, uint32_t _seed=12321):
        table_size(_table_size), empty_slot(_empty_slot), seed(_seed)
    {
        CUDA_CHECK(cudaMallocManaged(&table, sizeof(T)*table_size));
        SetEmpty<<<320, 512>>>(*this);
        CUDA_CHECK(cudaDeviceSynchronize())
  //      CUDA_CHECK(cudaMemset(table, empty_slot, sizeof(T)*table_size));
    }

    __host__ void release()
    {
        cudaFree(table);
    }

    __host__ void print()
    {
        cout << "table size: "<< table_size << " seed: "<< seed << " empty slot: "<< empty_slot << endl;
    }
};


