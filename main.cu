#include<iostream>
#include<string>

#include<cuda.h>

#include "my_tuple.cuh"
#include "hash_simple.cuh"
#include "launch.cuh"

#include "util/error_util.cuh"
#include "util/murmur.cuh"

using namespace std;

int main(int argc, char *argv[])
{
    CUDA_CHECK(cudaSetDevice(0));
    uint32_t table_size = 1<<10;

    SimpleHash<MyTuple<int, int>, MurmurHash32<int>> hashTable(table_size);
    hashTable.print();

//    MyTuple<int, int> * rand_array;
//    CUDA_CHECK(cudaMallocManaged(&rand_array, sizeof(MyTuple<int, int>)*table_size/2));
//    uint32_t seed = 98789;
//    srand(seed);
//    for(int i=0; i<(table_size/2); i++)
//    {
//        int r = rand();
//        MyTuple<int, int> rr(r,r);
//        rand_array[i] = rr; 
//    }

 //   launch_warp((1<<4), 
 //               [] __device__ (MyTuple<int, int> *rand, SimpleHash<MyTuple<int, int>, MurmurHash32<int>> map) {
 //                       MyTuple<int, int> my_tuple = rand[WARPID];
 //                       map.insert_warp(my_tuple);
 //                   }, 
 //               rand_array, hashTable);

//    CUDA_CHECK(cudaDeviceSynchronize());
    hashTable.release();

    return 0;
}
