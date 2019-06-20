#include<iostream>
#include<string>

#include<cuda.h>

#include "my_tuple.cuh"
#include "hash_simple.cuh"

#include "util/error_util.cuh"
#include "util/murmur.cuh"

using namespace std;

int main(int argc, char *argv[])
{
    CUDA_CHECK(cudaSetDevice(0));
    uint32_t table_size = 1<<20;

    MyTuple<int, int> empty_tuple(-1, 0);

    SimpleHash<MyTuple<int, int>, MurmurHash32<int>> hashTable(table_size, empty_tuple);
    hashTable.print();


    hashTable.release();

    return 0;
}
