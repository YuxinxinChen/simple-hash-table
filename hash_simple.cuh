#ifndef HASH_TABLE
#define HASH_TABLE

#include<iostream>

#include "util/error_util.cuh"
#include "util/murmur.cuh"
#include "util/comm.cuh"

#define USED ((unsigned short int)1)
#define FREE ((unsigned short int)0)
#define DEAD ((unsigned short int)2)

#define NO_FREE ((char)0)
#define FREE_FAIL ((char)1)
#define FREE_SUCCESS ((char)2)

using namespace std;

template<typename T, typename F>
struct SimpleHash
{
    T * table; // key table
    unsigned short int * flags; 
    uint32_t capacity;
    uint32_t seed;
    F hashFunc;

    SimpleHash(uint32_t _capacity, uint32_t _seed=12321):
         seed(_seed)
    {
        capacity = align_up(_capacity, 2);
        CUDA_CHECK(cudaMallocManaged(&table, sizeof(T)*capacity));
        CUDA_CHECK(cudaMallocManaged(&flags, sizeof(unsigned short int)*capacity));
        CUDA_CHECK(cudaMemset((void *)flags, FREE, sizeof(unsigned short int)*capacity/sizeof(int)));
    }

    
    __device__ char try_free_slot_warp(unsigned &mask, T &tuple, uint32_t prob_entry)
    {
        uint32_t flag = ((volatile uint32_t *)flags)[(prob_entry+LANE)%capacity];
        bool ifFree = (flag&flagMask0)&(flag&flagMask2);
        ifFree = ~ifFree;
        unsigned ifFreeMask = __ballot_sync(mask, ifFree);
        if(ifFreeMask != 0) // there are free slots
        {
            int laneID = __ffs(ifFreeMask)-1;
            bool success = 0;
            if(LANE == laneID)
            {
                uint32_t slot = prob_entry+laneID*2+1;
                if((flag&flagMask0)==0)
                    slot  = prob_entry+laneID*2;
                
                if(success = request_slot(slot)) write_tuple(slot, tuple);
            }
            success = __shfl_sync(mask, laneID, success);
            if(success)
                return FREE_SUCCESS;
            else return FREE_FAIL; 
        }
        return NO_FREE;
    }

    // single thread function
    __device__ bool request_slot(uint32_t &slot)
    {
        unsigned short int old = atomicCAS(flags+slot, FREE, USED);
        if(old == FREE)
            return true;
        else return false;
    }

    __device__ void write_tuple(uint32_t &slot, T &tuple)
    {
        table[slot] = tuple;
    }

    __device__ T get_tuple(uint32_t &prob_entry)
    {
        return table[(prob_entry+LANE)%capacity];
    }

    __device__ unsigned short int get_flag(uint32_t &prob_entry)
    {
        return flags[(prob_entry+LANE)%capacity];
    }

    __device__ bool insert_warp(const T & tuple) 
    {
        unsigned mask = __activemask();
        uint32_t hash = hashFunc(tuple.get_key, seed);
        uint32_t prob_entry = align_down(hash % capacity, 2);
        uint32_t prob = 0;
        char result = NO_FREE;
        do {
            result = try_free_slot_warp(mask, tuple, prob_entry+prob);
            if(result == NO_FREE)
                prob = prob + __popc(mask)*2;
        } while(result!=FREE_SUCCESS && prob < capacity);
        return (result==FREE_SUCCESS);
    }

    template<typename X>
    __device__ T find_warp(const X& key )
    {
        unsigned mask = __activemask();
        uint32_t hash = hashFunc(key, seed);
        uint32_t prob_entry = align_down(hash%capacity, 2);
        uint32_t prob = 0;
        unsigned success = 0;
        T item;
        do {
            item = get_tuple(prob_entry+prob);
            unsigned short int flg = get_flag(prob_entry+prob);
            bool find_key  = item.get_key() == key;
            success = __ballot_sync(mask, find_key);
            if(success!=0)
                item = __shfl_sync(mask, __ffs(success)-1, item);
            else {
                unsigned free_mask = __ballot_sync(mask, flg==FREE);
                success = (free_mask!=0);
                prob = prob+__popc(mask);
            }
        }while(!success && prob < capacity);
        if(!success) item.value = 0;
        return item.value;
    }

    __host__ void release()
    {
        cudaFree(table);
        cudaFree(flags);
    }

    __host__ void print()
    {
        cout << "table size: "<< capacity << " seed: "<< seed << endl;
    }
};

#endif
