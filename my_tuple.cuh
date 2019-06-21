#include <iostream>

using namespace std;

//each tuple is 64 bits, take this as a constrain
template<typename Key, typename Value>
struct MyTuple {
    Key key;
    Value value;

    __device__ __host__ MyTuple() {}
    __device__ __host__ MyTuple(Key _key, Value _value): key(_key), value(_value) {}

    __device__ __host__ MyTuple& operator= (const MyTuple & other_tuple)
    {
        key = other_tuple.key;
        value = other_tuple.value;
        return *this;
    }

    __device__ __host__ Key get_key()
    {
        return key;
    }

    __device__ __host__ Value get_value()
    {
        return value;
    }
}; //__attribute__((aligned (8)));

template<typename Key, typename Value>
ostream &operator<<(ostream &os, MyTuple<Key, Value> const & tuple)
{
    os << "(" << tuple.key << " , " << tuple.value << ")";
    return os;
}


