#include <iostream>

using namespace std;

template<typename Key, typename Value>
struct MyTuple {
    Key key;
    Value value;

    MyTuple(Key _key, Value _value): key(_key), value(_value) {}

    __device__ __host__ MyTuple& operator= (const MyTuple & other_tuple)
    {
        key = other_tuple.key;
        value = other_tuple.value;
        return *this;
    }
};

template<typename Key, typename Value>
ostream &operator<<(ostream &os, MyTuple<Key, Value> const & tuple)
{
    os << "(" << tuple.key << " , " << tuple.value << ")";
    return os;
}


