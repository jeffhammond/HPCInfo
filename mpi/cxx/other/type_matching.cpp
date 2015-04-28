#include <iostream>
#include <climits>
#include <cstdint>

class Foo
{
    public:
        //Foo(int i) { std::cout << "Foo(I=" << i << ")" << std::endl; }
        Foo(long i) { std::cout << "Foo(L=" << i << ")" << std::endl; }
        //Foo(long long i) { std::cout << "Foo(LL=" << i << ")" << std::endl; }
        Foo(unsigned int i) { std::cout << "Foo(UI=" << i << ")" << std::endl; }
        //Foo(unsigned long i) { std::cout << "Foo(UL=" << i << ")" << std::endl; }
        //Foo(unsigned long long i) { std::cout << "Foo(ULL=" << i << ")" << std::endl; }
};

int main(void)
{
    char     c = 7;
    short    s = 37;
    int      i = INT_MIN;
    unsigned u = UINT_MAX;
    Foo C(c);
    Foo S(s);
    Foo I(i);
    Foo U(u);
    return 0;
}
