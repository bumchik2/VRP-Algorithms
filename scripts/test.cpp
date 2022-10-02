#include <iostream>

using std::cout;
using std::endl;


class A {
public:
    void f() {
        g();
    }

    virtual void g() {
        cout << "A";
    }
};

class B : public A {
public:
    void g() override {
        cout << "B";
    }
};


void test() {
    B b;
    b.f();
}


int main() {
    test();
    return 0;
}
