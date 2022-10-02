//
// Created by eliseysudakov on 10/2/22.
//

#include "random_utils.h"

#include <cstdlib>
#include <cassert>

// rand generates random number between 0 and RAND_MAX inclusive

void fix_random_seed(int random_seed) {
    srand(random_seed);
}

int randint(int l, int r) {
    // random int between l inclusive and r exclusive
    assert(r > l);
    long long big_rand = static_cast<long long>(rand()) * (RAND_MAX - 1) + rand();
    return static_cast<int>(l + big_rand % (r - l));
}

float random_float() {
    // random float 0 to 1
    float big_rand = static_cast<float>(rand()) * (RAND_MAX - 1) + rand();
    float result = big_rand / (static_cast<float>(RAND_MAX) * RAND_MAX);
    assert(0 <= result && result <= 1);
    return result;
}

