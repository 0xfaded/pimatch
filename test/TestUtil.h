#ifndef PIMATCH_TEST_UTIL_H__
#define PIMATCH_TEST_UTIL_H__

#include <cmath>
#include <random>

namespace test_util {

void fill_random(int vstep, int width, int height, uint8_t *buffer,
    std::mt19937 &rng);

} /* namespace test_util */

#endif /* PIMATCH_TEST_UTIL_H__ */
