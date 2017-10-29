#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

#include "TestUtil.h"

namespace test_util {

void fill_random(int vstep, int width, int height, uint8_t *buffer,
    std::mt19937 &rng) {

  for (int i = 0; i < height; i += 1) {
    for (int j = 0; j < width; j += 1) {
      buffer[i*vstep+j] = rng();
    }
  }
}
} /* namespace test_util */
