#include <cmath>
#include <random>

#include "gtest/gtest.h"
#include "TestUtil.h"
#include "../include/HammingMatch.h"

namespace {

using ::testing::Combine;
using ::testing::Range;
using ::testing::Values;

#define RHADD(a, b) ((a >> 1) + (b >> 1) + ((a|b)&1))

class HammingMatchTest: public ::testing::TestWithParam<::std::tuple<int, int>> {};

void referenceDDD(uint32_t *matches, uint8_t *haystack,
    uint8_t *needle, size_t num_haystack,
    size_t num_needle, size_t haystack_base);

void referenceDDI(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base);

void referenceDID(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base);

void referenceIDI(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base);

void referenceDII(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *haystack_indices, size_t *needle_indices,
    size_t num_haystack, size_t num_needle, size_t haystack_base);

void referenceDDD2(uint32_t *matches, uint8_t *haystack,
    uint8_t *needle, size_t num_haystack,
    size_t num_needle, size_t haystack_base);

void referenceDDI2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base);

void referenceDID2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base);

void referenceIDI2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base);

void referenceDII2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *haystack_indices, size_t *needle_indices,
    size_t num_haystack, size_t num_needle, size_t haystack_base);

TEST_P(HammingMatchTest, randomMatch256ddd) {
  size_t num_descriptors = ::std::get<0>(GetParam());
  size_t num_queries = ::std::get<1>(GetParam());

  // use a > 16 bit number to test 24 bit indices
  size_t haystack_base = 0x12345;

  uint8_t *haystack = new uint8_t[num_descriptors*32];
  uint8_t *needle = new uint8_t[num_queries*32];
  uint32_t *matches = new size_t[num_queries+1];
  uint32_t *matches_reference = new size_t[num_queries];

  matches[num_queries] = 0xdeadbeef;

  std::mt19937 rng;
  test_util::fill_random(32, 32, num_descriptors, haystack, rng);
  test_util::fill_random(32, 32, num_queries, needle, rng);

  pimatch::hammingMatch256ddd(matches, haystack, needle,
      num_descriptors, num_queries, haystack_base);

  referenceDDD(matches_reference, haystack, needle,
      num_descriptors, num_queries, haystack_base);

  for (size_t i = 0; i < num_queries; i += 1) {
    EXPECT_EQ(matches[i], matches_reference[i]);
  }

  // check for overrun
  EXPECT_EQ(matches[num_queries], 0xdeadbeef);

  delete[] haystack;
  delete[] needle;
  delete[] matches;
  delete[] matches_reference;
}

TEST_P(HammingMatchTest, randomMatch256ddi) {
  size_t num_descriptors = ::std::get<0>(GetParam());
  size_t num_queries = ::std::get<1>(GetParam());

  // use a > 16 bit number to test 24 bit indices
  size_t haystack_base = 0x12345;

  uint8_t *haystack = new uint8_t[num_descriptors*32];
  uint8_t *needle = new uint8_t[num_queries*32];
  size_t *indices = new size_t[num_queries];
  uint32_t *matches = new size_t[num_queries+1];
  uint32_t *matches_reference = new size_t[num_queries];

  matches[num_queries] = 0xdeadbeef;

  std::mt19937 rng;
  test_util::fill_random(32, 32, num_descriptors, haystack, rng);
  test_util::fill_random(32, 32, num_queries, needle, rng);

  for (size_t i = 0; i < num_queries; i += 1) {
    indices[i] = i;
  }
  std::shuffle(indices, indices+num_queries, rng);

  pimatch::hammingMatch256ddi(matches, haystack, needle, indices,
      num_descriptors, num_queries, haystack_base);

  referenceDDI(matches_reference, haystack, needle, indices,
      num_descriptors, num_queries, haystack_base);

  for (size_t i = 0; i < num_queries; i += 1) {
    EXPECT_EQ(matches[i], matches_reference[i]);
  }

  // check for overrun
  EXPECT_EQ(matches[num_queries], 0xdeadbeef);

  delete[] haystack;
  delete[] needle;
  delete[] indices;
  delete[] matches;
  delete[] matches_reference;
}

TEST_P(HammingMatchTest, randomMatch256did) {
  size_t num_descriptors = ::std::get<0>(GetParam());
  size_t num_queries = ::std::get<1>(GetParam());

  // use a > 16 bit number to test 24 bit indices
  size_t haystack_base = 0x12345;

  uint8_t *haystack = new uint8_t[num_descriptors*32];
  uint8_t *needle = new uint8_t[num_queries*32];
  size_t *indices = new size_t[num_descriptors];
  uint32_t *matches = new size_t[num_queries+1];
  uint32_t *matches_reference = new size_t[num_queries];

  matches[num_queries] = 0xdeadbeef;

  std::mt19937 rng;
  test_util::fill_random(32, 32, num_descriptors, haystack, rng);
  test_util::fill_random(32, 32, num_queries, needle, rng);

  for (size_t i = 0; i < num_descriptors; i += 1) {
    indices[i] = i;
  }
  std::shuffle(indices, indices+num_descriptors, rng);

  pimatch::hammingMatch256did(matches, haystack, needle, indices,
      num_descriptors, num_queries, haystack_base);

  referenceDID(matches_reference, haystack, needle, indices,
      num_descriptors, num_queries, haystack_base);

  for (size_t i = 0; i < num_queries; i += 1) {
    EXPECT_EQ(matches[i], matches_reference[i]);
  }

  // check for overrun
  EXPECT_EQ(matches[num_queries], 0xdeadbeef);

  delete[] haystack;
  delete[] needle;
  delete[] indices;
  delete[] matches;
  delete[] matches_reference;
}

TEST_P(HammingMatchTest, randomMatch256idi) {
  size_t num_descriptors = ::std::get<0>(GetParam());
  size_t num_queries = ::std::get<1>(GetParam());

  // use a > 16 bit number to test 24 bit indices
  size_t haystack_base = 0x12345;

  uint8_t *haystack = new uint8_t[num_descriptors*32];
  uint8_t *needle = new uint8_t[num_queries*32];
  size_t *indices = new size_t[num_queries];
  uint32_t *matches = new size_t[num_queries+1];
  uint32_t *matches_reference = new size_t[num_queries];

  matches[num_queries] = 0xdeadbeef;

  std::mt19937 rng;
  test_util::fill_random(32, 32, num_descriptors, haystack, rng);
  test_util::fill_random(32, 32, num_queries, needle, rng);

  for (size_t i = 0; i < num_queries; i += 1) {
    indices[i] = i;
  }
  std::shuffle(indices, indices+num_queries, rng);

  pimatch::hammingMatch256idi(matches, haystack, needle, indices,
      num_descriptors, num_queries, haystack_base);

  referenceIDI(matches_reference, haystack, needle, indices,
      num_descriptors, num_queries, haystack_base);

  for (size_t i = 0; i < num_queries; i += 1) {
    EXPECT_EQ(matches[i], matches_reference[i]);
  }

  // check for overrun
  EXPECT_EQ(matches[num_queries], 0xdeadbeef);

  delete[] haystack;
  delete[] needle;
  delete[] indices;
  delete[] matches;
  delete[] matches_reference;
}

TEST_P(HammingMatchTest, randomMatch256dii) {
  size_t num_descriptors = ::std::get<0>(GetParam());
  size_t num_queries = ::std::get<1>(GetParam());

  // use a > 16 bit number to test 24 bit indices
  size_t haystack_base = 0x12345;

  uint8_t *haystack = new uint8_t[num_descriptors*32];
  uint8_t *needle = new uint8_t[num_queries*32];
  size_t *needle_indices = new size_t[num_queries];
  size_t *haystack_indices = new size_t[num_descriptors];
  uint32_t *matches = new size_t[num_queries+1];
  uint32_t *matches_reference = new size_t[num_queries];

  matches[num_queries] = 0xdeadbeef;

  std::mt19937 rng;
  test_util::fill_random(32, 32, num_descriptors, haystack, rng);
  test_util::fill_random(32, 32, num_queries, needle, rng);

  for (size_t i = 0; i < num_queries; i += 1) {
    needle_indices[i] = i;
  }

  for (size_t i = 0; i < num_descriptors; i += 1) {
    haystack_indices[i] = i;
  }

  std::shuffle(needle_indices, needle_indices+num_queries, rng);
  std::shuffle(haystack_indices, haystack_indices+num_descriptors, rng);

  pimatch::hammingMatch256dii(matches, haystack, needle,
      haystack_indices, needle_indices,
      num_descriptors, num_queries, haystack_base);

  referenceDII(matches_reference, haystack, needle,
      haystack_indices, needle_indices,
      num_descriptors, num_queries, haystack_base);

  for (size_t i = 0; i < num_queries; i += 1) {
    EXPECT_EQ(matches[i], matches_reference[i]);
  }

  // check for overrun
  EXPECT_EQ(matches[num_queries], 0xdeadbeef);

  delete[] haystack;
  delete[] needle;
  delete[] needle_indices;
  delete[] haystack_indices;
  delete[] matches;
  delete[] matches_reference;
}

TEST_P(HammingMatchTest, randomMatch256ddd2) {
  size_t num_descriptors = ::std::get<0>(GetParam());
  size_t num_queries = ::std::get<1>(GetParam());

  // use a > 16 bit number to test 24 bit indices
  size_t haystack_base = 0x12345;

  uint8_t *haystack = new uint8_t[num_descriptors*32];
  uint8_t *needle = new uint8_t[num_queries*32];
  uint32_t *matches = new size_t[num_queries*2+1];
  uint32_t *matches_reference = new size_t[num_queries*2];

  matches[num_queries*2] = 0xdeadbeef;

  std::mt19937 rng;
  test_util::fill_random(32, 32, num_descriptors, haystack, rng);
  test_util::fill_random(32, 32, num_queries, needle, rng);

  pimatch::hammingMatch256ddd2(matches, haystack, needle,
      num_descriptors, num_queries, haystack_base);

  referenceDDD2(matches_reference, haystack, needle,
      num_descriptors, num_queries, haystack_base);

  for (size_t i = 0; i < num_queries*2; i += 1) {
    EXPECT_EQ(matches[i], matches_reference[i]);
  }

  // check for overrun
  EXPECT_EQ(matches[num_queries*2], 0xdeadbeef);

  delete[] haystack;
  delete[] needle;
  delete[] matches;
  delete[] matches_reference;
}

TEST_P(HammingMatchTest, randomMatch256ddi2) {
  size_t num_descriptors = ::std::get<0>(GetParam());
  size_t num_queries = ::std::get<1>(GetParam());

  // use a > 16 bit number to test 24 bit indices
  size_t haystack_base = 0x12345;

  uint8_t *haystack = new uint8_t[num_descriptors*32];
  uint8_t *needle = new uint8_t[num_queries*32];
  size_t *indices = new size_t[num_queries];
  uint32_t *matches = new size_t[num_queries*2+1];
  uint32_t *matches_reference = new size_t[num_queries*2];

  matches[num_queries*2] = 0xdeadbeef;

  std::mt19937 rng;
  test_util::fill_random(32, 32, num_descriptors, haystack, rng);
  test_util::fill_random(32, 32, num_queries, needle, rng);

  for (size_t i = 0; i < num_queries; i += 1) {
    indices[i] = i;
  }
  std::shuffle(indices, indices+num_queries, rng);

  pimatch::hammingMatch256ddi2(matches, haystack, needle, indices,
      num_descriptors, num_queries, haystack_base);

  referenceDDI2(matches_reference, haystack, needle, indices,
      num_descriptors, num_queries, haystack_base);

  for (size_t i = 0; i < num_queries*2; i += 1) {
    EXPECT_EQ(matches[i], matches_reference[i]);
  }

  // check for overrun
  EXPECT_EQ(matches[num_queries*2], 0xdeadbeef);

  delete[] haystack;
  delete[] needle;
  delete[] indices;
  delete[] matches;
  delete[] matches_reference;
}


TEST_P(HammingMatchTest, randomMatch256idi2) {
  size_t num_descriptors = ::std::get<0>(GetParam());
  size_t num_queries = ::std::get<1>(GetParam());

  // use a > 16 bit number to test 24 bit indices
  size_t haystack_base = 0x12345;

  uint8_t *haystack = new uint8_t[num_descriptors*32];
  uint8_t *needle = new uint8_t[num_queries*32];
  size_t *indices = new size_t[num_queries];
  uint32_t *matches = new size_t[num_queries*2+1];
  uint32_t *matches_reference = new size_t[num_queries*2];

  matches[num_queries*2] = 0xdeadbeef;

  std::mt19937 rng;
  test_util::fill_random(32, 32, num_descriptors, haystack, rng);
  test_util::fill_random(32, 32, num_queries, needle, rng);

  for (size_t i = 0; i < num_queries; i += 1) {
    indices[i] = i;
  }
  std::shuffle(indices, indices+num_queries, rng);

  pimatch::hammingMatch256idi2(matches, haystack, needle, indices,
      num_descriptors, num_queries, haystack_base);

  referenceIDI2(matches_reference, haystack, needle, indices,
      num_descriptors, num_queries, haystack_base);

  for (size_t i = 0; i < num_queries*2; i += 1) {
    EXPECT_EQ(matches[i], matches_reference[i]);
  }

  // check for overrun
  EXPECT_EQ(matches[num_queries*2], 0xdeadbeef);

  delete[] haystack;
  delete[] needle;
  delete[] indices;
  delete[] matches;
  delete[] matches_reference;
}

TEST_P(HammingMatchTest, randomMatch256dii2) {
  size_t num_descriptors = ::std::get<0>(GetParam());
  size_t num_queries = ::std::get<1>(GetParam());

  // use a > 16 bit number to test 24 bit indices
  size_t haystack_base = 0x12345;

  uint8_t *haystack = new uint8_t[num_descriptors*32];
  uint8_t *needle = new uint8_t[num_queries*32];
  size_t *needle_indices = new size_t[num_queries];
  size_t *haystack_indices = new size_t[num_descriptors];
  uint32_t *matches = new size_t[num_queries*2+1];
  uint32_t *matches_reference = new size_t[num_queries*2];

  matches[num_queries*2] = 0xdeadbeef;

  std::mt19937 rng;
  test_util::fill_random(32, 32, num_descriptors, haystack, rng);
  test_util::fill_random(32, 32, num_queries, needle, rng);

  for (size_t i = 0; i < num_queries; i += 1) {
    needle_indices[i] = i;
  }

  for (size_t i = 0; i < num_descriptors; i += 1) {
    haystack_indices[i] = i;
  }

  std::shuffle(needle_indices, needle_indices+num_queries, rng);
  std::shuffle(haystack_indices, haystack_indices+num_descriptors, rng);

  pimatch::hammingMatch256dii2(matches, haystack, needle,
      haystack_indices, needle_indices,
      num_descriptors, num_queries, haystack_base);

  referenceDII2(matches_reference, haystack, needle,
      haystack_indices, needle_indices,
      num_descriptors, num_queries, haystack_base);

  for (size_t i = 0; i < num_queries*2; i += 1) {
    EXPECT_EQ(matches[i], matches_reference[i]);
  }

  // check for overrun
  EXPECT_EQ(matches[num_queries*2], 0xdeadbeef);

  delete[] haystack;
  delete[] needle;
  delete[] needle_indices;
  delete[] haystack_indices;
  delete[] matches;
  delete[] matches_reference;
}

INSTANTIATE_TEST_CASE_P(
    DimensionTest,
    HammingMatchTest,
    Combine(
      Values(1, 2, 3, 4, 5, 6, 7, 8, 9, 64, 101, 502, 1003),
      Values(1, 2, 3, 4, 5, 6, 7, 8, 9, 64, 101, 502, 1003)));

static uint32_t distance(uint8_t *a, uint8_t *b) {
  uint32_t *aa = reinterpret_cast<uint32_t *>(a);
  uint32_t *bb = reinterpret_cast<uint32_t *>(b);

  uint32_t distance = 0;
  for (size_t i = 0; i < 8; i += 1) {
    uint32_t c;
    uint32_t v = aa[i] ^ bb[i];

    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    c = (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;

    distance += c;
  }

  if (distance >= 256) {
    distance = 255;
  }

  return distance;
}

TEST_P(HammingMatchTest, randomMatch256did2) {
  size_t num_descriptors = ::std::get<0>(GetParam());
  size_t num_queries = ::std::get<1>(GetParam());

  // use a > 16 bit number to test 24 bit indices
  size_t haystack_base = 0x12345;

  uint8_t *haystack = new uint8_t[num_descriptors*32];
  uint8_t *needle = new uint8_t[num_queries*32];
  size_t *indices = new size_t[num_descriptors];
  uint32_t *matches = new size_t[num_queries*2+1];
  uint32_t *matches_reference = new size_t[num_queries*2];

  matches[num_queries*2] = 0xdeadbeef;

  std::mt19937 rng;
  test_util::fill_random(32, 32, num_descriptors, haystack, rng);
  test_util::fill_random(32, 32, num_queries, needle, rng);

  for (size_t i = 0; i < num_descriptors; i += 1) {
    indices[i] = i;
  }
  std::shuffle(indices, indices+num_descriptors, rng);

  pimatch::hammingMatch256did2(matches, haystack, needle, indices,
      num_descriptors, num_queries, haystack_base);

  referenceDID2(matches_reference, haystack, needle, indices,
      num_descriptors, num_queries, haystack_base);

  for (size_t i = 0; i < num_queries*2; i += 1) {
    EXPECT_EQ(matches[i], matches_reference[i]);
  }

  // check for overrun
  EXPECT_EQ(matches[num_queries*2], 0xdeadbeef);

  delete[] haystack;
  delete[] needle;
  delete[] indices;
  delete[] matches;
  delete[] matches_reference;
}

void referenceDDD(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t num_haystack, size_t num_indices, size_t haystack_base) {

  for (size_t i = 0; i < num_indices; i += 1) {
    uint32_t best = 0xffffffff;
    for (size_t j = 0; j < num_haystack; j += 1) {
      uint32_t score = distance(&haystack[j*32], &needle[i*32]);
      score = (score << 24) | (haystack_base + j);

      if (score < best) {
        best = score;
      }
    }
    matches[i] = best;
  }
}

void referenceDDI(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *indices, size_t num_haystack, size_t num_indices,
    size_t haystack_base) {

  for (size_t i = 0; i < num_indices; i += 1) {
    size_t index = indices[i];

    uint32_t best = 0xffffffff;
    for (size_t j = 0; j < num_haystack; j += 1) {
      uint32_t score = distance(&haystack[j*32], &needle[index*32]);
      score = (score << 24) | (haystack_base + j);

      if (score < best) {
        best = score;
      }
    }
    matches[i] = best;
  }
}

void referenceDID(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *indices, size_t num_haystack, size_t num_indices,
    size_t haystack_base) {

  for (size_t i = 0; i < num_indices; i += 1) {
    uint32_t best = 0xffffffff;
    for (size_t j = 0; j < num_haystack; j += 1) {
      size_t index = indices[j];

      uint32_t score = distance(&haystack[index*32], &needle[i*32]);
      score = (score << 24) | (haystack_base + index);

      if (score < best) {
        best = score;
      }
    }
    matches[i] = best;
  }
}

void referenceIDI(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *indices, size_t num_haystack, size_t num_indices,
    size_t haystack_base) {

  for (size_t i = 0; i < num_indices; i += 1) {
    size_t index = indices[i];

    uint32_t best = 0xffffffff;
    for (size_t j = 0; j < num_haystack; j += 1) {
      uint32_t score = distance(&haystack[j*32], &needle[index*32]);
      score = (score << 24) | (haystack_base + j);

      if (score < best) {
        best = score;
      }
    }
    matches[index] = best;
  }
}

void referenceDII(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *haystack_indices, size_t *needle_indices,
    size_t num_haystack, size_t num_needle, size_t haystack_base) {

  for (size_t i = 0; i < num_needle; i += 1) {
    size_t n = needle_indices[i];

    uint32_t best = 0xffffffff;
    for (size_t j = 0; j < num_haystack; j += 1) {
      size_t h = haystack_indices[j];

      uint32_t score = distance(&haystack[h*32], &needle[n*32]);
      score = (score << 24) | (haystack_base + h);

      if (score < best) {
        best = score;
      }
    }
    matches[i] = best;
  }
}

void referenceDDD2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t num_haystack, size_t num_indices, size_t haystack_base) {

  for (size_t i = 0; i < num_indices; i += 1) {
    uint32_t best = 0xffffffff;
    uint32_t best2 = 0xffffffff;
    for (size_t j = 0; j < num_haystack; j += 1) {
      uint32_t score = distance(&haystack[j*32], &needle[i*32]);
      score = (score << 24) | (haystack_base + j);

      if (score < best) {
        best2 = best;
        best = score;
      } else if (score < best2) {
        best2 = score;
      }
    }
    matches[2*i] = best;
    matches[2*i+1] = best2;
  }
}

void referenceDDI2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *indices, size_t num_haystack, size_t num_indices,
    size_t haystack_base) {

  for (size_t i = 0; i < num_indices; i += 1) {
    size_t index = indices[i];

    uint32_t best = 0xffffffff;
    uint32_t best2 = 0xffffffff;
    for (size_t j = 0; j < num_haystack; j += 1) {
      uint32_t score = distance(&haystack[j*32], &needle[index*32]);
      score = (score << 24) | (haystack_base + j);

      if (score < best) {
        best2 = best;
        best = score;
      } else if (score < best2) {
        best2 = score;
      }
    }
    matches[2*i] = best;
    matches[2*i+1] = best2;
  }
}

void referenceIDI2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *indices, size_t num_haystack, size_t num_indices,
    size_t haystack_base) {

  for (size_t i = 0; i < num_indices; i += 1) {
    size_t index = indices[i];

    uint32_t best = 0xffffffff;
    uint32_t best2 = 0xffffffff;
    for (size_t j = 0; j < num_haystack; j += 1) {
      uint32_t score = distance(&haystack[j*32], &needle[index*32]);
      score = (score << 24) | (haystack_base + j);

      if (score < best) {
        best2 = best;
        best = score;
      } else if (score < best2) {
        best2 = score;
      }
    }
    matches[2*index] = best;
    matches[2*index+1] = best2;
  }
}

void referenceDII2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *haystack_indices, size_t *needle_indices,
    size_t num_haystack, size_t num_needle, size_t haystack_base) {

  for (size_t i = 0; i < num_needle; i += 1) {
    size_t n = needle_indices[i];

    uint32_t best = 0xffffffff;
    uint32_t best2 = 0xffffffff;
    for (size_t j = 0; j < num_haystack; j += 1) {
      size_t h = haystack_indices[j];

      uint32_t score = distance(&haystack[h*32], &needle[n*32]);
      score = (score << 24) | (haystack_base + h);

      if (score < best) {
        best2 = best;
        best = score;
      } else if (score < best2) {
        best2 = score;
      }
    }
    matches[2*i] = best;
    matches[2*i+1] = best2;
  }
}

void referenceDID2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *indices, size_t num_haystack, size_t num_indices,
    size_t haystack_base) {

  for (size_t i = 0; i < num_indices; i += 1) {
    uint32_t best = 0xffffffff;
    uint32_t best2 = 0xffffffff;
    for (size_t j = 0; j < num_haystack; j += 1) {
      size_t index = indices[j];

      uint32_t score = distance(&haystack[index*32], &needle[i*32]);
      score = (score << 24) | (haystack_base + index);

      if (score < best) {
        best2 = best;
        best = score;
      } else if (score < best2) {
        best2 = score;
      }
    }
    matches[2*i] = best;
    matches[2*i+1] = best2;
  }
}


} /* namespace */
