#include <random>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <iostream>

#include "../include/HammingMatch.h"

constexpr int N = 100;

void print_delta(std::clock_t begin, std::clock_t end, const char *t) {
  std::cout << static_cast<double>(end - begin) / CLOCKS_PER_SEC * 1000 / N << t;
}

int main(int argc, char **argv) {
  size_t max_size = 2000;

  uint8_t *haystack = new uint8_t[max_size*32];
  uint8_t *needle = new uint8_t[max_size*32];
  size_t *haystack_indices = new size_t[max_size];
  size_t *needle_indices = new size_t[max_size];
  uint32_t *matches = new uint32_t[max_size*2];

  std::mt19937 rng;
  for (size_t i = 0; i < max_size*32; i += 1) {
    haystack[i] = rng();
    needle[i] = rng();
  }

  for (size_t i = 0; i < max_size; i += 1) {
    haystack_indices[i] = i;
    needle_indices[i] = i;
  }
  std::shuffle(haystack_indices, haystack_indices+max_size, rng);
  std::shuffle(needle_indices, needle_indices+max_size, rng);

  std::clock_t begin, end;

  std::cout << "N ddd ddi did idi dii ddd2 ddi2 did2 idi2 dii2" << std::endl;
  for (size_t i = 250; i <= max_size; i += 250) {
    std::cout << i << " ";

    begin = std::clock();
    for (int j = 0; j < N; j += 1) {
      pimatch::hammingMatch256ddd(matches, haystack, needle, i, i);
    }
    end = std::clock();
    print_delta(begin, end, " ");

    begin = std::clock();
    for (int j = 0; j < N; j += 1) {
      pimatch::hammingMatch256ddi(matches, haystack, needle,
          needle_indices, i, i);
    }
    end = std::clock();
    print_delta(begin, end, " ");

    begin = std::clock();
    for (int j = 0; j < N; j += 1) {
      pimatch::hammingMatch256did(matches, haystack, needle,
          haystack_indices, i, i);
    }
    end = std::clock();
    print_delta(begin, end, " ");

    begin = std::clock();
    for (int j = 0; j < N; j += 1) {
      pimatch::hammingMatch256idi(matches, haystack, needle,
          needle_indices, i, i);
    }
    end = std::clock();
    print_delta(begin, end, " ");

    begin = std::clock();
    for (int j = 0; j < N; j += 1) {
      pimatch::hammingMatch256dii(matches, haystack, needle,
          needle_indices, haystack_indices, i, i);
    }
    end = std::clock();
    print_delta(begin, end, " ");

    begin = std::clock();
    for (int j = 0; j < N; j += 1) {
      pimatch::hammingMatch256ddd2(matches, haystack, needle, i, i);
    }
    end = std::clock();
    print_delta(begin, end, " ");

    begin = std::clock();
    for (int j = 0; j < N; j += 1) {
      pimatch::hammingMatch256ddi2(matches, haystack, needle,
          needle_indices, i, i);
    }
    end = std::clock();
    print_delta(begin, end, " ");

    begin = std::clock();
    for (int j = 0; j < N; j += 1) {
      pimatch::hammingMatch256did2(matches, haystack, needle,
          haystack_indices, i, i);
    }
    end = std::clock();
    print_delta(begin, end, " ");

    begin = std::clock();
    for (int j = 0; j < N; j += 1) {
      pimatch::hammingMatch256idi2(matches, haystack, needle,
          needle_indices, i, i);
    }
    end = std::clock();
    print_delta(begin, end, " ");

    begin = std::clock();
    for (int j = 0; j < N; j += 1) {
      pimatch::hammingMatch256dii2(matches, haystack, needle,
          needle_indices, haystack_indices, i, i);
    }
    end = std::clock();
    print_delta(begin, end, "\n");
  }

  delete[] haystack;
  delete[] needle;
  delete[] matches;
  delete[] haystack_indices;
  delete[] needle_indices;
}
