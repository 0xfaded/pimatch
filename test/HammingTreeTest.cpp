#include <cmath>
#include <random>
#include <vector>
#include <stack>

#include "gtest/gtest.h"
#include "TestUtil.h"

#include "../include/HammingTree.h"

namespace {

using ::testing::Combine;
using ::testing::Range;
using ::testing::Values;

using Node = ::pimatch::HammingTree::Node;

//class HammingTreeTest: public ::testing::TestWithParam<::std::tuple<int, int>> {};

static uint32_t reference(std::vector<Node> tree,
    uint8_t *descriptors, uint8_t *needle);

TEST(HammingTreeTest, random) {
  size_t num_levels = 6;
  size_t num_queries = 1000;
  size_t num_children = 10;

  std::vector<Node> tree(1);

  std::stack<std::pair<size_t, size_t>> dfs;
  dfs.push(std::pair<size_t, size_t>(0, 0));

  size_t num_nodes = 0;
  while (!dfs.empty()) {
    size_t node_i = dfs.top().first;
    size_t level = dfs.top().second;

    dfs.pop();

    num_nodes += 1;

    tree[node_i].children = tree.size();

    if (level == num_levels) {
      tree[node_i].num_children = 0;
    } else {
      tree[node_i].num_children = num_children;

      for (int c = int(tree[node_i].num_children) - 1; c >= 0; c -= 1) {
        dfs.push(std::pair<size_t, size_t>(tree[node_i].children+c, level + 1));
        tree.emplace_back();
      }
    }
  }

  uint8_t *descriptors = new uint8_t[num_nodes*32];
  uint8_t *queries = new uint8_t[num_queries*32];
  uint32_t *matches = new uint32_t[num_queries];
  uint32_t *matches_reference = new uint32_t[num_queries];

  std::mt19937 rng;
  test_util::fill_random(32, 32, num_nodes, descriptors, rng);
  test_util::fill_random(32, 32, num_queries, queries, rng);

  // compute reference before passing tree to HammingTree
  for (size_t i = 0; i < num_queries; i += 1) {
    matches_reference[i] = reference(tree, descriptors, queries+i*32);
  }

  pimatch::HammingTree htree(tree, descriptors);
  htree.ApproxNN(matches, queries, num_queries);

  htree.BuildClassifier(2);

  for (size_t i = 0; i < num_queries; i += 1) {
    EXPECT_EQ(matches[i], matches_reference[i]);
  }

  delete[] queries;
  delete[] matches;
  delete[] matches_reference;
}


/*
INSTANTIATE_TEST_CASE_P(
    DimensionTest,
    HammingTreeTest,
    Combine(
      Values(1, 2, 3, 4, 5, 6, 7, 8, 9, 64, 101, 502, 1003),
      Values(1, 2, 3, 4, 5, 6, 7, 8, 9, 64, 101, 502, 1003)));
      */

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

uint32_t reference(std::vector<Node> tree,
    uint8_t *descriptors, uint8_t *needle) {

  Node *node = &tree[0];

  uint32_t best = 0xffffffff;
  while (node->num_children) {
    best = 0xffffffff;
    size_t end = node->children + node->num_children;
    for (size_t c = node->children; c < end; c += 1) {
      uint32_t dist = distance(needle, descriptors + c*32);
      uint32_t score = (dist << 24) | c;
      if (score < best) {
        best = score;
      }
    }
    node = &tree[best & 0xffffff];
  }
  return best;
}

} /* namespace */
