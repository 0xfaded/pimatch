#include <stdint.h>
#include <stddef.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stack>
#include <cstring>

#include "HammingTree.h"
#include "HammingMatch.h"

namespace pimatch {

HammingTree::HammingTree() : descriptors_(nullptr) {}

HammingTree::HammingTree(std::vector<HammingTree::Node> &tree,
    uint8_t *descriptors) : tree_(std::move(tree)),
    descriptors_(reinterpret_cast<uint8_t (*)[32]>(descriptors)) {}

HammingTree::~HammingTree() {
  delete[] reinterpret_cast<uint8_t *>(descriptors_);
}

void HammingTree::ApproxNN(uint32_t *matches, const uint8_t *needle,
    size_t num_indices) const {

  if (num_indices == 0) {
    return;
  }

  const Node &root = tree_[0];
  hammingMatch256ddd(matches, descriptors_[root.children], needle,
      root.num_children, num_indices, 1);

  size_t *indices2 = new size_t[num_indices];

  const size_t begin_child = root.children;
  const size_t end_child = begin_child + root.num_children;
  for (size_t c = begin_child; c < end_child; c += 1) {
    // aggregate indices assigned to this child
    size_t match_i = 0;
    for (size_t i = 0; i < num_indices; i += 1) {
      if (c == (matches[i] & 0xffffff)) {
        indices2[match_i] = i;
        match_i += 1;
      }
    }
    if (match_i != 0) {
      ApproxNNrec_(c, matches, indices2, needle, match_i);
    }
  }

  delete[] indices2;
}

void HammingTree::ApproxNNi(uint32_t *matches, size_t *indices,
    const uint8_t *needle, size_t num_indices) const {

  if (num_indices == 0) {
    return;
  }

  const Node &root = tree_[0];
  hammingMatch256ddi(matches, descriptors_[root.children], needle, indices,
      root.num_children, num_indices, 1);

  size_t *indices2 = new size_t[num_indices];

  const size_t begin_child = root.children;
  const size_t end_child = begin_child + root.num_children;
  for (size_t c = begin_child; c < end_child; c += 1) {
    // aggregate indices assigned to this child
    size_t match_i = 0;
    for (size_t i = 0; i < num_indices; i += 1) {
      if (c == (matches[i] & 0xffffff)) {
        indices2[match_i] = indices[i];
        match_i += 1;
      }
    }
    if (match_i != 0) {
      ApproxNNrec_(c, matches, indices2, needle, match_i);
    }
  }

  delete[] indices2;
}

HammingTree::Classifier HammingTree::BuildClassifier(size_t num_levels_down)
  const {

  std::stack<std::pair<size_t, size_t>> dfs;
  dfs.push(std::pair<size_t, size_t>(0, 0));

  Classifier classifier;

  // first boundary is implicitly zero
  bool first = true;
  while (!dfs.empty()) {
    size_t node_id = dfs.top().first;
    size_t depth = dfs.top().second;
    dfs.pop();

    if (depth == num_levels_down) {
      if (first) {
        first = false;
      } else {
        classifier.boundaries.push_back(tree_[node_id].children);
      }
    } else {
      int first_child = static_cast<int>(tree_[node_id].children);
      int last_child = first_child +
        static_cast<int>(tree_[node_id].num_children);

      // note[crc]: push children backwards so we traverse them off the stack
      // in the correct order.
      for (int c = last_child - 1; c >= first_child; c -= 1) {
        dfs.push(std::pair<size_t, size_t>(c, depth + 1));
      }
    }
  }

  return classifier;
}

void HammingTree::ApproxOneNN(size_t node_i, uint32_t *match,
    const uint8_t *needle) const {

  const Node &node = tree_[node_i];
  if (node.num_children == 0) {
    return;
  }

  // invert the problem, find the closest child (needle) to needle (haystack)
  uint32_t child_matches[node.num_children];
  hammingMatch256ddd(child_matches, needle, descriptors_[node.children],
      1, node.num_children);

  uint32_t best = 0xffffffff;
  uint32_t best_i = 0;
  for (size_t i = 0; i < node.num_children; i += 1) {
    if (child_matches[i] < best) {
      best = child_matches[i];
      best_i = i;
    }
  }

  best_i += node.children;
  best |= best_i;
  *match = best;

  ApproxOneNN(best_i, match, needle);
}

void HammingTree::ApproxNNrec_(size_t node_i, size_t *matches,
    size_t *indices, const uint8_t *needle, size_t num_indices) const {

  if (num_indices < 8) {
    for (size_t i = 0; i < num_indices; i += 1) {
      size_t index = indices[i];
      ApproxOneNN(node_i, &matches[index], &needle[index*32]);
    }
    return;
  }

  const Node *node = &tree_[node_i];

  // since an index can never appear in indices twice, and the original
  // allocator allocated num_indices, we are guaranteed enough space at
  // the end of indices for the next iterations indices.
  size_t *indices2 = indices + num_indices;

  uint8_t *descriptors = descriptors_[node->children];
  hammingMatch256idi(matches, descriptors, needle, indices,
      node->num_children, num_indices, node->children);

  const size_t begin_child = node->children;
  const size_t end_child = begin_child + node->num_children;

  for (size_t c = begin_child; c < end_child; c += 1) {
    // aggregate indices assigned to this child
    size_t match_i = 0;
    for (size_t i = 0; i < num_indices; i += 1) {
      if (c == (matches[indices[i]] & 0xffffff)) {
        indices2[match_i] = indices[i];
        match_i += 1;
      }
    }
    if (match_i != 0) {
      ApproxNNrec_(c, matches, indices2, needle, match_i);
    }
  }
}

bool HammingTree::Read(const std::string &filename) {
  std::ifstream f;
  f.open(filename, std::ios::in);

  if (!f.good()) {
    return false;
  }

  bool ok = Read(f);
  f.close();
  return ok;
}

bool HammingTree::Read(std::istream &in) {
  tree_.clear();
  weights_.clear();
  node_ids_.clear();

  delete[] reinterpret_cast<uint8_t *>(descriptors_);

  // dummy descriptor and weight for root node
  std::vector<uint8_t> descriptors(32);
  std::vector<float> weights(1);

  size_t node_id = 1;
  size_t parent_id;
  bool is_leaf;
  float weight;

  std::vector<std::vector<size_t>> children;

  std::string line;

  // skip header line
  std::getline(in, line);

  while (!in.eof()) {
    std::getline(in, line);

    std::stringstream ss;

    ss << line;
    ss >> parent_id >> is_leaf;

    if (in.eof() || ss.eof() || in.bad() || ss.bad()) {
      node_id -= 1;
      break;
    }

    if (parent_id >= children.size()) {
      children.resize(parent_id+1);
    }
    children[parent_id].push_back(node_id);

    descriptors.resize((node_id+1)*32);
    int bval = 0;
    for (size_t b = 0; b < 32; b += 1) {
      ss >> bval;
      descriptors[node_id*32+b] = bval;
    }

    ss >> weight;
    weights.push_back(weight);

    if (ss.bad()) {
      node_id -= 1;
      children[parent_id].resize(children[parent_id].size() - 1);
      descriptors.resize(descriptors.size() - 32);
      break;
    }

    node_id += 1;
  }

  if (node_id == 0) {
    return false;
  }

  descriptors_ = reinterpret_cast<uint8_t (*)[32]>(new uint8_t[node_id*32]);
  tree_.reserve(node_id);
  weights_.resize(node_id);

  // convert tree to sequential layout
  std::stack<std::pair<size_t, size_t>> dfs;

  dfs.push(std::pair<size_t, size_t>(0, 0));
  tree_.emplace_back();
  node_ids_.resize(node_id);

  while (!dfs.empty()) {
    std::pair<size_t, size_t> mapping = dfs.top();
    dfs.pop();

    size_t node_id = mapping.first;
    size_t index = mapping.second;

    tree_[index].children = tree_.size();
    tree_[index].num_children = children[node_id].size();
    node_ids_[index] = node_id;

    // move descriptor to correct location
    std::memcpy(descriptors_[index], &descriptors[node_id*32], 32);
    weights_[index] = weights[node_id];

    if (node_id < children.size()) {
      // push the indices in reverse order so that children are
      // processed in order.
      tree_.resize(tree_.size() + children[node_id].size());
      for (size_t c = 0; c < children[node_id].size(); c += 1) {
        size_t child_id = children[node_id].rbegin()[c];
        size_t child_index = tree_.size() - 1 - c;
        dfs.push(std::pair<size_t, size_t>(child_id, child_index));
      }
    }
  }

  return !tree_.empty();
}

bool HammingTree::ReadBinary(const std::string &filename) {
  std::ifstream f;
  f.open(filename, std::ios::in | std::ios::binary);

  std::cout << filename << " " << f.good() << std::endl;
  if (!f.good()) {
    return false;
  }


  bool ok = ReadBinary(f);
  f.close();
  return ok;
}

bool HammingTree::ReadBinary(std::istream &in) {
  tree_.clear();
  delete[] reinterpret_cast<uint8_t *>(descriptors_);

  size_t num_nodes = 0;
  in.read(reinterpret_cast<char *>(&num_nodes), sizeof(num_nodes));

  if (!in.good()) {
    return false;
  }

  tree_.resize(num_nodes);
  weights_.resize(num_nodes);
  node_ids_.resize(num_nodes);
  descriptors_ = reinterpret_cast<uint8_t (*)[32]>(new uint8_t[num_nodes*32]);

  in.read(reinterpret_cast<char *>(&tree_[0]), num_nodes*sizeof(Node));
  in.read(reinterpret_cast<char *>(descriptors_[0]), num_nodes*32);
  in.read(reinterpret_cast<char *>(&weights_[0]), num_nodes*sizeof(weights_[0]));
  in.read(reinterpret_cast<char *>(&node_ids_[0]), num_nodes*sizeof(node_ids_[0]));

  if (!in.good()) {
    tree_.clear();
    weights_.clear();
    node_ids_.clear();
    delete[] reinterpret_cast<uint8_t *>(descriptors_);
    descriptors_ = nullptr;
    return false;
  }
  return true;
}

bool HammingTree::WriteBinary(const std::string &filename) {
  std::ofstream f;
  f.open(filename, std::ios::out | std::ios::binary);

  if (!f.good()) {
    return false;
  }

  bool ok = WriteBinary(f);
  f.close();
  return ok;
}

bool HammingTree::WriteBinary(std::ostream &out) {
  size_t num_nodes = tree_.size();
  out.write(reinterpret_cast<char *>(&num_nodes), sizeof(num_nodes));
  out.write(reinterpret_cast<char *>(&tree_[0]), num_nodes*sizeof(Node));
  out.write(reinterpret_cast<char *>(descriptors_[0]), num_nodes*32);
  out.write(reinterpret_cast<char *>(&weights_[0]), num_nodes*sizeof(weights_[0]));
  out.write(reinterpret_cast<char *>(&node_ids_[0]), num_nodes*sizeof(node_ids_[0]));

  return out.good();
}

} /* namespace */
