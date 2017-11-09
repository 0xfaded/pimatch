#ifndef PIMATCH_HAMMING_TREE_H__
#define PIMATCH_HAMMING_TREE_H__

#include <stdint.h>
#include <stddef.h>
#include <iostream>
#include <algorithm>
#include <vector>

namespace pimatch {

/// k-means tree for approximate NN classification.
class HammingTree {
 public:
  struct Node;
  struct Classifier;

  HammingTree();
  HammingTree(std::vector<Node> &tree, uint8_t *descriptors);

  ~HammingTree();

  void ApproxNN(uint32_t *matches, const uint8_t *needle,
      size_t num_indices) const;
  void ApproxNNi(uint32_t *matches, size_t *indices, const uint8_t *needle,
      size_t num_indices) const;

  void ApproxOneNN(size_t node_i, uint32_t *match, const uint8_t *needle) const;

  float GetWeight(size_t node_i) const {
    return weights_[node_i];
  }

  bool IsStopWord(size_t node_i) const {
    return weights_[node_i] == 0;
  }

  /// Only used for DBoW compatibility. HammingTree uses node index
  /// in tree_ as an identifier.
  float GetNodeId(size_t node_i) const {
    return node_ids_[node_i];
  }

  size_t Size() const {
    return tree_.size();
  }

  /// Create a classifier which associates a classified leaf node index
  /// with its ancestor node num_levels_down from the root tree.
  /// This is useful for higher level clustering.
  Classifier BuildClassifier(size_t num_levels_down) const;

  bool Read(const std::string &filename);
  bool ReadBinary(const std::string &filename);
  bool WriteBinary(const std::string &filename);

  bool Read(std::istream &in);
  bool ReadBinary(std::istream &in);
  bool WriteBinary(std::ostream &out);

  struct Node {
    size_t children;
    size_t num_children;
  };

  struct Classifier {
    size_t Classify(size_t nn) {
      std::vector<size_t>::iterator low = std::lower_bound(
          boundaries.begin(), boundaries.end(), nn);

      return std::distance(boundaries.begin(), low);
    }

    std::vector<size_t> boundaries;
  };

 protected:
  void ApproxNNrec_(size_t node_i, uint32_t *matches, size_t *indices,
      const uint8_t *needle, size_t num_indices) const;

  // The entire tree, laid out in DFS order. This makes Classifier
  // very simple to implement.
  std::vector<Node> tree_;
  std::vector<float> weights_;
  std::vector<size_t> node_ids_;
  uint8_t (*descriptors_)[32];
};

} /* namespace */

#endif /* PIMATCH_HAMMING_TREE_H__ */
