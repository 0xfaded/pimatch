#include <iostream>
#include <fstream>
#include <random>
#include <set>
#include <vector>

#include <ctime>
#include <chrono>

#include <HammingTree.h>

#include "./DBoW2/DBoW2/FORB.h"
#include "./DBoW2/DBoW2/TemplatedVocabulary.h"

#include <opencv2/core/core.hpp>

const int measurements = 100;

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
  ORBVocabulary;

pimatch::HammingTree *doPiMatch(uint8_t *queries, size_t N, char *fname,
  DBoW2::BowVector &bow, DBoW2::FeatureVector &feat) {

  std::ifstream in;
  in.open(fname, std::ios::in);

  pimatch::HammingTree *tree = new pimatch::HammingTree;
  tree->Read(in);

  uint32_t *matches = new uint32_t[N];

  pimatch::HammingTree::Classifier classifier = tree->BuildClassifier(2);

  std::clock_t begin = std::clock();
  for (int m = 0; m < measurements; m += 1) {
    bow.clear();
    feat.clear();

    tree->ApproxNN(matches, queries, N);

    for (size_t i = 0; i < N; i += 1) {
      uint32_t match = matches[i] & 0xffffff;
      float weight = tree->GetWeight(match);

      // DBoW treats 0 as stop word
      if (weight > 0.f) {
        size_t classified = classifier.Classify(match);
        bow.addWeight(match, weight);
        feat.addFeature(classified, i);
      }
    }
  }
  std::clock_t end = std::clock();

  std::cout << "pimatch time: " << (end - begin) /
    (double)(CLOCKS_PER_SEC / 1000) / measurements << " ms" << std::endl;

  delete[] matches;

  return tree;
}

ORBVocabulary *doDBoW(uint8_t *queries, size_t N, char *fname,
  DBoW2::BowVector &bow, DBoW2::FeatureVector &feat) {

  std::vector<DBoW2::FORB::TDescriptor> descriptors(N);
  for (size_t i = 0; i < N; i += 1) {
    descriptors[i] = cv::Mat(cv::Size(32, 1), CV_8U, queries+i*32,
        cv::Mat::AUTO_STEP);
  }

  ORBVocabulary *tree = new ORBVocabulary();
  tree->loadFromTextFile(fname);

  std::clock_t begin = std::clock();
  for (int m = 0; m < measurements; m += 1) {
    tree->transform(descriptors, bow, feat, 4);
  }
  std::clock_t end = std::clock();

  std::cout << "dbow2 time: " << (end - begin) /
    (double)(CLOCKS_PER_SEC / 1000) / measurements << " ms" << std::endl;

  return tree;
}

void showCorrespondence(
    pimatch::HammingTree *pm, ORBVocabulary *db,
    DBoW2::BowVector bow1, DBoW2::FeatureVector feat1,
    DBoW2::BowVector bow2, DBoW2::FeatureVector feat2) {

  std::set<DBoW2::WordId> pmWords;
  std::set<DBoW2::WordId> dbWords;

  // pimatch only stores DBoW2 node_id for debugging purposes
  for (std::pair<DBoW2::WordId, double> kv : bow1) {
    pmWords.insert(pm->GetNodeId(kv.first));
  }

  for (std::pair<DBoW2::WordId, double> kv : bow2) {
    dbWords.insert(db->getParentNode(kv.first, 0));
  }

  if (pmWords != dbWords) {
    std::cout << "words not equal" << std::endl;
  }

  // find an equivalent set for each class
  for (std::pair<DBoW2::NodeId, std::vector<unsigned int>> i : feat1) {
    std::set<unsigned int> s1(i.second.begin(), i.second.end());

    bool found = false;
    for (std::pair<DBoW2::NodeId, std::vector<unsigned int>> j: feat2) {
      std::set<unsigned int> s2(j.second.begin(), j.second.end());
      if (s1 == s2) {
        found = true;
        break;
      }
    }

    if (!found) {
      std::cout << "No equivalent set found for";
      for (unsigned v : s1) {
        std::cout << " " << v;
      }
      std::cout << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage " << argv[0] << " Vocabulary.txt" << std::endl;
    return 1;
  }

  std::mt19937 rng;

  const size_t N = 1000;
  uint8_t *queries = new uint8_t[N*32];
  for (size_t i = 0; i < N*32; i += 1) {
    queries[i] = rng();
  }

  DBoW2::BowVector bow1;
  DBoW2::FeatureVector feat1;

  DBoW2::BowVector bow2;
  DBoW2::FeatureVector feat2;

  pimatch::HammingTree * pm = doPiMatch(queries, N, argv[1], bow1, feat1);
  ORBVocabulary *db = doDBoW(queries, N, argv[1], bow2, feat2);

  showCorrespondence(pm, db, bow1, feat1, bow2, feat2);

  delete pm;
  delete db;

  return 0;
}
