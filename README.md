# PiMatch

PiMatch uses ARM NEON instructions to implement fast binary descriptor matching.
Tests show that the core matching implementation is 4-5 times faster than the usual
bit-twiddling implementation.

## License

GPLv3

## Building

A copy of DBoW used by ORB_SLAM is included in the `demo/` directory. In order
to build the `HammingTreeDemo`, DBoW must be built first.

The default version of g++ installed on the raspberry pi may crash when compiling
DBoW. In this case, use g++-6 or later.

## Core Primitives
The library provides several primitive matching methods which differ only in how
the descriptors are arranged in memory.
Their basic function is for each descriptor in the array of descriptors `needle`,
to find the index of the closest matching descriptor in the array `haystack`.

Each variant is suffixed with three distinguishing letters, which are either `d` or `i`.
A `d` for dense indicates that the descriptors or matches are sequentially located
in memory. An `i` for indexed indicates that an additional array of indices will be
provided to index the input arrays. The three letters correspond to the three input
arrays `matches`, `haystack` and `needle` respectively. Therefore, the suffix
`ddi` indicates that `matches` and `haystack` are dense, but `needle` is indexed.

Indexing allows PiMatch to compute descriptor distances in parallel without requiring
the API consumer to construct densely arranged input arrays.

A second class of variants, suffixed with `2`, find the two best match indices,
as required by the commonly used best-match ratio test.

The currently available methods are

```
 hammingMatch256ddd
 hammingMatch256ddi
 hammingMatch256did
 hammingMatch256idi
 hammingMatch256dii
 hammingMatch256ddd2
 hammingMatch256ddi2
 hammingMatch256did2
 hammingMatch256idi2
 hammingMatch256dii2
```

## HammingTree
The `HammingTree` class implements approximate NN searches using a k-means tree.
The implementation is almost a drop-in replacement for DBoW, and `Demo.cpp` shows
how to compute compatible data structures.

Since PiMatch is part of a larger project to speedup [ORB_SLAM](https://github.com/raulmur/ORB_SLAM2) on the RaspberryPi,
the only file format currently understood by `HammingTree` is the textual format used by
ORB_SLAM. For testing, The `Vocabulary/ORBvoc.txt` from the ORB_SLAM repository
can be used.

## Performance
The best measure of real-world performance is illustrated by the `HammingTree` demo.
PiMatch is able to construct equivalent data structures 3.5x faster than DBoW.
If 1:1 compatibility is not needed, useful data structures can be computed 4x faster.

Measuring performance of the core primitives is more complicated, since some
use cases require the construction of index arrays. However, compared to the reference
implementations used by the test cases, PiMatch is 4-5 times faster. The below graph
shows computation costs on a RaspberryPi 3 for each of the core matching variants.

![Primitive Execution Time](doc/match_times.png?raw=true "Primative Execution Time")

Note that the x-axis shows `√comparisons` since the number of the comparisons grows `O(mn)`,
where `m` and `n` are usually approximately equal.
