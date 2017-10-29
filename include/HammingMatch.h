#define PIMATCH_HAMMING_MATCH_H__

#include <stdint.h>
#include <stddef.h>

namespace pimatch {

// pimatch achieves a speedup over other libraries by making bulk queries.
// However, the performance gains would be lost if the api consumer were
// required to rearrange their haystack and needle descriptors.
//
// Therefore, multiple methods that provide different access strategies.
// The three buffers matches, haystack, and needle, can optionally be indexed.
// The method name suffix describes how they will be indexed,
// with a 'd' implying dense access and 'i' for indexed access.
//
// The matches buffer, if indexed, uses the needle_indices.
//
// The variants suffixed with 2 return the two best matches.
// In this case the matches buffer should be twice as large.
//
// matches[2*i+0] = best_match(i)
// matches[2*i+1] = second_best_match(i)
//
// Sometimes they haystack may be an index into a larger haystack.
// In this case, haystack_base may be set to add an offset to the
// result indices outputted into the matches buffer.

/// Brute-force match 256 bit binary descriptors.
/// Both matches and needle are indexed sequentially, i.e
///    matches[i] = best_match(needle[i])
///
void hammingMatch256ddd(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t num_haystack, size_t num_needle, size_t haystack_base = 0);

/// Brute-force match 256 bit binary descriptors.
/// Matches are sequential, but needle is indexed. E.g. if indices[i] = 10,
///    matches[i] = best_match(needle[10])
///
void hammingMatch256ddi(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *needle_indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base = 0);

/// Brute-force match 256 bit binary descriptors.
/// Both matches and needle are indexed, e.g. if indices[i] = 10,
///    matches[10] = best_match(needle[10])
///
void hammingMatch256idi(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *needle_indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base = 0);

/// Brute-force match 256 bit binary descriptors.
/// Both haystack and needle are indexed, e.g. if indices[i] = 10,
///    matches[i] = best_match(needle[10]) for haystack index entries
///
void hammingMatch256dii(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *haystack_indices, size_t *needle_indices,
    size_t num_haystack, size_t num_needle, size_t haystack_base = 0);

/// Brute-force match 256 bit binary descriptors. Return two best matches.
/// Both matches and needle are indexed sequentially, i.e
///    matches[i] = best_match(needle[i])
///
void hammingMatch256ddd2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t num_haystack, size_t num_needle, size_t haystack_base = 0);

/// Brute-force match 256 bit binary descriptors. Return two best matches.
/// Matches are sequential, but needle is indexed. E.g. if indices[i] = 10,
///    matches[i] = best_match(needle[10])
///
void hammingMatch256ddi2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *needle_indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base = 0);

/// Brute-force match 256 bit binary descriptors. Return two best matches.
/// Both matches and needle are indexed, e.g. if indices[i] = 10,
///    matches[10] = best_match(needle[10])
///
void hammingMatch256idi2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *needle_indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base = 0);

/// Brute-force match 256 bit binary descriptors. Return two best matches.
/// Both haystack and needle are indexed, e.g. if indices[i] = 10,
///    matches[i] = best_match(needle[10]) for haystack index entries
///
void hammingMatch256dii2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *haystack_indices, size_t *needle_indices,
    size_t num_haystack, size_t num_needle, size_t haystack_base = 0);

} /* namespace */
