#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <iostream>

#include "HammingMatch.h"

namespace pimatch {

// classes defining input output patterns for matcher
namespace match_params {
struct DDD;
struct DDI;
struct IDI;
struct DII;

constexpr bool best1 = false;
constexpr bool best2 = true;
} /* namespace match_params */

using namespace match_params;

template <typename LS, bool n_bests>
void hammingMatch256Base(
    uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *haystack_indices, size_t *needle_indices,
    size_t num_haystack, size_t num_needle, size_t haystack_base);

void hammingMatch256ddd(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t num_haystack, size_t num_needle, size_t haystack_base) {

  return hammingMatch256Base<DDD, best1>(
      matches, haystack, needle, nullptr, nullptr,
      num_haystack, num_needle, haystack_base);
}

void hammingMatch256ddi(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *needle_indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base) {

  return hammingMatch256Base<DDI, best1>(
      matches, haystack, needle, nullptr, needle_indices,
      num_haystack, num_needle, haystack_base);
}

void hammingMatch256idi(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *needle_indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base) {

  return hammingMatch256Base<IDI, best1>(
      matches, haystack, needle, nullptr, needle_indices,
      num_haystack, num_needle, haystack_base);
}

void hammingMatch256dii(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *haystack_indices, size_t *needle_indices,
    size_t num_haystack, size_t num_needle, size_t haystack_base) {

  return hammingMatch256Base<DII, best1>(
      matches, haystack, needle, haystack_indices, needle_indices,
      num_haystack, num_needle, haystack_base);
}

void hammingMatch256ddd2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t num_haystack, size_t num_needle, size_t haystack_base) {

  return hammingMatch256Base<DDD, best2>(
      matches, haystack, needle, nullptr, nullptr,
      num_haystack, num_needle, haystack_base);
}

void hammingMatch256ddi2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *needle_indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base) {

  return hammingMatch256Base<DDI, best2>(
      matches, haystack, needle, nullptr, needle_indices,
      num_haystack, num_needle, haystack_base);
}

void hammingMatch256idi2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *needle_indices, size_t num_haystack, size_t num_needle,
    size_t haystack_base) {

  return hammingMatch256Base<IDI, best2>(
      matches, haystack, needle, nullptr, needle_indices,
      num_haystack, num_needle, haystack_base);
}

void hammingMatch256dii2(uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *haystack_indices, size_t *needle_indices,
    size_t num_haystack, size_t num_needle, size_t haystack_base) {

  return hammingMatch256Base<DII, best2>(
      matches, haystack, needle, haystack_indices, needle_indices,
      num_haystack, num_needle, haystack_base);
}

namespace match_params {

struct DenseMatch {

  template <bool best2>
  static inline
  void Store4(uint32_t *&matches, size_t *&indices,
      uint32x4_t &bests, uint32x4_t &bests2) {

    if (best2) {
      uint32x4x2_t zipped = vzipq_u32(bests, bests2);
      vst1q_u32(matches, zipped.val[0]); matches += 4;
      vst1q_u32(matches, zipped.val[1]); matches += 4;
    } else {
      vst1q_u32(matches, bests);
      matches += 4;
    }
  }

  template <bool best2>
  static inline
  void Store3(uint32_t *&matches, size_t *&indices,
      uint32x4_t &bests, uint32x4_t &bests2) {

    if (best2) {
      uint32x4x2_t zipped = vzipq_u32(bests, bests2);
      vst1q_u32(matches, zipped.val[0]);
      vst1_u32(matches+4, vget_low_u32(zipped.val[1]));
    } else {
      vst1_u32(matches, vget_low_u32(bests));
      vst1_lane_u32(matches+2, vget_high_u32(bests), 0);
    }
  }

  template <bool best2>
  static inline
  void Store2(uint32_t *&matches, size_t *&indices,
      uint32x4_t &bests, uint32x4_t &bests2) {

    if (best2) {
      uint32x2x2_t zipped = vzip_u32(vget_low_u32(bests), vget_low_u32(bests2));
      vst1_u32(matches, zipped.val[0]);
      vst1_u32(matches+2, zipped.val[1]);
    } else {
      vst1_u32(matches, vget_low_u32(bests));
    }
  }

  template <bool best2>
  static inline
  void Store1(uint32_t *&matches, size_t *&indices,
      uint32x4_t &bests, uint32x4_t &bests2) {

    if (best2) {
      uint32x2x2_t zipped = vzip_u32(vget_low_u32(bests), vget_low_u32(bests2));
      vst1_u32(matches, zipped.val[0]);
    } else {
      vst1_lane_u32(matches, vget_low_u32(bests), 0);
    }
  }
};

struct IndexMatch {

  template <bool best2>
  static inline
  void Store4(uint32_t *&matches, size_t *&indices,
      uint32x4_t &bests, uint32x4_t &bests2) {

    if (best2) {
      uint32x4x2_t zipped = vzipq_u32(bests, bests2);
      vst1_u32(&matches[indices[0]*2], vget_low_u32(zipped.val[0]));
      vst1_u32(&matches[indices[1]*2], vget_high_u32(zipped.val[0]));
      vst1_u32(&matches[indices[2]*2], vget_low_u32(zipped.val[1]));
      vst1_u32(&matches[indices[3]*2], vget_high_u32(zipped.val[1]));
    } else {
      vst1q_lane_u32(&matches[indices[0]], bests, 0);
      vst1q_lane_u32(&matches[indices[1]], bests, 1);
      vst1q_lane_u32(&matches[indices[2]], bests, 2);
      vst1q_lane_u32(&matches[indices[3]], bests, 3);
    }
    indices += 4;
  }

  template <bool best2>
  static inline
  void Store3(uint32_t *&matches, size_t *&indices,
      uint32x4_t &bests, uint32x4_t &bests2) {

    if (best2) {
      uint32x4x2_t zipped = vzipq_u32(bests, bests2);
      vst1_u32(&matches[indices[0]*2], vget_low_u32(zipped.val[0]));
      vst1_u32(&matches[indices[1]*2], vget_high_u32(zipped.val[0]));
      vst1_u32(&matches[indices[2]*2], vget_low_u32(zipped.val[1]));
    } else {
      vst1q_lane_u32(&matches[indices[0]], bests, 0);
      vst1q_lane_u32(&matches[indices[1]], bests, 1);
      vst1q_lane_u32(&matches[indices[2]], bests, 2);
    }
  }

  template <bool best2>
  static inline
  void Store2(uint32_t *&matches, size_t *&indices,
      uint32x4_t &bests, uint32x4_t &bests2) {

    if (best2) {
      uint32x2x2_t zipped = vzip_u32(vget_low_u32(bests), vget_low_u32(bests2));
      vst1_u32(&matches[indices[0]*2], zipped.val[0]);
      vst1_u32(&matches[indices[1]*2], zipped.val[1]);
    } else {
      vst1q_lane_u32(&matches[indices[0]], bests, 0);
      vst1q_lane_u32(&matches[indices[1]], bests, 1);
    }
  }

  template <bool best2>
  static inline
  void Store1(uint32_t *&matches, size_t *&indices,
      uint32x4_t &bests, uint32x4_t &bests2) {

    if (best2) {
      uint32x2x2_t zipped = vzip_u32(vget_low_u32(bests), vget_low_u32(bests2));
      vst1_u32(&matches[indices[0]*2], zipped.val[0]);
    } else {
      vst1q_lane_u32(&matches[indices[0]], bests, 0);
    }
  }
};

struct DenseHaystack {
  static inline
  uint8x16_t HaystackLo(size_t *&indices, uint8_t *&haystack, size_t index) {
      uint8x16_t ret = vld1q_u8(haystack);
      haystack += 16;
      return ret;
  }

  static inline
  uint8x16_t HaystackHi(size_t *&indices, uint8_t *&haystack, size_t index) {
    uint8x16_t ret = vld1q_u8(haystack);
    haystack += 16;
    return ret;
  }

  static inline
  uint16x4_t HaystackIndexLo(size_t *&indices, size_t index, size_t base) {
    return  vdup_n_u16(base + index);
  }

  static inline
  uint8x8_t HaystackIndexHi(size_t *&indices, size_t index, size_t base) {
    return vdup_n_u8((base + index) >> 16);
  }
};

struct IndexHaystack {
  static inline
  uint8x16_t HaystackLo(size_t *&indices, uint8_t *&haystack, size_t index) {
    return vld1q_u8(&haystack[indices[index]*32]);
  }

  static inline
  uint8x16_t HaystackHi(size_t *&indices, uint8_t *&haystack, size_t index) {
    return vld1q_u8(&haystack[indices[index]*32+16]);
  }

  static inline
  uint16x4_t HaystackIndexLo(size_t *&indices, size_t index, size_t base) {
    return vdup_n_u16(base + indices[index]);
  }

  static inline
  uint8x8_t HaystackIndexHi(size_t *&indices, size_t index, size_t base) {
    return vdup_n_u8((base + indices[index]) >> 16);
  }
};

struct DenseNeedle {
  static inline
  uint8x16_t NeedleLo(size_t *indices, uint8_t *&needle, size_t index) {
      uint8x16_t ret = vld1q_u8(needle);
      needle += 16;
      return ret;
  }

  static inline
  uint8x16_t NeedleHi(size_t *indices, uint8_t *&needle, size_t index) {
      uint8x16_t ret = vld1q_u8(needle);
      needle += 16;
      return ret;
  }
};

struct IndexNeedle {
  static inline
  uint8x16_t NeedleLo(size_t *indices, uint8_t *&needle, size_t index) {
      return vld1q_u8(&needle[indices[index]*32]);
  }

  static inline
  uint8x16_t NeedleHi(size_t *indices, uint8_t *&needle, size_t index) {
      return vld1q_u8(&needle[indices[index]*32+16]);
  }
};

struct BestUpdate {
  template <bool best2>
  static inline
  void Update(uint32x4_t &bests, uint32x4_t &bests2, uint32x4_t &candidate) {
    if (best2) {
      uint32x4_t a = vminq_u32(bests, candidate);
      uint32x4_t b = vminq_u32(bests2, candidate);
      bests2 = vsubq_u32(vaddq_u32(bests, b), a);
      bests = a;
    } else {
      bests = vminq_u32(bests, candidate);
    }
  }
};

struct DDD : public DenseMatch, public DenseHaystack, public DenseNeedle {};
struct DDI : public DenseMatch, public DenseHaystack, public IndexNeedle {
  // indices are advanced by store methods, but if dense storage is
  // used with indexed needles, we need to advance indices ourselves.
  template <bool n_bests>
  static inline
  void Store4(uint32_t *&matches, size_t *&indices,
      uint32x4_t &bests, uint32x4_t &bests2) {
    DenseMatch::template Store4<n_bests>(matches, indices, bests, bests2);
    indices += 4;
  }
};
struct IDI : public IndexMatch, public DenseHaystack, public IndexNeedle {};
struct DII : public DenseMatch, public IndexHaystack, public IndexNeedle {
  template <bool n_bests>
  static inline
  void Store4(uint32_t *&matches, size_t *&indices,
      uint32x4_t &bests, uint32x4_t &bests2) {
    DenseMatch::template Store4<n_bests>(matches, indices, bests, bests2);
    indices += 4;
  }
};

} /* namespace match_params */

template <typename LS, bool n_bests>
void hammingMatch256Base(
    uint32_t *matches, uint8_t *haystack, uint8_t *needle,
    size_t *haystack_indices, size_t *needle_indices,
    size_t num_haystack, size_t num_needle, size_t haystack_base) {
    
  if (num_haystack == 0 || num_needle == 0) {
    return;
  }

  const size_t hbase = haystack_base;

  uint8x16_t needle0l;
  uint8x16_t needle0h;
  uint8x16_t needle1l;
  uint8x16_t needle1h;
  uint8x16_t needle2l;
  uint8x16_t needle2h;
  uint8x16_t needle3l;
  uint8x16_t needle3h;

  for (size_t n = 3; n < num_needle; n += 4) {
    needle0l = LS::NeedleLo(needle_indices, needle, 0);
    needle0h = LS::NeedleHi(needle_indices, needle, 0);
    needle1l = LS::NeedleLo(needle_indices, needle, 1);
    needle1h = LS::NeedleHi(needle_indices, needle, 1);
    needle2l = LS::NeedleLo(needle_indices, needle, 2);
    needle2h = LS::NeedleHi(needle_indices, needle, 2);
    needle3l = LS::NeedleLo(needle_indices, needle, 3);
    needle3h = LS::NeedleHi(needle_indices, needle, 3);

    uint32x4_t bests = vdupq_n_u32(0xffffffff);
    uint32x4_t bests2 = vdupq_n_u32(0xffffffff);
      
    uint8_t *haystack_ptr = haystack;
    size_t h;
    uint8x16_t haystackl = LS::HaystackLo(haystack_indices, haystack_ptr, 0);
    uint8x16_t haystackh = LS::HaystackHi(haystack_indices, haystack_ptr, 0);

    for (h = 0; h < num_haystack - 1; h += 1) {
      // split index into high and low parts for zipping with scores
      uint8x8_t index_hi = LS::HaystackIndexHi(haystack_indices, h, hbase);
      uint16x4_t index_lo = LS::HaystackIndexLo(haystack_indices, h, hbase);

      uint8x16_t eor0l = veorq_u8(haystackl, needle0l);
      uint8x16_t eor0h = veorq_u8(haystackh, needle0h);
      uint8x16_t eor1l = veorq_u8(haystackl, needle1l);
      uint8x16_t eor1h = veorq_u8(haystackh, needle1h);
      uint8x16_t eor2l = veorq_u8(haystackl, needle2l);
      uint8x16_t eor2h = veorq_u8(haystackh, needle2h);
      uint8x16_t eor3l = veorq_u8(haystackl, needle3l);
      uint8x16_t eor3h = veorq_u8(haystackh, needle3h);

      uint8x16_t cnt0l = vcntq_u8(eor0l);
      uint8x16_t cnt0h = vcntq_u8(eor0h);
      uint8x16_t cnt1l = vcntq_u8(eor1l);
      uint8x16_t cnt1h = vcntq_u8(eor1h);
      uint8x16_t cnt2l = vcntq_u8(eor2l);
      uint8x16_t cnt2h = vcntq_u8(eor2h);
      uint8x16_t cnt3l = vcntq_u8(eor3l);
      uint8x16_t cnt3h = vcntq_u8(eor3h);

      uint8x8_t cnt0ll = vpadd_u8(vget_low_u8(cnt0l), vget_high_u8(cnt0l));
      uint8x8_t cnt0hh = vpadd_u8(vget_low_u8(cnt0h), vget_high_u8(cnt0h));
      uint8x8_t cnt1ll = vpadd_u8(vget_low_u8(cnt1l), vget_high_u8(cnt1l));
      uint8x8_t cnt1hh = vpadd_u8(vget_low_u8(cnt1h), vget_high_u8(cnt1h));
      uint8x8_t cnt2ll = vpadd_u8(vget_low_u8(cnt2l), vget_high_u8(cnt2l));
      uint8x8_t cnt2hh = vpadd_u8(vget_low_u8(cnt2h), vget_high_u8(cnt2h));
      uint8x8_t cnt3ll = vpadd_u8(vget_low_u8(cnt3l), vget_high_u8(cnt3l));
      uint8x8_t cnt3hh = vpadd_u8(vget_low_u8(cnt3h), vget_high_u8(cnt3h));

      uint8x8_t cnt01l = vpadd_u8(cnt0ll, cnt1ll);
      uint8x8_t cnt01h = vpadd_u8(cnt0hh, cnt1hh);
      uint8x8_t cnt23l = vpadd_u8(cnt2ll, cnt3ll);
      uint8x8_t cnt23h = vpadd_u8(cnt2hh, cnt3hh);

      uint8x8_t cnt0123l(vpadd_u8(cnt01l, cnt23l));
      uint8x8_t cnt0123h(vpadd_u8(cnt01h, cnt23h));

      // counts are still in pairs
      cnt0123l = vpadd_u8(cnt0123l, cnt0123l);
      cnt0123h = vpadd_u8(cnt0123h, cnt0123h);

      // saturated add to avoid overflow
      uint8x8_t score = vqadd_u8(cnt0123l, cnt0123h);

      // zip in high part
      uint8x8x2_t score_index1 = vzip_u8(index_hi, score);

      // zip in lo part
      uint16x4x2_t score_index2 = vzip_u16(index_lo,
          vreinterpret_u16_u8(score_index1.val[0]));

      uint32x4_t score_index = vcombine_u32(
          vreinterpret_u32_u16(score_index2.val[0]),
          vreinterpret_u32_u16(score_index2.val[1]));

      BestUpdate::Update<n_bests>(bests, bests2, score_index);

      haystackl = LS::HaystackLo(haystack_indices, haystack_ptr, h + 1);
      haystackh = LS::HaystackHi(haystack_indices, haystack_ptr, h + 1);
    }
    
    { // last haystack
      // split index into high and low parts for zipping with scores
      uint8x8_t index_hi = LS::HaystackIndexHi(haystack_indices, h, hbase);
      uint16x4_t index_lo = LS::HaystackIndexLo(haystack_indices, h, hbase);

      uint8x16_t eor0l = veorq_u8(haystackl, needle0l);
      uint8x16_t eor0h = veorq_u8(haystackh, needle0h);
      uint8x16_t eor1l = veorq_u8(haystackl, needle1l);
      uint8x16_t eor1h = veorq_u8(haystackh, needle1h);
      uint8x16_t eor2l = veorq_u8(haystackl, needle2l);
      uint8x16_t eor2h = veorq_u8(haystackh, needle2h);
      uint8x16_t eor3l = veorq_u8(haystackl, needle3l);
      uint8x16_t eor3h = veorq_u8(haystackh, needle3h);

      uint8x16_t cnt0l = vcntq_u8(eor0l);
      uint8x16_t cnt0h = vcntq_u8(eor0h);
      uint8x16_t cnt1l = vcntq_u8(eor1l);
      uint8x16_t cnt1h = vcntq_u8(eor1h);
      uint8x16_t cnt2l = vcntq_u8(eor2l);
      uint8x16_t cnt2h = vcntq_u8(eor2h);
      uint8x16_t cnt3l = vcntq_u8(eor3l);
      uint8x16_t cnt3h = vcntq_u8(eor3h);

      uint8x8_t cnt0ll = vpadd_u8(vget_low_u8(cnt0l), vget_high_u8(cnt0l));
      uint8x8_t cnt0hh = vpadd_u8(vget_low_u8(cnt0h), vget_high_u8(cnt0h));
      uint8x8_t cnt1ll = vpadd_u8(vget_low_u8(cnt1l), vget_high_u8(cnt1l));
      uint8x8_t cnt1hh = vpadd_u8(vget_low_u8(cnt1h), vget_high_u8(cnt1h));
      uint8x8_t cnt2ll = vpadd_u8(vget_low_u8(cnt2l), vget_high_u8(cnt2l));
      uint8x8_t cnt2hh = vpadd_u8(vget_low_u8(cnt2h), vget_high_u8(cnt2h));
      uint8x8_t cnt3ll = vpadd_u8(vget_low_u8(cnt3l), vget_high_u8(cnt3l));
      uint8x8_t cnt3hh = vpadd_u8(vget_low_u8(cnt3h), vget_high_u8(cnt3h));

      uint8x8_t cnt01l = vpadd_u8(cnt0ll, cnt1ll);
      uint8x8_t cnt01h = vpadd_u8(cnt0hh, cnt1hh);
      uint8x8_t cnt23l = vpadd_u8(cnt2ll, cnt3ll);
      uint8x8_t cnt23h = vpadd_u8(cnt2hh, cnt3hh);

      uint8x8_t cnt0123l(vpadd_u8(cnt01l, cnt23l));
      uint8x8_t cnt0123h(vpadd_u8(cnt01h, cnt23h));

      // counts are still in pairs
      cnt0123l = vpadd_u8(cnt0123l, cnt0123l);
      cnt0123h = vpadd_u8(cnt0123h, cnt0123h);

      // saturated add to avoid overflow
      uint8x8_t score = vqadd_u8(cnt0123l, cnt0123h);

      // zip in high part
      uint8x8x2_t score_index1 = vzip_u8(index_hi, score);

      // zip in lo part
      uint16x4x2_t score_index2 = vzip_u16(index_lo,
          vreinterpret_u16_u8(score_index1.val[0]));

      uint32x4_t score_index = vcombine_u32(
          vreinterpret_u32_u16(score_index2.val[0]),
          vreinterpret_u32_u16(score_index2.val[1]));

      BestUpdate::Update<n_bests>(bests, bests2, score_index);
    }

    LS::template Store4<n_bests>(matches, needle_indices, bests, bests2);
  }

  size_t remainder = num_needle % 4;
  if (remainder == 0) {
    return;
  } else if (remainder == 1) {
    needle0l = LS::NeedleLo(needle_indices, needle, 0);
    needle0h = LS::NeedleHi(needle_indices, needle, 0);

    uint32x4_t bests = vdupq_n_u32(0xffffffff);
    uint32x4_t bests2 = vdupq_n_u32(0xffffffff);

    uint8_t *haystack_ptr = haystack;
    size_t h;
    uint8x16_t haystackl = LS::HaystackLo(haystack_indices, haystack_ptr, 0);
    uint8x16_t haystackh = LS::HaystackHi(haystack_indices, haystack_ptr, 0);

    for (h = 0; h < num_haystack - 1; h += 1) {
      // split index into high and low parts for zipping with scores
      uint8x8_t index_hi = LS::HaystackIndexHi(haystack_indices, h, hbase);
      uint16x4_t index_lo = LS::HaystackIndexLo(haystack_indices, h, hbase);

      uint8x16_t eor0l = veorq_u8(haystackl, needle0l);
      uint8x16_t eor0h = veorq_u8(haystackh, needle0h);

      uint8x16_t cnt0l = vcntq_u8(eor0l);
      uint8x16_t cnt0h = vcntq_u8(eor0h);

      uint8x8_t cnt0ll = vpadd_u8(vget_low_u8(cnt0l), vget_high_u8(cnt0l));
      uint8x8_t cnt0hh = vpadd_u8(vget_low_u8(cnt0h), vget_high_u8(cnt0h));

      uint8x8_t cnt01l = vpadd_u8(cnt0ll, cnt0ll);
      uint8x8_t cnt01h = vpadd_u8(cnt0hh, cnt0hh);

      uint8x8_t cnt0123l(vpadd_u8(cnt01l, cnt01l));
      uint8x8_t cnt0123h(vpadd_u8(cnt01h, cnt01h));

      // counts are still in pairs
      cnt0123l = vpadd_u8(cnt0123l, cnt0123l);
      cnt0123h = vpadd_u8(cnt0123h, cnt0123h);

      // saturated add to avoid overflow
      uint8x8_t score = vqadd_u8(cnt0123l, cnt0123h);

      // zip in high part
      uint8x8x2_t score_index1 = vzip_u8(index_hi, score);

      // zip in lo part
      uint16x4x2_t score_index2 = vzip_u16(index_lo,
          vreinterpret_u16_u8(score_index1.val[0]));

      uint32x4_t score_index = vcombine_u32(
          vreinterpret_u32_u16(score_index2.val[0]),
          vreinterpret_u32_u16(score_index2.val[1]));

      BestUpdate::Update<n_bests>(bests, bests2, score_index);

      haystackl = LS::HaystackLo(haystack_indices, haystack_ptr, h + 1);
      haystackh = LS::HaystackHi(haystack_indices, haystack_ptr, h + 1);
    }
    
    { // last haystack
      // split index into high and low parts for zipping with scores
      uint8x8_t index_hi = LS::HaystackIndexHi(haystack_indices, h, hbase);
      uint16x4_t index_lo = LS::HaystackIndexLo(haystack_indices, h, hbase);

      uint8x16_t eor0l = veorq_u8(haystackl, needle0l);
      uint8x16_t eor0h = veorq_u8(haystackh, needle0h);

      uint8x16_t cnt0l = vcntq_u8(eor0l);
      uint8x16_t cnt0h = vcntq_u8(eor0h);

      uint8x8_t cnt0ll = vpadd_u8(vget_low_u8(cnt0l), vget_high_u8(cnt0l));
      uint8x8_t cnt0hh = vpadd_u8(vget_low_u8(cnt0h), vget_high_u8(cnt0h));

      uint8x8_t cnt01l = vpadd_u8(cnt0ll, cnt0ll);
      uint8x8_t cnt01h = vpadd_u8(cnt0hh, cnt0hh);

      uint8x8_t cnt0123l(vpadd_u8(cnt01l, cnt01l));
      uint8x8_t cnt0123h(vpadd_u8(cnt01h, cnt01h));

      // counts are still in pairs
      cnt0123l = vpadd_u8(cnt0123l, cnt0123l);
      cnt0123h = vpadd_u8(cnt0123h, cnt0123h);

      // saturated add to avoid overflow
      uint8x8_t score = vqadd_u8(cnt0123l, cnt0123h);

      // zip in high part
      uint8x8x2_t score_index1 = vzip_u8(index_hi, score);

      // zip in lo part
      uint16x4x2_t score_index2 = vzip_u16(index_lo,
          vreinterpret_u16_u8(score_index1.val[0]));

      uint32x4_t score_index = vcombine_u32(
          vreinterpret_u32_u16(score_index2.val[0]),
          vreinterpret_u32_u16(score_index2.val[1]));

      BestUpdate::Update<n_bests>(bests, bests2, score_index);
    }

    LS::template Store1<n_bests>(matches, needle_indices, bests, bests2);
  } else if (remainder == 2) {
    needle0l = LS::NeedleLo(needle_indices, needle, 0);
    needle0h = LS::NeedleHi(needle_indices, needle, 0);
    needle1l = LS::NeedleLo(needle_indices, needle, 1);
    needle1h = LS::NeedleHi(needle_indices, needle, 1);

    uint32x4_t bests = vdupq_n_u32(0xffffffff);
    uint32x4_t bests2 = vdupq_n_u32(0xffffffff);
      
    uint8_t *haystack_ptr = haystack;
    size_t h;
    uint8x16_t haystackl = LS::HaystackLo(haystack_indices, haystack_ptr, 0);
    uint8x16_t haystackh = LS::HaystackHi(haystack_indices, haystack_ptr, 0);

    for (h = 0; h < num_haystack - 1; h += 1) {
      // split index into high and low parts for zipping with scores
      uint8x8_t index_hi = LS::HaystackIndexHi(haystack_indices, h, hbase);
      uint16x4_t index_lo = LS::HaystackIndexLo(haystack_indices, h, hbase);

      uint8x16_t eor0l = veorq_u8(haystackl, needle0l);
      uint8x16_t eor0h = veorq_u8(haystackh, needle0h);
      uint8x16_t eor1l = veorq_u8(haystackl, needle1l);
      uint8x16_t eor1h = veorq_u8(haystackh, needle1h);

      uint8x16_t cnt0l = vcntq_u8(eor0l);
      uint8x16_t cnt0h = vcntq_u8(eor0h);
      uint8x16_t cnt1l = vcntq_u8(eor1l);
      uint8x16_t cnt1h = vcntq_u8(eor1h);

      uint8x8_t cnt0ll = vpadd_u8(vget_low_u8(cnt0l), vget_high_u8(cnt0l));
      uint8x8_t cnt0hh = vpadd_u8(vget_low_u8(cnt0h), vget_high_u8(cnt0h));
      uint8x8_t cnt1ll = vpadd_u8(vget_low_u8(cnt1l), vget_high_u8(cnt1l));
      uint8x8_t cnt1hh = vpadd_u8(vget_low_u8(cnt1h), vget_high_u8(cnt1h));

      uint8x8_t cnt01l = vpadd_u8(cnt0ll, cnt1ll);
      uint8x8_t cnt01h = vpadd_u8(cnt0hh, cnt1hh);

      uint8x8_t cnt0123l(vpadd_u8(cnt01l, cnt01l));
      uint8x8_t cnt0123h(vpadd_u8(cnt01h, cnt01h));

      // counts are still in pairs
      cnt0123l = vpadd_u8(cnt0123l, cnt0123l);
      cnt0123h = vpadd_u8(cnt0123h, cnt0123h);

      // saturated add to avoid overflow
      uint8x8_t score = vqadd_u8(cnt0123l, cnt0123h);

      // zip in high part
      uint8x8x2_t score_index1 = vzip_u8(index_hi, score);

      // zip in lo part
      uint16x4x2_t score_index2 = vzip_u16(index_lo,
          vreinterpret_u16_u8(score_index1.val[0]));

      uint32x4_t score_index = vcombine_u32(
          vreinterpret_u32_u16(score_index2.val[0]),
          vreinterpret_u32_u16(score_index2.val[1]));

      BestUpdate::Update<n_bests>(bests, bests2, score_index);

      haystackl = LS::HaystackLo(haystack_indices, haystack_ptr, h + 1);
      haystackh = LS::HaystackHi(haystack_indices, haystack_ptr, h + 1);
    }
    
    { // last haystack
      // split index into high and low parts for zipping with scores
      uint8x8_t index_hi = LS::HaystackIndexHi(haystack_indices, h, hbase);
      uint16x4_t index_lo = LS::HaystackIndexLo(haystack_indices, h, hbase);

      uint8x16_t eor0l = veorq_u8(haystackl, needle0l);
      uint8x16_t eor0h = veorq_u8(haystackh, needle0h);
      uint8x16_t eor1l = veorq_u8(haystackl, needle1l);
      uint8x16_t eor1h = veorq_u8(haystackh, needle1h);

      uint8x16_t cnt0l = vcntq_u8(eor0l);
      uint8x16_t cnt0h = vcntq_u8(eor0h);
      uint8x16_t cnt1l = vcntq_u8(eor1l);
      uint8x16_t cnt1h = vcntq_u8(eor1h);

      uint8x8_t cnt0ll = vpadd_u8(vget_low_u8(cnt0l), vget_high_u8(cnt0l));
      uint8x8_t cnt0hh = vpadd_u8(vget_low_u8(cnt0h), vget_high_u8(cnt0h));
      uint8x8_t cnt1ll = vpadd_u8(vget_low_u8(cnt1l), vget_high_u8(cnt1l));
      uint8x8_t cnt1hh = vpadd_u8(vget_low_u8(cnt1h), vget_high_u8(cnt1h));

      uint8x8_t cnt01l = vpadd_u8(cnt0ll, cnt1ll);
      uint8x8_t cnt01h = vpadd_u8(cnt0hh, cnt1hh);

      uint8x8_t cnt0123l(vpadd_u8(cnt01l, cnt01l));
      uint8x8_t cnt0123h(vpadd_u8(cnt01h, cnt01h));

      // counts are still in pairs
      cnt0123l = vpadd_u8(cnt0123l, cnt0123l);
      cnt0123h = vpadd_u8(cnt0123h, cnt0123h);

      // saturated add to avoid overflow
      uint8x8_t score = vqadd_u8(cnt0123l, cnt0123h);

      // zip in high part
      uint8x8x2_t score_index1 = vzip_u8(index_hi, score);

      // zip in lo part
      uint16x4x2_t score_index2 = vzip_u16(index_lo,
          vreinterpret_u16_u8(score_index1.val[0]));

      uint32x4_t score_index = vcombine_u32(
          vreinterpret_u32_u16(score_index2.val[0]),
          vreinterpret_u32_u16(score_index2.val[1]));

      BestUpdate::Update<n_bests>(bests, bests2, score_index);
    }

    LS::template Store2<n_bests>(matches, needle_indices, bests, bests2);

  } else if (remainder == 3) {
    needle0l = LS::NeedleLo(needle_indices, needle, 0);
    needle0h = LS::NeedleHi(needle_indices, needle, 0);
    needle1l = LS::NeedleLo(needle_indices, needle, 1);
    needle1h = LS::NeedleHi(needle_indices, needle, 1);
    needle2l = LS::NeedleLo(needle_indices, needle, 2);
    needle2h = LS::NeedleHi(needle_indices, needle, 2);

    uint32x4_t bests = vdupq_n_u32(0xffffffff);
    uint32x4_t bests2 = vdupq_n_u32(0xffffffff);
      
    uint8_t *haystack_ptr = haystack;
    size_t h;
    uint8x16_t haystackl = LS::HaystackLo(haystack_indices, haystack_ptr, 0);
    uint8x16_t haystackh = LS::HaystackHi(haystack_indices, haystack_ptr, 0);

    for (h = 0; h < num_haystack - 1; h += 1) {
      // split index into high and low parts for zipping with scores
      uint8x8_t index_hi = LS::HaystackIndexHi(haystack_indices, h, hbase);
      uint16x4_t index_lo = LS::HaystackIndexLo(haystack_indices, h, hbase);

      uint8x16_t eor0l = veorq_u8(haystackl, needle0l);
      uint8x16_t eor0h = veorq_u8(haystackh, needle0h);
      uint8x16_t eor1l = veorq_u8(haystackl, needle1l);
      uint8x16_t eor1h = veorq_u8(haystackh, needle1h);
      uint8x16_t eor2l = veorq_u8(haystackl, needle2l);
      uint8x16_t eor2h = veorq_u8(haystackh, needle2h);

      uint8x16_t cnt0l = vcntq_u8(eor0l);
      uint8x16_t cnt0h = vcntq_u8(eor0h);
      uint8x16_t cnt1l = vcntq_u8(eor1l);
      uint8x16_t cnt1h = vcntq_u8(eor1h);
      uint8x16_t cnt2l = vcntq_u8(eor2l);
      uint8x16_t cnt2h = vcntq_u8(eor2h);

      uint8x8_t cnt0ll = vpadd_u8(vget_low_u8(cnt0l), vget_high_u8(cnt0l));
      uint8x8_t cnt0hh = vpadd_u8(vget_low_u8(cnt0h), vget_high_u8(cnt0h));
      uint8x8_t cnt1ll = vpadd_u8(vget_low_u8(cnt1l), vget_high_u8(cnt1l));
      uint8x8_t cnt1hh = vpadd_u8(vget_low_u8(cnt1h), vget_high_u8(cnt1h));
      uint8x8_t cnt2ll = vpadd_u8(vget_low_u8(cnt2l), vget_high_u8(cnt2l));
      uint8x8_t cnt2hh = vpadd_u8(vget_low_u8(cnt2h), vget_high_u8(cnt2h));

      uint8x8_t cnt01l = vpadd_u8(cnt0ll, cnt1ll);
      uint8x8_t cnt01h = vpadd_u8(cnt0hh, cnt1hh);
      uint8x8_t cnt23l = vpadd_u8(cnt2ll, cnt2ll);
      uint8x8_t cnt23h = vpadd_u8(cnt2hh, cnt2hh);

      uint8x8_t cnt0123l(vpadd_u8(cnt01l, cnt23l));
      uint8x8_t cnt0123h(vpadd_u8(cnt01h, cnt23h));

      // counts are still in pairs
      cnt0123l = vpadd_u8(cnt0123l, cnt0123l);
      cnt0123h = vpadd_u8(cnt0123h, cnt0123h);

      // saturated add to avoid overflow
      uint8x8_t score = vqadd_u8(cnt0123l, cnt0123h);

      // zip in high part
      uint8x8x2_t score_index1 = vzip_u8(index_hi, score);

      // zip in lo part
      uint16x4x2_t score_index2 = vzip_u16(index_lo,
          vreinterpret_u16_u8(score_index1.val[0]));

      uint32x4_t score_index = vcombine_u32(
          vreinterpret_u32_u16(score_index2.val[0]),
          vreinterpret_u32_u16(score_index2.val[1]));

      BestUpdate::Update<n_bests>(bests, bests2, score_index);

      haystackl = LS::HaystackLo(haystack_indices, haystack_ptr, h + 1);
      haystackh = LS::HaystackHi(haystack_indices, haystack_ptr, h + 1);
    }
    
    { // last haystack
      // split index into high and low parts for zipping with scores
      uint8x8_t index_hi = LS::HaystackIndexHi(haystack_indices, h, hbase);
      uint16x4_t index_lo = LS::HaystackIndexLo(haystack_indices, h, hbase);

      uint8x16_t eor0l = veorq_u8(haystackl, needle0l);
      uint8x16_t eor0h = veorq_u8(haystackh, needle0h);
      uint8x16_t eor1l = veorq_u8(haystackl, needle1l);
      uint8x16_t eor1h = veorq_u8(haystackh, needle1h);
      uint8x16_t eor2l = veorq_u8(haystackl, needle2l);
      uint8x16_t eor2h = veorq_u8(haystackh, needle2h);

      uint8x16_t cnt0l = vcntq_u8(eor0l);
      uint8x16_t cnt0h = vcntq_u8(eor0h);
      uint8x16_t cnt1l = vcntq_u8(eor1l);
      uint8x16_t cnt1h = vcntq_u8(eor1h);
      uint8x16_t cnt2l = vcntq_u8(eor2l);
      uint8x16_t cnt2h = vcntq_u8(eor2h);

      uint8x8_t cnt0ll = vpadd_u8(vget_low_u8(cnt0l), vget_high_u8(cnt0l));
      uint8x8_t cnt0hh = vpadd_u8(vget_low_u8(cnt0h), vget_high_u8(cnt0h));
      uint8x8_t cnt1ll = vpadd_u8(vget_low_u8(cnt1l), vget_high_u8(cnt1l));
      uint8x8_t cnt1hh = vpadd_u8(vget_low_u8(cnt1h), vget_high_u8(cnt1h));
      uint8x8_t cnt2ll = vpadd_u8(vget_low_u8(cnt2l), vget_high_u8(cnt2l));
      uint8x8_t cnt2hh = vpadd_u8(vget_low_u8(cnt2h), vget_high_u8(cnt2h));

      uint8x8_t cnt01l = vpadd_u8(cnt0ll, cnt1ll);
      uint8x8_t cnt01h = vpadd_u8(cnt0hh, cnt1hh);
      uint8x8_t cnt23l = vpadd_u8(cnt2ll, cnt2ll);
      uint8x8_t cnt23h = vpadd_u8(cnt2hh, cnt2hh);

      uint8x8_t cnt0123l(vpadd_u8(cnt01l, cnt23l));
      uint8x8_t cnt0123h(vpadd_u8(cnt01h, cnt23h));

      // counts are still in pairs
      cnt0123l = vpadd_u8(cnt0123l, cnt0123l);
      cnt0123h = vpadd_u8(cnt0123h, cnt0123h);

      // saturated add to avoid overflow
      uint8x8_t score = vqadd_u8(cnt0123l, cnt0123h);

      // zip in high part
      uint8x8x2_t score_index1 = vzip_u8(index_hi, score);

      // zip in lo part
      uint16x4x2_t score_index2 = vzip_u16(index_lo,
          vreinterpret_u16_u8(score_index1.val[0]));

      uint32x4_t score_index = vcombine_u32(
          vreinterpret_u32_u16(score_index2.val[0]),
          vreinterpret_u32_u16(score_index2.val[1]));

      BestUpdate::Update<n_bests>(bests, bests2, score_index);
    }

    LS::template Store3<n_bests>(matches, needle_indices, bests, bests2);
  }
}

} /* namespace */
