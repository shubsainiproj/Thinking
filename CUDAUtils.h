#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

__host__ __forceinline__ void add256_u64(const uint64_t a[4], uint64_t b, uint64_t out[4]);
__host__ __forceinline__ void add256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]);
__host__ __forceinline__ void sub256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]);
__host__ __forceinline__ void inc256(uint64_t a[4], uint64_t inc);
__host__ void divmod_256_by_u64(const uint64_t value[4], uint64_t divisor, uint64_t quotient[4], uint64_t &remainder);

__device__ __forceinline__ void divmod_256_by_u64_device(const uint64_t value[4], uint64_t divisor, uint64_t quotient[4], uint64_t &remainder);
__device__ __forceinline__ void ge256(const uint64_t a[4], const uint64_t b[4], bool &result);
__device__ __forceinline__ void inc256_device(uint64_t a[4], uint64_t inc);
__device__ __forceinline__ bool ge256_u64(const uint64_t a[4], uint64_t b);
__device__ __forceinline__ void sub256_u64_inplace(uint64_t a[4], uint64_t dec);
__device__ __forceinline__ unsigned long long warp_reduce_add_ull(unsigned long long v);
__device__ __forceinline__ uint32_t load_u32_le(const uint8_t* p);
__device__ __forceinline__ bool hash160_matches_prefix_then_full(const uint8_t* h, const uint8_t* target, const uint32_t target_prefix_le);
__device__ __forceinline__ bool hash160_prefix_equals(const uint8_t* h, uint32_t target_prefix);

bool hexToLE64(const std::string& h_in, uint64_t w[4]);
bool hexToHash160(const std::string& h, uint8_t hash160[20]);
std::string formatHex256(const uint64_t limbs[4]);
std::string formatCompressedPubHex(const uint64_t Rx[4], const uint64_t Ry[4]);
std::string human_bytes(double bytes);
long double ld_from_u256(const uint64_t v[4]);

#endif // CUDA_UTILS_H
