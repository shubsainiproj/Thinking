#ifndef CUDA_HASH_CUH
#define CUDA_HASH_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <cstring>

struct MatchResult {
    int found;           
    uint8_t publicKey[33];
    uint8_t sha256[32];
    uint8_t ripemd160[20];
};

__device__ __forceinline__ uint32_t ror32(uint32_t x, int n);
__device__ __forceinline__ uint32_t bigS0(uint32_t x);
__device__ __forceinline__ uint32_t bigS1(uint32_t x);
__device__ __forceinline__ uint32_t smallS0(uint32_t x);
__device__ __forceinline__ uint32_t smallS1(uint32_t x);
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z);
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z);
__device__ __forceinline__ void SHA256Initialize(uint32_t s[8]);
__device__ __forceinline__ void SHA256Transform(uint32_t state[8], uint32_t W_in[64]);
__device__ __forceinline__ void RIPEMD160Initialize(uint32_t s[5]);
__device__ __forceinline__ void RIPEMD160Transform(uint32_t s[5], uint32_t* w);
__device__ __forceinline__ bool compare20(const uint8_t* h, const uint8_t* ref);
__device__ __forceinline__ uint32_t bswap32(uint32_t x);
__device__ __forceinline__ uint32_t pack_be4(uint8_t a, uint8_t b, uint8_t c, uint8_t d);
__device__ void getSHA256_33bytes(const uint8_t* pubkey33, uint8_t sha[32]);
__device__ void getRIPEMD160_32bytes(const uint8_t* sha, uint8_t ripemd[20]);
__device__ void getHash160_33bytes(const uint8_t* pubkey33, uint8_t* hash20);
__device__ void addBigEndian32(uint8_t* data32, uint64_t offset);
__device__ void RIPEMD160_from_SHA256_state(const uint32_t sha_state_be[8], uint8_t ripemd20[20]);
__device__ void getHash160_33_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4], uint8_t out20[20]);
__host__ void getHash160_33_from_limbs_host(uint8_t prefix02_03, const uint64_t x_be_limbs[4], uint8_t out20[20]);

// SHA256 round constants
__device__ extern const uint32_t K[64];

#endif
