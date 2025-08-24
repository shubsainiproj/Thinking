#include "CUDAUtils.h"
#include <sstream>
#include <iomanip>
#include <cmath>

__host__ void divmod_256_by_u64(const uint64_t value[4], uint64_t divisor, uint64_t quotient[4], uint64_t &remainder) {
    remainder = 0;
    for (int i = 3; i >= 0; --i) {
        __uint128_t cur = (__uint128_t(remainder) << 64) | value[i];
        quotient[i] = (uint64_t)(cur / divisor);
        remainder = (uint64_t)(cur % divisor);
    }
}

__device__ __forceinline__ void divmod_256_by_u64_device(const uint64_t value[4], uint64_t divisor, uint64_t quotient[4], uint64_t &remainder) {
    remainder = 0;
    for (int i = 3; i >= 0; --i) {
        unsigned __int128 cur = ((unsigned __int128)remainder << 64) | value[i];
        quotient[i] = (uint64_t)(cur / divisor);
        remainder = (uint64_t)(cur % divisor);
    }
}

__device__ __forceinline__ void ge256(const uint64_t a[4], const uint64_t b[4], bool &result) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] > b[i]) {
            result = true;
            return;
        }
        if (a[i] < b[i]) {
            result = false;
            return;
        }
    }
    result = true; // Equal case
}

bool hexToLE64(const std::string& h_in, uint64_t w[4]) {
    std::string h = h_in;
    if (h.size() >= 2 && (h[0] == '0') && (h[1] == 'x' || h[1] == 'X')) h = h.substr(2);
    if (h.size() > 64) return false;
    if (h.size() < 64) h = std::string(64 - h.size(), '0') + h;
    if (h.size() != 64) return false;
    for (int i = 0; i < 4; ++i) {
        std::string part = h.substr(i * 16, 16);
        w[3 - i] = std::stoull(part, nullptr, 16);
    }
    return true;
}

bool hexToHash160(const std::string& h, uint8_t hash160[20]) {
    if (h.size() != 40) return false;
    for (int i = 0; i < 20; ++i) {
        std::string byteStr = h.substr(i * 2, 2);
        hash160[i] = (uint8_t)std::stoul(byteStr, nullptr, 16);
    }
    return true;
}

std::string formatHex256(const uint64_t limbs[4]) {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    for (int i = 3; i >= 0; --i) {
        oss << std::setw(16) << limbs[i];
    }
    return oss.str();
}

std::string human_bytes(double bytes) {
    static const char* u[]={"B","KB","MB","GB","TB","PB"};
    int k=0;
    while(bytes>=1024.0 && k<5){ bytes/=1024.0; ++k; }
    std::ostringstream o; o.setf(std::ios::fixed); o<<std::setprecision(bytes<10?2:1)<<bytes<<" "<<u[k];
    return o.str();
}

std::string formatCompressedPubHex(const uint64_t Rx[4], const uint64_t Ry[4]) {
    uint8_t out[33];
    out[0] = (Ry[0] & 1ULL) ? 0x03 : 0x02;
    int off=1;
    for (int limb=3; limb>=0; --limb) {
        uint64_t v = Rx[limb];
        out[off+0]=(uint8_t)(v>>56); out[off+1]=(uint8_t)(v>>48);
        out[off+2]=(uint8_t)(v>>40); out[off+3]=(uint8_t)(v>>32);
        out[off+4]=(uint8_t)(v>>24); out[off+5]=(uint8_t)(v>>16);
        out[off+6]=(uint8_t)(v>> 8); out[off+7]=(uint8_t)(v>> 0);
        off+=8;
    }
    static const char* hexd="0123456789ABCDEF";
    std::string s; s.resize(66);
    for (int i=0;i<33;++i){ s[2*i]=hexd[(out[i]>>4)&0xF]; s[2*i+1]=hexd[out[i]&0xF]; }
    return s;
}

long double ld_from_u256(const uint64_t v[4]) {
    return std::ldexp((long double)v[3],192) + std::ldexp((long double)v[2],128) + std::ldexp((long double)v[1],64) + (long double)v[0];
}
