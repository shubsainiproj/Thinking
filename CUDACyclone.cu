#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include <cmath>
#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// Base58 alphabet
static const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static __device__ const char BASE58_ALPHABET_DEVICE[58] = {
    '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
};

// Unified toBase58Minikey for host and device
__host__ __device__ void toBase58Minikey(const uint64_t num_in[4], char out[32]) {
    uint64_t num[4];
    for (int i = 0; i < 4; ++i) num[i] = num_in[i];
    int pos = 30;
    out[31] = '\0';
    while (num[0] || num[1] || num[2] || num[3]) {
        uint64_t r;
#ifdef __CUDA_ARCH__
        divmod_256_by_u64_device(num, 58, num, r);
#else
        divmod_256_by_u64(num, 58, num, r);
#endif
        out[pos--] = BASE58_ALPHABET[r];
    }
    while (pos >= 1) {
        out[pos--] = '1'; // Pad with '1'
    }
    out[0] = 'S';
}

// Device-compatible SHA256 for minikey validation
__device__ void sha256_minikey(const char* minikey, int length, uint8_t out[32]) {
    uint8_t block[64] = {0};
    memcpy(block, minikey, length);
    block[length] = '?';
    block[length + 1] = 0x80;
    uint64_t bitlen = (uint64_t)(length + 1) * 8ull;
    block[63] = (uint8_t)(bitlen);
    block[62] = (uint8_t)(bitlen >> 8);
    block[61] = (uint8_t)(bitlen >> 16);
    block[60] = (uint8_t)(bitlen >> 24);
    block[59] = (uint8_t)(bitlen >> 32);
    block[58] = (uint8_t)(bitlen >> 40);
    block[57] = (uint8_t)(bitlen >> 48);
    block[56] = (uint8_t)(bitlen >> 56);

    uint32_t M[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        M[i] = ((uint32_t)block[4 * i + 0] << 24) |
               ((uint32_t)block[4 * i + 1] << 16) |
               ((uint32_t)block[4 * i + 2] << 8) |
               ((uint32_t)block[4 * i + 3]);
    }

    uint32_t W[64];
#pragma unroll
    for (int i = 0; i < 16; ++i) W[i] = M[i];
#pragma unroll
    for (int t = 16; t < 64; ++t) {
        W[t] = smallS1(W[t - 2]) + W[t - 7] + smallS0(W[t - 15]) + W[t - 16];
    }

    uint32_t state[8];
    SHA256Initialize(state);

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

#pragma unroll
    for (int t = 0; t < 64; ++t) {
        uint32_t T1 = h + bigS1(e) + Ch(e, f, g) + K[t] + W[t];
        uint32_t T2 = bigS0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[4 * i + 0] = (uint8_t)(state[i] >> 24);
        out[4 * i + 1] = (uint8_t)(state[i] >> 16);
        out[4 * i + 2] = (uint8_t)(state[i] >> 8);
        out[4 * i + 3] = (uint8_t)(state[i]);
    }
}

// CUDA kernel for minikey generation and matching
__global__ void minikeySearchKernel(
    uint64_t start[4], uint64_t stride, int length,
    uint8_t* target_hash160s, int num_targets,
    FoundResult* results, int* result_count,
    uint64_t max_iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t scalar[4];
    uint64_t offset = idx * stride;

    // Compute starting scalar for this thread
    add256_u64(start, offset, scalar);

    char minikey[32];
    uint8_t privkey[32];
    uint8_t pubkey[33];
    uint8_t hash160[20];
    uint64_t Rx[4], Ry[4];

    for (uint64_t i = 0; i < max_iterations; ++i) {
        // Generate minikey
        toBase58Minikey(scalar, minikey);

        // Validate minikey
        sha256_minikey(minikey, length, privkey);
        if (privkey[0] != 0) {
            inc256_device(scalar, stride);
            continue;
        }

        // Compute public key
        scalarMulBaseAffine(scalar, Rx, Ry);

        // Create compressed public key
        pubkey[0] = (Ry[0] & 1ULL) ? 0x03 : 0x02;
        int off = 1;
        for (int limb = 3; limb >= 0; --limb) {
            uint64_t v = Rx[limb];
            pubkey[off + 0] = (uint8_t)(v >> 56); pubkey[off + 1] = (uint8_t)(v >> 48);
            pubkey[off + 2] = (uint8_t)(v >> 40); pubkey[off + 3] = (uint8_t)(v >> 32);
            pubkey[off + 4] = (uint8_t)(v >> 24); pubkey[off + 5] = (uint8_t)(v >> 16);
            pubkey[off + 6] = (uint8_t)(v >> 8); pubkey[off + 7] = (uint8_t)(v);
            off += 8;
        }

        // Compute hash160
        getHash160_33bytes(pubkey, hash160);

        // Check against target hash160s
        for (int t = 0; t < num_targets; ++t) {
            if (compare20(hash160, target_hash160s + t * 20)) {
                int pos = atomicAdd(result_count, 1);
                if (pos < 1000) { // Safety limit to prevent buffer overflow
                    FoundResult& res = results[pos];
                    res.threadId = idx;
                    res.iter = i;
                    memcpy(res.scalar, scalar, sizeof(scalar));
                    memcpy(res.Rx, Rx, sizeof(Rx));
                    memcpy(res.Ry, Ry, sizeof(Ry));
                }
            }
        }

        // Increment scalar
        inc256_device(scalar, stride);
    }
}

// Load target hash160s from file
bool loadTargetHash160s(const std::string& filename, std::vector<uint8_t>& targets) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open input file " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (line.empty()) continue;

        uint8_t hash160[20];
        bool success = false;
        if (line[0] == '1') {
            // Try Base58 P2PKH address
            success = decode_p2pkh_address(line, hash160);
        } else {
            // Try hex hash160
            success = hexToHash160(line, hash160);
        }
        if (success) {
            targets.insert(targets.end(), hash160, hash160 + 20);
        } else {
            std::cerr << "Warning: Invalid hash160 or address in line: " << line << std::endl;
        }
    }
    file.close();
    return true;
}

// Save found result to file
void saveFoundResult(const FoundResult& res, int length) {
    std::ofstream file("CASAFOUND.txt", std::ios::app);
    char minikey[32];
    toBase58Minikey(res.scalar, minikey);
    std::string pubkeyHex = formatCompressedPubHex(res.Rx, res.Ry);
    file << "Minikey: " << minikey << "\n";
    file << "Public Key: " << pubkeyHex << "\n";
    file << "Private Key: ";
    uint8_t privkey[32];
    host_sha256::sha256((uint8_t*)minikey, length, privkey);
    for (int i = 0; i < 32; ++i) {
        file << std::hex << std::setw(2) << std::setfill('0') << (int)privkey[i];
    }
    file << "\n";
    file << "Hash160: ";
    uint8_t hash160[20];
    getHash160_33_from_limbs_host((res.Ry[0] & 1ULL) ? 0x03 : 0x02, res.Rx, hash160);
    for (int i = 0; i < 20; ++i) {
        file << std::hex << std::setw(2) << std::setfill('0') << (int)hash160[i];
    }
    file << "\n\n";
    file.close();
}

int main(int argc, char* argv[]) {
    std::string input_file, range_start, range_end;
    int minikey_length = 0;
    int threads_per_block = 256;
    int batches_per_sm = 8;
    bool verbose = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "--range" && i + 1 < argc) {
            std::string range = argv[++i];
            size_t colon = range.find(':');
            if (colon != std::string::npos) {
                range_start = range.substr(0, colon);
                range_end = range.substr(colon + 1);
            }
        } else if (arg == "--length" && i + 1 < argc) {
            minikey_length = std::stoi(argv[++i]);
        } else if (arg == "--grid" && i + 1 < argc) {
            std::string grid = argv[++i];
            size_t comma = grid.find(',');
            if (comma != std::string::npos) {
                threads_per_block = std::stoi(grid.substr(0, comma));
                batches_per_sm = std::stoi(grid.substr(comma + 1));
            }
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --input <file>            Target Bitcoin address pubkeys file\n"
                      << "  --range <start>:<end>     Mini key range to search\n"
                      << "  --length <22|23|26|30>    Mini key length\n"
                      << "  --grid <A,B>              Threads per block and batches per SM\n"
                      << "  --verbose                 Enable detailed logging\n"
                      << "  --help, -h                Show this help message\n"
                      << "Examples:\n"
                      << "  " << argv[0] << " --input targets.txt --length 30 --range S000:Szzz\n";
            return 0;
        }
    }

    // Validate arguments
    if (input_file.empty()) {
        std::cerr << "Error: --input file is required\n";
        return 1;
    }
    if (minikey_length != 22 && minikey_length != 23 && minikey_length != 26 && minikey_length != 30) {
        std::cerr << "Error: Invalid minikey length. Must be 22, 23, 26, or 30\n";
        return 1;
    }
    if (range_start.empty() || range_end.empty()) {
        std::cerr << "Error: --range <start>:<end> is required\n";
        return 1;
    }

    // Validate range
    if (range_start[0] != 'S' || range_end[0] != 'S' || range_start.size() != minikey_length || range_end.size() != minikey_length) {
        std::cerr << "Error: Invalid range. Must start with 'S' and match the specified length\n";
        return 1;
    }

    // Load target hash160s
    std::vector<uint8_t> target_hash160s;
    if (!loadTargetHash160s(input_file, target_hash160s)) {
        return 1;
    }
    int num_targets = target_hash160s.size() / 20;
    if (num_targets == 0) {
        std::cerr << "Error: No valid hash160s found in input file\n";
        return 1;
    }

    // Convert range to 256-bit numbers
    uint64_t start_scalar[4] = {0}, end_scalar[4] = {0};
    std::vector<uint8_t> start_bytes, end_bytes;
    if (!base58_decode(range_start, start_bytes) || !base58_decode(range_end, end_bytes)) {
        std::cerr << "Error: Invalid Base58 range\n";
        return 1;
    }
    if (start_bytes.size() > 32 || end_bytes.size() > 32) {
        std::cerr << "Error: Range values too large\n";
        return 1;
    }
    for (size_t i = 0; i < start_bytes.size(); ++i) {
        start_scalar[i / 8] |= (uint64_t)start_bytes[start_bytes.size() - 1 - i] << ((i % 8) * 8);
    }
    for (size_t i = 0; i < end_bytes.size(); ++i) {
        end_scalar[i / 8] |= (uint64_t)end_bytes[end_bytes.size() - 1 - i] << ((i % 8) * 8);
    }

    // Initialize CUDA
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int blocks = sm_count * batches_per_sm;
    int total_threads = blocks * threads_per_block;
    uint64_t stride = total_threads;

    // Calculate total keys
    uint64_t total_keys[4];
    sub256(end_scalar, start_scalar, total_keys);
    long double total_keys_ld = ld_from_u256(total_keys);

    // Allocate device memory
    uint8_t* d_target_hash160s;
    FoundResult* d_results;
    int* d_result_count;
    CUDA_CHECK(cudaMalloc(&d_target_hash160s, num_targets * 20 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_results, total_threads * sizeof(FoundResult)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_target_hash160s, target_hash160s.data(), num_targets * 20 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_result_count, 0, sizeof(int)));

    // Print GPU information
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    double mem_utilization = (double)(total_mem - free_mem) / total_mem * 100.0;
    std::cout << "===== PrePhase: GPU Information ====================\n"
              << "Device               : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n"
              << "SM                   : " << sm_count << "\n"
              << "ThreadsPerBlock      : " << threads_per_block << "\n"
              << "Blocks               : " << blocks << "\n"
              << "Points batch size    : " << threads_per_block << "\n"
              << "Batches/SM           : " << batches_per_sm << "\n"
              << "Memory utilization   : " << std::fixed << std::setprecision(1) << mem_utilization
              << "% (" << human_bytes(total_mem - free_mem) << " / " << human_bytes(total_mem) << ")\n"
              << "---------------------------------------------------\n"
              << "Total threads        : " << total_threads << "\n\n"
              << "======== Phase-1: Brute Force Mini Keys ===============\n"
              << "Mini Key Length : " << minikey_length << "\n"
              << "Range Start     : " << range_start << "\n"
              << "Range End       : " << range_end << "\n"
              << "Total Keys      : " << std::fixed << std::setprecision(0) << total_keys_ld << "\n";

    // Progress reporting
    uint64_t current_scalar[4];
    memcpy(current_scalar, start_scalar, sizeof(start_scalar));
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t keys_processed = 0;
    const uint64_t batch_size = total_threads * 1000; // Process 1000 keys per thread per batch
    bool done;

    ge256(current_scalar, end_scalar, done);
    while (!done) {
        // Launch kernel
        minikeySearchKernel<<<blocks, threads_per_block>>>(current_scalar, stride, minikey_length,
                                                          d_target_hash160s, num_targets,
                                                          d_results, d_result_count, 1000);

        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back
        int h_result_count;
        std::vector<FoundResult> h_results(total_threads);
        CUDA_CHECK(cudaMemcpy(&h_result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_result_count > 0) {
            CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, h_result_count * sizeof(FoundResult), cudaMemcpyDeviceToHost));
            for (int i = 0; i < h_result_count; ++i) {
                saveFoundResult(h_results[i], minikey_length);
                if (verbose) {
                    char minikey[32];
                    toBase58Minikey(h_results[i].scalar, minikey);
                    std::cout << "Found match: Minikey = " << minikey << ", Pubkey = "
                              << formatCompressedPubHex(h_results[i].Rx, h_results[i].Ry) << "\n";
                }
            }
            CUDA_CHECK(cudaMemset(d_result_count, 0, sizeof(int)));
        }

        // Update progress
        keys_processed += batch_size;
        inc256(current_scalar, batch_size);
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double speed = keys_processed / elapsed / 1e6; // Mkeys/s
        double progress = keys_processed / total_keys_ld * 100.0;
        double eta = (total_keys_ld - keys_processed) / (speed * 1e6);

        char current_minikey[32];
        toBase58Minikey(current_scalar, current_minikey);

        std::cout << "Time: " << std::fixed << std::setprecision(1) << elapsed
                  << " s | Speed: " << speed << " Mkeys/s | Count: " << keys_processed
                  << " | Progress: " << progress << " % | ETA: " << eta / 3600.0
                  << " h | Current: " << current_minikey << "\r" << std::flush;

        ge256(current_scalar, end_scalar, done);
    }

    std::cout << "\nSearch completed.\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_target_hash160s));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_result_count));

    return 0;
}
