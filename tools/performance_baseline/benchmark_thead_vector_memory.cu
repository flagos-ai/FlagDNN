// Thead PPU FP32 vector and streaming device-memory microbenchmarks.
// Derived from benchmark_nvidia_vector_memory.cu; uses the PPU CUDA compatibility
// compiler/runtime. No matrix or tensor path. Volatile PTX FMA preserves the work.
// Build: /usr/local/PPU_SDK/CUDA_SDK/bin/nvcc -O3 -std=c++17 -arch=sm_80 \
//          tools/performance_baseline/benchmark_thead_vector_memory.cu -o /tmp/flagdnn_thead_vector_memory
// Run:   /tmp/flagdnn_thead_vector_memory > vector_memory.csv 2> vector_memory.log
// Profile only: --profile (one FP32 FMA, copy, and triad launch plus initialization).
// All throughput uses decimal TFLOP/s or GB/s. Memory figures are logical
// bytes requested by kernels, not a claim about DRAM-controller transaction bytes.
// CUDA-event samples include kernel execution and exclude validation/transfers.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) throw std::runtime_error(std::string(#call) + ": " + \
      cudaGetErrorString(e) + " at line " + std::to_string(__LINE__)); \
} while (0)

constexpr int kChains = 8;
constexpr int kUnroll = 32;

struct FP32 {
  using T = float;
  static constexpr int lanes = 1;
  static const char* name() { return "fp32"; }
  static T mul() { return -1.0f; }
  static T add() { return 0.125f; }
  __host__ __device__ static T seed(unsigned tid, int c) {
    return 1.0f + float(c + (tid & 7)) * 0.0625f;
  }
  __device__ static __forceinline__ T step(T x, T a, T b) {
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x) : "f"(x), "f"(a), "f"(b));
    return x;
  }
};
// Eight independent register chains expose instruction throughput. Runtime a/b
// kernel arguments and volatile PTX preserve every FMA. No mma/wmma code.
// Each chain does iters * 32 instructions; FMA counts 2 FLOPs per scalar lane.
// a=-1 and b=0.125 form an exact two-cycle recurrence for these binary seeds;
// after an even instruction count the expected output is exactly the seed.
template <class Op>
__global__ void compute_fma(typename Op::T* out, int iters,
                            typename Op::T a, typename Op::T b) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  typename Op::T acc[kChains];
  #pragma unroll
  for (int c = 0; c < kChains; ++c) acc[c] = Op::seed(tid, c);
  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      #pragma unroll
      for (int c = 0; c < kChains; ++c) acc[c] = Op::step(acc[c], a, b);
    }
  }
  #pragma unroll
  for (int c = 0; c < kChains; ++c) out[size_t(tid) * kChains + c] = acc[c];
}

__host__ __device__ uint32_t mix_bits(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  return x ^ (x >> 16);
}
__host__ __device__ uint4 initial_vector(size_t i, uint32_t salt) {
  uint32_t j = uint32_t(i) * 4u + salt;
  // Finite float values in [0.5,1), with all 23 mantissa bits populated.
  return make_uint4(0x3f000000u | (mix_bits(j) & 0x7fffffu),
                    0x3f000000u | (mix_bits(j + 1) & 0x7fffffu),
                    0x3f000000u | (mix_bits(j + 2) & 0x7fffffu),
                    0x3f000000u | (mix_bits(j + 3) & 0x7fffffu));
}
__host__ __device__ uint4 write_vector(size_t i) {
  uint32_t x = uint32_t(i);
  return make_uint4(x * 1664525u + 1013904223u, x ^ 0xa5a5a5a5u,
                    x * 747796405u + 2891336453u, x ^ 0x3c6ef372u);
}
__device__ __forceinline__ uint4 load_cg(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void store_cs(uint4* p, uint4 v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
               :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__global__ void initialize_memory(uint4* a, uint4* b, uint4* c, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += size_t(gridDim.x) * blockDim.x) {
    a[i] = initial_vector(i, 0);
    b[i] = initial_vector(i, 0x13579bdu);
    c[i] = make_uint4(0, 0, 0, 0);
  }
}
__global__ void memory_copy(const uint4* a, uint4* c, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += size_t(gridDim.x) * blockDim.x) store_cs(c + i, load_cg(a + i));
}
__global__ void memory_write(uint4* c, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += size_t(gridDim.x) * blockDim.x) store_cs(c + i, write_vector(i));
}
__global__ void memory_read(const uint4* a, uint4* sink, size_t n) {
  size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint4 sum = make_uint4(0, 0, 0, 0);
  for (size_t i = tid; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint4 v = load_cg(a + i);
    sum.x ^= v.x; sum.y ^= v.y; sum.z ^= v.z; sum.w ^= v.w;
  }
  sink[tid] = sum;  // Observable output prevents removal of the read stream.
}
__global__ void memory_triad(const uint4* a, const uint4* b, uint4* c,
                             size_t n, float alpha) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint4 av = load_cg(a + i), bv = load_cg(b + i);
    uint4 cv = make_uint4(__float_as_uint(fmaf(alpha, __uint_as_float(bv.x), __uint_as_float(av.x))),
                          __float_as_uint(fmaf(alpha, __uint_as_float(bv.y), __uint_as_float(av.y))),
                          __float_as_uint(fmaf(alpha, __uint_as_float(bv.z), __uint_as_float(av.z))),
                          __float_as_uint(fmaf(alpha, __uint_as_float(bv.w), __uint_as_float(av.w))));
    store_cs(c + i, cv);
  }
}

struct Options {
  int device = 0, warmup = 20, rounds = 5, repeats = 20, iters = 256;
  int threads = 0, blocks_per_sm = 0, size_mib = 0;
  bool profile = false;
  std::string suite = "all";
};
struct DeviceBuffer {
  void* p = nullptr;
  explicit DeviceBuffer(size_t n) { CUDA_CHECK(cudaMalloc(&p, n)); }
  ~DeviceBuffer() { if (p) cudaFree(p); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  template <class T> T* as() { return static_cast<T*>(p); }
};
struct Events {
  std::vector<cudaEvent_t> begin, end;
  explicit Events(int n) : begin(n), end(n) {
    for (int i = 0; i < n; ++i) {
      CUDA_CHECK(cudaEventCreate(&begin[i])); CUDA_CHECK(cudaEventCreate(&end[i]));
    }
  }
  ~Events() { for (auto e : begin) cudaEventDestroy(e); for (auto e : end) cudaEventDestroy(e); }
};

double median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  size_t n = values.size();
  return n % 2 ? values[n / 2] : (double(values[n / 2 - 1]) + values[n / 2]) / 2;
}
void emit(const std::string& record, const std::string& resource, const std::string& dtype,
          const std::string& config, uint64_t work, int round, int repeat,
          const std::vector<float>& times, double divisor, const char* unit, bool checked) {
  double med = median(times);
  std::cout << record << ',' << resource << ',' << dtype << ',' << config << ',' << work
            << ',' << round << ',' << repeat << ',';
  if (record == "sample") std::cout << times[0];
  std::cout << ',' << med << ',' << *std::min_element(times.begin(), times.end())
            << ',' << *std::max_element(times.begin(), times.end()) << ','
            << double(work) / (med * 1e-3) / divisor << ',' << unit << ','
            << (checked ? "pass" : "pending") << '\n';
}

template <class Launch, class Validate>
void measure(const Options& o, const std::string& resource, const std::string& dtype,
             const std::string& config, uint64_t work, double divisor, const char* unit,
             Launch launch, Validate validate) {
  std::cerr << "Running " << resource << ' ' << dtype << ' ' << config << '\n';
  Events events(o.repeats);
  for (int w = 0; w < o.warmup; ++w) launch();
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> all;
  for (int r = 0; r < o.rounds; ++r) {
    for (int j = 0; j < o.repeats; ++j) {
      CUDA_CHECK(cudaEventRecord(events.begin[j]));
      launch();
      CUDA_CHECK(cudaEventRecord(events.end[j]));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(events.end.back()));
    std::vector<float> times(o.repeats);
    for (int j = 0; j < o.repeats; ++j) {
      CUDA_CHECK(cudaEventElapsedTime(&times[j], events.begin[j], events.end[j]));
      if (!(times[j] > 0)) throw std::runtime_error("Nonpositive CUDA event duration");
    }
    validate();
    for (int j = 0; j < o.repeats; ++j)
      emit("sample", resource, dtype, config, work, r, j, {times[j]}, divisor, unit, true);
    emit("round", resource, dtype, config, work, r, -1, times, divisor, unit, true);
    all.insert(all.end(), times.begin(), times.end());
  }
  emit("summary", resource, dtype, config, work, -1, -1, all, divisor, unit, true);
  std::cout.flush();
}

template <class Op>
void run_compute(const Options& o, int blocks, int threads, int bpsm) {
  using T = typename Op::T;
  size_t count = size_t(blocks) * threads * kChains;
  DeviceBuffer output(count * sizeof(T));
  std::vector<T> host(count);
  std::string config = "blocks_per_sm=" + std::to_string(bpsm) + ";threads=" +
      std::to_string(threads) + ";iters=" + std::to_string(o.iters) + ";chains=8;unroll=32";
  uint64_t work = uint64_t(count) * o.iters * kUnroll * Op::lanes * 2;
  measure(o, "vector_fp32", Op::name(), config, work, 1e12, "TFLOP/s",
      [&] { compute_fma<Op><<<blocks, threads>>>(output.as<T>(), o.iters, Op::mul(), Op::add()); },
      [&] {
        CUDA_CHECK(cudaMemcpy(host.data(), output.p, count * sizeof(T), cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < count; ++j) {
          if (host[j] != Op::seed(unsigned(j / kChains), int(j % kChains)))
            throw std::runtime_error(std::string("FMA validation failed: ") + Op::name() +
                                     " element " + std::to_string(j));
        }
      });
}

bool equal4(uint4 a, uint4 b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
float as_float(uint32_t bits) {
  float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}
uint32_t as_bits(float f) {
  uint32_t bits; std::memcpy(&bits, &f, sizeof(bits)); return bits;
}
uint4 expected_triad(size_t i) {
  uint4 a = initial_vector(i, 0), b = initial_vector(i, 0x13579bdu);
  return make_uint4(as_bits(std::fma(0.75f, as_float(b.x), as_float(a.x))),
                    as_bits(std::fma(0.75f, as_float(b.y), as_float(a.y))),
                    as_bits(std::fma(0.75f, as_float(b.z), as_float(a.z))),
                    as_bits(std::fma(0.75f, as_float(b.w), as_float(a.w))));
}

void run_memory(const Options& o, int sm_count, int size_mib) {
  size_t bytes = size_t(size_mib) << 20, n = bytes / sizeof(uint4);
  DeviceBuffer a(bytes), b(bytes), c(bytes);
  initialize_memory<<<sm_count * 8, 256>>>(a.as<uint4>(), b.as<uint4>(), c.as<uint4>(), n);
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
  auto threads_values = o.threads ? std::vector<int>{o.threads} : std::vector<int>{128, 256};
  auto bpsm_values = o.blocks_per_sm ? std::vector<int>{o.blocks_per_sm} : std::vector<int>{2, 4, 8};
  for (int threads : threads_values) for (int bpsm : bpsm_values) {
    int blocks = sm_count * bpsm;
    size_t total_threads = size_t(blocks) * threads;
    DeviceBuffer sink(total_threads * sizeof(uint4));
    std::string config = "buffer_mib=" + std::to_string(size_mib) + ";blocks_per_sm=" +
        std::to_string(bpsm) + ";threads=" + std::to_string(threads) + ";vector_bytes=16";
    for (int mode = 0; mode < 4; ++mode) {
      if (o.profile && mode != 0 && mode != 3) continue;
      const char* names[] = {"hbm_copy", "hbm_read", "hbm_write", "hbm_triad"};
      int streams[] = {2, 1, 1, 3};
      // Read work excludes the small output sink, explicitly exposed in config.
      std::string this_config = config;
      if (mode == 1) this_config += ";sink_bytes=" + std::to_string(total_threads * sizeof(uint4));
      measure(o, names[mode], mode == 3 ? "fp32" : "uint128", this_config,
          uint64_t(bytes) * streams[mode], 1e9, "GB/s",
          [&] {
            if (mode == 0) memory_copy<<<blocks, threads>>>(a.as<uint4>(), c.as<uint4>(), n);
            if (mode == 1) memory_read<<<blocks, threads>>>(a.as<uint4>(), sink.as<uint4>(), n);
            if (mode == 2) memory_write<<<blocks, threads>>>(c.as<uint4>(), n);
            if (mode == 3) memory_triad<<<blocks, threads>>>(a.as<uint4>(), b.as<uint4>(), c.as<uint4>(), n, 0.75f);
          },
          [&] {
            if (mode == 1) {
              // Check 64 complete per-thread reductions, spread across the grid.
              std::vector<uint4> result(total_threads);
              CUDA_CHECK(cudaMemcpy(result.data(), sink.p, total_threads * sizeof(uint4), cudaMemcpyDeviceToHost));
              for (size_t sample = 0; sample < 64; ++sample) {
                size_t tid = sample * (total_threads - 1) / 63;
                uint4 expected = make_uint4(0, 0, 0, 0);
                for (size_t i = tid; i < n; i += total_threads) {
                  uint4 v = initial_vector(i, 0);
                  expected.x ^= v.x; expected.y ^= v.y; expected.z ^= v.z; expected.w ^= v.w;
                }
                if (!equal4(expected, result[tid])) throw std::runtime_error("Read checksum validation failed");
              }
            } else {
              // Check contiguous windows at the head, middle, and tail (768 vectors).
              constexpr size_t window = 256;
              for (size_t offset : {size_t(0), n / 2, n - window}) {
                std::vector<uint4> result(window);
                CUDA_CHECK(cudaMemcpy(result.data(), c.as<uint4>() + offset,
                                      window * sizeof(uint4), cudaMemcpyDeviceToHost));
                for (size_t j = 0; j < window; ++j) {
                  size_t i = offset + j;
                  uint4 expected = mode == 0 ? initial_vector(i, 0) :
                                     mode == 2 ? write_vector(i) : expected_triad(i);
                  if (!equal4(expected, result[j]))
                    throw std::runtime_error(std::string(names[mode]) + " validation failed at vector " + std::to_string(i));
                }
              }
            }
          });
    }
  }
}

void usage() {
  std::cout << "Usage: benchmark_thead_vector_memory [options]\n"
      "  --device N                  CUDA device ordinal (default 0)\n"
      "  --suite all|compute|memory  Select benchmark suite\n"
      "  --warmup N --rounds N --repeats N  Defaults: 20, 5, 20\n"
      "  --iters N                   Compute loop iterations (default 256)\n"
      "  --threads N                 Default scan: 128,256\n"
      "  --blocks-per-sm N           Default scan: 2,4,8\n"
      "  --size-mib N                Default scan: 256,1024 per buffer\n"
      "  --profile                   One FP32 FMA/copy/triad launch; 256 threads,\n"
      "                              4 blocks/SM, 1024 MiB buffers, no warmup\n"
      "CSV stdout includes individual samples, round summaries, and all-sample summaries.\n"
      "stderr includes device/configuration metadata. Memory work is logical bytes;\n"
      "FMA work is scalar FLOPs (two per scalar FMA).\n"
      "Profile filter: --kernel-name 'regex:compute_fma|memory_copy|memory_triad'\n";
}

int main(int argc, char** argv) {
  try {
    Options o;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") { usage(); return 0; }
      if (arg == "--profile") { o.profile = true; continue; }
      if (i + 1 == argc) throw std::runtime_error("Missing value for " + arg);
      std::string value = argv[++i];
      if (arg == "--suite") { o.suite = value; continue; }
      size_t parsed = 0;
      int v = std::stoi(value, &parsed);
      if (parsed != value.size()) throw std::runtime_error("Invalid numeric value: " + value);
      if (arg == "--device") o.device = v;
      else if (arg == "--warmup") o.warmup = v;
      else if (arg == "--rounds") o.rounds = v;
      else if (arg == "--repeats") o.repeats = v;
      else if (arg == "--iters") o.iters = v;
      else if (arg == "--threads") o.threads = v;
      else if (arg == "--blocks-per-sm") o.blocks_per_sm = v;
      else if (arg == "--size-mib") o.size_mib = v;
      else throw std::runtime_error("Unknown option: " + arg);
    }
    if (o.profile) {
      o.warmup = 0; o.rounds = 1; o.repeats = 1;
      if (!o.threads) o.threads = 256;
      if (!o.blocks_per_sm) o.blocks_per_sm = 4;
      if (!o.size_mib) o.size_mib = 1024;
      o.suite = "all";
    }
    if (o.device < 0 || o.warmup < 0 || o.rounds < 1 || o.repeats < 1 || o.iters < 1 ||
        o.threads < 0 || o.threads > 1024 || (o.threads && o.threads % 32) ||
        o.blocks_per_sm < 0 || o.blocks_per_sm > 32 || o.size_mib < 0 ||
        (o.suite != "all" && o.suite != "compute" && o.suite != "memory"))
      throw std::runtime_error("Invalid configuration; see --help");
    CUDA_CHECK(cudaSetDevice(o.device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, o.device));
    if (std::string(prop.name).find("PPU") == std::string::npos)
      throw std::runtime_error("This collector requires a Thead PPU device");
    int runtime = 0, driver = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
    std::cerr << "device=" << o.device << " name=" << prop.name << " sm=" << prop.major << prop.minor
              << " multiprocessors=" << prop.multiProcessorCount << " L2_bytes=" << prop.l2CacheSize
              << " total_memory_bytes=" << prop.totalGlobalMem << " runtime=" << runtime << " driver=" << driver
              << " warmup=" << o.warmup << " rounds=" << o.rounds << " repeats=" << o.repeats << '\n';
    if (o.size_mib && (size_t(o.size_mib) << 20) <= size_t(prop.l2CacheSize))
      throw std::runtime_error("Memory buffer must exceed L2 for an HBM measurement");
    std::cout << std::setprecision(10)
              << "record,resource,dtype,configuration,work_per_launch,round,repeat,elapsed_ms,median_ms,min_ms,max_ms,throughput,unit,validation\n";
    auto threads_values = o.threads ? std::vector<int>{o.threads} : std::vector<int>{128, 256};
    auto bpsm_values = o.blocks_per_sm ? std::vector<int>{o.blocks_per_sm} : std::vector<int>{2, 4, 8};
    if (o.suite != "memory") {
      for (int threads : threads_values) for (int bpsm : bpsm_values) {
        int blocks = prop.multiProcessorCount * bpsm;
        run_compute<FP32>(o, blocks, threads, bpsm);
      }
    }
    if (o.suite != "compute") {
      auto sizes = o.size_mib ? std::vector<int>{o.size_mib} : std::vector<int>{256, 1024};
      for (int size : sizes) run_memory(o, prop.multiProcessorCount, size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cerr << "Completed: all compute outputs and sampled memory outputs passed validation.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
