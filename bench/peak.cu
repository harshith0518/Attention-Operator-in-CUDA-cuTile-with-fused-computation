// Register-only tensor-core microbenchmark: the ceiling both kernels are measured against.
//
// Every accumulator lives in registers, the loop contains nothing but mma.sync, and there is no memory
// traffic inside the timed region. That measures the instruction, not the kernel.
//
// Build:  nvcc -arch=sm_86 -O3 -o /tmp/peak bench/peak.cu
// Run:    /tmp/peak            (optional argument: iterations, default 20000)
//
// Expected on an RTX 3050 Laptop (sm_86, 16 SMs): about 12.3 TFLOP/s with fp32 accumulate and about
// 24.5 with fp16. The fp16 : fp32 ratio should be 0.5, which is what the hardware specifies. Both of
// this project's kernels issue the fp32 form, so their percentages use the fp32 number.

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>

#define ACCS 4        // independent accumulators per warp, to hide mma latency
#define ITERS 20000

// D += A * B, fp16 inputs, fp32 accumulate.  D is 4 floats, A is 4 x 32-bit, B is 2 x 32-bit.
__device__ __forceinline__ void mma_f32(float* d, const unsigned int* a, const unsigned int* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The fp16-accumulate form: D and C are two 32-bit registers, each holding two halves.
__device__ __forceinline__ void mma_f16(unsigned int* d, const unsigned int* a, const unsigned int* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void peak_f32(float* out, int iters) {
    unsigned int a[4] = {threadIdx.x, threadIdx.x + 1u, threadIdx.x + 2u, threadIdx.x + 3u};
    unsigned int b[2] = {threadIdx.x + 4u, threadIdx.x + 5u};
    float acc[ACCS][4];

    #pragma unroll
    for (int i = 0; i < ACCS; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int i = 0; i < ACCS; ++i) mma_f32(acc[i], a, b);
    }

    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < ACCS; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) s += acc[i][j];
    if (s == 12345.678f) out[0] = s;      // never true, but keeps the loop from being removed
}

__global__ void peak_f16(float* out, int iters) {
    unsigned int a[4] = {threadIdx.x, threadIdx.x + 1u, threadIdx.x + 2u, threadIdx.x + 3u};
    unsigned int b[2] = {threadIdx.x + 4u, threadIdx.x + 5u};
    unsigned int acc[ACCS][2];

    #pragma unroll
    for (int i = 0; i < ACCS; ++i) { acc[i][0] = 0u; acc[i][1] = 0u; }

    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int i = 0; i < ACCS; ++i) mma_f16(acc[i], a, b);
    }

    unsigned int s = 0u;
    #pragma unroll
    for (int i = 0; i < ACCS; ++i) s += acc[i][0] + acc[i][1];
    if (s == 0xdeadbeefu) out[0] = (float)s;
}

static float run(int mode, int iters, int blocks, int threads) {
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // one warm launch, then the timed one
    if (mode == 0) peak_f32<<<blocks, threads>>>(d_out, iters);
    else           peak_f16<<<blocks, threads>>>(d_out, iters);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    if (mode == 0) peak_f32<<<blocks, threads>>>(d_out, iters);
    else           peak_f16<<<blocks, threads>>>(d_out, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main(int argc, char** argv) {
    const int iters = (argc > 1) ? atoi(argv[1]) : ITERS;
    const int blocks = 16 * 4;                   // four blocks per SM on a 16-SM part
    const int threads = 128;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device %s, %d SMs, %d accumulators per warp, %d iterations\n",
           prop.name, prop.multiProcessorCount, ACCS, iters);

    // mma.m16n8k16 does 16*8*16 = 2048 multiply-accumulate operations, which is 4096 FLOP
    const double instr = (double)blocks * (threads / 32.0) * ACCS * iters;
    const double flop = instr * 4096.0;

    float ms32 = run(0, iters, blocks, threads);
    float ms16 = run(1, iters, blocks, threads);

    double t32 = flop / (ms32 * 1e-3) / 1e12;
    double t16 = flop / (ms16 * 1e-3) / 1e12;

    printf("\naccumulate type   time(ms)   TFLOP/s\n");
    printf("f32 (what the kernels use)  %8.2f  %8.2f\n", ms32, t32);
    printf("f16                         %8.2f  %8.2f\n", ms16, t16);
    printf("ratio f32/f16: %.3f  (hardware specifies 0.500)\n", t32 / t16);
    return 0;
}
