// matmul_naive_with_verification.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define EPSILON 1e-3f // tolerance for float comparison
// Update this path to your folder with .bin files
const char* basePath = "E:/ME_Computer/Semester-2/CMPE-214GPU/Assingnment-2/optmized_Matrix/";

// CUDA kernel: Naive matrix multiplication
__global__ void matmul_naive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Load matrix from binary float file
void load_matrix(const char *fname, float *matrix, int N) {
    FILE *f = fopen(fname, "rb");
    if (!f) { printf("Cannot open %s\n", fname); exit(1); }
    size_t got = fread(matrix, sizeof(float), N*N, f);
    if (got != N*N) { printf("File size mismatch in %s\n", fname); exit(1); }
    fclose(f);
}

// Verify computed result vs. reference CPU output
bool verify(const float *ref, const float *test, int N) {
    for (int i = 0; i < N*N; ++i) {
        if (fabs(ref[i] - test[i]) > EPSILON) return false;
    }
    return true;
}

int main() {
    int sizes[] = {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    printf("Size,Time(s),Verification\n");

    for (int idx = 0; idx < num_sizes; ++idx) {
        int N = sizes[idx];
        size_t bytes = N*N*sizeof(float);
        float *hA = (float*)malloc(bytes);
        float *hB = (float*)malloc(bytes);
        float *hC = (float*)malloc(bytes);
        float *hRef = (float*)malloc(bytes);

        // Prepend basePath to file names
        char fileA[512], fileB[512], fileP[512];
        sprintf(fileA, "%sM_%d.bin", basePath, N);
        sprintf(fileB, "%sN_%d.bin", basePath, N);
        sprintf(fileP, "%sP_%d.bin", basePath, N);
        load_matrix(fileA, hA, N);
        load_matrix(fileB, hB, N);
        load_matrix(fileP, hRef, N);

        float *dA, *dB, *dC;
        cudaMalloc((void**)&dA, bytes);
        cudaMalloc((void**)&dB, bytes);
        cudaMalloc((void**)&dC, bytes);
        cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16);
        dim3 gridDim((N+15)/16, (N+15)/16);
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        matmul_naive<<<gridDim, blockDim>>>(dA, dB, dC, N);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

        bool ok = verify(hRef, hC, N);
        printf("%d,%.6f,%s\n", N, elapsed.count(), ok ? "Success" : "Failed");

        free(hA); free(hB); free(hC); free(hRef);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    return 0;
}
