#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define TILE_SIZE 32 // Safe max for your GPU
#define EPSILON 1e-3f // tolerance for float comparison

const char* basePath = "E:/ME_Computer/Semester-2/CMPE-214GPU/Assingnment-2/optmized_Matrix/";

__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // load tile for A
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        // load tile for B
        if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < N && col < N)
        C[row * N + col] = sum;
}

void load_matrix(const char *fname, float *matrix, int N) {
    FILE *f = fopen(fname, "rb");
    if (!f) { printf("Cannot open %s\n", fname); exit(1); }
    size_t got = fread(matrix, sizeof(float), N*N, f);
    if (got != N*N) { printf("File size mismatch in %s\n", fname); exit(1); }
    fclose(f);
}

bool verify(const float *ref, const float *test, int N) {
    for (int i = 0; i < N*N; ++i) {
        if (fabs(ref[i] - test[i]) > EPSILON) return false;
    }
    return true;
}

void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
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

        char fileA[512], fileB[512], fileP[512];
        sprintf(fileA, "%sM_%d.bin", basePath, N);
        sprintf(fileB, "%sN_%d.bin", basePath, N);
        sprintf(fileP, "%sP_%d.bin", basePath, N);
        load_matrix(fileA, hA, N);
        load_matrix(fileB, hB, N);
        load_matrix(fileP, hRef, N);

        float *dA, *dB, *dC;
        cudaMalloc((void**)&dA, bytes); check_cuda_error("cudaMalloc dA");
        cudaMalloc((void**)&dB, bytes); check_cuda_error("cudaMalloc dB");
        cudaMalloc((void**)&dC, bytes); check_cuda_error("cudaMalloc dC");
        cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice); check_cuda_error("cudaMemcpy hA->dA");
        cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice); check_cuda_error("cudaMemcpy hB->dB");

        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);

        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        matmul_tiled<<<gridDim, blockDim>>>(dA, dB, dC, N);
        check_cuda_error("matmul_tiled kernel launch");
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost); check_cuda_error("cudaMemcpy dC->hC");

        bool ok = verify(hRef, hC, N);
        printf("%d,%.6f,%s\n", N, elapsed.count(), ok ? "Success" : "Failed");

        free(hA); free(hB); free(hC); free(hRef);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    return 0;
}
