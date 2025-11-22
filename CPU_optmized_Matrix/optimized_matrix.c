#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Function prototypes
int** create_matrix(int size);
void free_matrix(int **mat, int size);
void save_matrix(const char *fname, int **mat, int size);
void multiply_tiled(int **A, int **B, int **C, int size);
double get_wall_time();

// Get high-precision time
double get_wall_time() {
#ifdef _WIN32
    LARGE_INTEGER time, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&time);
    return (double)time.QuadPart / freq.QuadPart;
#else
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
#endif
}

// Allocate size x size 2D matrix
int** create_matrix(int size) {
    int **mat = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; i++) {
        mat[i] = (int*)malloc(size * sizeof(int));
    }
    return mat;
}

// Free size x size 2D matrix
void free_matrix(int **mat, int size) {
    for (int i = 0; i < size; i++) free(mat[i]);
    free(mat);
}

// Save matrix to binary file
void save_matrix(const char *fname, int **mat, int size) {
    FILE *f = fopen(fname, "wb");
    if (!f) { printf("Cannot open %s\n", fname); exit(1); }
    for (int i = 0; i < size; i++) {
        fwrite(mat[i], sizeof(int), size, f);
    }
    fclose(f);
}

// Blocked matrix multiplication
#define BLOCK_SIZE 64
void multiply_tiled(int **A, int **B, int **C, int size) {
    for (int i0 = 0; i0 < size; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < size; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < size; k0 += BLOCK_SIZE) {
                for (int i = i0; i < i0 + BLOCK_SIZE && i < size; i++) {
                    for (int j = j0; j < j0 + BLOCK_SIZE && j < size; j++) {
                        int sum = (k0 == 0) ? 0 : C[i][j];
                        for (int k = k0; k < k0 + BLOCK_SIZE && k < size; k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
}

int main() {
    srand((unsigned int)time(NULL));
    int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("MatrixSize,CPU_TiledTimeSeconds\n");
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        int **M = create_matrix(size);
        int **N = create_matrix(size);
        int **P = create_matrix(size);
        // Fill matrices with random integer (0–9)
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                M[i][j] = rand() % 10;
                N[i][j] = rand() % 10;
            }
        }
        double t0 = get_wall_time();
        multiply_tiled(M, N, P, size);
        double t1 = get_wall_time();
        printf("%d,%lf\n", size, t1 - t0);
        // Save matrices for GPU verification
        char fn_m[32], fn_n[32], fn_p[32];
        sprintf(fn_m, "M_%d.bin", size);
        sprintf(fn_n, "N_%d.bin", size);
        sprintf(fn_p, "P_%d.bin", size);
        save_matrix(fn_m, M, size);
        save_matrix(fn_n, N, size);
        save_matrix(fn_p, P, size);
        free_matrix(M, size);
        free_matrix(N, size);
        free_matrix(P, size);
    }
    return 0;
}
