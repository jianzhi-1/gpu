#include <stdio.h>
#include <math.h>
#include <assert.h> // for checkCuda

#define N  64

inline cudaError_t checkCuda(cudaError_t result){
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__
void matMul(int* a, int* b, int* c){
    int val{0};
    int row = blockIdx.x;
    int col = threadIdx.x;
    for (int k = 0; k < N; k++){
        val += a[row*N + k]*b[k*N + col];
    }
    c[row*N + col] = val;
}

int main(){
    int *a, *b, *c;

    int size = N * N * sizeof (int);

    checkCuda(cudaMallocManaged(&a, size));
    checkCuda(cudaMallocManaged(&b, size));
    checkCuda(cudaMallocManaged(&c, size));

  // Initialize memory; create 2D matrices
    for(int row = 0; row < N; row++){
        for(int col = 0; col < N; col++){
            a[row*N + col] = row;
            b[row*N + col] = col+2;
            c[row*N + col] = 0;
        }
    }

    dim3 threads_per_block {64, 1, 1};
    dim3 number_of_blocks {64, 1, 1};

    matMul<<<number_of_blocks, threads_per_block>>>(a, b, c);

    checkCuda(cudaGetLastError()); // check this
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaFree(a));
    checkCuda(cudaFree(b));
    checkCuda(cudaFree(c));
}
