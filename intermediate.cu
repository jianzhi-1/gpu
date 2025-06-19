#include <stdio.h>
#include <math.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result){
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__
void init(float num, float* a, int N){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    for (int i = index; i < N; i += stride){
        a[i] = num;
    }
}

__global__
void add(float* result, float* a, float* b, int N){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for (int i = index; i < N; i += stride){
        result[i] = a[i] + b[i];
    }
}

int main(){
    int deviceId;
    cudaGetDevice(&deviceId);
    
    int numberOfSMs; // streaming multiprocessors
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    const int N = 2 << 24;
    size_t size = N*sizeof(float);

    float *device_a, *device_b, *device_c, *host_c;

    cudaMalloc(&device_a, size);
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_c, size);
    cudaMallocHost(&host_c, size);

    size_t threadsPerBlock = 256;
    size_t numberOfBlocks = 32*numberOfSMs;

    cudaError_t addVectorsErr;
    cudaError_t asyncErrInit;
    cudaError_t asyncErrAdd;

    cudaStream_t stream_a, stream_b, stream_c;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    
    initWith<<<numberOfBlocks, threadsPerBlock, 0, stream_a>>>(3, device_a, N);
    initWith<<<numberOfBlocks, threadsPerBlock, 0, stream_b>>>(4, device_b, N);
    initWith<<<numberOfBlocks, threadsPerBlock, 0, stream_c>>>(0, device_c, N);

    checkCuda(cudaDeviceSynchronize());

    for (int i = 0; i < 4; i++){
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        addVectorsInto<<<numberOfBlocks/4, threadsPerBlock, 0, stream>>>(&device_c[i*N/4], &device_a[i*N/4], &device_b[i*N/4], N/4);
        cudaMemcpyAsync(&host_c[i*N/4], &device_c[i*N/4], size/4, cudaMemcpyDeviceToHost, stream);
        cudaStreamDestroy(stream);
    }

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    
    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);
    cudaStreamDestroy(stream_c);

    checkCuda(cudaFree(device_a));
    checkCuda(cudaFree(device_b));
    checkCuda(cudaFree(device_c));
    checkCuda(cudaFreeHost(host_c));
}
