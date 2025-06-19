#include <stdio.h>
#include <math.h>
#include <assert.h>

int main(){
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
}

