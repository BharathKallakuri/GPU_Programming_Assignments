#include <iostream>
#include <cuda_runtime.h>

__device__ void globalBarrier(int *d_barrier_count, int num_blocks)
{
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(d_barrier_count, 1);
        printf("Block: %d\n", blockIdx.x);
        while (atomicAdd(d_barrier_count, 0) < num_blocks)
        {
            // spin
        }
    }
    __syncthreads();
}

__global__ void testKernel(int *d_barrier_count, int *d_data)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    d_data[global_tid] = global_tid;
    globalBarrier(d_barrier_count, gridDim.x);
    int last_tid = (gridDim.x * blockDim.x) - 1;

    if (global_tid == 0)
    {
        printf("Thread 0 successfully read data from thread %d : %d\n", last_tid, d_data[last_tid]);
    }
}

int main()
{
    int num_blocks = 4;
    int threads_per_block = 256;
    int total_threads = num_blocks * threads_per_block;

    int *d_barrier_count;
    int *d_data;

    cudaMalloc(&d_barrier_count, sizeof(int));
    cudaMemset(d_barrier_count, 0, sizeof(int));
    cudaMalloc(&d_data, total_threads * sizeof(int));

    std::cout << "Kernel with Blocks: " << num_blocks
              << " Threads: " << threads_per_block << std::endl;

    testKernel<<<num_blocks, threads_per_block>>>(d_barrier_count, d_data);

    cudaDeviceSynchronize();
    cudaFree(d_barrier_count);
    cudaFree(d_data);

    return 0;
}