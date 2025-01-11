#include <stdio.h>
#include <cuda.h>
#include <float.h>
#include "utility_functions.h" // Include utility functions for readability

// Kernel to calculate maximum and sum of the matrix
__global__ void computeMaxAndSum(const int *A, int *max_result, int *sum_result, int N) {
    extern __shared__ int sharedData[];
    int *sharedMax = sharedData;   // Shared memory for max
    float *sharedSum = (float*)&sharedData[blockDim.x]; // Shared memory for sum

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    // Initialize local values
    int localMax = INT_MIN;
    float localSum = 0.0f;

    // Process elements in chunks
    while (tid < N) {
        int val = A[tid];
        localMax = findMax(localMax, val); // Calculate local max
        localSum += val;  // Calculate local sum
        tid += blockDim.x * gridDim.x; // Stride to next portion
    }

    // Store local results in shared memory
    sharedMax[cacheIndex] = localMax;
    sharedSum[cacheIndex] = localSum;

    __syncthreads();

    // Perform reduction for max and sum in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (cacheIndex < stride) {
            sharedMax[cacheIndex] = findMax(sharedMax[cacheIndex], sharedMax[cacheIndex + stride]);
            sharedSum[cacheIndex] += sharedSum[cacheIndex + stride];
        }
        __syncthreads();
    }

    // First thread writes the block's result to the global memory
    if (cacheIndex == 0) {
        atomicMax(max_result, sharedMax[0]);  // Atomic update for max
        atomicAdd(sum_result, sharedSum[0]); // Atomic update for sum
    }
}

// Kernel to compute matrix B
__global__ void computeMatrixB(const int *A, float *B, float m, int amax, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N * N) {
        B[tid] = (m - A[tid]) / amax;
        tid += blockDim.x * gridDim.x; 
    }
}
__global__ void findMinInMatrixB(float *B, float *minValue, int N) {
    extern __shared__ float sharedMin[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int blockTid = threadIdx.x;
  // Initialize local minimum with a large value
    float localMin = FLT_MAX;

    // Process elements in chunks using a strided loop
    for (int idx = tid; idx < N * N; idx += blockDim.x * gridDim.x) {
        localMin = findMinFloat(localMin, B[idx]); // Update local minimum
    }

    // Store the local minimum in shared memory
    sharedMin[blockTid] = localMin;
    __syncthreads();

    // Perform binary tree reduction to find the minimum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (blockTid < stride) {
            // Compare and store the minimum in shared memory
            sharedMin[blockTid] = findMinFloat(sharedMin[blockTid], sharedMin[blockTid + stride]);
        }
        __syncthreads(); // Ensure all threads have updated the shared memory
    }

    // Write the minimum value from each block to global memory
    if (blockTid == 0) {
        minValue[blockIdx.x] = sharedMin[0];
    }
}
__global__ void computeMatrixC(const int *A, float *C, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N * N) {
        int row = tid / N;
        int col = tid % N;

        int left = col == 0 ? A[row * N + (N - 1)] : A[row * N + (col - 1)];
        int right = col == (N - 1) ? A[row * N] : A[row * N + (col + 1)];

        C[tid] = (A[row * N + col] + left + right) / 3.0f;
        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <ThreadsPerBlock>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Input validation
    int N = ValidateInput(argv[1], 1, 30 * 1024, "N must be between 1 and 30 * 1024.");
    int ThreadsPerBlock = ValidateInput(argv[2], 1, 1024, "ThreadsPerBlock must be between 1 and 1024.");
    if (!isMultipleOf2(ThreadsPerBlock))
    {
        fprintf(stderr, "Threads Per Block must be a multiple of 2 for binary tree reductions");
        return EXIT_FAILURE;
    } 
    int matrixSize = N * N;
    // TOTAL THREADS == MATRIX SIZE (N^2) 
    int BlocksPerGrid = (matrixSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
   
    // Host memory
    int *h_A = createIntMatrix(matrixSize);
    initializeArray(h_A, matrixSize);
    printf("Input Matrix A:\n");
    // PRINT MATRIX IF IT IS SMALL...
    if (N < 10) print2D(h_A, N, N, stdout, 'i');

    // Device memory
    int *d_A, *d_max, *d_sum;
    HANDLE_ERROR(cudaMalloc((void **)&d_A, matrixSize * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&d_max, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&d_sum, sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_A, h_A, matrixSize * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize max and sum on the device
    int initialMax = INT_MIN;
    int initialSum = 0;
    HANDLE_ERROR(cudaMemcpy(d_max, &initialMax, sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_sum, &initialSum, sizeof(int), cudaMemcpyHostToDevice));

    // Shared memory size for reduction
    int sharedMemorySize = ThreadsPerBlock * (sizeof(int) + sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // Start timing for max and sum kernel
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // Kernel: Compute max and sum
    computeMaxAndSum<<<BlocksPerGrid, ThreadsPerBlock, sharedMemorySize>>>(d_A, d_max, d_sum, matrixSize);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Stop timing for max and sum kernel
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float maxSumTime;
    HANDLE_ERROR(cudaEventElapsedTime(&maxSumTime, start, stop));
    printf("Time for computeMaxAndSum kernel: %.3f ms\n", maxSumTime);

    // Retrieve results from device
    int amax;
    int totalSum;
    HANDLE_ERROR(cudaMemcpy(&amax, d_max, sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&totalSum, d_sum, sizeof(int), cudaMemcpyDeviceToHost));

    float m = totalSum / (float)matrixSize;
    printf("Matrix Mean (m): %.2f, Max Element (amax): %d\n", m, amax);

    if (amax > N * m) {
        float *d_B;
        HANDLE_ERROR(cudaMalloc((void **)&d_B, matrixSize * sizeof(float)));

        // Start timing for computeMatrixB kernel
        HANDLE_ERROR(cudaEventRecord(start, 0));

        // Compute Matrix B
        computeMatrixB<<<BlocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, m, amax, N);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Stop timing for computeMatrixB kernel
        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float matrixBTime;
        HANDLE_ERROR(cudaEventElapsedTime(&matrixBTime, start, stop));
        printf("Time for computeMatrixB kernel: %.3f ms\n", matrixBTime);

        // Find minimum in Matrix B using binary tree reduction
        float *d_minValues;
        HANDLE_ERROR(cudaMalloc((void **)&d_minValues, BlocksPerGrid * sizeof(float)));
        
        // Start timing for findMinInMatrixB kernel
        HANDLE_ERROR(cudaEventRecord(start, 0));

        findMinInMatrixB<<<BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(float)>>>(d_B, d_minValues, N);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Stop timing for findMinInMatrixB kernel
        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float minBTime;
        HANDLE_ERROR(cudaEventElapsedTime(&minBTime, start, stop));
        printf("Time for findMinInMatrix B kernel: %.3f ms\n", minBTime);

        // Host reduction for global minimum
        float *h_minValues = createFloatMatrix(BlocksPerGrid);
        HANDLE_ERROR(cudaMemcpy(h_minValues, d_minValues, BlocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

        //Compute Global minimum on Host.
        float globalMin = h_minValues[0];
        for (int i = 1; i < BlocksPerGrid; i++) {
            globalMin = findMinFloatHost(globalMin, h_minValues[i]);
        }

        printf("Global Minimum of Matrix B: %.2f\n", globalMin);

        // Retrieve Matrix B for printing
        float *h_B = createFloatMatrix(matrixSize);
        HANDLE_ERROR(cudaMemcpy(h_B, d_B, matrixSize * sizeof(float), cudaMemcpyDeviceToHost));

        // PRINT MATRIX IF IT IS SMALL...
        if (N < 10)
        {
            printf("Matrix B:\n");
            print2D(h_B, N, N, stdout, 'f');
        }

        // Cleanup for B-related resources
        free(h_B);
        free(h_minValues);
        HANDLE_ERROR(cudaFree(d_B));
        HANDLE_ERROR(cudaFree(d_minValues));
    } else {
        float *d_C;
        HANDLE_ERROR(cudaMalloc((void **)&d_C, matrixSize * sizeof(float)));

        // Start timing for computeMatrixC kernel
        HANDLE_ERROR(cudaEventRecord(start, 0));

        // Compute Matrix C
        computeMatrixC<<<BlocksPerGrid, ThreadsPerBlock, ThreadsPerBlock * sizeof(int)>>>(d_A, d_C, N);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Stop timing for computeMatrixC kernel
        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float matrixCTime;
        HANDLE_ERROR(cudaEventElapsedTime(&matrixCTime, start, stop));
        printf("Time for computeMatrixC kernel: %.3f ms\n", matrixCTime);

        // Retrieve Matrix C for printing
        float *h_C = createFloatMatrix(matrixSize);
        HANDLE_ERROR(cudaMemcpy(h_C, d_C, matrixSize * sizeof(float), cudaMemcpyDeviceToHost));
        // PRINT MATRIX IF IT IS SMALL...
        if (N < 10)
        {
            printf("Matrix C:\n");
            print2D(h_C, N, N, stdout, 'f');
        }
        // Cleanup for C-related resources
        free(h_C);
        HANDLE_ERROR(cudaFree(d_C));
    }

    // Cleanup
    free(h_A);
    HANDLE_ERROR(cudaFree(d_A));
    HANDLE_ERROR(cudaFree(d_max));
    HANDLE_ERROR(cudaFree(d_sum));

    // Destroy CUDA events
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return 0;
}
