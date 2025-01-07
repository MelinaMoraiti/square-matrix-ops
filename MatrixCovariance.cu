#include <stdio.h>
#include <cuda.h>
#include <float.h>
#include "utility_functions.h" 
#define blockSide 32

__global__ void computeAndSubtractAverages(const int *A, float *colAverages, float* newA, int N) {
    // Declare dynamically allocated shared memory for matrix block (each thread loads a matrix element into shared memory)
    __shared__ float sharedMem[blockSide][blockSide];

    // Calculate global row and column indices
    int globalCol = blockIdx.x * blockSide + threadIdx.x;
    int globalRow = blockIdx.y * blockSide + threadIdx.y;
    // Local row and column indices within the block
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;

    // Load matrix element into shared memory (only load if within matrix bounds)
    if (globalRow < N && globalCol < N) {
        sharedMem[localRow][localCol] = A[globalRow * N + globalCol];
    }
    __syncthreads();

    for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) 
    {
        if (localRow < stride) {
            sharedMem[localRow][localCol] += sharedMem[localRow + stride][localCol];
        }
        __syncthreads();
    }
    
    if (localRow == 0 && globalCol < N){
		//colSums[globalCol] = sharedMem[0][localCol];
        colAverages[globalCol] = (float)sharedMem[0][localCol]/N;
        //float average = colAverages[globalCol];
	}
    __syncthreads();

    if (globalRow < N && globalCol < N) {
        newA[globalRow * N + globalCol] = A[globalRow * N + globalCol] - colAverages[globalCol];
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
    if (!isMultipleOf2(ThreadsPerBlock)) {
        fprintf(stderr, "Threads Per Block must be a multiple of 2 for binary tree reductions");
        return EXIT_FAILURE;
    }
    int matrixSize = N * N;

    // Define grid and block dimensions dynamically
    //int blockSide = sqrt(ThreadsPerBlock);
    if (blockSide * blockSide != ThreadsPerBlock) {
        fprintf(stderr, "Threads Per Block must allow a square block (e.g., 256 = 16x16 or 1024 = 32x32).\n");
        return EXIT_FAILURE;
    }

    // Host memory
    int *h_A = createIntMatrix(matrixSize);
    initializeRandArray(h_A, matrixSize);
    // PRINT MATRIX IF IT IS SMALL...
    if (N < 10) {
        printf("Input Matrix A:\n");
        print2D(h_A, N, N, stdout, 'i');
    }

    // Allocate device memory
    int *d_A;
    float *d_adjustedA, *d_colAverages;
    //float *d_colSums;
    cudaMalloc((void **)&d_A, matrixSize * sizeof(int));
    cudaMalloc((void **)&d_adjustedA, matrixSize * sizeof(float));
    cudaMalloc((void **)&d_colAverages, N * sizeof(float));
    //cudaMalloc((void **)&d_colSums, N * sizeof(int));  // Allocate colSums

    // Define grid and block dimensions
    dim3 dimGrid((N + blockSide - 1) / blockSide, (N + blockSide - 1) / blockSide);
    dim3 dimBlock(blockSide, blockSide);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, matrixSize * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // Start timing for max and sum kernel
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // Launch kernel
    computeAndSubtractAverages<<<dimGrid, dimBlock>>>(d_A, d_colAverages, d_adjustedA, N);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Stop timing for max and sum kernel
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float timeElapsed;
    HANDLE_ERROR(cudaEventElapsedTime(&timeElapsed, start, stop));
    printf("Time for kernel: %.3f ms\n", timeElapsed);

    // Copy results back to host
    float *h_colAverages = (float *)malloc(N * sizeof(float));
    float *h_newA = (float *)malloc(matrixSize * sizeof(float));
    //int *h_colSums = (int *)malloc(N * sizeof(int));
    cudaMemcpy(h_colAverages, d_colAverages, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_newA, d_adjustedA, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_colSums, d_colSums, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("\n\nAverages:\n");
    if (N < 10) print2D(h_colAverages, N, 1, stdout, 'f');
    printf("\n\nNew A (subtracted by averages):\n");
    if (N < 10) print2D(h_newA, N, N, stdout, 'f');

    // Cleanup
    free(h_A);
    free(h_colAverages);
    free(h_newA);
    //free(h_colSums);
    cudaFree(d_A);
    cudaFree(d_colAverages);
    cudaFree(d_adjustedA);
    //cudaFree(d_colSums);  
    return 0;
}
