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
        colAverages[globalCol] = (float)sharedMem[0][localCol]/N;
	}
    __syncthreads();

    if (globalRow < N && globalCol < N) {
        newA[globalRow * N + globalCol] = A[globalRow * N + globalCol] - colAverages[globalCol];
    }

}
// NO SHARED MEMORY
/*
__global__ void CovarianceMatrix(const float *A, float *covariance, int N) {
    // Calculate global row and column indices
    int globalCol = blockIdx.x * blockSide + threadIdx.x;
    int globalRow = blockIdx.y * blockSide + threadIdx.y;

    // Load matrix element into shared memory (only load if within matrix bounds)
    if (globalRow < N && globalCol < N) {
        covariance[globalCol * N + globalRow] = A[globalRow * N + globalCol];
    }
}
*/

//SHARED MEMORY
__global__ void TransposeMatrix(const float *A, float *transpose, int N) {
    // Declare dynamically allocated shared memory for matrix block (each thread loads a matrix element into shared memory)
    __shared__ float sharedMem[blockSide][blockSide];

    // Calculate global row and column indices
    int globalCol = blockIdx.x * blockSide + threadIdx.x;
    int globalRow = blockIdx.y * blockSide + threadIdx.y;
    // Local row and column indices within the block
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;

    if (globalRow < N && globalCol < N) {
        sharedMem[localRow][localCol] = A[globalRow * N + globalCol];
    }
    __syncthreads();
    // Write transposed data back to covariance matrix (coalesced write)
    int transposedCol = blockIdx.y * blockSide + threadIdx.x;
    int transposedRow = blockIdx.x * blockSide + threadIdx.y;

    if (transposedRow < N && transposedCol < N) {
        transpose[transposedRow * N + transposedCol] = sharedMem[localCol][localRow];
    }
}

__global__ void MatrixMul(float* Md, float* Nd, float* Pd, int Width)
{
  // declare cache in the shared memory
  __shared__ float Mds[blockSide][blockSide];
  __shared__ float Nds[blockSide][blockSide];
 
  // keep track of column index of the Pd element using thread index
  int x = threadIdx.x + blockIdx.x * blockDim.x; // x is column
  // keep track of row index of the Pd element using thread index
  int y = threadIdx.y + blockIdx.y * blockDim.y; // y is row

  float Pvalue = 0;
  // Loop over the Md and Nd block dimension required to compute the Pd element
  for (int m = 0; m < Width/blockSide; m++){
	
    // collaboratively loading of Md and Nd blocks into shared memory	 
    Mds[threadIdx.y][threadIdx.x] = Md[y * Width + (m * blockSide + threadIdx.x)];
    Nds[threadIdx.y][threadIdx.x] = Md[(m * blockSide + threadIdx.y) * Width + x];
    __syncthreads();
    
    // keep track of the running sum    
    for (int k = 0; k < blockSide; k++)
      Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
    __syncthreads();
  }
   Pd[y * Width + x] = Pvalue;
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
    // Define grid and block dimensions
    dim3 dimGrid((N + blockSide - 1) / blockSide, (N + blockSide - 1) / blockSide);
    dim3 dimBlock(blockSide, blockSide);

     /* i) και ii) ΑΝΑΣΤΡΟΦΗ Α ΚΑΙ ΥΠΟΛΟΓΙΣΜΟΣ ΑΤ * Α */
    // Allocate device memory
    int *d_A;
    float *d_adjustedA, *d_colAverages;
    cudaMalloc((void **)&d_A, matrixSize * sizeof(int));
    cudaMalloc((void **)&d_adjustedA, matrixSize * sizeof(float));
    cudaMalloc((void **)&d_colAverages, N * sizeof(float));

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

    // Copy results back to host
    float *h_colAverages = (float *)malloc(N * sizeof(float));
    float *h_newA = (float *)malloc(matrixSize * sizeof(float));
    cudaMemcpy(h_colAverages, d_colAverages, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_newA, d_adjustedA, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("\n\nAverages:\n");
    if (N < 10) print2D(h_colAverages, 1, N, stdout, 'f');
    printf("\n\nNew A (subtracted by averages):\n");
    if (N < 10) print2D(h_newA, N, N, stdout, 'f');

    /* iii) ΑΝΑΣΤΡΟΦΗ Α ΚΑΙ ΥΠΟΛΟΓΙΣΜΟΣ COVARIANCE = Α * ΑΤransposed */
    // Allocate memory for transposed matrix
    float *d_transposedA, *h_transposedA, *d_covarianceA, *h_covarianceA;
    cudaMalloc((void **)&d_transposedA, matrixSize * sizeof(float));
    h_transposedA = (float *)malloc(matrixSize * sizeof(float));

    // Start timing 
    HANDLE_ERROR(cudaEventRecord(start, 0));
    // TransposeMatrix KERNEL LAUNCH
    TransposeMatrix<<<dimGrid, dimBlock>>>(d_adjustedA, d_transposedA, N);
    HANDLE_ERROR(cudaDeviceSynchronize());
    // Stop timing 
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float timeElapsed2;
    HANDLE_ERROR(cudaEventElapsedTime(&timeElapsed2, start, stop));
    
    // Copy transposed matrix back to host
    cudaMemcpy(h_transposedA, d_transposedA, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Print transposed matrix if small
    printf("\n\nTransposed New A:\n");
    if (N < 10) print2D(h_transposedA, N, N, stdout, 'f');

    // CovarianceMatrix KERNEL LAUNCH
    cudaMalloc((void **)&d_covarianceA, matrixSize * sizeof(float));
    h_covarianceA = (float *)malloc(matrixSize * sizeof(float));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    MatrixMul<<<dimGrid, dimBlock>>>(d_adjustedA, d_transposedA, d_covarianceA, N);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float timeElapsed3;
    HANDLE_ERROR(cudaEventElapsedTime(&timeElapsed3, start, stop));
    cudaMemcpy(h_covarianceA, d_covarianceA, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Print transposed matrix if small
    printf("\n\nCovariance A:\n");
    if (N < 10) print2D(h_covarianceA, N, N, stdout, 'f');

    printf("-------------------------------------------EXECUTION TIMES FOR KERNELS----------------------------------\n");
    printf("Time for kernel computeAndSubtractAverages: %.3f ms\n", timeElapsed);
    printf("Time for kernel TransposeMatrix: %.3f ms\n", timeElapsed2);
    printf("Time for kernel MatrixMul: %.3f ms\n", timeElapsed3);

    // Cleanup
    free(h_A);
    free(h_colAverages);
    free(h_newA);
    free(h_transposedA);
    free(h_covarianceA);
    cudaFree(d_A);
    cudaFree(d_colAverages);
    cudaFree(d_adjustedA);
    cudaFree(d_transposedA);
    cudaFree(d_covarianceA);
    return 0;
}
