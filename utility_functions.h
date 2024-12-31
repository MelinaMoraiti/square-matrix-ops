#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <time.h>

// Function to allocate memory for a 1D Integer array in CPU
static int* createIntMatrix(unsigned int size) {
    int *A = (int*)malloc(size * sizeof(int));
    if (A == NULL) {
        printf("Memory allocation failed for A[%d x %d].\n", size, size);
        exit(EXIT_FAILURE);
    }
    return A;
}

// Function to allocate memory for a 1D Float array in CPU
static float* createFloatMatrix(unsigned int size) {
    float *A = (float*)malloc(size * sizeof(float));
    if (A == NULL) {
        printf("Memory allocation failed for A[%d x %d].\n", size, size);
        exit(EXIT_FAILURE);
    }
    return A;
}

// Function to print a 2D array of type int or float (num_of_rows x num_of_columns)
static void print2D(void* array, int num_of_rows, int num_of_columns, FILE* out, char type) {
    for (int i = 0; i < num_of_rows; i++) {
        for (int j = 0; j < num_of_columns; j++) {
            if (type == 'i') {
                fprintf(out, "|%4d| ", ((int*)array)[i * num_of_columns + j]);
            } else if (type == 'f') {
                fprintf(out, "|%6.2f| ", ((float*)array)[i * num_of_columns + j]);
            } else {
                fprintf(out, "Unsupported type\n");
                return;
            }
        }
        fprintf(out, "\n");
    }
}

// Function to check if an integer is within specified limits
static int isIntWithinLimits(int value, int min, int max) {
    return (value >= min && value <= max);
}

// Function to parse and validate input
static int ValidateInput(const char *arg, int minLimit, int maxLimit, const char *errorMessage) {
    char *endPtr;
    int value = strtol(arg, &endPtr, 10);

    if (*endPtr != '\0' || !isIntWithinLimits(value, minLimit, maxLimit)) {
        fprintf(stderr, "Error: %s\n", errorMessage);
        exit(EXIT_FAILURE);
    }

    return value;
}

// Function to initialize a 1D array with random values
static void initializeRandArray(int *array, int N) {
    for (int i = 0; i < N; i++) {
        array[i] = rand() % 100; 
    }
}
// Function to initialize a 1D array with random values
static void initializeArray(int *array, int N) {
    for (int i = 0; i < N; i++) {
        array[i] = i; 
    }
}
static int isMultipleOf2(int num)
{
    return num%2 == 0;
}
// Function to find the maximum value between two integers
__device__ int findMax(int a, int b) {
    return (a > b) ? a : b;
}

// Function to find the minimum value between two integers
__device__ int findMin(int a, int b) {
    return (a < b) ? a : b;
}

// Function to find the maximum value between two floating-point numbers
__device__ float findMaxFloat(float a, float b) {
    return (a > b) ? a : b;
}

// Function to find the minimum value between two floating-point numbers
__device__ float findMinFloat(float a, float b) {
    return (a < b) ? a : b;
}

float findMinFloatHost(float a, float b) {
    return (a < b) ? a : b;
}

// Error handling macro and function for CUDA
static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(a)                                                                 \
    if (a == NULL) {                                                                   \
        printf("Host memory failed in %s at line %d\n", __FILE__, __LINE__);           \
        exit(EXIT_FAILURE);                                                            \
    }

#endif // UTILITY_FUNCTIONS_H
