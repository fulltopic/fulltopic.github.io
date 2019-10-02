# Ch3 Solution
## 3.1
### a
``` c
void matrixAdd (float* output, float* inputA, float* inputB, int dimN)
{
    int size = dimN * dimN * sizeof(float);
    float* devInputA, devInputB, devOutput;

    cudaMalloc((void**) &devInputA, size);
    cudaMemcpy(devInputA, inputA, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &devInputB, size);
    cudaMemcpy(devInputB, inputB, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &devOutput, size);

    //Kernel invocation

    cudaMemcpy(output, devOuput, size, cudaMemcpyDeviceToHost);

    cudaFree(devInputA);
    cudaFree(devInputB);
    cudaFree(devOutput);
}
```
### b
``` c
__global__
void matrixAddKernel(float* output, float* inputA, float* inputB, int dimN)
{
    int index = dimN * (blockDim.y * blockIdx.y + threadIdx.y) +
                blockDim.y * blockIdx.y + threadIdx.y;

    if (index < (dimN * dimN))
    {
        output[index] = inputA[index] + inputB[index];
    }
}
```
### c
``` c
__global__
void matrixAddKernel(float* output, float* inputA, float* inputB, int dimN)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int index = 0;
    if (idx < dimN)
    {
        for (index = idx * dimN; index < idx * dimN + dimN; index ++)
        {
            output[index] = inputA[index] + inputB[index];
        }
    }
}
```
### d
``` c
__global__
void matrixAddKernel(float* output, float* inputA, float* inputB, int dimN)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int index = 0;
    if (idx < dimN)
    {
        for (index = idx; index < dimN * dimN; index += dimN)
        {
            output[index] = inputA[index] + inputB[index];
        }
    }
}
```
### e
#### Solution Element
* Simple and straightforward
* Maximum parallel
* Overhead of __*if*__ statement
* Low calculation/memory load ratio
#### Solution Row
* Better calculation/memory ratio
* Less parallelism
* More register overhead
* Poor memory coalescing
#### Solution Column
* Better calculation/memory ratio
* Better memory coalescing
* Less parallelism
* More register overhead
## 3.2
``` c
__global__
void vecMatMulKernel(float* output, float* vec, float* mat, int dimN)
{
    int idx = blockDim.x * blockIdx.x + threadIx.x;
    if (idx < dimN)
    {
        index = idx * dimN;
        int i = 0;
        float prod = 0.0f;

        for (i = 0; i < dimN; i ++)
        {
            prod += mat[index + i] * vec[i];
        }

        output[idx] = prod;
    }
}
```
``` c
void vecMatrixMul(float* output, float* vec, float* mat, int dimN)
{
    int matSize = dimN * dimN * sizeof(float);
    int vecSize = dimN * sizeof(float);
    float* devOutput, devVec, devMat;

    cudaMalloc((void**)&devVec, vecSize);
    cudaMalloc((void**)&devMat, matSize);
    cudaMalloc((void**)&devOutput, vecSize);
    cudaMemcpy(devVec, vec, vecSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devMat, mat, matSize, cudaMemcpyHostToDevice);

    vecMatMulKernel(output, vec, mat, dimN);

    cudaMemcpy(output, devOutput, vecSize, cudaMemcpyDeviceToHost);

    cudaFree(devVec);
    cudaFree(devMat);
    cudaFree(devOutput);
}
```
## 3.3
* By default, all functions in CUDA program are host functions if they do not have any of the CUDA keywords in their declaration.
* One can use both *\__host__* and *\__device__* in a function declaration to generate two versions of object files for same function. One for host_execution/host_invocation, another for device_kernel_execution/device_invocation.
## 3.4
Suppose the problem want the parts of Figure3.5 to be completed.
It is very similar to above host stub codes.
## 3.5
C
## 3.6
C
## 3.7
C