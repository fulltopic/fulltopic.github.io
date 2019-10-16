#Ch9
## Notes
[ref1](http://www.cs.ucr.edu/~nael/217-f15/lectures/217-lec12.pdf)

[ref2](https://tschmidt23.github.io/cse599i/CSE%20599%20I%20Accelerated%20Computing%20-%20Programming%20GPUs%20Lecture%208.pdf)

[ref3](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)
### Inefficient algorithm
To prove the correctness of the algorithm. 

To prove it by mathematical induction. The theory to be proved is:
When a thread completed working on stride n, it has all 2 * n prefix summed. 
For example, when thread 31 completed stride 4, it has elements (31, 30, 29, 28, 27, 26, 25, 24) summed up in 31st element.

Stride = 1, it is obvious true

Suppose Stride = n / 2, it is true.

When stride = n, the element XY\[i] += XY\[i - n].
Before this step, XY\[i] has summed up XY\[i] ~ XY\[i - n / 2 + 1], XY\[i - n / 2] has had summed up XY\[i - n / 2] ~ XY\[i - n + 1].
After this step, XY\[i] has summed up XY\[i] ~ XY\[i - n + 1]. Proved.

So, for any index i, the process is to add prefix (1, 2, 4, ... , 2<sup>log<sub>2</sub>n</sup>).
### Inefficient algorithm performance
For example, if we execute the kernel on a parallel
machine with four times the execution resources as a sequential machine,
the parallel machine executing the parallel kernel can end up with only
half the performance of the sequential machine executing the sequential
code. Second, all the extra work consumes additional energy.  
### Efficient algorithm
To prove the correctness of efficient algorithm.

After reduction tree scan, the fact is that 
the SHM element with index i = (2<sup>n</sup> - 1) holds the sum of elements
of (i, i - 1, i - 2, ..., i - 2<sup>n</sup> + 1)

Then in each iteration, this type of elements push the partial sum to elements that 2<sup>n/2</sup> away,
Before the iteration, the element with index = i + 2<sup>n/2</sup> holds partial sum of 
elements (i + 2<sup>n/2</sup> ~ i + 2<sup>n/2</sup> - 2<sup>2/n</sup> + 1) = (i + 2<sup>n/2</sup> ~ i + 1). 
After this one iteration, the element with index = i + 2<sup>n/2</sup> holds sum of elements 
(i + 2<sup>n/2</sup> ~ i + 1, i ~ i - 2<sup>n</sup> + 1)
= (i + 2<sup>n/2</sup> ~ i - 2<sup>n</sup> + 1). 

Define the __*fixed*__ element with index = i as an element has the final sum of (element\[0] ~ element\[i]) 


Now mark the element with its real order. That is, the first element has the index = 1, second element has the index = 2, etc.
 
1) Suppose that the index of an element is represented in bits as b<sub>k</sub>b<sub>k-1</sub>...b<sub>1</sub>b<sub>0</sub>, and name the data set as array *data*.
2) Before the inverse reduction, we have data\[b<sub>k</sub>00...00], i.e.,
 data\[100...00] and the fake element data\[000...00] = 0, fixed.
3) For data\[b<sub>k</sub>b<sub>k-1</sub>0...00], if b<sub>k-1</sub> = 1, data\[b<sub>k</sub>10...00] += data\[b<sub>k</sub>00...00];
else leave data\[b<sub>k</sub>00..00] alone.
Then data\[b<sub>k</sub>b<sub>k-1</sub>0...00] fixed.
The value of (0b<sub>k-1</sub>0...0) is the corresponding stride in figure 9.7
4) for data\[b<sub>k</sub>b<sub>k-1</sub>b<sub>k-2</sub>...00], 
if b<sub>k-2</sub> = 1, data\[b<sub>k</sub>b<sub>k-1</sub>1...00] += data\[b<sub>k</sub>b<sub>k-1</sub>0...00];
else leave data\[b<sub>k</sub>b<sub>k-1</sub>0...00] alone.
Then data\[b<sub>k</sub>b<sub>k-1</sub>b<sub>k-2</sub>...00] fixed.
5) Iterate above step in decrease order, till the last bit, when stride = 1
6) Now all elements of *data* fixed.
 
## Solutions
### 9.1
In each iteration, the thread executing addition has threadIdx.x >= stride;
the idle thread has threadIdx.x < stride.

When stride >= 32, the idle threads have threadIdx.x (0 ~ 31), (0 ~ 63) ... .
That is, threadIdx.x that splits tasks = n * 32 (n = 1, 2, ...). 
Then the threads in the same warp are executing same operations.
### 9.2
#### Figure 9.7
Suppose the capacity >= 2048 threads per block.

In reduction phase, in each iteration 
```c
index = (threadIdx.x + 1) * 2 * stride -1;
index < blockDim.x
==>
(threadIdx.x + 1) * 2 * stride - 1 < blockDim.x
threadIdx < (blockDim.x + 1) / (2 * stride) - 1 
```
In all iterations, threads that executed add operations are:
```c
(blockDim.x + 1) / 2 - 1
(blockDim.x + 1) / 4 - 1
...
(blockDim.x + 1) / (blockDim.x) - 1

sum up into
(blockDim.x + 1) * (1/2 + 1/4 + ... + 1/blockDim.x) - log2(blockDim.x)
```

In inverse phase, in each iteration:
```c
index = (threadIdx.x + 1) * stride * 2 - 1;
index + stride < BLOCK_SIZE
==> index < blockDim.x - stride
==> (threadIdx.x + 1) * stride * 2 - 1 < blockDim.x - stride
==> threadIdx.x < (blockDim.x - stride) / (stride * 2) + 1
==> threadIdx.x < blockDim.x / (stride * 2) + 1/2
```
In all iterations, threads that executed add operations are:
```c
blockDim.x / 2 + 1/2
blockDim.x / 4 + 1/2
...
blockDim.x / blockDim.x + 1/2

sum up into
blockDim.x * (1/2 + 1/4 + ... + 1/blockDim.x) + (1/2) * log2(blockDim.x)
```
With both phases, the time that add operations are executed is about
```c
2 * blockDim.x * (1/2 + 1/4 + ... + 1/blockDim.x) - (1/2) * log2(blockDim.x)
~ 2 * blockDIm.x
``` 
With blockDim.x = 2048, the answer is a
#### More efficient
If the capacity = 1024 threads per block.

Before the reduction tree phase, each thread added the neighbour element. That is 1024 add operations.

Then the reduction tree and inverse phases of 1024 elements introduced about 2 * 1024 add operations.

After inverse phase, each threads push the prefix inside element to neighbour element. 
That is more 1024 add operations.

So the number of add operations is also about 4 * 1024 = 2 * 2048
### 9.3
The number of add operations is N * log<sub>2</sub>N - (N - 1), N = 2048

==> about 1024 * 10
### 9.4
```c
__global__ void work_inefficient_scan_kernel(float *X, float *Y, int inputSize) 
{
    __shared__ float XY[SECTION_SIZE];

    int i = blockIdx.x + blockDim.x + threadIdx.x;
    if (i < inputSize)
    {
        if (i == 0)
        {
            XY[i] = 0;
        }else {
            XY[i] = X[i - 1];
        }
    }

    for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2)
    {
        __synchthreads();
        XY[threadIdx.x] += XY[threadIdx.x - stride];
    }

    Y[i] = XY[i];
}
```
### 9.5
```c
#define ThreadPerBlock 1024
void prefixSum(float* output, float* input, int len)
{
    int arraySize = len * sizeof(float);
    int blockNum = ceil((double)len / ThreadPerBlock);
    int blockSumSize = sizeof(float) * blockNum;
    float *devOutput;
    float *devInput;
    float *blockSum;
    float* blockPrefix;

    cudaMalloc((void**)&devInput, arraySize);
    cudaMalloc((void**)&devOutput, arraySize);
    cudaMalloc((void**)&blockSum, blockSumSize);
    cudaMalloc((void**)&blockPrefix, blockSumSize):
    
    cudaMemcpy(devInput, input, arraySize, cudaMemcpyHostToDeivice):

    prefixSum<<<blockNum, ThreadPerBlock>>>(output, input, blockSum, len);

    //blockNum for block sum is 1, or there are hierarchy hierarchy algorithm 
    blockPrefix<<<1, ThreadPerBlock>>>(blockPrefix, blockSum, blockNum);

    distributePrefix<<<blockNum, ThreadPerBlock>>>(output, blockPrefix, len, blockNum);

    cudaMemcpy(output, devOutput, arraySize, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
    cudaFree(blockSum);
    cudaFree(blockPrefix);
}
```
```c
__global__
void prefixSum(float* output, float* input, float* blockSum, int len)
{
    __shared__ float x[ThreadPerBlock];

    int i = blockIdx.x * blockDim.x + threadIdx;
    if (i < len)
    {
        x[threadIdx.x] = input[i];
    } else {
        x[threadIdx.x] = 0;
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __synchthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x)
        {
            x[index] += x[index - stride];
        } 
    }

    for (int stride = blockDim.x / 4; stride > 0; stride /= 2)
    {
        __synchthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x)
        {
            x[index + stride] += x[index];
        }
    }
    __synchthreads();

    if (i < len)
    {
        output[i] = x[threadIdx.x];
    }
    
    if (threadIdx.x == 0)
    {
        //Anyway, the last  element get the final sum
        blockSum[blockIdx.x] = x[blockDim.x - 1];
    }
}
```
```c
__global__
void blockPrefix (blockPrefix, blockSum, blockNum)
{
    __shared__ x[BlockDim.x];
    if (threadIdx.x < blockNum)
    {
        x[threadIdx.x] = blockSum[threadIdx.x];
    }

    for (unsigned int stride = 1; stride < blockNum; stride *= 2)
    {
        __synchthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockNum)
        {
            x[index] += x[index - stride];
        } 
    }

    for (int stride = blockNum / 4; stride > 0; stride /= 2)
    {
        __synchthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < blockNum)
        {
            x[index + stride] += x[index];
        }
    }
    __synchthreads();
    
    blockPrefix[threadIdx.x] = x[threadIdx.x];
}
```
```c
__global__
void distributePrefix<<<blockNum, ThreadPerBlock>>>(output, blockPrefix, len, blockNum)
{
    if (blockIdx.x > 0)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < len)
        {
            output[index] += blockPrefix[blockIdx.x - 1];
        }
    }
}
```
### 9.6
Split the input into m blocks, each with n elements. N = m * n.

In each block, run efficient algorithm. The number of add operations is 2 * (n - 1).

Run prefix sum on block sum array. It is about 2 * (m - 1).

Distribute block sum prefix into each element. It is N.

Finally, it is 2 * (n - 1) * m + 2 * (m - 1) + N = 2 * n * m - 2 * m + 2 * m - 2 + N = 3 * N - 4.
### 9.7
| index/stride | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--------------|---|---|---|---|---|---|---|---|
| 0            | 4 | 6 | 7 | 1 | 2 | 8 | 5 | 2 |
| 1            | 4 |10 |13 | 8 | 3 |10 |13 | 7 |
| 2            | 4 |10 |17 |18 |16 |18 |16 |17 |
| 4            | 4 |10 |17 |18 |20 |28 |33 |35 |

![9.7](./images/9_7.jpg)
### 9.8
Reduction tree

| index/stride | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--------------|---|---|---|---|---|---|---|---|
| 0            | 4 | 6 | 7 | 1 | 2 | 8 | 5 | 2 |
| 1            | 4 |10 | 7 | 8 | 2 |10 | 5 | 7 |
| 2            | 4 |10 | 7 |18 | 2 |10 | 5 |17 |
| 4            | 4 |10 | 7 |18 | 2 |10 | 5 |35 |

![reduction_tree](./images/efficient_reduction_tree.jpg)

Inverse Reduction Tree

| index/stride | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--------------|---|---|---|---|---|---|---|---|
| 4            | 4 |10 | 7 |18 | 2 |10 | 5 |35 |
| 2            | 4 |10 | 7 |18 | 2 |28 | 5 |35 |
| 1            | 4 |10 |17 |18 |20 |28 |33 |35 |

![inverse](./images/inverse_reduction_tree.jpg)
### 9.9
Two-level hierarchy scan means that the block-sum prefix-sum could be covered in one block.
That is, the max number of threads per block is the max number of blocks in data set scan.
The other factors that only impact on performance by deciding how many warps could be executed in parallel:
* Max number of threads per SM: is must be >= max number of threads per block
* Register file capacity: it impacts on performance not the capacity of thread number
* Memory bandwidth: it impacts only on performance.
* SHM capacity: it impacts only on performance.

Suppose that the max number of threads per block is N. 

The number of blocks defined in host codes that invoke the kernel is in logical concept.
The factors that impact on number of threads/elements could be executed are
* max number of thread per block.
* global memory capacity

The other factors that only impact on performance are:
* number of SM per GPU
* max number of blocks per SM
* max number of threads per SM
* max number of SHM per SM
* bandwidth capacity
* register file capacity    

The capacity of the data set is N * N 