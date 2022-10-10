# Chapter 5
## Notes
[number setting](https://zhuanlan.zhihu.com/p/84511202)
### 5.1 Threads
* Because of huge register file, switching threads has effectively zero overhead.
* Lock-step: Each instruction in the instruction queue is dispatched to every SP within an SM. That is, 
all SP execute the same instruction at the same time. No next instruction until current instruction completed. [Wiki](https://en.wikipedia.org/wiki/Lockstep_(computing))
* The hardware limit of threads per block is 512/1024
### 5.2 Blocks
* Total block number: 65536
* 1024 threads per block (2.x device)
* Total SM number
* Block number per SM is limited
* Seemed that **Block** is the unit for distribution (among hardware/SM).
* Thread number is limited by hardware thread number and number of registers provided
* Block number is limited by thread number per SM and SM number.
* Block may be idle or running, the registers and other resources are always occupied by them before instruction complete. So, number of register is a factor to be considered during block grid arrangement.
* Running one block of huge thread number is bad idea. In practice, for multi-core environment, it is always better to have multiple blocks running on each core to avoid idle states. [stackoverflow](https://stackoverflow.com/questions/40465633/why-smaller-block-size-same-overall-thread-count-exposes-more-parallelism)
* Tiny blocks don't make full use of hardware. Why?
### 5.3 Grids
* Warp is the unit for scheduling
* The width value of array must always be a multiple of warp size. Padded if not
* Be careful when parallelizing loops so that the access pattern always runs sequentially through memory in rows and never columns.
* Threads within same block can communicate using shared memory.
* Cache line is 128 bytes
#### index
```cpp
const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;
```
* thread_idx = y * W + x
* W = blockDim.x * gridDim.x
* y = block_index_y * thread_per_block_y + threadIdx.y
* block_index_y = blockIdx.y * gridDim.y
* thread_per_block_y = blockDim.y
* y = blockIdx.y * gridDim.y * blockDim.y + threadIdx.y
* gridDim.y = 1 (suppose)
* idy = y = blockIdx.y * blockDim.y + threadIdx.y
* x is similar
* thread_idx = (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x) + (blockDim.x * blockIdx.x + threadIdx.x)
### 5.4 Warps
* Warp is basic unit of execution
* CUDA model uses huge numbers of threads to hide memory latency
* you should not get too tied up concerning the number of warps, as they are really just
  a measure of the overall number of threads present on the SMs. It’s this total number that is really the interesting part, along with the percentage
  utilization
### 5.5 Block Scheduling
