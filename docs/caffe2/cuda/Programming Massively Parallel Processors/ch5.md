# Ch5
## Notes
* In CUDA, pointers are used to point to data objects in global memory.
  There are two typical ways in which pointer usage arises in kernel and
  device functions. First, if an object is allocated by a host function, the
  pointer to the object is initialized by cudaMalloc() and can be passed to
  the kernel function as a parameter. For example, the parameters d_M , d_N ,
  and d_P in Figure 5.1 are such pointers. The second type of usage is to
  assign the address of a variable declared in the global memory to a pointer
  variable. For example, the statement {float *ptr = &GlobalVar;} in a
  kernel function assigns the address of GlobalVar into an automatic pointer
  variable ptr.
## Solutions
### 5.1
Use shared memory does not reduce global memory bandwidth consumption.
* Each element of every thread is used once by one thread. No re-use of data
* In both case, there are one CPU --> GPU and one GPU --> CPU copy for each element.
* By pure global memory, the GPU load a bulk of data from CPU into GPU that is accessible to every thread
* In global memory case, it is Load --> Execution --> Copy back. A simple sequence.
* By shared memory, a small bulk of data is loaded into Global memory then SHM when corresponding block is running. And then are refreshed in next block execution.
* In SHM case, it is Load --> Execution --> Copy back --> Load --> .... It is a loop.

So, it seemed that SHM introduced more overhead.

### 5.2
#### 2x2 Tiles
|Thread  | element0 | element1 | element2 | element3 | element4 | element5 | element6 | element7 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|
|thread<sub>0,0</sub>|M<sub>0,0</sub> * N<sub>0,0</sub>|M<sub>0,1</sub> * N<sub>1,0</sub>|M<sub>0,2</sub> * N<sub>2,0</sub>|M<sub>0,3</sub> * N<sub>3,0</sub>|M<sub>0,4</sub> * N<sub>4,0</sub>|M<sub>0,5</sub> * N<sub>5,0</sub>|M<sub>0,6</sub> * N<sub>6,0</sub>|M<sub>0,7</sub> * N<sub>7,0</sub>|
|thread<sub>0,1</sub>|M<sub>0,0</sub> * N<sub>0,1</sub>|M<sub>0,1</sub> * N<sub>1,1</sub>|M<sub>0,2</sub> * N<sub>2,1</sub>|M<sub>0,3</sub> * N<sub>3,1</sub>|M<sub>0,4</sub> * N<sub>4,1</sub>|M<sub>0,5</sub> * N<sub>5,1</sub>|M<sub>0,6</sub> * N<sub>6,1</sub>|M<sub>0,7</sub> * N<sub>7,1</sub>|
|thread<sub>1,0</sub>|M<sub>1,0</sub> * N<sub>0,0</sub>|M<sub>1,1</sub> * N<sub>1,0</sub>|M<sub>1,2</sub> * N<sub>2,0</sub>|M<sub>1,3</sub> * N<sub>3,0</sub>|M<sub>1,4</sub> * N<sub>4,0</sub>|M<sub>1,5</sub> * N<sub>5,0</sub>|M<sub>1,6</sub> * N<sub>6,0</sub>|M<sub>1,7</sub> * N<sub>7,0</sub>|
|thread<sub>1,1</sub>|M<sub>1,0</sub> * N<sub>0,1</sub>|M<sub>1,1</sub> * N<sub>1,1</sub>|M<sub>1,2</sub> * N<sub>2,1</sub>|M<sub>1,3</sub> * N<sub>3,1</sub>|M<sub>1,4</sub> * N<sub>4,1</sub>|M<sub>1,5</sub> * N<sub>5,1</sub>|M<sub>1,6</sub> * N<sub>6,1</sub>|M<sub>1,7</sub> * N<sub>7,1</sub>|
#### 4x4 Tiles
|Thread  | element0 | element1 | element2 | element3 | element4 | element5 | element6 | element7 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|
|thread<sub>0,0</sub>|M<sub>0,0</sub> * N<sub>0,0</sub>|M<sub>0,1</sub> * N<sub>1,0</sub>|M<sub>0,2</sub> * N<sub>2,0</sub>|M<sub>0,3</sub> * N<sub>3,0</sub>|M<sub>0,4</sub> * N<sub>4,0</sub>|M<sub>0,5</sub> * N<sub>5,0</sub>|M<sub>0,6</sub> * N<sub>6,0</sub>|M<sub>0,7</sub> * N<sub>7,0</sub>|
|thread<sub>0,1</sub>|M<sub>0,0</sub> * N<sub>0,1</sub>|M<sub>0,1</sub> * N<sub>1,1</sub>|M<sub>0,2</sub> * N<sub>2,1</sub>|M<sub>0,3</sub> * N<sub>3,1</sub>|M<sub>0,4</sub> * N<sub>4,1</sub>|M<sub>0,5</sub> * N<sub>5,1</sub>|M<sub>0,6</sub> * N<sub>6,1</sub>|M<sub>0,7</sub> * N<sub>7,1</sub>|
|thread<sub>0,2</sub>|M<sub>0,0</sub> * N<sub>0,2</sub>|M<sub>0,1</sub> * N<sub>1,2</sub>|M<sub>0,2</sub> * N<sub>2,2</sub>|M<sub>0,3</sub> * N<sub>3,2</sub>|M<sub>0,4</sub> * N<sub>4,2</sub>|M<sub>0,5</sub> * N<sub>5,2</sub>|M<sub>0,6</sub> * N<sub>6,2</sub>|M<sub>0,7</sub> * N<sub>7,2</sub>|
|thread<sub>0,3</sub>|M<sub>0,0</sub> * N<sub>0,3</sub>|M<sub>0,1</sub> * N<sub>1,3</sub>|M<sub>0,2</sub> * N<sub>2,3</sub>|M<sub>0,3</sub> * N<sub>3,3</sub>|M<sub>0,4</sub> * N<sub>4,3</sub>|M<sub>0,5</sub> * N<sub>5,3</sub>|M<sub>0,6</sub> * N<sub>6,3</sub>|M<sub>0,7</sub> * N<sub>7,3</sub>|
|thread<sub>1,0</sub>|M<sub>1,0</sub> * N<sub>0,0</sub>|M<sub>1,1</sub> * N<sub>1,0</sub>|M<sub>1,2</sub> * N<sub>2,0</sub>|M<sub>1,3</sub> * N<sub>3,0</sub>|M<sub>1,4</sub> * N<sub>4,0</sub>|M<sub>1,5</sub> * N<sub>5,0</sub>|M<sub>1,6</sub> * N<sub>6,0</sub>|M<sub>1,7</sub> * N<sub>7,0</sub>|
|thread<sub>1,1</sub>|M<sub>1,0</sub> * N<sub>0,1</sub>|M<sub>1,1</sub> * N<sub>1,1</sub>|M<sub>1,2</sub> * N<sub>2,1</sub>|M<sub>1,3</sub> * N<sub>3,1</sub>|M<sub>1,4</sub> * N<sub>4,1</sub>|M<sub>1,5</sub> * N<sub>5,1</sub>|M<sub>1,6</sub> * N<sub>6,1</sub>|M<sub>1,7</sub> * N<sub>7,1</sub>|
|thread<sub>1,2</sub>|M<sub>1,0</sub> * N<sub>0,2</sub>|M<sub>1,1</sub> * N<sub>1,2</sub>|M<sub>1,2</sub> * N<sub>2,2</sub>|M<sub>1,3</sub> * N<sub>3,2</sub>|M<sub>1,4</sub> * N<sub>4,2</sub>|M<sub>1,5</sub> * N<sub>5,2</sub>|M<sub>1,6</sub> * N<sub>6,2</sub>|M<sub>1,7</sub> * N<sub>7,2</sub>|
|thread<sub>1,3</sub>|M<sub>1,0</sub> * N<sub>0,3</sub>|M<sub>1,1</sub> * N<sub>1,3</sub>|M<sub>1,2</sub> * N<sub>2,3</sub>|M<sub>1,3</sub> * N<sub>3,3</sub>|M<sub>1,4</sub> * N<sub>4,3</sub>|M<sub>1,5</sub> * N<sub>5,3</sub>|M<sub>1,6</sub> * N<sub>6,3</sub>|M<sub>1,7</sub> * N<sub>7,3</sub>|
|thread<sub>2,0</sub>|M<sub>2,0</sub> * N<sub>0,0</sub>|M<sub>2,1</sub> * N<sub>1,0</sub>|M<sub>2,2</sub> * N<sub>2,0</sub>|M<sub>2,3</sub> * N<sub>3,0</sub>|M<sub>2,4</sub> * N<sub>4,0</sub>|M<sub>2,5</sub> * N<sub>5,0</sub>|M<sub>2,6</sub> * N<sub>6,0</sub>|M<sub>2,7</sub> * N<sub>7,0</sub>|
|thread<sub>2,1</sub>|M<sub>2,0</sub> * N<sub>0,1</sub>|M<sub>2,1</sub> * N<sub>1,1</sub>|M<sub>2,2</sub> * N<sub>2,1</sub>|M<sub>2,3</sub> * N<sub>3,1</sub>|M<sub>2,4</sub> * N<sub>4,1</sub>|M<sub>2,5</sub> * N<sub>5,1</sub>|M<sub>2,6</sub> * N<sub>6,1</sub>|M<sub>2,7</sub> * N<sub>7,1</sub>|
|thread<sub>2,2</sub>|M<sub>2,0</sub> * N<sub>0,2</sub>|M<sub>2,1</sub> * N<sub>1,2</sub>|M<sub>2,2</sub> * N<sub>2,2</sub>|M<sub>2,3</sub> * N<sub>3,2</sub>|M<sub>2,4</sub> * N<sub>4,2</sub>|M<sub>2,5</sub> * N<sub>5,2</sub>|M<sub>2,6</sub> * N<sub>6,2</sub>|M<sub>2,7</sub> * N<sub>7,2</sub>|
|thread<sub>2,3</sub>|M<sub>2,0</sub> * N<sub>0,3</sub>|M<sub>2,1</sub> * N<sub>1,3</sub>|M<sub>2,2</sub> * N<sub>2,3</sub>|M<sub>2,3</sub> * N<sub>3,3</sub>|M<sub>2,4</sub> * N<sub>4,3</sub>|M<sub>2,5</sub> * N<sub>5,3</sub>|M<sub>2,6</sub> * N<sub>6,3</sub>|M<sub>2,7</sub> * N<sub>7,3</sub>|
|thread<sub>3,0</sub>|M<sub>3,0</sub> * N<sub>0,0</sub>|M<sub>3,1</sub> * N<sub>1,0</sub>|M<sub>3,2</sub> * N<sub>2,0</sub>|M<sub>3,3</sub> * N<sub>3,0</sub>|M<sub>3,4</sub> * N<sub>4,0</sub>|M<sub>3,5</sub> * N<sub>5,0</sub>|M<sub>3,6</sub> * N<sub>6,0</sub>|M<sub>3,7</sub> * N<sub>7,0</sub>|
|thread<sub>3,1</sub>|M<sub>3,0</sub> * N<sub>0,1</sub>|M<sub>3,1</sub> * N<sub>1,1</sub>|M<sub>3,2</sub> * N<sub>2,1</sub>|M<sub>3,3</sub> * N<sub>3,1</sub>|M<sub>3,4</sub> * N<sub>4,1</sub>|M<sub>3,5</sub> * N<sub>5,1</sub>|M<sub>3,6</sub> * N<sub>6,1</sub>|M<sub>3,7</sub> * N<sub>7,1</sub>|
|thread<sub>3,2</sub>|M<sub>3,0</sub> * N<sub>0,2</sub>|M<sub>3,1</sub> * N<sub>1,2</sub>|M<sub>3,2</sub> * N<sub>2,2</sub>|M<sub>3,3</sub> * N<sub>3,2</sub>|M<sub>3,4</sub> * N<sub>4,2</sub>|M<sub>3,5</sub> * N<sub>5,2</sub>|M<sub>3,6</sub> * N<sub>6,2</sub>|M<sub>3,7</sub> * N<sub>7,2</sub>|
|thread<sub>3,3</sub>|M<sub>3,0</sub> * N<sub>0,3</sub>|M<sub>3,1</sub> * N<sub>1,3</sub>|M<sub>3,2</sub> * N<sub>2,3</sub>|M<sub>3,3</sub> * N<sub>3,3</sub>|M<sub>3,4</sub> * N<sub>4,3</sub>|M<sub>3,5</sub> * N<sub>5,3</sub>|M<sub>3,6</sub> * N<sub>6,3</sub>|M<sub>3,7</sub> * N<sub>7,3</sub>|


In matrix multiplication, an element M<sub>p,q</sub> would be read by all threads that calculate output P<sub>p,i</sub> (i = 0 ~ N_COL)
In pure global memory case, that means read global memory *COLNUM* times in unit of *sizeof(M<sub>p,q</sub>)*
In SHM + Tiling case, once the element has been loaded into SHM, it would be shared by all threads in the block that compute output P<sub>p,i</sub>.
The number of these threads are often *TileDim.COL*.
That is 2 in *2x2 Tiles* case, or 4 in *4x4 Tiles* case.

So, for each element of *M*
* in naive version, the time of read is number of column of *N*.
* In Tiling case, the number of read is (number of column of *N*) / (number of column of column of tile)

--> The global memory bandwidth cost of reading *M* is proportional to corresponding dimension size of tile.

And the same to *N*

### 5.3
In figure 5.12, line 12:
``` c
for (int k = 0; k < TILE_WIDTH; k ++)
{
    Pvalue += Mds[ty][k] * Nds[k][tx];
}
```
The calculation read *Mds* and *Nds* in range of *TILE_WIDTH*.
These elements are loaded by other threads except one.

Without *__synchthreads* in line 11, current thread may read *Mds* element that has not been loaded.
Without *__synchthreads* in line 14, current thread may update *Mds* element that has not been read by other threads.

For example
``` c
ty = 0, tx = 0, blockIdx.x = 0, blockIdx.y = 0

Without __synchthreads in line 11, the Mds[0][1] may be still initial value, not the value to be loaded (M[0][1]).

Without __synchthreads in line 14, when Pvalue+ loop completed, this value go on loading Mds[0][2],
but thread(0,1) is still computing the tile((0,0), (0,1), (1,0), (1,1)).
Then thread(0,1) will read M[0][2] instead of M[0][0].
```
### 5.4
The SHM could be shared between threads in the same block,
which means reuse and communication.

Make the kernel in figure 5.12 as the example.
### 5.5
C

Refer to analysis at the end of section 5.4 or solution 5.2
### 5.6
d

Local variable inside kernel is a register variable.
It has lifetime of thread.
So A variable would be created for each thread.

--> 1000 * 512 = 512,000
### 5.7
b

It is bound to block
### 5.8
* L1 cache is cache for global memory. The scope of L1 cache is SM (or kernel lie in the SM), scope of SHM is block.
* SHM is manipulated by kernel explicitly; L1 cache is controlled by hardware, unpredictable.
### 5.9
#### a
N
#### b
N / T
### 5.10
#### a
Consider GFLOPS capacity, max threads running in parallel is (200G / 36) > 5G;

For memory bandwidth, max threads running in parallel is (100G / (7 * 4)) < 4G.

The bottleneck lies in bandwidth, it is memory-bound case.
#### b
300G / 36 = 8.33G < 250G / (7 * 4) = 8.93

It is compute-bound case.
### 5.11
Still don't know specification of capacityX.x

* Check capacity of number of threads per block
* Check if SHM occupied by one block within the capacity of SHM per SM

If above conditions matched, the case is possible.

The limitation would be:
* Capacity of number of blocks per SM
* Capacity of number of threads per SM
* Capacity of SHM per SM

They all limit number of blocks running in parallel.
