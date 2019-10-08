# Ch6
## Notes
### Coalesced Access
This technique takes advantage of the fact that threads in a warp execute
the same instruction at any given point in time. When all threads in a warp execute a load instruction,
the hardware detects whether they access consecutive global memory locations.
That is, the most favorable access pattern is achieved when all threads in a warp
access consecutive global memory locations.
In this case, the hardware combines, or coalesces,
all these accesses into a consolidated access to consecutive DRAM locations.
### Coalesced Load Pattern
The consolidated copy in figure 6.11 is like :

![dm](./images/6_11_dm.jpg)

The second row is copied into second column of *Mds*

![mds](./images/6_11_mds.jpg)

The sequence of load:

![mds](./images/cosolidated_load.jpg)

#### Implementation:
* tx decides column index of *d_M* and *d_N*:
``` c
// d_M
Mds[tx][ty] = d_M[Row * Width + m * TILE_WIDTH + tx];
// = d_M[Row][m * TILE_WIDTH + tx]
```

``` c
//d_N
int Col = bx * TILE_WIDTH + tx
Nds[tx][ty] = d_N[(m * TILE_WIDTH + ty) * Width + Col];
// = d_N[m * TILE_WIDTH + ty][Col];
// = d_N[m * TILE_WIDTH + ty][bx * TILE_WIDTH + tx];
```
* tx decides row index of *Mds*/*Nds*, ty decides column index of *Mds*/*Nds*.
*Mds* and *Nds* load transpose of original tile.
Which ensures the relative sequence of elements in original tile.
* *d_M* is loaded by row, *d_N* is loaded by column
* Each block computes (*blockIdx.x*<sub>th</sub> rows of *d_M*) * (*blockIdx.y*<sub>th</sub> columns of *d_N*)
with (number of rows) = (number of columns) = *TILE_WIDTH*
* __Don't know why *Mds* and *Nds* are indexed by *\[tx]\[ty]* instead of *\[ty]\[tx]*__
## Solutions
### 6.1
[ref](http://www.csce.uark.edu/~mqhuang/courses/5013/f2011/lecture/HSCoDesign_Lecture_6.pdf)
#### Figure 6.2
``` c
__shared__ float partialSum[];
unsigned int t = threadIdx.x;

partial[t] += partial[t + blockDim.x];
__synchthreads();

for (unsinged int stride = 1; stride < blockDim.x; stride *= 2)
{
    __synchthreads();

   if (t % (2 * stride) == 0)
   {
    partialSum[t] += partialSum[t + stride];
   }
}
```
* blockIdx.x = half original blockIdx.x
* Extra arithmetic operation: The extra part deals with right part of partialSum.
When the blockDim.x reduced, this part could be concealed with the first loop of the original design.
So, no extra cost introduced.
* Resource limitation:
    1) No extra register introduced
    2) Element read per thread doubled, while the total number remains. No global bandwidth limitation addressed.
    3) Threads per block reduced.
    4) No extra block introduced.
    5) No extra SHM required.
#### Figure 6.4
``` c
__shared__ float partialSum[];
unsigned int t = threadIdx.x;
for (unsigned int stride = blockDim.x; stride > 0; stride >> 1) {
    __syncthreads();
    if (t < stride) {
        partialSum[t] += partialSum[t+stride];
    }
}
```

Others questions: the same to that of *Figure 6.2*
### 6.2
The modification of *Figure 6.2* removed the __>>__ operation in *for* loop.