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

