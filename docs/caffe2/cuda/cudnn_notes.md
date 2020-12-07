# cuDNN Developer Guide Notes
[The guide](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/)
## Overview
* Closed source library
* Primitives, not framework
* Works on a single device, requires other frameworks for multi-GPU cases

## General Description

### Tensor Descriptor
* Tensor descriptor is merely a descriptor of memory layout. It does not hold or point to memory.
* By coordinating memory solely at the descriptor, Caffe2 retains control over memory and communication for efficiency

#### Packed Descriptor

##### Fully-packed
According to the definition:
``` c
* the number of tensor dimensions is equal to the number of letters preceding the fully-packed suffix.
* the stride of the i-th dimension is equal to the product of the (i+1)-th dimension by the (i+1)-th stride.
* the stride of the last dimension is 1.
```
Suppose a NCHW fully-packed tensor:
* dim(index1) = dim(N) = 4
* dim(index2) = dim(C) = 3
* dim(index3) = dim(H) = 2
* dim(index4) = dim(W) = 1

So,
* stride(W) = 1
* stride(H) = dim(index3) * stride(W) = 2
* stride(C) = dim(index2) * stride(H) = 6
* stride(N) = dim(index1) * stride(C) = 24

Refer to layout described at [NCHW Layout](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/#data-layout-formats),
the stride is in fact gap between adjacent memory pointer in the same dimension.

For example, suppose data type is int, sizeof(data unit) = 4, then
* p(W0) = 0
* p(W1) = p(W0) + stride(W) = 1
* p(H0) = 0
* p(H1) = p(H0) + stride(H) = 2
* p(C0) = 0
* p(C1) = p(C0) + stride(C) = 6

So, fully-packed tensor means a tensor with each element filled.

Partially-packed tensor is a tensor that follows above fully-packed rules
in some adjacent dimensions but not all dimensions.

Partially-packed tensor specifies which part of dimensions are fully-packed.
Spatially-packed tensor does not. It is just a general alias for partially-packed tensor.

##### Overlapping
When a tensor is NOT a *-packed tensor, it is an overlapping tensor.

In an overlapping tensor, an data element could be mapped to more than one index combinations.

For example, data\[0]\[1]\[2]\[3] and data\[0]\[2]\[1]\[0] could point to the same data element.

I don't know a factual case of this type of layout, suppose a circular layout matches the definition.

### Thread-safe
The library is thread safe and its functions can be called from multiple host threads, as long as threads to do not share the same cuDNN handle simultaneously.

The handle is a pointer to cuDNN library context.
The context is associated with only one GPU device, however multiple contexts can be created on the save GPU device

### Reproducibility
Atomic operation on floating-point is not reproducible.

Floating-point addition is not associative, due to rounding error that occur when adding numbers with different exponents.
It leads to the absorption of the lower bits of the sum. The numerical non-reproducibility of floating-point atomic additions
is due to the combination of two phenomena: rounding-error and the order in which operations are executed.

[Reproducible floating-point atomic addition in data-parallel environment](https://annals-csis.org/proceedings/2015/pliks/86.pdf)

### RNN Functions

#### Persistent Algorithm
The standard algorithm loads W(parameters) from off-chip memory for each time step calculation.
The bandwidth between off/on-chip memory becomes a bottleneck of performance.
A solution to ease the bottleneck is to increase batch size.

While as the mediate result of each step calculation also increases, the solution is not a good choice:
* It increase the amount of memory needed to train a network
* Memory(register) per thread increases, which causes decreases of amount of threads running in parallel
* It may require multiple user streams, so that to complicates the deployment of the model

Then, the persistent RNN algorithm is to fix the model parameters inside the chip memory to remove communication cost in small batch size.

The idea is to assign this (relative large) block of memory to one or small size of threads,
and to always keep the thread(s) active in GPU while still preemptive.

[Persistent RNNs](https://svail.github.io/persistent_rnns/)

[Persistent RNNs: Stashing Recurrent Weights On-Chip](http://proceedings.mlr.press/v48/diamos16.pdf)

## Usage
[The tutorial guide](http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/)

### Convolution Algorithm

#### FFT

##### Convolution and Fourier Transform
Suppose f(x) is input, h(x) is filter. Σ(i) means summary on index i.

The calculation of convolution is in fact cross-correlation, as:
```
g(x) = Σ(i)f(x + i)h(i): 0 <= i <= N, N is length of h
 = Σ(i)f(i)h(i - x): (0 + x) <= i <= (N + x), 0 <= (i - x) <= N
```
Fourier transform is like:
```
g(x) = Σ(i)f(i)h(x - i): 0 <= i <= x, 0 <= (x - i) <= N
```
So, index of h in cross-correlation and Fourier transform differs in order of sequence, not in scope.

As kernel in convolution network is something to be learned, the order is of no significance.
In other words, just thinking that we are learning a flipped kernel.
So, the Fourier transform algorithm and corresponding theories could be applied in convolution network calculation.

##### Why FFT
The standard convolution is not a simple and straightforward calculation. And
```
*: Convolution
.: Dot product
FT: Fourier Transform
IFT: Inverse Fourier Transform
f(x)*g(x) = IFT(FT(fx)) dot FT(g(x)))
```
In 2D input case, set number of elements of input as N^2, number of elements of kernel as K^2
* The standard convolution requires multiplication: O(N^2 * K^2).
* Although dot product is a promising alternative, the multiplication of FT is still of O(N^3)) for complexity of DFT:

    1) For each element, execute DFT in row (O(N)), execute DFT in column (O(N)).

    2) For all elements (O(N ^ 2)), get complexity as O(N ^ 3).

    3) Dot: O(N ^ 2)

    4) IDFT: O(N ^ 3)
* FFT

    1) For each element, execute FFT in row (O(log(N)), in column (O(log(N))

    2) For all elements (O(N ^ 2)), get complexity as O(N ^ 2 * log(N))

##### FFT Algorithm
Suppose in 1D case, number of element is N.

Set W<sub>N</sub> = e<sup>-j2π/N</sup>

W<sub>N</sub><sup>k(N-n)</sup> = W<sub>N</sub><sup>kN</sup> . W<sub>N</sub><sup>k(-n)</sup> = e<sup>-j2πkN/N</sup> * W<sub>N</sub><sup>k(-n)</sup> = e<sup>-j2πk</sup> * W<sub>N</sub><sup>k(-n)</sup> = W<sub>N</sub><sup>(-kn)</sup>

W<sub>N</sub><sup>2</sup> = e<sup>-j2π*2/N</sup> = e<sup>-j2π/(N/2)</sup> = W<sub>N/2</sub>

X\[k]

= Σ<sub>(n = 0 ~ (N - 1))</sub>X\[n]W<sub>N</sub><sup>kn</sup>

= Σ<sub>(n.even)</sub>X\[n]W<sub>N</sub><sup>kn</sup> + Σ<sub>(n.odd)</sub>X\[n]W<sub>N</sub><sup>kn</sup>

= Σ<sub>(r = 0 ~ (N / 2 - 1))</sub>X\[2r]W<sub>N</sub><sup>2kr</sup> + Σ<sub>(r = 0 ~ (N / 2 - 1))</sub>X\[2r + 1]W<sub>N</sub><sup>(2r + 1)k</sup>

= Σ<sub>(r = 0 ~ (N / 2 - 1))</sub>X\[2r](W<sub>N</sub><sup>2</sup>)<sup>kr</sup> + Σ<sub>(r = 0 ~ (N / 2 - 1))</sub>X\[2r + 1](W<sub>N</sub><sup>2</sup>)<sup>kr</sup> * W<sub>N</sub><sup>k</sup>

= Σ<sub>(r)</sub>X\[2r]W<sub>N/2</sub><sup>kr</sup> + W<sub>N</sub><sup>k</sup> * Σ<sub>(r)</sub>X\[2r + 1]W<sub>N/2</sub><sup>kr</sup>

= X<sub>even</sub>\[k] + W<sub>N</sub><sup>k</sup> * X<sub>odd</sub>\[k]

Then, the complexity of X\[k] is of O(N/2)

Split the problem recursively, the final complexity is O(logN) for each element

##### Reference

[Convolution and FFT](https://www.cs.princeton.edu/courses/archive/spring05/cos423/lectures/05fft.pdf)

[The Fast Fourier Transform Algorithm](https://www.youtube.com/watch?v=EsJGuI7e_ZQ)

[How the 2D FFT works](https://www.youtube.com/watch?v=v743U7gvLq0)

[2D Fourier transforms and applications](http://www.robots.ox.ac.uk/~az/lectures/ia/lect2.pdf)


#### WINOGRAD
In small K case, it is possible that K^2 < log(N), FFT does not improve the performance.
Winograd algorithm is a further optimization for small kernel.

##### 1D Case
For a convolution of input = R<sup>4</sup>, filter = R<sup>3</sup>, output = R<sup>2</sup>,
the standard algorithm requires 6 multiplications.

F(2,3) = | d<sub>0</sub> d<sub>1</sub> d<sub>2</sub> |
         | d<sub>1</sub> d<sub>2</sub> d<sub>3</sub> | * | g<sub>0</sub> g<sub>1</sub> g<sub>2</sub> |<sup>T</sup>

= | m<sub>1</sub> + m<sub>2</sub> + m<sub>3</sub> |
  | m<sub>2</sub> - m<sub>3</sub> + m<sub>4</sub> |

m<sub>1</sub> = (d<sub>0</sub> - d<sub>2</sub>) * g<sub>0</sub>

m<sub>2</sub> = (d<sub>1</sub> + d<sub>2</sub>) * (g<sub>0</sub> + g<sub>1</sub> + g<sub>2</sub>) / 2

m<sub>3</sub> = (d<sub>2</sub> - d<sub>1</sub>) * (g<sub>0</sub> - g<sub>1</sub> + g<sub>2</sub>) / 2

m<sub>4</sub> = (d<sub>1</sub> + d<sub>3</sub>) * g<sub>2</sub>

By Winograd FFT, it requires 4 multiplications.

##### 2D Case
To solve the problem of F(4, 9) = F(2 * 2, 3 * 3), and W = R<sup>9</sup>

Split the input into 2 * 3 blocks, named K; each block in size of F(2, 3). Split W into 3 blocks, named W; each block in size of 3.

Then, in format, the calculation could be expressed by K and W as described in 1D case

Each element in this psudo result is a 1D F(2, 3) case.

Since in 1D case, the performance improves by (6 multiplications) / (4 multiplications) = 1.5,
in 2D case, the improvement is 1.5 ^ 2 = 2.25

##### Conv Network
In convolution network, the input is treated similar to i2col,
tiles are extracted from original input to expected sizes, and calculate the convolution respectively.

##### Reference
[卷积神经网络中的Winograd快速卷积算法](https://www.cnblogs.com/shine-lee/p/10906535.html)

[Fast Algorithm for Convolutional Neural Networks](https://homes.cs.washington.edu/~cdel/presentations/Fast_Algorithms_for_Convolutional_Neural_Networks_Slides_reading_group_uw_delmundo_slides.pdf)