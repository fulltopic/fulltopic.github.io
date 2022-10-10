# Exercises
## 1
* A CUDA stream refers to a sequence of asynchronous CUDA operations that execute on a device 
in the order issued by the host code. 
* Kernels and memory transfer. i.e. almost all the operations that issued by host but executed by GPU 
* Supports asynchronous operations and parallelism
## 2
* Events would be enqueued in CUDA stream. It can indicate beginning or ending of certain stream
* Event split granularity as kernel, or the host operations or other kernel will have to synchronize with a stream
## 3
* False dependency: There is no dependency between two kernels, while the hardware and some synchronous operations prevent one kernel from issued because of anohter kernel.
* Fermi has only one hardware queue, which is root cause of some false dependency. Kepler has [8, 32] hardware queue enabled by Hyper-Q technology.
### 4
* Explicit synchronization: when an API declared synchronization explicitly and user uses it intently.
* For example, in Fermi, memory transfer is synchronized globally by hardware
### 5
* In depth-first ordering, the scheduler issued the first task, then tried to issued the second one. But if the second task depends on the first one, the whole scheduler blocked. For most tasks, where is inner dependency between sub-tasks of the same task. 
On the other hand, in width-first ordering, the adjacent sub-tasks were from different tasks, they were independent in most cases. So the parallelism performance is higher.
### 6
The overlaps include:
* CPU and GPU: launch asynchronous kernel by host
* GPU and GPU: launch multiple asynchronous kernels 
* CPU and CPU-GPU data transfer: launch asynchronous data transfer API
* GPU and CPU-GPU data transfer: launch multiple asynchronous kernels and use asynchronous data transfer API for these kernels.
### 7
All 32 kernels run concurrently if no other hardware resource limits
### 8
Concurrency is limited by number of queues
### 9
The callback function is similar to event which provides a synchronization point of stream and queue. 
For Fermi device, the four kernels of the same stream may be issued almost concurrently, the callback will block the following kernels of the following stream. Then the output will be printed in sequence.
For Kepler device, the streams would be issued concurrently, the order of output is undetermined.
