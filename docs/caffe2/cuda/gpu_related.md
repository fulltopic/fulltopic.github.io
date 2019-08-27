# CUDA Programming
## Threads Related
### Terminology
![Terminology](./images/gpu_1.jpg)


* a: A kernel is a unit of program that to be executed in GPU.
A kernel can requires N thread blocks, each block is combined by same number of threads.
Kernel is a software concept.
* b: Thread block is unit of SM scheduling.
    1) A block can be assigned to only one SM.
    The hardware architecture and scheduling policy decide in which SM the block would be assigned.
    Block does not migrate.
    2) Threads in the same block can communicate with each other via SHM,
    barrier or other synchronization primitives such atomic operations;
    as they reside in the same SM and have access to same cache, SHM or register file.
    3) Number of Thread per block is concept to programmer.
    Threads in the same block may not run physically in parallel when resources (computing resource or memory access) is not ready.
* c: One GPU contains N SM. It is hardware concept.
* d: One SM contains N SP. It is hardware concept.
* e: Warp is unit of hardware scheduling.
    Threads in one warp executed in parallel physically.
    One Warp = 32 threads.
* f: A block contains a lots of threads (.../512/1024/or more).
    1) A block/kernel assign tasks to threads by programming logic.
    2) The threads in one block are assigned to one SM, and managed by Warp
    3) SM/GPU schedules warp into SP according to it life cycle status.
* g: A Warp is combined by 32 threads.
    1) A Warp is SIMT
    2) Two Warps are MIMD, can do branching, loops, etc.
* h: SP is the hardware unit that actually executes thread.
    One thread per SP. SP does not control life cycle of thread.
* i: Core is alias of SP

_*_A block would be executed after occupying block completed? That is something related to STREAM_*_

### Dynamic Parameters
For certain GPU, the fixed parameters are:

* Number of SM
* Number of SP per SM
* Maximum number of blocks per SM
* Maximum number of threads per SM
* Maximum number of threads per Block

The dynamic parameters are
* Number of blocks
* Number of threads per block

Consequently:
* Total number of Threads = (Number of Thread per Block) * (Number of Block)
* Total number of Blocks = (Max Number of Block per SM) * (Number of SM)

For a certain kernel, Number of Blocks:
* It depends on max number of Block per GPU = (max number of Block per SM) * (Number of SM)
* (Number of Block per SM) <= Floor((Max number of Threads per SM) / (Number of Threads per Block))
* But it is OK to set (Number of blocks) > (Number of blocks per GPU).
    Although some of them are definitely not able to executed in parallel,
    this type of setting helps hide the potential latency.

Number of Thread per Block (When number of Block is not decided yet):
* Number of Thread per Block <= Max number of Thread per Block
* It decides Number of Block: (Total Number of Threads) / (Number of Thread per Block)
* It decides Number of Block per SM = Floor((Max number of Thread per SM) / (Number of Thread per Block)).
* When calculating Number of Warp per SM, the Number of Block never exceeds hardware capacity of SM.
* It decides Number of Warp per SM: (Block per SM) * (Number of Thread per Block) / 32
    1) if (Number of Block per SM) <= (Max number of Block per SM), (Number of Warp per SM) = (Number of Block per SM) * (Number of Thread per Block) / 32
    2) if (Number of Block per SM) > (Max Number of Block per SM), (Number of Warp per SM) = (Max number of Block per SM) * (Number of Thread per Block) / 32
