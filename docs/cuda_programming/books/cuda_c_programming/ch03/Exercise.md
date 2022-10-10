#Chapter 3
## Exercises
### 1. Unrolling
#### 1.1 Cause of performance improvement
By reducing frequency of branches and loop maintenance instructions.

--> reducing instruction overhead and creating more independent instructions to scheduler.

--> higher saturation of instruction and memory bandwidth.
#### 1.2 Improve instruction throughput
##### Unrolled loop
1) No loop stop divergence
2) No register for tmp counter
##### Unrolled warp
1) No tmp index calculation
2) No extra synchronization
##### Unrolled data block
1) Suppose address calculation is less
### 2. Unrolling 8 VS 16
The outputs are similar:
```bash
$ ./ex02 8
Starting reduce...
Cpu reduce = 2131059349: 0.060291 
Gpu unroll8 -------> <<<4096, 512>>> 
Gpu interleave 2131059349: 0.000350 
(cuda2) [zf@192 ch3]$ ./ex02 16
Starting reduce...
Cpu reduce = 2131059349: 0.060688 
Gpu unroll16 -------> <<<65536, 16>>> 
Gpu interleave 2131059349: 0.000376 
```
The default profile outputs: [unroll8](./profile0208.txt) and [unroll16](./profile0216.txt)

The performance of unroll8 was hurdled by memory throughput. Not enough memory for computing so warps stalled.
```bash
WRN   This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance 
          of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate    
          latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.
```
The performance of unroll16 was hurdled by memory access conflicts. 
```bash
    WRN   Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the    
          DRAM bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the       
          bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or  
          whether there are values you can (re)compute. 
```
Both cases has only half warp valid by stride setting.
### 3. Unroll Loop
Output
```bash
(cuda2) [zf@192 ch3]$ nvcc -I../ -rdc=true ./ex03.cu -o ex03
(cuda2) [zf@192 ch3]$ ./ex03
Starting reduce...
Cpu reduce = 2131059349: 0.059698 
Gpu unroll8 -------> <<<4096, 512>>> 
Gpu interleave 2131059349: 0.000356 
(cuda2) [zf@192 ch3]$ nvcc -I../ -rdc=true ./ex03.cu -o ex03
(cuda2) [zf@192 ch3]$ ./ex03
Starting reduce...
Cpu reduce = 2131059349: 0.060892 
Gpu unroll8 -------> <<<4096, 512>>> 
Gpu interleave 2131059349: 0.000334 
(cuda2) [zf@192 ch3]$ 
```
The default profile output: [unroll8](./profile03) and [unroll8Loop](./profile03Loop).

They both have greater than 80% memory performance and about 30% compute performance. They spent too much time on memory access.
### 4. No volatile
The performance:
```bash
Starting reduce...
Cpu reduce = 2131059349: 0.061238 
Gpu unroll8 warp -------> <<<4096, 512>>> 
Gpu interleave 2131059349: 0.000374 
Gpu unroll8 warp sync -------> <<<4096, 512>>> 
Gpu interleave 2131059349: 0.000334 
Gpu unroll8 warp sync unroll -------> <<<4096, 512>>> 
Gpu interleave 2131059349: 0.000339 
Gpu unroll8 warp sync unroll volatile -------> <<<4096, 512>>> 
Gpu interleave 2131059349: 0.000361 
```
*sync* version is slightly better than *volatile* version.

The profiling [volatile](./profile04.txt) and [syncthreads](./profile04sync.txt) shows:
1. *sync* has slightly better memory utility: 82% vs 73%
2. *sync* has slightly better compute utility: 29% vs 20%
3. *sync* has far better *Achieved Occupancy*: 92% vs 47%
4. *sync* has better *Active Warps*: 29% vs 15%

Although *sync* has better performance in almost all tags, the overall performance improvement is limited.
It is supposed to be caused by bad memory performance. 

Furthermore *volatile* version has < 50% *Achieved Occupancy*, it is supposed to be caused by memory access conflict.
For example:
```cu
    if (tid < 8) {
        volatile int *vData = blockIData;
        vData[tid] += vData[tid + 8];
        vData[tid] += vData[tid + 4];
        vData[tid] += vData[tid + 2];
        vData[tid] += vData[tid + 1];
    }
```
In loop 0:
```cu
    vData[7] += vData[15];  vData[3] += vData[11];
```
Loop 1:
```cu
    vData[7] += vData[11]; vData[3] += vData[7];
```
In this round, *vData[7] += vData[11]* is un-desired side effect. 
At the same time, *vData[3]* requires access to global data *vData[7]*. 
It may cause an implicit warp divergence.

It coud be proved by [access volatile by tid ni range](./profile04tidrange.txt):
the *Achieved Occupancy* improved from 47% into 84%.
### 5. Float
### 6. Float Performance
Refer to [interleave](./profile05.txt) and [unroll](./profile05unroll.txt), the memory efficiency contributes greatly to their performance difference.
```bash
Starting reduce...
Cpu reduce = 2131059328.000000: 0.055530 
Gpu interleave -------> <<<32768, 512>>> 
Gpu interleave 2131057408.000000: 0.001135 
Gpu interleave unroll8 -------> <<<4096, 512>>> 
Gpu unroll 2131059328.000000: 0.000362 
```
### 7
After child kernel completed and control returned to parent kernel.