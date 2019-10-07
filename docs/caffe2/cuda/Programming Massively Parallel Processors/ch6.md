# Ch6
## Notes
* This technique takes advantage of the fact that threads in a warp execute
    the same instruction at any given point in time. When all threads in a warp execute a load instruction,
    the hardware detects whether they access consecutive global memory locations.
    That is, the most favorable access pattern is achieved when all threads in a warp
    access consecutive global memory locations.
    In this case, the hardware combines, or coalesces,
    all these accesses into a consolidated access to consecutive DRAM locations.
* The consolidated copy in figure 6.11 is like :
![dm](./images/6_11_dm.jpg)

The second row is copied into second column of *Mds*

![mds](./images/6_11_mds.jpg)

The sequence of load:

![mds](./images/cosolidated_load.jpg)