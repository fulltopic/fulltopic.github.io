#Coroutine
## Reference
1. [cppreference](https://en.cppreference.com/w/cpp/language/coroutines)
2. [under the cover 2016](https://www.bilibili.com/video/BV19g4y1b7qF/)
3. [coroutine concepts in python](https://www.bilibili.com/video/BV1VM4y1c7yW/?spm_id_from=pageDriver)
4. [Asio (want ppt)](https://www.bilibili.com/video/BV1s64y1r7iD/?spm_id_from=333.337.search-card.all.click)
5. [reference-params](https://toby-allsopp.github.io/2017/04/22/coroutines-reference-params.html)
6. [coroutine tutorial](https://www.scs.stanford.edu/~dm/blog/c++-coroutines.html)
7. [C++2a Coroutines and dangling references](https://quuxplusone.github.io/blog/2019/07/10/ways-to-get-dangling-references-with-coroutines/#boring-old-way)
8. [The lifetime of objects involved in the coroutine function (win)](https://devblogs.microsoft.com/oldnewthing/20210412-00/?p=105078)
9. [stackoverflow: what is coroutine](https://stackoverflow.com/questions/553704/what-is-a-coroutine)
10. [Baker's gitio: Asymmetric Transfer](https://lewissbaker.github.io/)
11. [Raymond Chen's series: C++ coroutines](https://devblogs.microsoft.com/oldnewthing/20191209-00/?p=103195)
12. [coroutine theory](https://lewissbaker.github.io/2017/09/25/coroutine-theory)
13. [awaiter](https://lewissbaker.github.io/2017/11/17/understanding-operator-co-await)
14. [design purpose?](https://stackoverflow.com/questions/71153205/c-coroutine-when-how-to-use#)
## Concepts
General `coroutine` concept is different from `coroutine syntax library`. 
Then, why coroutine?

Is it supposed to be widely used in FSM?

This ability to execute logic after the coroutine enters the ‘suspended’ state allows the coroutine to be scheduled for resumption without the need for synchronisation that would otherwise be required if the coroutine was scheduled for resumption prior to entering the ‘suspended’ state due to the potential for suspension and resumption of the coroutine to race [12].
### C++ std Coroutine
#### Theory
##### definition
Roughly speaking, coroutines are functions that can invoke
each other but do not share a stack, so can flexibly suspend their execution at any point to enter a different coroutine.
In the true spirit of C++, C++20 coroutines are implemented 
as a nice little nugget buried underneath heaps of garbage that you have to 
wade through to access this nice part [6].

##### stackful vs. stackless
The implementation of coroutines described 
in the current draft of the TS is “stackless”, 
which means that the entire stack is not saved 
when the coroutine is suspended; instead, 
only local variables are saved. 
This is different to “stackful” coroutines 
in which the entire call stack is saved and restored. 
The advantage of the “stackless” approach in the TS 
is that coroutines can be very light-weight, 
only needing storage for any local variables [5].
#### Underneath
What compiler has done, like, create all those pieces of functions and objects(on heap).
#### Object lifetime in coroutine
In a coroutine, you can’t rely on references passed as arguments remaining valid for the life of the coroutine. You need to think of them like captures in a lambda [5].
#### Synchronization in coroutine
[13]
#### Implement simple coroutine case by other framework(mutex, condition etc)
e.g.:
* assembly 
* multi-thread communication
* break pointer/synchronization pointer
* thread-local, boost.asio previous versions
### Boost.Coroutine
#### Transfer between threads
### Proactor/Reactor
### Promise/Future
### Asynchronized IO
### Exception Handle
### epoll integration
### Others
* `coroutine_handle<>` is shorthand for `coroutine_handle<void>`