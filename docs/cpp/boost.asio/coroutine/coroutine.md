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
15. [demystifying](https://www.cnblogs.com/coolcpp/p/demystifying_cpp20_coroutines.html)
16. [introduction](https://zhuanlan.zhihu.com/p/446993465)
## Concepts
General `coroutine` concept is different from `coroutine syntax library`. 
Then, why coroutine?

Is it supposed to be widely used in FSM?

This ability to execute logic after the coroutine enters the ‘suspended’ state allows the coroutine to be scheduled for resumption without the need for synchronisation that would otherwise be required if the coroutine was scheduled for resumption prior to entering the ‘suspended’ state due to the potential for suspension and resumption of the coroutine to race [12].
### C++ std Coroutine
#### Theory
##### definition
`Roughly speaking, coroutines are functions that can invoke
each other but do not share a stack, so can flexibly suspend their execution at any point to enter a different coroutine.
In the true spirit of C++, C++20 coroutines are implemented 
as a nice little nugget buried underneath heaps of garbage that you have to 
wade through to access this nice part [6].`

##### stackful vs. stackless
`The implementation of coroutines described 
in the current draft of the TS is “stackless”, 
which means that the entire stack is not saved 
when the coroutine is suspended; instead, 
only local variables are saved. 
This is different to “stackful” coroutines 
in which the entire call stack is saved and restored. 
The advantage of the “stackless” approach in the TS 
is that coroutines can be very light-weight, 
only needing storage for any local variables [5].`
#### Underneath
What compiler has done, like, create all those pieces of functions and objects(on heap).
#### Object lifetime in coroutine
In a coroutine, you can’t rely on references passed as arguments remaining valid for the life of the coroutine. You need to think of them like captures in a lambda [5].
#### Synchronization in coroutine
[13]
`
比如协程 A 调用了协程 B，如果只有 B 完成之后才能调用 A 那么这个协程就是 Stackful，此时 A/B 是非对称协程；如果 A/B 被调用的概率相同那么这个协程就是 Stackless，此时 A/B 是对称协程[16]
`

`
Note that because the coroutine is fully suspended before entering awaiter.await_suspend(), that function is free to transfer the coroutine handle across threads, with no additional synchronization. For example, it can put it inside a callback, scheduled to run on a threadpool when async I/O operation completes. In that case, since the current coroutine may have been resumed and thus executed the awaiter object's destructor, all concurrently as await_suspend() continues its execution on the current thread, await_suspend() should treat *this as destroyed and not access it after the handle was published to other threads.[1]
`

So, c++20 coroutine has synchronization issues?

`
协程天生有栈属性，而且是 lock free[16]
`
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