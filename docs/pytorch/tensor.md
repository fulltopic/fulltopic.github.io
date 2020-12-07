# Tensor Implementation (CPU)
## Structure
![tensor_cpu](./images/tensor_cpu.jpg)

* _Tensor_ is a wrapper class, it does not provide actual implementation.
* The functions of _Tensor_ implemented in _TensorImpl_. 
* The functions of _TensorImpl_ could be grouped into 3 categories:
    1. For memory storage, implemented in _Storage_ class.
    2. For autograd engine, key class is _AutogradMeta_.
    3. Shortcut of _torch_ functions (e.g. _torch::add_ and _Tensor.add_)     
## Memory
![torch::zeros](./images/torch__zeros.jpg)
* Dispatcher dispatched the _zeros_ to native cpu handle: _zeros_ defined in _TensorFactories.cpp_
* To create an empty tensor.
* To decide type meta: 
    1. type element size
    2. _Newer_ and Deleter_
    3. The _TypeMeta_ is still a wrapper class, it wraps a _TypeMetaData_ pointer. 
    4. The default dtype is a static object, the _TypeMetaData_ member is a raw pointer, and the get/create functions return object instead of smart pointer or reference. 
    There is no cost of deep clone of _TypeMetaData_, while it is still cost of object copy.
    5. Return _TypeMeta_ of _float_ + _CPU_.
* Get CPU allocator as allocator.
* Allocate _dataptr_ by posix library. There is alternative to allocate large SHM file and allocate _dataptr_ from SHM.
* Create _StorageImpl_ by above _TypeMeta_, _dataptr_ and _allocator_ (for delete).
* Create _Tensor_ by above _StorageImpl_
* Set the dispatcher key of the _Tensor_ as CPU
* Call _Tensor.zero_ to fill the Tensor with 0s
* Dispatcher dispatched the function to _fill_kernel_ defined in _FillKernel.cpp_
* The kernel implements fill by vector instructions.  

In general, to create a all zeros tensor:
* Decide type meta to dispatch to proper allocator.
* Decide dispatcher key to dispatch to proper fill kernel

Ref:

[philosophy](http://blog.ezyang.com/2019/05/pytorch-internals/)

[memory internal](https://www.miracleyoo.com/2019/12/11/Pytorch-Core-Code-Research/)

[OS related](https://zhuanlan.zhihu.com/p/34629243)
## Autograd
### Forward
![forward](./images/tensor_add.jpg)

* Before _add_ forward execution, prepare the _Variable_ for backward in advance.
* Find backward kernel _AddBackward0_
* Create _Node_ with backward kernel
* Set addends as next edges of current _Node_
* Call CPU kernel for add operation
* Set the add result as input of current _Node_
* Add the _Node_ into the _add_result_tensor.autogradMeta_
### Backward
![tensor_backward](./images/tensor_backward.jpg)

* Get root _Edge_. An _Edge_ includes a _Node_ pointer and a sequence number. 
A _Node_ includes a backward function and next edges. 
* Execute backward DAG with root _Edge_ and root grad outputs.
* Create _GraphTask_ and _GraphRoot_. 
    1. _GraphRoot_ is a special _Node_, which defines no actual backward function.
    2. _GraphTask_ keeps all _Node_ to be executed, storing in _not_ready_ map or _dependencies_ map.
*  _Engine_ traverses the graph by _Node_ --> _Edge_ --> _Node_ --> ... --> Leaf. 
_Engine_ traverses the graph in BFS or topology order and keeps _NodeTask_ in maps mentioned above.
* If execution thread pool has not been initialized, create the thread pool
* When there is no outstanding task in local _ready_queue_, just run dummy task.
* Create _NodeTask_ with _GraphTask_ mentioned above and push into local _ready_queue_ and return _Future_
* Working thread detects the ready _GraphTask_ and turns from idle state into working state
* Working thread execute kernel of current _NodeTask_
* Get all dependent _NodeTask_ of current _NodeTask_ by its _edge_list_.
* Check _dependencies_ of current _GraphTask_, if dependent _Node_ has dependent-number == 0, 
remove it from _dependencies_ and push corresponding _NodeTask_ into thread local _ready_queue_.
* Check _not_ready_ map of current _GraphTask_, if dependent _Node_ is ready, 
erase it from _not_ready_ map and push the task into thread local _ready_queue_.
* If a _NodeTask_ that was in dependencies_ had reached (dependent-number == 0), push it into _not_ready_
* If no outstanding task in _GraphTask_ remained, mark the _GraphTask_ as Complete
* Notify caller of _backward_ that is waiting on _Future_

#### Key Files
* _aten/src/ATen/core/Variadic.h_
* _aten/src/ATen/core/dispatch/Dispatcher.h_
* _aten/src/ATen/core/Tensor.cpp_
* _aten/src/ATen/native/BinaryOps.cpp_
* _aten/src/ATen/native/TensorFactories.h_
* _aten/src/Aten/native/TensorFactories.cpp_
* _aten/src/ATen/native/cpu/BinaryOpsKernel.cpp_
* _aten/src/ATen/templates/TensorBody.h_

* _c10/core/TensorOptions.h_
* _c10/core/ScalarType.h_
* _c10/core/Scalar.h_
* _c10/core/DispatchKeySet.h_
* _c10/core/Allocator.h_
* _c10/core/TensorImpl.cpp_

* _torch/include/c10/core/TensorImpl.h_
* _torch/include/c10/core/Storage.h_
* _torch/include/c10/core/StorageImpl.h_

* _torch/csrc/autograd/variable.h_
* _torch/csrc/autograd/variable.cpp_
* _torch/csrc/autograd/VariableTypeManual.cpp_
* _torch/csrc/autograd/input_buffer.h_
* _torch/csrc/autograd/edge.h_
* _torch/csrc/autograd/engine.h_
* _torch/csrc/autograd/engine.cpp_
* _torch/csrc/autograd/function.h_
* _torch/csrc/autograd/autograd.cpp_
* _torch/csrc/autograd/functions/utils.h_
* _torch/csrc/autograd/functions/basic_ops.h_
* _torch/csrc/autograd/generated/Functions.cpp_
* _torch/csrc/autograd/generated/Functions.h_
* _torch/csrc/autograd/generated/variable_factories.h_
* _torch/csrc/autograd/generated/VariableTypeEverything.cpp_

### Use Case
Take following block as example:
``` c++
	Tensor t0 = torch::rand({2, 2}, TensorOptions().requires_grad(true));
	Tensor t1 = torch::rand({2, 2}, TensorOptions().requires_grad(true));

	Tensor a = torch::mm(t0, t1);
	Tensor b = a + t1;
	std::cout << "x add" << std::endl;
	Tensor c = b + t0;
	Tensor d = torch::sin(c);
	Tensor e = d.mean();

	e.backward();
```
#### Forward Phase
![usecase0_forward](./images/usecase0_forward.jpg)

In forward phase, the operation result _Variable_ remembers the backward function 
and set forward operation input _Variables_ as next edges in backward phase.
#### Backward Phase
##### Initiation
Prepare _GraphTask_ and _GraphNode_ for task execution.

``` c++
auto Engine::execute(const edge_list& roots,
                     const variable_list& inputs,
                     bool keep_graph,
                     bool create_graph,
                     const edge_list& outputs) -> variable_list {
// ...

  auto graph_task = std::make_shared<GraphTask>(
      keep_graph,
      create_graph,
      worker_device == NO_DEVICE ? 0 : total_depth + 1);

  auto graph_root = std::make_shared<GraphRoot>(roots, inputs);
 
// ...
}
```

After initiation:

![initiation](./images/usecase0_backward0.jpg)
##### Compute Dependencies
```c++
auto Engine::compute_dependencies(Node* root, GraphTask& task) -> void {
  std::unordered_set<Node*> seen;
  std::vector<Node*> queue { root };

  auto& dependencies = task.dependencies_;
  while (!queue.empty()) {
    auto fn = queue.back(); queue.pop_back();
    for (const auto& edge : fn->next_edges()) {
      if (auto next_ptr = edge.function.get()) {
        dependencies[next_ptr] += 1;
        const bool was_inserted = seen.insert(next_ptr).second;
        if (was_inserted) queue.push_back(next_ptr);
      }
    }
  }
}
```
The process of BFS:

![compute_dependencies](./images/usecase0_backward_compute_dependencies.jpg)
##### Execution
```c++
void Engine::evaluate_function(
    std::shared_ptr<GraphTask>& graph_task,
    Node* func,
    InputBuffer& inputs) { 
  const auto opt_parent_stream = (*func).stream(c10::DeviceType::CUDA);
  c10::OptionalStreamGuard parent_stream_guard{opt_parent_stream};

  auto outputs = call_function(graph_task, func, inputs);
  auto& fn = *func;

  int num_outputs = outputs.size();
  if (num_outputs == 0) { 
    return;
  }

  std::lock_guard<std::mutex> lock(graph_task->mutex_);
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = outputs[i];
    const auto& next = fn.next_edge(i);

    bool is_ready = false;
    auto& dependencies = graph_task->dependencies_;
    auto it = dependencies.find(next.function.get());

    if (--it->second == 0) {
      dependencies.erase(it);
      is_ready = true;
    }

    auto& not_ready = graph_task->not_ready_;
    auto not_ready_it = not_ready.find(next.function.get());
    if (not_ready_it == not_ready.end()) {
      InputBuffer input_buffer(next.function->num_inputs());

      // Accumulates into buffer
      const auto opt_next_stream = next.function->stream(c10::DeviceType::CUDA);
      input_buffer.add(next.input_nr,
                       std::move(output),
                       opt_parent_stream,
                       opt_next_stream);

      if (is_ready) {
        auto& queue = ready_queue(input_buffer.device());
        queue.push(
            NodeTask(graph_task, next.function, std::move(input_buffer)));
      } else {
        not_ready.emplace(next.function.get(), std::move(input_buffer));
      }
    } else {
      auto &input_buffer = not_ready_it->second;

      // Accumulates into buffer
      const auto opt_next_stream = next.function->stream(c10::DeviceType::CUDA);
      input_buffer.add(next.input_nr,
                       std::move(output),
                       opt_parent_stream,
                       opt_next_stream);
      if (is_ready) {
        auto& queue = ready_queue(input_buffer.device());
        queue.push(
            NodeTask(graph_task, next.function, std::move(input_buffer)));
        not_ready.erase(not_ready_it);
      }
    }
  }
}
```

The process of topology DAG traverse

![backward_threadmain](./images/usecase0_backward_threadmain.jpg)
### Detach
![detach](./images/detach.jpg)

_Tensor.detach()_ implements a shallow clone of original _Tensor_. 
The detached _Tensor_ is invalid to grad immediately after _detach()_ as its _AutogradMeta_ has been set as _nullptr_.
## Some Codes
### intrusive_ptr.h
_pytorch_ defines _intrusive_ptr_ as an alternative to _shared_ptr_ for:
* It has better performance because it does the refcounting intrusively (i.e. in a member of the object itself). **Why better performance?**
* It provides _intrusive_ptr_ only by _make_intrusive_, which newed a heap object. 
Otherwise, the stack object would be detected as the _refcount == 0_ and therefore be released.

[boost intrusive_ptr](https://www.boost.org/doc/libs/1_60_0/libs/smart_ptr/intrusive_ptr.html)
#### intrusive_ptr_target
The type that could be wrapped by _intrusive_ptr_ should inherit class _intrusive_ptr_target_.
```c++
class C10_API intrusive_ptr_target {
```  
It provides mutable atomic _size_t_ type members for concurrent counting.
```c++
  mutable std::atomic<size_t> refcount_;
  mutable std::atomic<size_t> weakcount_;
```  
The counters are not public members, define friend function to grant accession. 
```c++
  template <typename T, typename NullType>
  friend class intrusive_ptr;
  friend inline void raw::intrusive_ptr::incref(intrusive_ptr_target* self);
```
Define destructor as protected to prevent end user from deleting the target. 
```c++
 protected:
  // protected destructor. We never want to destruct intrusive_ptr_target*
  // directly.
  virtual ~intrusive_ptr_target() {
```
Define the function that executes the real cleanup job as private virtual:
```c++
 private:
  virtual void release_resources() {}
```
[extra cleanup virtual function](https://github.com/fulltopic/fulltopic.github.io/blob/master/docs/caffe2/shared_ptr/std__shared_ptr.md#_sp_counted_ptr)
#### intrusive_ptr
Define a null type for _nullptr_ with _noexcept_ _constexpr_ _static_ function _singleton_.
```c++
namespace detail {
template <class TTarget>
struct intrusive_target_default_null_type final {
  static constexpr TTarget* singleton() noexcept {
    return nullptr;
  }
};
```
**Why an extra null type defined?**

The _intrusive_ptr_ class defined as:
```c++
template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class intrusive_ptr final {
```
Declare the inner resource:
```c++
  TTarget* target_;
```
Allow other types specification of _intrusive_ptr_ as friend for implicit conversion that was allowed between types of wrapped pointer.
```c++
  template <class TTarget2, class NullType2>
  friend class intrusive_ptr;
```  
Add reference counter. If the counter reached 0, report error as 0 reference followed by object destruction.
_intrusive_ptr_ is friend class of corresponding target type _T_.
```c++
  void retain_() {
    if (target_ != NullType::singleton()) {
      size_t new_refcount = ++target_->refcount_;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          new_refcount != 1,
          "intrusive_ptr: Cannot increase refcount after it reached zero.");
    }
  }
```
Remove reference of current object. _NullType::singleton()_ used.
```c++
  void reset_() noexcept {
    if (target_ != NullType::singleton() && --target_->refcount_ == 0) {
      const_cast<std::remove_const_t<TTarget>*>(target_)->release_resources();

      if (--target_->weakcount_ == 0) {
        delete target_;
      }
    }
    target_ = NullType::singleton();
  }
```
Declare the constructor as _private explicit_, as _intrusive_ptr_ is only allowed to be created by _make_intrusive_ with counter management.
```c++
  explicit intrusive_ptr(TTarget* target) noexcept : target_(target) {}
``` 
Type alias:
```c++
 public:
  using element_type = TTarget;
```
Default constructor set target as _NullType_. Maybe for container usage.
```c++
  intrusive_ptr() noexcept : intrusive_ptr(NullType::singleton()) {}
```
The move constructor, set moved in *rhs.target* as _NullType_. 
**Why not just swap?** 
```c++
  intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }
```  
If target pointers are convertible, the _intrusive_ptr_ objects are convertible.
The moved-in object _rhs_ has _nullptr_ set.
```c++
  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr move constructor got pointer of wrong type.");
    rhs.target_ = FromNullType::singleton();
  }
``` 
Copy constructor, increase reference counter.
```c++
  intrusive_ptr(const intrusive_ptr& rhs) : target_(rhs.target_) {
    retain_();
  }
```
Destructor, decrease reference counter, release resources if no more references.
```c++
  ~intrusive_ptr() noexcept {
    reset_();
  }
```
The assignment operator:
* _tmp_ is created by copy constructor mentioned above. _tmp_ has _target_ set, _rhs_ has _nullptr_ set.
* The copy-and-swap idiom. 
* At the end of _operator=()_, destructor of _tmp_ takes responsibility of counting and resource management.
* The first _operator=()_ reuse implementation of the second one.
```c++
  intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept {
	  //zf: Why specialize TTarget?
    return operator=<TTarget, NullType>(std::move(rhs));
  }

  template <class From, class FromNullType>
      intrusive_ptr& operator=(intrusive_ptr<From, FromNullType>&& rhs) &
      noexcept {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr move assignment got pointer of wrong type.");
    intrusive_ptr tmp = std::move(rhs);
    swap(tmp);
    return *this;
  }
```
The class _intrusive_ptr_ overloaded _bool()_ operator. It's seemed not a good practice, while simple and straightforward.

[Safe Bool Idiom](https://www.artima.com/cppsource/safebool.html)
```c++
  operator bool() const noexcept {
    return target_ != NullType::singleton();
  }
``` 
To construct _intrusive_ptr_ object, end-user must take use of static function _make_. 
That is why constructor has been declared as private.
```c++
  template <class... Args>
  static intrusive_ptr make(Args&&... args) {
    auto result = intrusive_ptr(new TTarget(std::forward<Args>(args)...));
    // We can't use retain_(), because we also have to increase weakcount
    // and because we allow raising these values from 0, which retain_()
    // has an assertion against.
    ++result.target_->refcount_;
    ++result.target_->weakcount_;

    return result;
  }
``` 
Other operators have been defined as non-member functions. 

### Tensor.h Template
### Some Knowledge
#### Anonymous Namespace
* The definition is treated as a definition of a namespace with unique name and a using-directive in the current scope that nominates this unnamed namespace.
* This is the same as the C way of having a static global variable or static function but it can be used for class definitions as well. 
* It also avoids making global static variable.
* Usage in _aten/src/ATen/core/op_registration/README.md_

[cppreference](https://en.cppreference.com/w/cpp/language/namespace#Unnamed_namespaces)

[stackoverflow](https://stackoverflow.com/questions/357404/why-are-unnamed-namespaces-used-and-what-are-their-benefits)

[local linkage](https://stackoverflow.com/questions/4181059/linkage-of-symbols-within-anonymous-namespace-within-a-regular-namespace)

#### template keyword
```c++
template<class Key, class Value, class Iterator>
class DictEntryRef final {
public:
  explicit DictEntryRef(Iterator iterator)
  : iterator_(std::move(iterator)) {}

  Key key() const {
    return iterator_->first.template to<Key>();
  }
//...
```
#### const
```c++
  const DictEntryRef<Key, Value, Iterator>& operator*() const {
      return entryRef_;
  }
```
*entryRef_* is not a const object. Declaring the return type as *const* enables declaring the function as *const*
#### copy-and-swap
```c++
class  Blob final : public c10::intrusive_ptr_target {
 public:
  Blob() noexcept : meta_(), pointer_(nullptr), has_ownership_(false) {}
  ~Blob() {
    Reset();
  }

  Blob(Blob&& other) noexcept : Blob() {
    swap(other);
  }

  void Reset() {
    free_();
    pointer_ = nullptr;
    meta_ = TypeMeta();
    has_ownership_ = false;
  }
 
  void swap(Blob& rhs) {
    using std::swap;
    swap(meta_, rhs.meta_);
    swap(pointer_, rhs.pointer_);
    swap(has_ownership_, rhs.has_ownership_);
  }
}
```

The copy-and-swap idiom application:
```c++
  Blob& operator=(Blob&& other) noexcept {
    Blob(std::move(other)).swap(*this);
    return *this;
  }
```

The object _other_ is moved into the _operator=()_ function, this function take responsibility of resource of _other_ and _self_.
A pure _swap_ is not able to guarantee the _noexcept_ constraint. 
* A Blob created by copy constructor. Name the object _tmp_. Object _tmp_ holds *pointer_* of _other_. 
Object _other_ holds *pointer_* of _nullptr_. 
Object _other_ then destructed at the end of copy constructor. 
* Swap objects _tmp_ and _self_. 
Object _self_ holds pointer of original _other_, object _tmp_ holds pointer of original _self_.
* Object _tmp_ destructed at the end of _operator=()_, *pointer_* hold by original _self_ released.
* The temporary object has been created for destruction of original _self_ resources.
* Even if _swap_ or copy constructor is not noexcept, this temporary object guarantees destruction of resources hold by _other_ or _self_.    

[copy-and-swap](https://cpppatterns.com/patterns/copy-and-swap.html)

[stackoverflow](https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom)
#### Variadic Templates
An example (_torch/aten/src/ATen/core/Variadic.h_)
```c++
  template <typename... Args>
  inline F& apply() {
    return self();
  }

  template <typename T, typename... Args>
  inline F& apply(T&& arg, Args&&... args) {
    self()(std::forward<T>(arg));
    if (self().short_circuit()) {
      return self();
    } else {
      return apply(std::forward<Args>(args)...);
    }
  }
``` 
[variadic template](https://eli.thegreenplace.net/2014/variadic-templates-in-c/)