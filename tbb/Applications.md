These are some applications/libraries/frameworks that use TBB...

# TBLIS

Source: https://github.com/devinamatthews/tblis.git (in `develop` branch right now)

```c++
$ git grep tbb
src/external/tci/tci/communicator.c:    tbb::task_group tg;
src/external/tci/tci/communicator.c:    tbb::task_group tg;
src/external/tci/tci/task_set.c:    set->comm = (tci_comm*)new tbb::task_group();
src/external/tci/tci/task_set.c:    ((tbb::task_group*)set->comm)->wait();
src/external/tci/tci/task_set.c:    delete (tbb::task_group*)set->comm;
src/external/tci/tci/task_set.c:    ((tbb::task_group*)set->comm)->run(
src/external/tci/tci/tci_global.h:#include <tbb/tbb.h>
```

# RAJA

Source: https://github.com/LLNL/RAJA

Applications/proxies/benchmarks that use RAJA:
- https://github.com/LLNL/RAJAperf
- https://github.com/LLNL/Kripke
- https://github.com/LLNL/RAJAProxies - This is where LULESH@RAJA lives, but also includes Kripke as a subproject.

```c++
$ git grep "tbb::"
include/RAJA/policy/tbb/forall.hpp:using tbb_static_partitioner = tbb::static_partitioner;
include/RAJA/policy/tbb/forall.hpp:using tbb_static_partitioner = tbb::auto_partitioner;
include/RAJA/policy/tbb/forall.hpp:  using brange = ::tbb::blocked_range<decltype(iter.begin())>;
include/RAJA/policy/tbb/forall.hpp:  ::tbb::parallel_for(brange(begin(iter), end(iter), p.grain_size),
include/RAJA/policy/tbb/forall.hpp:  using brange = ::tbb::blocked_range<decltype(iter.begin())>;
include/RAJA/policy/tbb/forall.hpp:  ::tbb::parallel_for(brange(begin(iter), end(iter), ChunkSize),
include/RAJA/policy/tbb/policy.hpp:using policy::tbb::tbb_for_dynamic;
include/RAJA/policy/tbb/policy.hpp:using policy::tbb::tbb_for_exec;
include/RAJA/policy/tbb/policy.hpp:using policy::tbb::tbb_for_static;
include/RAJA/policy/tbb/policy.hpp:using policy::tbb::tbb_reduce;
include/RAJA/policy/tbb/policy.hpp:using policy::tbb::tbb_segit;
include/RAJA/policy/tbb/reduce.hpp:  std::shared_ptr<tbb::combinable<T>> data;
include/RAJA/policy/tbb/reduce.hpp:    data = std::shared_ptr<tbb::combinable<T>>(
include/RAJA/policy/tbb/reduce.hpp:        std::make_shared<tbb::combinable<T>>([=]() { return initializer; }));
include/RAJA/policy/tbb/scan.hpp:  scan_adapter(scan_adapter& b, tbb::split)
include/RAJA/policy/tbb/scan.hpp:  void operator()(const tbb::blocked_range<Index_type>& r, Tag)
include/RAJA/policy/tbb/scan.hpp:  void operator()(const tbb::blocked_range<Index_type>& r, Tag)
include/RAJA/policy/tbb/scan.hpp:  tbb::parallel_scan(tbb::blocked_range<Index_type>{0,
include/RAJA/policy/tbb/scan.hpp:  tbb::parallel_scan(tbb::blocked_range<Index_type>{0,
include/RAJA/policy/tbb/scan.hpp:  tbb::parallel_scan(tbb::blocked_range<Index_type>{0,
include/RAJA/policy/tbb/scan.hpp:  tbb::parallel_scan(tbb::blocked_range<Index_type>{0,
```

# MADNESS

Source: https://github.com/m-a-d-n-e-s-s/madness

```c++
$ git grep "tbb::"
src/madness/tensor/systolic.h:            tbb::parallel_for(0, nthread, [=](const int id) {
src/madness/tensor/systolic.h:                    tbb::parallel_for(0, nthread,
src/madness/tensor/systolic.h:            tbb::parallel_for(0, nthread, [=](const int id) {
src/madness/tensor/systolic.h:                done = tbb::parallel_reduce(tbb::blocked_range<int>(0,nthread), true,
src/madness/tensor/systolic.h:                    [=] (const tbb::blocked_range<int>& range, bool init) -> bool {
src/madness/world/range.h:    typedef tbb::split Split;
src/madness/world/taskfn.h:        virtual tbb::task* execute() {
src/madness/world/thread.cc:    tbb::task_scheduler_init* ThreadPool::tbb_scheduler = 0;
src/madness/world/thread.cc:            tbb_scheduler = new tbb::task_scheduler_init(nthreads+2);
src/madness/world/thread.cc:            tbb_scheduler = new tbb::task_scheduler_init(nthreads+1);
src/madness/world/thread.h:            public tbb::task,
src/madness/world/thread.h:        tbb::task* execute() {
src/madness/world/thread.h:             return ::operator new(size, tbb::task::allocate_root());
src/madness/world/thread.h:                tbb::task::destroy(*reinterpret_cast<tbb::task*>(p));
src/madness/world/thread.h:        static tbb::task_scheduler_init* tbb_scheduler; ///< \todo Description needed.
src/madness/world/thread.h:                tbb::task::spawn(*task);
src/madness/world/thread.h:                tbb::task::enqueue(*task);
src/madness/world/thread.h:            tbb::task& waiter = *new( tbb::task::allocate_root() ) tbb::empty_task;
src/madness/world/thread.h:            tbb::task& dummy = *new( waiter.allocate_child() ) tbb::empty_task;
src/madness/world/thread.h:            tbb::task::enqueue(dummy);
src/madness/world/thread.h:            tbb::task::destroy(waiter);
src/madness/world/world_task_queue.h:            virtual tbb::task* execute() {
src/madness/world/world_task_queue.h:                    tbb::parallel_reduce(range_, true,
src/madness/world/worldrmi.cc:    tbb::task* RMI::tbb_rmi_parent_task = nullptr;
src/madness/world/worldrmi.cc:                new (tbb::task::allocate_root()) tbb::empty_task;
src/madness/world/worldrmi.cc:            tbb::task::enqueue(*task_ptr, tbb::priority_high);
src/madness/world/worldrmi.cc:            tbb::task* empty_root =
src/madness/world/worldrmi.cc:                new (tbb::task::allocate_root()) tbb::empty_task;
src/madness/world/worldrmi.cc:                tbb::task* empty =
src/madness/world/worldrmi.cc:                    new (empty_root->allocate_child()) tbb::empty_task;
src/madness/world/worldrmi.cc:                tbb::task::enqueue(*empty, tbb::priority_high);
src/madness/world/worldrmi.cc:            tbb::task::destroy(*empty_root);
src/madness/world/worldrmi.h:                : public tbb::task, private madness::Mutex
src/madness/world/worldrmi.h:            tbb::task* execute() {
src/madness/world/worldrmi.h:        static tbb::task* tbb_rmi_parent_task;
src/madness/world/worldrmi.h:                tbb::task::destroy(*tbb_rmi_parent_task);
```

# Moose

Source: https://github.com/idaholab/moose

```c++
$ git grep "tbb::"
framework/include/utils/ParallelUniqueId.h:  static tbb::concurrent_bounded_queue<unsigned int> _ids;
framework/src/utils/ParallelUniqueId.C:tbb::concurrent_bounded_queue<unsigned int> ParallelUniqueId::_ids;
```

# deal.II

Source: https://github.com/dealii/dealii

Details: https://www.dealii.org/current/doxygen/deal.II/group__threads.html
