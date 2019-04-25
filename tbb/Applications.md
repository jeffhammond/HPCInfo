These are some applications/libraries/frameworks that use TBB...

# TBLIS

Source: https://github.com/devinamatthews/tblis.git (in `develop` branch right now)

```c++
$ git grep tbb
.travis.yml:  - RUN_TEST=0 THREADING="tbb" BUILD_CONFIG="auto"
.travis.yml:        brew update && brew install gcc@6 tbb llvm@3.9;
.travis.yml:    - libtbb-dev
src/external/tci/configure:                          none, auto, task, openmp, pthreads, tbb, omptask,
src/external/tci/configure:for ac_header in tbb/tbb.h
src/external/tci/configure:  ac_fn_cxx_check_header_mongrel "$LINENO" "tbb/tbb.h" "ac_cv_header_tbb_tbb_h" "$ac_includes_default"
src/external/tci/configure:if test "x$ac_cv_header_tbb_tbb_h" = xyes; then :
src/external/tci/configure:for ac_lib in '' tbb; do
src/external/tci/configure:if test x"$ac_cv_header_tbb_tbb_h" = xyes && \
src/external/tci/configure:    have_tbb=yes
src/external/tci/configure:    have_tbb=no
src/external/tci/configure:    elif test x"$have_tbb" = xyes; then
src/external/tci/configure:        thread_model=tbb
src/external/tci/configure:    if test x"$have_tbb" = xyes; then
src/external/tci/configure:        thread_model=tbb
src/external/tci/configure:elif test x"$thread_model" = xtbb; then
src/external/tci/configure:    if test x"$have_tbb" = xno; then
src/external/tci/configure:        as_fn_error $? "tbb requested but not available" "$LINENO" 5
src/external/tci/configure.ac:    openmp, pthreads, tbb, omptask, winthreads, ppl, or dispatch @<:@default=auto@:>@]),
src/external/tci/configure.ac:AC_CHECK_HEADERS([tbb/tbb.h])
src/external/tci/configure.ac:AC_SEARCH_LIBS([TBB_runtime_interface_version], [tbb])
src/external/tci/configure.ac:if test x"$ac_cv_header_tbb_tbb_h" = xyes && \
src/external/tci/configure.ac:    have_tbb=yes
src/external/tci/configure.ac:    have_tbb=no
src/external/tci/configure.ac:    elif test x"$have_tbb" = xyes; then
src/external/tci/configure.ac:        thread_model=tbb
src/external/tci/configure.ac:    if test x"$have_tbb" = xyes; then
src/external/tci/configure.ac:        thread_model=tbb
src/external/tci/configure.ac:elif test x"$thread_model" = xtbb; then
src/external/tci/configure.ac:    if test x"$have_tbb" = xno; then
src/external/tci/configure.ac:        AC_MSG_ERROR([tbb requested but not available])
src/external/tci/tci/communicator.c:    tbb::task_group tg;
src/external/tci/tci/communicator.c:    tbb::task_group tg;
src/external/tci/tci/task_set.c:    set->comm = (tci_comm*)new tbb::task_group();
src/external/tci/tci/task_set.c:    ((tbb::task_group*)set->comm)->wait();
src/external/tci/tci/task_set.c:    delete (tbb::task_group*)set->comm;
src/external/tci/tci/task_set.c:    ((tbb::task_group*)set->comm)->run(
src/external/tci/tci/tci_global.h:#include <tbb/tbb.h>
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
