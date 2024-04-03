#include <ATen/cuda/CachingHostAllocator.h>

#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/detail/CUDAHooks.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/thread_pool.h>
#include <c10/cuda/CUDAAllocatorConfig.h>

#include <cuda_runtime_api.h>
#include <future>

namespace at::cuda {
namespace {

using Block = HostBlock<CUDAStream>;
using Comparator = ComparatorSize<Block>;

// Note: cudaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling cudaEventCreate/cudaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<
      at::cuda::CUDAEvent,
      std::function<void(at::cuda::CUDAEvent*)>>;
  EventPool() : pools_(at::cuda::device_count()) {}

  Event get(DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](at::cuda::CUDAEvent* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<at::cuda::CUDAEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(
        std::make_unique<at::cuda::CUDAEvent>(cudaEventDisableTiming).release(),
        destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<at::cuda::CUDAEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

/**
 * Note [CUDAHostAllocator design]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * We have three key data structures - the free list which stores blocks that
 * are not currently used, the block list which stores all blocks that have been
 * allocated, and the event queue which stores CUDA events and their
 * corresponding blocks.
 *
 * Each of these are protected by a separate mutex. The key design principles
 * are to 1) only hold each mutex for the minimal amount of time possible, 2)
 * never do any possible expensive operations (such as CUDA runtime API calls)
 * while holding the lock.
 *
 * There are three public methods: allocate, free, and record_event. In the
 * allocate path, we first check to see if we can service our request from this
 * free list, and otherwise we create a new block with cudaHostAlloc. In the
 * free path, we insert events (if required) into the event queue, and if
 * possible insert our block back into the free list. In allocate, we first
 * eagerly query events until we find one that is not ready, and insert the
 * corresponding block onto the free list if all the events recorded for a
 * block are ready. In the record_event path, we simply insert the given
 * stream into the set of streams tracked by the specified block. This set of
 * streams is then consumed in the free path.
 *
 * Some of the invariants here are less strict than they could be - for example,
 * we do not enforce that free(Block* block) => block->event_count == 0. This is
 * for compatibility reasons, and we can explore enforcing these in subsequent
 * versions.
 */

using AllocatorImplInterface =
    CachingHostAllocatorImplInterface<CUDAStream, EventPool::Event, Block, Comparator>;

struct CUDAHostAllocatorImpl : public AllocatorImplInterface {

private:
  void allocate_host_memory(size_t size, void** ptr) override {
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, since we assume unified addressing. So we grab any existing
    // primary context, if available. See pytorch/pytorch#21081.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index =
        c10::cuda::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(
          at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
    }

    if (c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
            pinned_use_cuda_host_register()) {
      allocWithCudaHostRegister(ptr, size);
    } else {
      // Use cudaHostAlloc for allocating pinned memory (global lock in driver)
      C10_CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
    }
  }

  void free_block(Block* block) override {
    if (c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
              pinned_use_cuda_host_register()) {
      void* ptr = block->ptr_;
      AT_CUDA_CHECK(cudaHostUnregister(ptr));
      free(ptr);
    } else {
      AT_CUDA_CHECK(cudaFreeHost(block->ptr_));
    }
  }

  void record_stream(
      c10::optional<std::vector<EventPool::Event>>& events,
      CUDAStream stream) override {
    auto event = create_event_internal(stream.device_index());
    event->record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(EventPool::Event& event) override {
    cudaError_t err = cudaEventQuery(*event);
    if(err == cudaErrorNotReady) {
      (void)cudaGetLastError(); // clear CUDA error
      return false;
    } else if (err!= cudaSuccess){
      C10_CUDA_CHECK(err);
    }
    return true;
  }

  EventPool::Event create_event_internal(DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  TaskThreadPool* getThreadPool() {
    static TaskThreadPool* pool = new TaskThreadPool(
        c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
            pinned_max_register_threads());
    return pool;
  }

  void mapPagesForRegister(
      const void* ptr,
      size_t size,
      size_t i,
      size_t numThreads,
      size_t pageSize) {
    uintptr_t start = (uintptr_t)ptr + (size * i / numThreads);
    uintptr_t end = (uintptr_t)start + (size / numThreads);
    if (i == (numThreads - 1)) {
      end = (uintptr_t)ptr + size;
    }

    // pre-fault/map the pages by setting the first byte of the page
    uintptr_t alignedStart =
        (((uintptr_t)start + pageSize - 1) & ~(pageSize - 1));
    for (uintptr_t p = alignedStart; p < ((uintptr_t)end); p += pageSize) {
      memset((void*)p, 0, 1);
    }
  }

  void registerPages(const void* ptr, size_t size) {
    AT_CUDA_CHECK(
        cudaHostRegister((void*)ptr, (size_t)size, cudaHostRegisterDefault));

    // If host and device pointer don't match, give a warning and exit
    void* devptr;
    AT_CUDA_CHECK(cudaHostGetDevicePointer(&devptr, (void*)ptr, 0));
    TORCH_CHECK(
        (void*)devptr == (void*)ptr,
        "Host and device pointer dont match with cudaHostRegister. "
        "Please dont use this feature by setting "
        "PYTORCH_CUDA_ALLOC_CONF=use_cuda_host_register:False (default)",
        "");
  }

  inline void allocWithCudaHostRegister(void** ptr, size_t roundSize) {
    // Here we do regular allocation, pre-fault/map the pages, and then do
    // cudaHostRegister with GPU mapping flags to lock the pages, so we
    // can minimize the cost for the cuda global lock.
    *ptr = malloc(roundSize);

    // Parallelize the mapping/registering of pages to reduce wall time
    size_t pageSize = (1 << 12); // 4kB pages
    size_t numMapThreads = c10::cuda::CUDACachingAllocator::
        CUDAAllocatorConfig::pinned_num_register_threads();
    if ((numMapThreads > 1) && (roundSize >= (pageSize * numMapThreads))) {
      // parallelize the mapping of pages with a threadpool
      auto* pool = getThreadPool();
      std::vector<std::promise<void>> promises;
      std::vector<std::future<void>> futures;
      promises.reserve(numMapThreads);
      futures.reserve(numMapThreads);

      for (size_t i = 0; i < numMapThreads; i++) {
        promises.emplace_back();
        futures.push_back(promises[i].get_future());
        auto task = [this, i, ptr, roundSize, numMapThreads, pageSize, &promises]() mutable {
          mapPagesForRegister(
            *ptr,
            roundSize,
            i, // thread task-id
            numMapThreads,
            pageSize);
          // set the promise when mapping pages are done
          promises[i].set_value();
        };
        pool->run(task);
      }
      for (auto& future : futures) {
        future.wait();
      }
    } else {
      // Map pages in the same thread
      mapPagesForRegister(*ptr, roundSize, 0, 1, pageSize);
    }

    // Register the mapped pages using cudaHostRegister
    registerPages(*ptr, roundSize);
  }
};

} // namespace

void raw_local_deleter(void* ptr);

struct CUDAHostAllocator final
    : public HostAllocatorInterface<CUDAHostAllocatorImpl> {
  at::DataPtr allocate(size_t size) override {
    auto ptr_and_ctx = impl_->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &raw_local_deleter,
        at::DeviceType::CPU};
  }
};

static CUDAHostAllocator host_allocator;

void raw_local_deleter(void* ptr) {
  host_allocator.free(ptr);
}

static inline  CUDAHostAllocator& getCUDAHostAllocator() {
  return host_allocator;
}

bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    at::cuda::CUDAStream stream) {
  return getCUDAHostAllocator().record_event(ptr, ctx, stream);
}

// Releases cached pinned memory allocations via cudaHostFree
void CachingHostAllocator_emptyCache() {
  getCUDAHostAllocator().empty_cache();
}

at::Allocator* getCachingHostAllocator() {
  return &host_allocator;
}

} // namespace at::cuda
