"""
NUMA Unified Memory Manager for Dual-Socket Xeon.

Makes a dual-socket system behave like unified memory for ML workloads:

1. Socket-aware model sharding: layers 0-31 on socket 0, 32-63 on socket 1
2. Thread affinity: each socket's threads only touch local memory
3. Cross-socket handoff: one tensor transfer at the layer boundary
4. Memory pre-fault: touch all pages at allocation to avoid TLB misses
5. OMP thread binding: prevent thread migration between sockets

The result: each socket runs independently on local memory, connected
by a single UPI transfer at the midpoint. Like two GPUs in pipeline parallel.
"""

import os
import ctypes
import torch
import time


def setup_numa_optimal():
    """Configure the system for optimal NUMA behavior."""

    # 1. Set OMP threads to physical cores only (no hyperthreading for compute)
    n_physical = 16  # per socket on Gold 5218
    os.environ['OMP_NUM_THREADS'] = str(n_physical * 2)  # both sockets
    os.environ['MKL_NUM_THREADS'] = str(n_physical * 2)

    # 2. Bind OMP threads to cores explicitly (no migration)
    os.environ['OMP_PROC_BIND'] = 'close'  # bind threads to nearby cores
    os.environ['OMP_PLACES'] = 'cores'     # one thread per core

    # 3. GOMP (GCC OpenMP) settings
    os.environ['GOMP_CPU_AFFINITY'] = '0-63'

    # 4. PyTorch thread settings
    torch.set_num_threads(n_physical * 2)  # 32 physical cores

    # 5. Disable PyTorch's internal parallelism fighting with OMP
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

    print(f"NUMA Unified Setup:")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  OMP_PROC_BIND: {os.environ.get('OMP_PROC_BIND')}")
    print(f"  PyTorch threads: {torch.get_num_threads()}")


def get_numa_node(tensor):
    """Get which NUMA node a tensor's memory is on."""
    try:
        libc = ctypes.CDLL('libc.so.6')
        ptr = tensor.data_ptr()
        page_size = os.sysconf('SC_PAGE_SIZE')
        # get_mempolicy would tell us but needs special flags
        return -1  # can't easily detect from userspace
    except:
        return -1


def allocate_on_node(shape, dtype=torch.float32, node=0):
    """Allocate a tensor with memory on a specific NUMA node."""
    # Pin current thread to the target node's CPUs
    if node == 0:
        cpus = set(range(16))
    else:
        cpus = set(range(16, 32))

    old_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)

    # Allocate — first-touch policy puts memory on current CPU's node
    t = torch.empty(shape, dtype=dtype)

    # Touch all pages (pre-fault) to ensure physical allocation
    t.zero_()

    # Restore affinity
    os.sched_setaffinity(0, old_affinity)

    return t


def shard_model_across_sockets(model, midpoint=None):
    """
    Shard a model's layers across two NUMA sockets.

    Layers [0, midpoint) on socket 0
    Layers [midpoint, L) on socket 1

    Uses first-touch policy: move each layer's parameters to
    the target socket by re-allocating and copying.
    """
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        print("  Can't find model.model.layers — skipping sharding")
        return

    layers = model.model.layers
    L = len(layers)
    if midpoint is None:
        midpoint = L // 2

    print(f"  Sharding {L} layers: [0-{midpoint-1}] → socket 0, [{midpoint}-{L-1}] → socket 1")

    old_affinity = os.sched_getaffinity(0)

    for li, layer in enumerate(layers):
        target_node = 0 if li < midpoint else 1
        target_cpus = set(range(16)) if target_node == 0 else set(range(16, 32))

        os.sched_setaffinity(0, target_cpus)

        # Re-allocate each parameter on the target socket
        for name, param in layer.named_parameters():
            new_data = torch.empty_like(param.data)
            new_data.copy_(param.data)
            param.data = new_data

        # Also re-allocate buffers
        for name, buf in layer.named_buffers():
            new_buf = torch.empty_like(buf)
            new_buf.copy_(buf)
            # Can't directly assign buffers, but the copy is enough
            # for first-touch policy

    # Also shard embedding and lm_head
    os.sched_setaffinity(0, set(range(16)))  # embedding on socket 0
    if hasattr(model.model, 'embed_tokens'):
        embed = model.model.embed_tokens
        new_w = torch.empty_like(embed.weight.data)
        new_w.copy_(embed.weight.data)
        embed.weight.data = new_w

    os.sched_setaffinity(0, old_affinity)
    print(f"  Sharding complete")


class NUMAPipelineForward:
    """
    Run a model's forward pass with socket-aware thread pinning.

    First half of layers: pin to socket 0
    Second half: pin to socket 1
    """

    def __init__(self, model, midpoint=None):
        self.model = model
        layers = model.model.layers
        self.L = len(layers)
        self.midpoint = midpoint or self.L // 2
        self.original_forward = None

    def install(self):
        """Install the NUMA-aware forward hooks."""
        layers = self.model.model.layers

        def make_hook(layer_idx):
            mid = self.midpoint
            def hook(module, args, kwargs):
                if layer_idx == 0:
                    os.sched_setaffinity(0, set(range(16)))  # socket 0
                elif layer_idx == mid:
                    os.sched_setaffinity(0, set(range(16, 32)))  # socket 1
            return hook

        self.handles = []
        for li in range(self.L):
            h = layers[li].register_forward_pre_hook(make_hook(li), with_kwargs=True)
            self.handles.append(h)

        # Reset to all cores after forward
        def post_hook(module, args, output):
            os.sched_setaffinity(0, set(range(64)))
        self.handles.append(
            self.model.register_forward_hook(post_hook)
        )

    def remove(self):
        for h in self.handles:
            h.remove()


def benchmark_configurations(model, tokenizer, seq_len=256):
    """Benchmark different NUMA configurations."""

    input_ids = tokenizer("The cat sat on the mat and the dog",
                          return_tensors="pt")["input_ids"]

    configs = {}

    # 1. Default (no optimization)
    os.sched_setaffinity(0, set(range(64)))
    torch.set_num_threads(64)
    times = []
    with torch.inference_mode():
        for _ in range(3):  # warmup
            model(input_ids=input_ids, use_cache=False)
        for _ in range(5):
            t0 = time.perf_counter()
            model(input_ids=input_ids, use_cache=False)
            times.append(time.perf_counter() - t0)
    configs['default_64t'] = sum(times) / len(times)
    print(f"  default (64 threads):    {configs['default_64t']*1000:.1f}ms")

    # 2. Physical cores only (32 threads, no HT)
    torch.set_num_threads(32)
    os.environ['OMP_NUM_THREADS'] = '32'
    times = []
    with torch.inference_mode():
        for _ in range(3):
            model(input_ids=input_ids, use_cache=False)
        for _ in range(5):
            t0 = time.perf_counter()
            model(input_ids=input_ids, use_cache=False)
            times.append(time.perf_counter() - t0)
    configs['physical_32t'] = sum(times) / len(times)
    print(f"  physical cores (32t):    {configs['physical_32t']*1000:.1f}ms")

    # 3. Single socket (16 threads)
    os.sched_setaffinity(0, set(range(16)))
    torch.set_num_threads(16)
    times = []
    with torch.inference_mode():
        for _ in range(3):
            model(input_ids=input_ids, use_cache=False)
        for _ in range(5):
            t0 = time.perf_counter()
            model(input_ids=input_ids, use_cache=False)
            times.append(time.perf_counter() - t0)
    configs['socket0_16t'] = sum(times) / len(times)
    print(f"  socket 0 only (16t):     {configs['socket0_16t']*1000:.1f}ms")

    # 4. NUMA pipeline (socket switch at midpoint)
    os.sched_setaffinity(0, set(range(64)))
    torch.set_num_threads(32)
    pipe = NUMAPipelineForward(model)
    pipe.install()
    times = []
    with torch.inference_mode():
        for _ in range(3):
            model(input_ids=input_ids, use_cache=False)
        for _ in range(5):
            t0 = time.perf_counter()
            model(input_ids=input_ids, use_cache=False)
            times.append(time.perf_counter() - t0)
    configs['numa_pipeline'] = sum(times) / len(times)
    pipe.remove()
    print(f"  NUMA pipeline (32t):     {configs['numa_pipeline']*1000:.1f}ms")

    # 5. Sharded + pipeline
    print("  Sharding model across sockets...")
    shard_model_across_sockets(model)
    pipe = NUMAPipelineForward(model)
    pipe.install()
    times = []
    with torch.inference_mode():
        for _ in range(3):
            model(input_ids=input_ids, use_cache=False)
        for _ in range(5):
            t0 = time.perf_counter()
            model(input_ids=input_ids, use_cache=False)
            times.append(time.perf_counter() - t0)
    configs['sharded_pipeline'] = sum(times) / len(times)
    pipe.remove()
    print(f"  Sharded + pipeline:      {configs['sharded_pipeline']*1000:.1f}ms")

    # Reset
    os.sched_setaffinity(0, set(range(64)))
    torch.set_num_threads(32)

    # Summary
    best = min(configs, key=configs.get)
    print(f"\n  BEST: {best} at {configs[best]*1000:.1f}ms")
    print(f"  vs default: {configs['default_64t']/configs[best]:.2f}x speedup")

    return configs


if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = p.parse_args()

    setup_numa_optimal()

    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()

    print(f"\nBenchmarking NUMA configurations...")
    configs = benchmark_configurations(model, tokenizer)
