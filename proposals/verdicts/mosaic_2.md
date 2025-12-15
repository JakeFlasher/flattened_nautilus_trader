## Comparative Gap & Novelty Table (as of 2025‑12‑15)

| Candidate | Claimed gap (one-liner) | Closest existing work found (papers / repos / docs) | What still looks *truly missing* | Verdict |
|---|---|---|---|---|
| **1) FractalGen** (whole-loop compilation of `generate()` as nested scans/ETDG) | DAG compilers can’t optimize across autoregressive loops; need a compiler that treats the *entire* loop as a first-class operator and emits persistent execution. | **FractalTensor (SOSP’24)** already introduces ETDG + `map/reduce/scan` + compiler passes for nested parallelism. ([microsoft.com](https://www.microsoft.com/en-us/research/publication/uncovering-nested-data-parallelism-and-data-reuse-in-dnn-computation-with-fractaltensor/?utm_source=openai))  **TensorRT‑LLM** (and others) reduce per-token overhead via **CUDA Graphs** (record/replay). ([nvidia.github.io](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html?utm_source=openai)) | A compiler that (a) captures real-world LLM `generate()` control flow (EOS + sampling + KV growth) and (b) **provably** lowers it into a *single reusable* high-order operator with cross-iteration optimization (not just CUDA Graph replay). | **Yellow** |
| **2) Spec-Graph** (compiler IR for *tree-structured* speculative decoding) | Tree speculation is “runtime-managed”; need an IR that fuses verification + optimizes tree memory/layout. | **SpecInfer (ASPLOS’24)** already does tree-based speculative inference + parallel verification. ([asplos-conference.org](https://www.asplos-conference.org/asplos2024/main-program/abstracts/index.html?utm_source=openai))  **DeFT (ICLR’25)** targets tree-structured inference with Flash Tree-Attention. ([arxiv.org](https://arxiv.org/abs/2404.00242?utm_source=openai))  **TensorRT‑LLM** and **vLLM** both ship speculative decoding features (Medusa/EAGLE/etc.). ([nvidia.github.io](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html?utm_source=openai)) | A *compiler-first* abstraction that makes token trees first-class **and** yields advantages beyond “better tree attention kernels” (e.g., end-to-end fused verify+sample pipelines, generalized to many tree policies, with strong correctness proofs). | **Yellow** |
| **3) Mosaic** (cross-kernel fusion via symbolic layout propagation on TileLang) | Big inefficiency is *between* kernels (layout discontinuity). Need compiler to “negotiate” layouts across kernels and emit fused megakernels. | **TileLang (2025)** provides tiled DSL + layout inference *within* kernels. ([arxiv.org](https://arxiv.org/abs/2504.17577?utm_source=openai))  **TensorRT‑LLM** fuses many patterns but admits complex fusions (e.g., FlashAttention-like) may require explicit plugins. ([nvidia.github.io](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html?utm_source=openai))  **TileFusion** is a macro-kernel/tile library (not graph-level cross-kernel layout negotiation). ([github.com](https://github.com/microsoft/TileFusion)) | A **general cross-kernel** layout constraint system that can fuse independently authored “hero kernels” without rewriting them monolithically, keeping intermediates in registers/SMEM where legal. | **Green** |
| **4) Triton‑V** (virtualized tiling for ragged/paged tensors; automate indirection/prefetch) | High-perf ragged kernels (PagedAttention/MoE) require painful manual pointer math + scheduling; need compiler support. | **vLLM** uses a dedicated CUDA attention kernel compatible with paged KV caches. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/paged_attention.html?utm_source=openai))  **FlashInfer** exposes paged KV attention APIs (with page tables). ([docs.flashinfer.ai](https://docs.flashinfer.ai/api/attention.html?utm_source=openai))  Triton community is still discussing **PagedAttention optimization gaps** (esp. ROCm). ([github.com](https://github.com/triton-lang/triton/issues/8281?utm_source=openai)) | A Triton-level IR/type abstraction for ragged/paged tensors with middle-end passes that **systematically** hoist/overlap indirection and preserve coalescing guarantees. | **Green / Yellow** (green on idea, yellow on feasibility/upstream risk) |
| **5) Forge** (GPU-fused FSM mask+softmax+sampling for constrained generation) | Current constrained decoding does host-device ping-pong; fuse FSM traversal into sampling kernel. | **XGrammar** is an open-source structured generation engine targeting near-zero overhead; integrated into **vLLM, SGLang, TensorRT‑LLM** etc. ([github.com](https://github.com/mlc-ai/xgrammar?utm_source=openai))  **SGLang** already lists “compressed finite state machines” for structured decoding. ([papers.nips.cc](https://papers.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html?utm_source=openai)) | Only niche: a **GPU-resident DFA** design that materially outperforms XGrammar across engines *and* avoids its approach (overlap/co-design) by truly in-kernel masking+sampling—hard to justify now. | **Red** |
| **6) BitStream** (layout-aware type promotion for mixed-precision fusion; Inductor + Triton) | Compilers can’t fuse across quantized layouts; need layout-aware type system and auto-fused kernels. | **TorchAO** already defines quantized tensor types (e.g., Int4*) and integrates quantization flows + kernels; docs show configs that use **gemlite Triton kernels** with specific packing formats. ([docs.pytorch.org](https://docs.pytorch.org/ao/stable/quantization_overview.html?utm_source=openai))  **GemLite** provides low-bit Triton matmul kernels and compares vs TorchAO/vLLM kernels. ([github.com](https://github.com/dropbox/gemlite?utm_source=openai)) | A *compiler-wide* (not library-specific) analysis that chooses fusion scopes, reconciles multiple packing formats, and fuses quantized ops with downstream epilogues in a principled way—could still be publishable but “already crowded.” | **Yellow** |
| **7) Auto‑Jagged** (compile nested parallelism → block-sparse Triton kernels for dynamic MoEs) | Jagged MoE workloads rely on hand-written “hero kernels”; need compiler synthesis to Triton from nested IR. | **MegaBlocks** is an MoE training library with grouped GEMM paths and strong performance claims. ([github.com](https://github.com/databricks/megablocks?utm_source=openai))  **vLLM** documents multiple **Triton MoE** kernel “flavors” and compatibility constraints. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/moe_kernel_features.html?utm_source=openai))  PyTorch blog describes a **Triton persistent cache-aware grouped GEMM kernel** for MoE with up to **2.62×** speedup on H100. ([pytorch.ac.cn](https://pytorch.ac.cn/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/?utm_source=openai)) | Compiler synthesis from a *nested-parallel* IR (ETDG-style) into Triton block pointers, especially for ragged routing + fusion around routing boundaries—still nontrivial, but the “no middle-end support exists” claim is weaker now. | **Yellow** |

---

## Candidate-by-candidate deep gap analysis

### 1) **FractalGen** — “Compile the entire autoregressive loop as nested scans”

#### (1) Claimed Gap
You claim: DAG compilers (XLA/Inductor-like) see only one step of decode, so they can’t optimize across iterations; the remedy is to **lift the while-loop** into a high-order operator (nested `scan`) and do whole-loop optimization (prefill+decode fusion, persistent kernels, paged layouts via indirect access).

#### (2) Reality (Evidence)
- **FractalTensor already exists and explicitly targets nested parallelism** using a FractalTensor ADT + `map/reduce/scan`, extracting an **Extended Task Dependence Graph (ETDG)** and applying compiler transformations like graph coarsening, data reordering, and access materialization. This is not hypothetical—there is a SOSP’24 paper and an active open-source repo. ([microsoft.com](https://www.microsoft.com/en-us/research/publication/uncovering-nested-data-parallelism-and-data-reuse-in-dnn-computation-with-fractaltensor/?utm_source=openai))  
- Meanwhile, the *systems* community has attacked the “per-token overhead” problem largely through **CUDA Graph record/replay** (and related scheduling tricks). TensorRT‑LLM explicitly documents CUDA Graphs as a way to reduce CPU-side kernel launch overhead and discusses padding strategies to maximize graph reuse. ([nvidia.github.io](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html?utm_source=openai))  

So: the “kernel launch overhead crisis” is real, but there is already a mainstream remedy (CUDA Graphs), and the nested-IR angle is already established in FractalTensor.

#### (3) What’s still *open* (and could be publishable)
If you push beyond “we can represent a loop,” the real novelty could be:
- **Correct loop lifting for realistic generation**: EOS termination, sampling, KV-cache growth, request batching, and dynamic attention shapes—captured as a *single* reusable operator with a correctness story (not just tracing + replay).  
- **Cross-iteration memory planning**: e.g., prove that certain intermediate states can remain on-chip across iterations (or across micro-iterations within a persistent kernel), with measurable wins beyond CUDA Graph replay.

#### (4) Verdict: **Yellow**
Strong intellectual framing, but the core primitives (ETDG + scan) are already “owned” by FractalTensor, and the deployment pain point (launch overhead) is already widely mitigated via CUDA Graphs. ([microsoft.com](https://www.microsoft.com/en-us/research/publication/uncovering-nested-data-parallelism-and-data-reuse-in-dnn-computation-with-fractaltensor/?utm_source=openai))  

---

### 2) **Spec-Graph** — “Compiler IR for tree-structured speculation”

#### (1) Claimed Gap
You claim: tree-based speculative decoding is held back by **runtime-managed** control flow and many small kernels; a compiler IR with `spec.fork/spec.verify` would allow tree-aware fusion and better layouts.

#### (2) Reality (Evidence)
This space is already very active, and multiple systems/papers directly target tree-structured decoding:

- **SpecInfer** (ASPLOS’24; arXiv 2023) explicitly introduces tree-based speculative inference and a “tree-based parallel decoding mechanism” for verification. ([asplos-conference.org](https://www.asplos-conference.org/asplos2024/main-program/abstracts/index.html?utm_source=openai))  
- **DeFT** (arXiv 2024; ICLR 2025) targets “tree-structured LLM inference” and proposes **Flash Tree-attention** with prefix-aware and load-balanced KV partitions, reporting substantial speedups by reducing KV-cache IO. ([arxiv.org](https://arxiv.org/abs/2404.00242?utm_source=openai))  
- On the production side, **TensorRT‑LLM** provides speculative decoding and includes Medusa/Medusa Tree as a documented option. ([nvidia.github.io](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html?utm_source=openai))  
- **vLLM** also documents speculative decoding support (including EAGLE-based draft models) and discusses limitations/ongoing performance investigations. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/spec_decode.html?utm_source=openai))  

So the claimed “nobody has solved tree verification efficiently” is not accurate. The open question is whether a *compiler IR* can systematically outperform a specialized systems approach like SpecInfer/DeFT.

#### (3) What’s still *open*
A credible “Green” path exists, but it must clear a high bar:
- Show that your IR enables **general fusion beyond attention** (e.g., gather/scatter of tree nodes, KV layout management, verification + sampling) and not just yet-another tree attention kernel.  
- Demonstrate **policy generality**: support Medusa-like trees, EAGLE-like proposals, and new research policies without bespoke kernel rewrites—something systems papers often struggle with.  
- Provide a strong correctness argument: tree masking correctness is subtle; reviewers will look for formalization (or mechanized testing).

#### (4) Verdict: **Yellow**
Because tree-structured inference has already been tackled (SpecInfer, DeFT, vendor engines), the “gap” must be reframed: *not* “tree speculation is inefficient,” but “there is no compiler IR that makes tree strategies easy to innovate on while preserving near-SoTA performance.” ([asplos-conference.org](https://www.asplos-conference.org/asplos2024/main-program/abstracts/index.html?utm_source=openai))  

---

### 3) **Mosaic** — “Layout-aware cross-kernel fusion using TileLang symbolic propagation”

#### (1) Claimed Gap
You claim: even when each kernel is optimal, **layout discontinuity** between kernels forces global memory spills, because compilers treat custom kernels as opaque and don’t understand their internal fragment/layout semantics.

#### (2) Reality (Evidence)
- **TileLang** exists as a 2025 paper + active open-source project, emphasizing explicit tiling, layouts, fragments, pipelining, and hardware-aware scheduling, largely at the **single-kernel** level. ([arxiv.org](https://arxiv.org/abs/2504.17577?utm_source=openai))  
- Mainstream graph compilers *can* fuse many patterns, but there are still “hard cases.” Even TensorRT‑LLM notes that some complex fusions (FlashAttention-like) often cannot be automatically discovered and may require explicit plugins. That’s consistent with your “opaque kernel” diagnosis. ([nvidia.github.io](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html?utm_source=openai))  
- Macro-kernel libraries (e.g., Microsoft’s **TileFusion**) exist and explicitly provide a tile-level abstraction for building macro-kernels and fused dataflows—however, this still largely assumes you’re building a macro-kernel intentionally, not auto-fusing two independently-authored kernels inside a graph compiler. ([github.com](https://github.com/microsoft/TileFusion))  

In other words: the *general* kernel fusion story exists, but “fuse across two specialized layout choices without forcing a canonical relayout” still looks like a real gap.

#### (3) Why Mosaic still looks **genuinely novel**
Your key novelty is not “fusion exists,” but:
- **Fusion under nontrivial layout constraints** (Tensor Core swizzles / register fragments).  
- **Grey-box kernel composition**: instead of treating kernels as black boxes (can’t fuse) or requiring a monolithic rewrite (painful), you stitch IRs and solve layout constraints across boundaries.

I did not find an existing, widely-cited system that does this *generally* for GPU kernel DSL outputs (especially not with register-fragment semantics carried across operators). TileFusion and CUTLASS-style epilogues cover important special cases, but not general cross-kernel layout negotiation. ([github.com](https://github.com/microsoft/TileFusion))  

#### (4) What you must nail for OSDI/MLSys
Reviewers will likely focus on:
- **Correctness rules**: when is it legal to keep intermediates in registers vs shared memory?  
- **Synchronization synthesis**: barrier insertion, warpgroup semantics, and portability (NVIDIA vs AMD).  
- **Cost model**: when does fusion hurt (occupancy/register pressure)?  
- **End-to-end wins**: prove it on real subgraphs (GEMM+RMSNorm+activation; attention epilogues; quantized epilogues).

#### (5) Verdict: **Green**
This is the cleanest “compiler middle-end gap” among your candidates: it targets a pain point that *production systems acknowledge* (hard-to-discover complex fusions), and it exploits a new enabling substrate (TileLang layout objects) that makes the approach believable. ([nvidia.github.io](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html?utm_source=openai))  

---

### 4) **Triton‑V** — “Virtualized tiling for ragged/paged tensors”

#### (1) Claimed Gap
You claim: ragged/indirect workloads (PagedAttention, MoE routing) break Triton’s static tiling assumptions; developers must do manual page-table loads and pointer math.

#### (2) Reality (Evidence)
- **vLLM’s current paged KV-cache attention uses a custom CUDA kernel** designed specifically for paged KV caches. Their own docs explicitly point to a CUDA source file and a “specially designed memory layout and access method.” ([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/paged_attention.html?utm_source=openai))  
- **FlashInfer** exposes attention APIs that take paged KV cache structures (page tables / indices / indptrs) and even accounts for speculative decoding shapes in its attention API signatures. ([docs.flashinfer.ai](https://docs.flashinfer.ai/api/attention.html?utm_source=openai))  
- Triton itself: there are still active discussions about closing the performance gap for **Paged Attention** in Triton (notably on ROCm), which is evidence that this is nontrivial and not “solved by the compiler.” ([github.com](https://github.com/triton-lang/triton/issues/8281?utm_source=openai))  

So the “manual indirection + performance gap” claim is consistent with reality: the best-known implementations are hand-tuned kernels or specialized libraries, and Triton-level solutions are still evolving.

#### (3) What remains open
- A first-class **ragged/paged type system** in a kernel compiler that can reason about (a) indirection latency, (b) coalescing across pages, (c) safe vectorization boundaries, and (d) pipelined address translation.  
- The compelling part is your proposed **middle-end**: “indirection-aware pipelining” and hoisting table loads.

#### (4) Risks / feasibility notes
- Modifying Triton’s compiler stack (IR + passes + backend) is heavy engineering and may be difficult to upstream/maintain.  
- You must show robust wins across multiple models and across NVIDIA/AMD (or at least explain portability).

#### (5) Verdict: **Green / Yellow**
Green on “real gap + impactful workload,” yellow on execution risk (deep compiler surgery, strong baselines exist, and upstream Triton/engine communities are actively iterating). ([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/paged_attention.html?utm_source=openai))  

---

### 5) **Forge** — “Fused FSM kernels for constrained text generation”

#### (1) Claimed Gap
You claim: constrained decoding still does host-device ping-pong (CPU FSM traversal, GPU logits), so fuse DFA traversal + masking + sampling in one GPU kernel.

#### (2) Reality (Evidence)
This appears **directly pre-empted** by XGrammar and partially by SGLang’s own contributions:

- **XGrammar** is an open-source library explicitly targeting “structured generation” (JSON/regex/CFG), claiming “near-zero overhead,” and it is integrated into **vLLM, SGLang, TensorRT‑LLM** (and others). ([github.com](https://github.com/mlc-ai/xgrammar?utm_source=openai))  
- **SGLang (NeurIPS 2024)** lists “compressed finite state machines” as a runtime optimization for faster structured output decoding. ([papers.nips.cc](https://papers.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html?utm_source=openai))  

Even if Forge’s mechanism differs (GPU-resident DFA vs XGrammar’s approach), the *problem statement* “this is a big unsolved bottleneck” is no longer credible.

#### (3) Verdict: **Red**
To be publishable, Forge would have to convincingly beat a widely adopted, purpose-built structured generation engine that already claims near-zero overhead and is integrated across major inference stacks. That’s an extremely high bar for novelty and “gap validity.” ([github.com](https://github.com/mlc-ai/xgrammar?utm_source=openai))  

---

### 6) **BitStream** — “Layout-aware type promotion for mixed-precision fusion (Inductor + Triton), evaluated with PyTorchSim”

#### (1) Claimed Gap
You claim: general-purpose compilers cannot reason about sub-byte packing layouts (INT4/NF4/etc.) and therefore fail to fuse dequantization + matmul + epilogues.

#### (2) Reality (Evidence)
A lot of this is already being industrialized:

- **TorchAO** documents a quantization stack with quantized tensor types (e.g., Int4-derived tensors) and quantization primitives/kernels, designed for composability with `torch.compile`. ([docs.pytorch.org](https://docs.pytorch.org/ao/stable/quantization_overview.html?utm_source=openai))  
- TorchAO’s API docs show configs like **GemliteUIntXWeightOnlyConfig**, explicitly describing “weight-only 4 or 8 bit integer quantization” and that it “utilizes the gemlite triton kernel and its associated weight packing format.” ([docs.pytorch.org](https://docs.pytorch.org/ao/stable/_modules/torchao/quantization/quant_api.html?utm_source=openai))  
- Independent projects like **GemLite** exist to provide fast low-bit matmul kernels in Triton and compare end-to-end performance vs TorchAO/vLLM kernels. ([github.com](https://github.com/dropbox/gemlite?utm_source=openai))  

So the underlying *capability* (register-local dequant inside matmul kernels, with specific packing formats) is not only present—it’s becoming standard practice via TorchAO + specialized kernels.

On evaluation: **PyTorchSim** is real and reputable (MICRO’25) and provides a simulation-based workflow for PyTorch graphs and ISA experiments. ([psal.postech.ac.kr](https://www.psal.postech.ac.kr/pytorchsim-tutorial?utm_source=openai))  
But “we used PyTorchSim” is not novelty by itself; it’s a methodology choice.

#### (3) What remains open
There’s still a plausible research contribution if you can show:
- A **compiler-general** representation of packing layouts (not tied to one kernel lib), and  
- A **graph-level fusion** strategy that reliably fuses quantized ops with downstream ops *without* requiring users to opt into a specific kernel backend.

#### (4) Verdict: **Yellow**
The gap exists in some forms, but the space is now crowded and fast-moving (TorchAO + GemLite + many other vendor/kernel efforts). Your positioning must be carefully differentiated as “compiler middle-end generality,” not “we fused dequant into matmul.” ([docs.pytorch.org](https://docs.pytorch.org/ao/stable/_modules/torchao/quantization/quant_api.html?utm_source=openai))  

---

### 7) **Auto‑Jagged** — “Compile nested parallelism → Triton block-sparse kernels for dynamic MoEs”

#### (1) Claimed Gap
You claim: dynamic MoE/jagged tensors force padding or require hero kernels; compilers can’t fuse around routing; Auto-Jagged synthesizes Triton kernels from nested parallel IR (ETDG).

#### (2) Reality (Evidence)
This domain is also highly active, and many “hero kernels” are now reusable libraries or even Triton-based:

- **MegaBlocks** (Databricks) is a public MoE training library emphasizing efficient dropless MoE, grouped GEMM paths, and strong performance comparisons. ([github.com](https://github.com/databricks/megablocks?utm_source=openai))  
- **vLLM** documents its **Fused MoE kernel** ecosystem, including Triton-based expert implementations, activation support, quantization formats, and modular interfaces. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/moe_kernel_features.html?utm_source=openai))  
- PyTorch (blog) describes an **optimized Triton grouped GEMM kernel** for MoE with “persistent cache-aware” design and reports up to **2.62×** speedup on H100 vs a Python loop baseline. ([pytorch.ac.cn](https://pytorch.ac.cn/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/?utm_source=openai))  

So the “only hand-written CUDA hero kernels exist” claim is outdated.

#### (3) What remains open
Auto-Jagged can still be publishable if it delivers something these works don’t:
- Automatic **end-to-end compilation** from a higher-level nested-parallel IR (ETDG-like) into Triton kernels *without* users hand-authoring grouped GEMMs, routing permutations, and boundary logic.  
- **Fusion across routing boundaries** (router + permute + experts + unpermute + epilogue), not just “a faster grouped GEMM.”

#### (4) Verdict: **Yellow**
There is still a compiler gap, but the baseline landscape is now strong and includes high-quality open-source kernels and even official Triton-based MoE kernel work. You’ll need a sharper “automation + generality” story. ([github.com](https://github.com/databricks/megablocks?utm_source=openai))  

---

## Final Recommendation: **Pick Mosaic (Candidate 3)**

### Why Mosaic is the best “top-tier-safe” bet
Among the seven, **Mosaic** best satisfies the *top-tier conference triangle*:

1. **Clear, still-valid gap**: Production compilers can fuse many things, but complex/hardware-specific layouts still create “fusion discontinuities,” and even vendor stacks acknowledge that some complex fusions aren’t discovered automatically and require plugins. ([nvidia.github.io](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html?utm_source=openai))  
2. **New enabling substrate**: TileLang gives you explicit layout/fragment abstractions and a real compiler toolchain to build on, which makes “layout-aware fusion” technically plausible rather than aspirational. ([arxiv.org](https://arxiv.org/abs/2504.17577?utm_source=openai))  
3. **High impact with measurable end-to-end wins**: Cross-kernel fusion that eliminates global-memory round trips can plausibly deliver the kind of 1.4–2× end-to-end wins that MLSys/OSDI reviewers like—*especially* on inference pipelines dominated by memory traffic.

### How to position Mosaic to beat reviewers
To make it a “slam dunk,” I would explicitly position Mosaic **against two existing paradigms**:

- **Epilogue fusion (CUTLASS/TensorRT)**: great, but limited to specific operator families; Mosaic generalizes by solving layout constraints across a broader subgraph. ([developer.nvidia.com](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/?utm_source=openai))  
- **Macro-kernel libraries (TileFusion-like)**: powerful but typically require building a macro-kernel intentionally; Mosaic auto-fuses separately-authored kernels by IR composition + constraint solving. ([github.com](https://github.com/microsoft/TileFusion))  

### If you need a “runner-up”
If Mosaic feels too “compiler-y” and you want a more systems-facing LLM-serving story, **Triton‑V** is the runner-up: the gap around ragged/paged compilation is real and still not solved at the DSL/compiler level, but it carries higher implementation and baselining risk. ([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/paged_attention.html?utm_source=openai))  

---

## One-line verdict list (for quick decision)
- **FractalGen**: Yellow (good story, but ETDG/scan is already established; CUDA Graphs already mitigate key pain). ([microsoft.com](https://www.microsoft.com/en-us/research/publication/uncovering-nested-data-parallelism-and-data-reuse-in-dnn-computation-with-fractaltensor/?utm_source=openai))  
- **Spec-Graph**: Yellow (crowded: SpecInfer + DeFT + vendor implementations; must differentiate as *compiler generality*). ([arxiv.org](https://arxiv.org/abs/2305.09781?utm_source=openai))  
- **Mosaic**: **Green** (cleanest unsolved middle-end gap; strong publishability). ([nvidia.github.io](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html?utm_source=openai))  
- **Triton‑V**: Green/Yellow (real gap, high engineering risk). ([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/paged_attention.html?utm_source=openai))  
- **Forge**: **Red** (XGrammar + SGLang already occupy this space). ([github.com](https://github.com/mlc-ai/xgrammar?utm_source=openai))  
- **BitStream**: Yellow (TorchAO + gemlite + many kernels already; needs sharper compiler-general angle). ([docs.pytorch.org](https://docs.pytorch.org/ao/stable/_modules/torchao/quantization/quant_api.html?utm_source=openai))  
- **Auto‑Jagged**: Yellow (MoE kernels + Triton grouped GEMM are well-developed; must emphasize automation + fusion). ([github.com](https://github.com/databricks/megablocks?utm_source=openai))

---
Learn more:
1. [Uncovering Nested Data Parallelism and Data Reuse in DNN Computation with FractalTensor - Microsoft Research](https://www.microsoft.com/en-us/research/publication/uncovering-nested-data-parallelism-and-data-reuse-in-dnn-computation-with-fractaltensor/?utm_source=openai)
2. [Architecture Overview — TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html?utm_source=openai)
3. [Paper Abstracts – ASPLOS 2024](https://www.asplos-conference.org/asplos2024/main-program/abstracts/index.html?utm_source=openai)
4. [DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference](https://arxiv.org/abs/2404.00242?utm_source=openai)
5. [Speculative Sampling — TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html?utm_source=openai)
6. [TileLang: A Composable Tiled Programming Model for AI Systems](https://arxiv.org/abs/2504.17577?utm_source=openai)
7. [Model Definition — TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html?utm_source=openai)
8. [GitHub - microsoft/TileFusion: TileFusion is an experimental C++ macro kernel template library that elevates the abstraction level in CUDA C for tile processing.](https://github.com/microsoft/TileFusion)
9. [Paged Attention - vLLM](https://docs.vllm.ai/en/latest/design/paged_attention.html?utm_source=openai)
10. [FlashInfer Attention Kernels - FlashInfer 0.5.2 documentation](https://docs.flashinfer.ai/api/attention.html?utm_source=openai)
11. [\[RFC\]\[AMD\] Optimizations for Paged Attention: Proposal with Multiple Features · Issue #8281 · triton-lang/triton · GitHub](https://github.com/triton-lang/triton/issues/8281?utm_source=openai)
12. [GitHub - mlc-ai/xgrammar: Fast, Flexible and Portable Structured Generation](https://github.com/mlc-ai/xgrammar?utm_source=openai)
13. [SGLang: Efficient Execution of Structured Language Model Programs](https://papers.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html?utm_source=openai)
14. [Quantization Overview — torchao 0.13 documentation](https://docs.pytorch.org/ao/stable/quantization_overview.html?utm_source=openai)
15. [GitHub - dropbox/gemlite: Fast low-bit matmul kernels in Triton](https://github.com/dropbox/gemlite?utm_source=openai)
16. [GitHub - databricks/megablocks](https://github.com/databricks/megablocks?utm_source=openai)
17. [Fused MoE Kernel Features - vLLM](https://docs.vllm.ai/en/latest/design/moe_kernel_features.html?utm_source=openai)
18. [通过 Triton 持久化缓存感知分组 GEMM 内核加速 MoE – PyTorch - PyTorch 框架](https://pytorch.ac.cn/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/?utm_source=openai)
19. [Speculative Decoding - vLLM](https://docs.vllm.ai/en/latest/features/spec_decode.html?utm_source=openai)
20. [torchao.quantization.quant\_api — torchao 0.13 documentation](https://docs.pytorch.org/ao/stable/_modules/torchao/quantization/quant_api.html?utm_source=openai)
21. [Tutorial | PSAL at POSTECH](https://www.psal.postech.ac.kr/pytorchsim-tutorial?utm_source=openai)
22. [CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/?utm_source=openai)
23. [SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/abs/2305.09781?utm_source=openai)
