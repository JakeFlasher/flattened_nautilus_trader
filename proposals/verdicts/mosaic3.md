## Gap analysis (web-validated) for 7 proposals (as of **2025-12-15**)

You asked for a novelty / overlap check against *current* literature, arXiv, and active repos, to decide which proposal has the highest value for an OSDI/MLSys submission. I treated each proposal as a *claims → evidence → remaining gap* exercise, and I biased toward **recent (2024–2025)** systems that could make a proposal obsolete.

Key sources I relied on include: FractalTensor (SOSP’24), SpecInfer (arXiv’23 / ASPLOS’24 + FlexFlow), TileLang (arXiv’25 + GitHub), FlashInfer (paged attention + speculative support), XGrammar (arXiv’24 + MLSys’25 poster + integrations), PyTorch control-flow compilation primitives, torchao + GemLite for quantization kernels, and the recent PyTorch Triton MoE kernel blogs. ([microsoft.com](https://www.microsoft.com/en-us/research/publication/uncovering-nested-data-parallelism-and-data-reuse-in-dnn-computation-with-fractaltensor/))

---

## Comparative table (quick decision aid)

| Candidate | Claimed gap (one-liner) | Strongest existing overlap (papers/systems) | What’s *already* solved today | What’s plausibly still novel | Verdict |
|---|---|---|---|---|---|
| **1) FractalGen** | Compile the *entire* autoregressive loop (while/EOS) into nested scans (ETDG), enabling “whole-loop” fusion + paged KV + persistent kernels | FractalTensor ETDG/scan abstraction (SOSP’24); CUDA Graph based loop overhead reduction in practice; PyTorch structured control-flow ops (`cond`, `while_loop`) | Nested-parallel IR (ETDG) exists; many deployments reduce overhead via CUDA Graphs; structured control-flow ops exist but must be written explicitly and are still evolving | Automatic lifting of Python “generate()” loops into structured ops + aggressive whole-loop optimizations on GPU backends (beyond CUDA Graph replay constraints) | **Yellow** |
| **2) Spec-Graph** | Compiler IR for **tree-structured speculative decoding**; fuse tree verification + better memory layout than paging | **SpecInfer** (arXiv’23 / ASPLOS’24, FlexFlow); TensorRT-LLM speculative sampling (Medusa tree, EAGLE); FlashInfer speculative-aware attention interfaces | Tree-based verification is already a full system (SpecInfer); vendors support multiple speculative methods in-engine; kernels exist for “multi-token” attention shapes | A **general compiler IR** that unifies tree topology + KV layout + fusion across frameworks (not just one engine) | **Yellow** |
| **3) Mosaic** | **Cross-kernel** fusion by symbolic propagation/unification of TileLang `Layout`/`Fragment` across kernel boundaries | TileLang already does **intra-kernel** layout inference with `Layout`/`Fragment`/`LayoutMap`; other “mega-kernel” fusion papers (ClusterFusion) exist but not layout negotiation between *separately authored* kernels | Intra-kernel layout inference and fragment layouts are in TileLang today | **Inter-kernel** / graph-level layout constraint solving + automatic AST stitching of independently developed kernels | **Green** |
| **4) Triton-V** | Make ragged/paged tensors first-class in Triton IR; auto-hoist page-table indirections + pipelining | vLLM/PagedAttention (SOSP’23); FlashInfer paged attention + index-prefetch; Triton has descriptors/gather/scatter primitives but not ragged types | High-perf paged attention exists as libraries; some indirection latency hiding is hand-done in kernels | Compiler/IR-level **virtualized tiling** for indirection-heavy kernels (PagedAttention, MoE gathers), with productivity + correctness wins | **Green** |
| **5) Forge** | GPU-fused FSM traversal + masking + sampling for constrained decoding; compress DFA via vocab clustering | **XGrammar** (arXiv’24; MLSys’25 poster) integrated into vLLM/SGLang/TensorRT-LLM; SGLang compressed FSM | Structured generation overhead has been heavily attacked; engines already integrate XGrammar; masking application can be GPU kernel | Fully GPU-resident matcher + sampler *might* still be new, but practical gap is much smaller now | **Red** (at best Yellow with a pivot) |
| **6) BitStream** | Inductor pass: layout-aware type promotion to fuse dequant in registers; validate via PyTorchSim | torchao quantized tensor stack + low-bit kernels; GemLite integrated into torchao; Hexcute DSL targets mixed-type ops; PyTorchSim exists (MICRO’25) | Many fused low-bit kernels exist and are shipping; “layout” concepts already appear in torchao’s quantized tensor types and kernel configs | A *general* compiler analysis that fuses across mixed-precision boundaries beyond matmul-only, plus simulator-driven ISA exploration | **Yellow** |
| **7) Auto-Jagged** | Compile nested parallelism → block-sparse Triton kernels for dynamic MoE/SSM; replace “hero kernels” | MegaBlocks; Triton grouped GEMM tutorial; PyTorch Triton MoE kernels (persistent cache-aware grouped GEMM); Triton-distributed (MoE operators) | Triton-based MoE kernels and grouped GEMM are mainstream; strong hand-tuned and semi-automated solutions exist | Automatic synthesis from a higher-level nested-IR (ETDG → Triton) + jagged fusion analysis could still be publishable | **Yellow** |

---

# 1) FractalGen (whole-loop compilation via nested scans)

### 1) Claimed gap
FractalGen claims compilers are stuck in “single-step DAG tracing” for decoding, leaving the **autoregressive while loop** (EOS termination + KV-cache growth) as a runtime problem. It proposes representing generation as nested `scan` operators (via ETDG / FractalTensor), enabling **whole-loop fusion**, **paged layout synthesis**, and **persistent kernels**.

### 2) Reality (evidence)
- **FractalTensor already provides the core conceptual machinery**: nested list ADT + high-order ops (map/reduce/scan) and an **Extended Task Dependence Graph (ETDG)** that enables whole-program analysis and optimizations across operator boundaries (graph coarsening, data reordering, access materialization). ([microsoft.com](https://www.microsoft.com/en-us/research/publication/uncovering-nested-data-parallelism-and-data-reuse-in-dnn-computation-with-fractaltensor/))
- **In deployed inference engines**, a major practical mitigation for decode overhead is **CUDA Graphs**: capture a stable graph and replay it to reduce per-kernel launch overhead. NVIDIA documents meaningful gains in token eval by reducing launch gaps (illustrated for llama.cpp). ([developer.nvidia.com](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/?utm_source=openai))
- **But** CUDA Graphs (and similar capture/replay techniques) fundamentally struggle with **data-dependent control flow**. PyTorch explicitly notes control-flow like `torch.cond` is “typically banned during stream capture” because it can induce device→host sync (e.g., `.item()` on a predicate). ([github.com](https://github.com/pytorch/pytorch/issues/168911?utm_source=openai))
- PyTorch is moving toward **structured control-flow operators** (`torch.cond`, `torch._higher_order_ops.while_loop`) that preserve control-flow in graphs, but they are **prototype** features and generally require the user to write models using these ops rather than expecting Dynamo to infer them from arbitrary Python while loops. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/cond.html?utm_source=openai))  
  (There are also ongoing failure reports with `while_loop` under `torch.compile`, indicating maturity gaps.) ([github.com](https://github.com/pytorch/pytorch/issues/160939?utm_source=openai))

### 3) What remains genuinely open (and could be your “publishable delta”)
If you pursue FractalGen, the novel core must be **not** “scan exists,” but:

1. **Automatic loop lifting**: robustly turning “imperative HF `generate()`-style Python” into structured control flow / scan, including EOS and dynamic max length. PyTorch’s current story is “use `while_loop` explicitly”; FractalGen would automate that lift. ([docs.pytorch.org](https://docs.pytorch.org/xla/master/perf/fori_loop.html?utm_source=openai))  
2. **Whole-loop memory planning for KV cache**: compilers still don’t treat KV cache as a first-class, growing state through the decode loop (engines do). You’d need a memory planning + indirection strategy competitive with PagedAttention-style paging. ([arxiv.org](https://arxiv.org/abs/2309.06180?utm_source=openai))  
3. **GPU execution model beyond capture/replay**: something like persistent kernels or device-side scheduling that remains correct under termination and batching.

### 4) Verdict: **Yellow**
There is real novelty potential, but parts of the “missing abstraction” story are weakened by (a) FractalTensor already existing for nested parallelism, and (b) PyTorch introducing structured control flow ops. ([microsoft.com](https://www.microsoft.com/en-us/research/publication/uncovering-nested-data-parallelism-and-data-reuse-in-dnn-computation-with-fractaltensor/))  
This can still be top-tier if you demonstrate an end-to-end system that compilers *still* can’t do, but it’s a high-scope, high-risk build.

---

# 2) Spec-Graph (compiler IR for tree-structured speculation)

### 1) Claimed gap
Spec-Graph claims tree-based speculative decoding has **algorithmic** speedups but is bottlenecked by runtime orchestration: building/verifying token trees incurs CPU overhead and kernel launch overhead. It proposes an IR (e.g., MLIR dialect) with `spec.fork/spec.verify`, plus fusion and a tree-local memory allocator (“arena” instead of paging).

### 2) Reality (evidence)
- **SpecInfer already exists** and is *explicitly* about token-tree speculative inference + parallel verification. It’s on arXiv (2023) and appeared at ASPLOS 2024; it’s integrated into FlexFlow and is open-source. ([arxiv.org](https://arxiv.org/abs/2305.09781?utm_source=openai))
- **TensorRT-LLM has production support for speculative decoding** and explicitly discusses Medusa tree, EAGLE, etc. The docs also state some speculative methods perform **draft token acceptance inside the TensorRT engine** (i.e., not a Python-level orchestrated outer loop). ([nvidia.github.io](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html?utm_source=openai))
- **FlashInfer provides “speculative decoding–aware” attention interfaces**, where the query tensor includes a `q_seq_len` dimension “if using speculative decoding.” That’s not a full speculative system, but it shows the kernel libraries already expose shapes and kernels aligned with speculation. ([docs.flashinfer.ai](https://docs.flashinfer.ai/api/attention.html?utm_source=openai))

### 3) What’s still open / how to reframe to be novel
To keep Spec-Graph from feeling “reinventing SpecInfer,” you need to pin the novelty on **compiler generality** and **interoperability**:

- **A reusable IR** that can express *multiple* speculative strategies (draft model, Medusa-style multihead trees, EAGLE-like dynamic trees) and lower them to efficient kernels across backends—while SpecInfer is one system and TensorRT-LLM is vendor-specific. ([arxiv.org](https://arxiv.org/abs/2305.09781?utm_source=openai))  
- A genuinely new piece could be **tree-aware KV layouts** and fusion with paged attention kernels (or avoidance of paging). But you must show this is not already effectively handled by existing specialized engines for your target regime.

### 4) Verdict: **Yellow**
Spec-Graph is directionally interesting, but **tree speculation systems already exist** (SpecInfer), and **vendor stacks already “compile” speculation into their engines** to some extent (TensorRT-LLM). ([arxiv.org](https://arxiv.org/abs/2305.09781?utm_source=openai))  
To be top-tier, you’ll need a strong “compiler IR unlocks *new* portability + new fusion + new correctness guarantees” narrative and results that beat or generalize beyond these baselines.

---

# 3) Mosaic (layout-aware cross-kernel fusion with TileLang)

### 1) Claimed gap
Mosaic targets the “**layout discontinuity**” problem: kernels produce/consume non-standard layouts (swizzled shared memory layouts, MMA accumulator fragments in registers). Existing compilers treat custom kernels as opaque; so intermediate results spill to HBM between kernels.

### 2) Reality (evidence)
- **TileLang already has explicit layout and fragment abstractions** and an internal “Layout Inference” mechanism. The TileLang paper describes:
  - explicit allocations to shared/fragment (register-file) memory,  
  - operator interfaces including `InferLayout`,  
  - and a `LayoutMap` used to record layout information across buffers, with the observation that “multiple tile operators often share the same buffers” so layout/thread-binding decisions are interdependent. ([ar5iv.org](https://ar5iv.org/pdf/2504.17577))  
- TileLang is not hypothetical: it is open-source and actively developed. ([github.com](https://github.com/tile-ai/tilelang?utm_source=openai))
- Separately, there is active research on expanding fusion scope for LLM inference with mega-kernels (e.g., **ClusterFusion**). This indicates the community agrees “bigger fusion scope” is important—but ClusterFusion’s novelty is cluster-level collectives, not symbolic layout negotiation between kernels. ([arxiv.org](https://arxiv.org/abs/2508.18850?utm_source=openai))

### 3) The real novelty wedge
Mosaic can still be **Green** if you draw a bright line:

- TileLang’s layout inference is **intra-kernel**: it infers layouts while lowering one TileLang program that already includes multiple tile operators in the same kernel scope. ([ar5iv.org](https://ar5iv.org/pdf/2504.17577))  
- Mosaic would do **inter-kernel / graph-level** layout constraint solving: take *two separately authored kernels* (or independently generated TileLang ASTs) and automatically “stitch” them into a megakernel while preserving producer layouts (fragments/swizzles) so the consumer can read them directly.

This “grey-box composition” is not something current mainstream compilers do well: graph compilers typically fuse **high-level ops**, not arbitrary custom kernels with specialized internal register layouts.

### 4) Verdict: **Green**
TileLang gives you (a) a real substrate and (b) a real “layout algebra” story; Mosaic can plausibly publish as the missing middle-end that finally fuses across kernel boundaries without forcing developers to rewrite monolithic kernels by hand. ([ar5iv.org](https://ar5iv.org/pdf/2504.17577))

**High-value second-order implication:** Because TileLang targets **both NVIDIA and AMD** (per repo claims), a cross-kernel layout solver that works across backends is a strong story for MLSys (portability + performance). ([github.com](https://github.com/tile-ai/tilelang?utm_source=openai))

---

# 4) Triton-V (virtualized tiling for ragged tensors / PagedAttention)

### 1) Claimed gap
Triton-V claims Triton’s tiling model assumes affine contiguous layouts; ragged/paged KV-cache workloads require manual page table lookups and scheduling. Triton-V proposes a first-class `RaggedTensor` in IR and compiler passes like indirection-aware pipelining and software-MMU injection.

### 2) Reality (evidence)
- **PagedAttention and vLLM are established**: KV cache stored in pages, block tables, etc. ([arxiv.org](https://arxiv.org/abs/2309.06180?utm_source=openai))
- **FlashInfer is a strong prior for high-performance paged attention kernels**, including:
  - paged KV cache wrappers and APIs, explicitly described as “first proposed in vLLM,” ([docs.flashinfer.ai](https://docs.flashinfer.ai/api/attention.html?utm_source=openai))  
  - and discussion of prefetching page indices into GPU shared memory to reduce page-table overhead. ([flashinfer.ai](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html?utm_source=openai))  
  - It also highlights “JIT compilation for attention variants,” which partially overlaps with “ease of expressing variants.” ([github.com](https://github.com/flashinfer-ai/flashinfer?utm_source=openai))
- **But these are libraries, not compiler abstractions.** They ship kernels and wrappers; they don’t give you a generalized IR type for “ragged/paged tensor” with correctness-preserving transformations.

- Triton itself has low-level primitives like `tensor_descriptor` with `gather`/`scatter`, but this is not the same as *compiler-managed indirection reasoning* for raggedness. ([triton-lang.org](https://triton-lang.org/main/python-api/generated/triton.language.tensor_descriptor.html?utm_source=openai))

### 3) What’s still open / defensible novelty
A good Triton-V paper would not claim “paged attention doesn’t exist”; it clearly does. ([arxiv.org](https://arxiv.org/abs/2309.06180?utm_source=openai))  
Instead, the novelty would be:

- A **compiler middle-end** that can *prove* safety and schedule quality in the presence of indirection (page-table loads, variable pages), and can automatically generate pipelined code patterns that today are handwritten in kernels.  
- A **productivity + variant velocity** angle: e.g., show that implementing novel attention variants on paged KV (sliding window, soft-capping, GQA quirks) is dramatically simpler and remains performant.

### 4) Verdict: **Green**
Even with FlashInfer and vLLM, there is still a credible gap: “we have good hand kernels; we don’t have an IR + compiler that treats indirection/raggedness as first-class and optimizes it automatically.” ([github.com](https://github.com/flashinfer-ai/flashinfer?utm_source=openai))

---

# 5) Forge (GPU-fused FSM kernels for constrained generation)

### 1) Claimed gap
Forge claims constrained decoding is bottlenecked by CPU-side FSM traversal and host-device transfers of logits/masks; it proposes GPU-resident DFA tables via vocabulary clustering and a fused FSM+softmax+sampling kernel.

### 2) Reality (evidence)
This space moved *fast* in 2024–2025:

- **SGLang (NeurIPS 2024)** explicitly includes “compressed finite state machines for faster structured output decoding.” ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html?utm_source=openai))  
- **XGrammar (arXiv 2024, MLSys 2025 poster)** is directly aimed at flexible and efficient structured generation, claiming near-zero overhead in low-latency serving scenarios, and it is already integrated into major inference engines (vLLM, SGLang, TensorRT-LLM, etc.). ([arxiv.org](https://arxiv.org/abs/2411.15100?utm_source=openai))  
- XGrammar’s docs show the typical workflow: generate a **token bitmask** (allocated on CPU, filled with CPU logic), then apply it to logits (GPU kernel if logits are on CUDA). ([xgrammar.mlc.ai](https://xgrammar.mlc.ai/docs/tutorials/workflow_of_xgrammar.html?utm_source=openai))  
- vLLM’s docs show structured outputs are a first-class feature with xgrammar as a backend option. ([docs.vllm.ai](https://docs.vllm.ai/en/v0.8.4/features/structured_outputs.html?utm_source=openai))

### 3) What’s left?
You can still argue “mask generation is CPU, and copying per-step is undesirable,” but reviewers will immediately ask: **why isn’t XGrammar already enough?** Since the XGrammar paper explicitly targets near-zero overhead by co-design and overlap, it directly attacks Forge’s core “logit bottleneck” narrative. ([arxiv.org](https://arxiv.org/abs/2411.15100?utm_source=openai))

To salvage Forge, you’d need a pivot such as:
- **GPU-only matcher** that eliminates CPU work entirely for disaggregated serving / CPU-saturated frontends, plus a demonstration that overlap is insufficient in real deployments, or  
- Support for a constrained decoding regime that XGrammar doesn’t handle efficiently (but XGrammar covers regex/JSON/CFG broadly). ([github.com](https://github.com/mlc-ai/xgrammar?utm_source=openai))

### 4) Verdict: **Red**
Given XGrammar’s scope, claims, and widespread integrations, Forge as written is very likely to be judged “solved / already surpassed.” ([github.com](https://github.com/mlc-ai/xgrammar?utm_source=openai))

---

# 6) BitStream (layout-aware type promotion for mixed-precision fusion)

### 1) Claimed gap
BitStream proposes a PyTorch Inductor pass that treats quantization as a layout constraint (“packing density”), enabling fusion of dequantization into compute loops (especially INT4/FP8) and using PyTorchSim (cycle-accurate NPU sim) for evaluation.

### 2) Reality (evidence)
- **torchao** is now a fairly complete quantization stack that includes:
  - quantized tensor types (e.g., Int4Tensor / preshuffled variants), ([docs.pytorch.org](https://docs.pytorch.org/ao/stable/quantization_overview.html?utm_source=openai))  
  - “quantization primitive ops / efficient kernels,” and it explicitly mentions layout concepts like `TensorCoreTiledLayout` in the stack description. ([docs.pytorch.org](https://docs.pytorch.org/ao/stable/quantization.html?utm_source=openai))  
  - It has configuration hooks for using **GemLite Triton kernels** for weight-only quantization, suggesting this ecosystem already fuses low-bit packing/unpacking in custom kernels, with Inductor-related tuning knobs. ([docs.pytorch.org](https://docs.pytorch.org/ao/stable/_modules/torchao/quantization/quant_api.html?utm_source=openai))
- Third-party projects like **GemLite** exist and emphasize low-bit matmul efficiency in Triton with large end-to-end gains. ([github.com](https://github.com/dropbox/gemlite?utm_source=openai))
- There is also *new compiler/DSL research* in this direction: **Hexcute (arXiv 2025)** explicitly targets mixed-type matrix multiplication, with automatic layout and task-mapping synthesis via a type-inference-based algorithm. ([arxiv.org](https://arxiv.org/abs/2504.16214?utm_source=openai))
- **PyTorchSim is real** (MICRO’25; GitHub/tutorial), so the evaluation infrastructure piece is credible and timely. ([psal.postech.ac.kr](https://www.psal.postech.ac.kr/pytorchsim-tutorial?utm_source=openai))

### 3) What remains open
BitStream can still be publishable if it **goes beyond** “we added another int4 matmul kernel,” because that’s already a crowded space.

A stronger novelty wedge would be:
- A **general fusion analysis** across quantization boundaries in *multi-op subgraphs* (e.g., quantized GEMM → activation → normalization → residual), deciding when to keep packed data in registers/shared vs when to materialize.  
- Demonstrate that *today’s* Inductor + torchao still breaks fusion in important patterns, or relies on bespoke kernels per pattern, and BitStream unifies it.

### 4) Verdict: **Yellow**
There is meaningful overlap with torchao and emerging DSL work (Hexcute). ([docs.pytorch.org](https://docs.pytorch.org/ao/stable/quantization.html?utm_source=openai))  
But if you reposition BitStream as “the missing compiler analysis layer that systematically fuses and chooses representations,” it can still be a solid MLSys paper.

---

# 7) Auto-Jagged (ETDG → Triton block-sparse kernels for dynamic MoE)

### 1) Claimed gap
Auto-Jagged claims dynamic MoE/jagged tensors force either padding (waste) or hand-written “hero kernels” (MegaBlocks, etc.). It proposes compiling nested parallelism (ETDG) into Triton kernels using block pointers and jaggedness analysis, enabling fusion across routing boundaries.

### 2) Reality (evidence)
- **MegaBlocks exists** as a widely used MoE training library; it frames MoE as block-sparse ops to avoid token dropping and improve efficiency. ([github.com](https://github.com/databricks/megablocks?utm_source=openai))
- **Triton-based MoE kernels are now mainstream**:
  - PyTorch/Meta/IBM published a Triton “persistent cache-aware grouped GEMM” kernel for MoE (DeepSeekv3), claiming up to 2.62× speedup vs a Python-loop baseline on H100. ([pytorch.com.tw](https://pytorch.com.tw/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/?utm_source=openai))  
  - PyTorch also published an earlier MoE inference optimization blog with code. ([pytorch.ac.cn](https://pytorch.ac.cn/blog/accelerating-moe-model/?utm_source=openai))  
  - Triton itself documents grouped GEMM. ([triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html?utm_source=openai))  
- Industrial Triton ecosystems like **ByteDance-Seed/Triton-distributed** explicitly list MoE operators and “MegaTritonKernel,” indicating substantial prior engineering and overlapping goals (compiler + fused kernels for LLM workloads). ([github.com](https://github.com/ByteDance-Seed/Triton-distributed?utm_source=openai))
- FractalTensor gives you the ETDG angle, but it’s already a published compiler abstraction. ([microsoft.com](https://www.microsoft.com/en-us/research/publication/uncovering-nested-data-parallelism-and-data-reuse-in-dnn-computation-with-fractaltensor/))

### 3) What’s still open
Auto-Jagged is not “MoE kernels in Triton” — those exist. The novelty would have to be:

- **Automatic synthesis** of these kernels from higher-level nested-IR (ETDG), including scheduling decisions that humans currently tune (e.g., cache-aware expert grouping, persistent execution decisions).  
- A **fusion analysis** across MoE routing + expert compute + recombine, delivered automatically.

### 4) Verdict: **Yellow**
The prior art is strong and recent; you’ll need a very crisp “automation/abstraction” story and performance parity with existing MoE kernels to win. ([pytorch.com.tw](https://pytorch.com.tw/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/?utm_source=openai))

---

# Final recommendation (single best candidate)

## Pick **Mosaic** (Layout-aware cross-kernel fusion on TileLang) as the highest-value OSDI/MLSys bet

### Why Mosaic is the best “top-tier safe” choice
1. **The gap is real and not already closed by a single dominant system**: Kernel-level DSLs (TileLang, Triton) help you write one fast kernel, but cross-kernel layout discontinuities still force spills and prevent fusion in practice. TileLang itself explicitly frames layout/thread-binding as a constraint inference problem across operators sharing buffers *within* a kernel. Mosaic’s leap is to take that reasoning **across kernel boundaries**, which is a qualitatively different compiler problem. ([ar5iv.org](https://ar5iv.org/pdf/2504.17577))  
2. **Strong substrate + feasibility**: TileLang is open-source and actively developed; it already has `Layout`/`Fragment` concepts and layout inference machinery you can extend, which lowers implementation risk compared to inventing an entirely new compiler stack. ([github.com](https://github.com/tile-ai/tilelang?utm_source=openai))  
3. **Clear evaluation path**: you can benchmark real LLM inference subgraphs where intermediate layout mismatches cause global memory traffic, and show end-to-end speedups (TPOT, tokens/sec, DRAM traffic), on H100 and MI300X—exactly the kind of story MLSys likes. TileLang already targets both vendors, which is a differentiator. ([github.com](https://github.com/tile-ai/tilelang?utm_source=openai))  
4. **Defensible novelty vs 2024–2025 fusion work**: ClusterFusion/FLUX show “mega-kernel fusion” is valuable, but they don’t solve “layout negotiation between independently authored kernels” as a general compiler capability. Mosaic can frame itself as the missing middle-end that makes mega-kernels *reusable and composable*. ([arxiv.org](https://arxiv.org/abs/2508.18850?utm_source=openai))  

### How to position Mosaic so reviewers can’t dismiss it as “just write one fused kernel”
Your strongest rhetorical stance is:

- **Input**: a graph of *existing* high-performance kernels (TileLang ASTs / IR) with known internal layouts, treated as grey boxes rather than black boxes.  
- **Output**: an automatically generated megakernel that (a) preserves producer layouts (fragments/swizzles) and (b) synthesizes consumer access in the same layout, inserting minimal adapters when required.

TileLang already has the language of “Layout inference,” “Fragment,” and “LayoutMap”; Mosaic is “lifting these invariants to the graph level.” ([ar5iv.org](https://ar5iv.org/pdf/2504.17577))

---

## If you want a backup “second best”
**Triton-V** is the runner-up: it’s also Green because “ragged/paged tensors as first-class compiler objects” is still not a mainstream compiler capability, despite strong kernel libraries like FlashInfer. ([github.com](https://github.com/flashinfer-ai/flashinfer?utm_source=openai))

---

## One-sentence kill shots (what I would write on your internal decision doc)

- **Choose Mosaic** if you want the **highest novelty-to-feasibility ratio**: it’s a true middle-end compiler gap, enabled by a real substrate (TileLang) that already formalizes layouts/fragments. ([ar5iv.org](https://ar5iv.org/pdf/2504.17577))  
- **Avoid Forge** unless you pivot hard: XGrammar + engine integrations have largely eaten the “structured decoding overhead” space. ([arxiv.org](https://arxiv.org/abs/2411.15100?utm_source=openai))  
- **Treat Spec-Graph and FractalGen as high-risk/high-reward**: both have real ideas, but the “systems baseline” is already very strong (SpecInfer/TensorRT-LLM for speculation; CUDA Graphs + structured control-flow ops for loops). ([arxiv.org](https://arxiv.org/abs/2305.09781?utm_source=openai))

---
Learn more:
1. [Uncovering Nested Data Parallelism and Data Reuse in DNN Computation with FractalTensor - Microsoft Research](https://www.microsoft.com/en-us/research/publication/uncovering-nested-data-parallelism-and-data-reuse-in-dnn-computation-with-fractaltensor/)
2. [Optimizing llama.cpp AI Inference with CUDA Graphs | NVIDIA Technical Blog](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/?utm_source=openai)
3. [Add support for cuda graph with torch.cond · Issue #168911 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/168911?utm_source=openai)
4. [Control Flow - Cond — PyTorch 2.9 documentation](https://docs.pytorch.org/docs/stable/cond.html?utm_source=openai)
5. [Strange failure modes of while\_loop under torch.compile -- possibly due to internal array stride · Issue #160939 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/160939?utm_source=openai)
6. [Optimize Memory Utilization with while\_loop — PyTorch/XLA master documentation](https://docs.pytorch.org/xla/master/perf/fori_loop.html?utm_source=openai)
7. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180?utm_source=openai)
8. [SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/abs/2305.09781?utm_source=openai)
9. [Speculative Sampling — TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html?utm_source=openai)
10. [FlashInfer Attention Kernels - FlashInfer 0.5.2 documentation](https://docs.flashinfer.ai/api/attention.html?utm_source=openai)
11. [\[2504.17577\] TileLang: A Composable Tiled Programming Model for AI Systems](https://ar5iv.org/pdf/2504.17577)
12. [GitHub - tile-ai/tilelang: Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels](https://github.com/tile-ai/tilelang?utm_source=openai)
13. [ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive](https://arxiv.org/abs/2508.18850?utm_source=openai)
14. [Accelerating Self-Attentions for LLM Serving with FlashInfer | FlashInfer](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html?utm_source=openai)
15. [GitHub - flashinfer-ai/flashinfer: FlashInfer: Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer?utm_source=openai)
16. [triton.language.tensor\_descriptor — Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.tensor_descriptor.html?utm_source=openai)
17. [SGLang: Efficient Execution of Structured Language Model Programs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html?utm_source=openai)
18. [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100?utm_source=openai)
19. [Workflow of XGrammar — XGrammar 0.1.28 documentation](https://xgrammar.mlc.ai/docs/tutorials/workflow_of_xgrammar.html?utm_source=openai)
20. [Structured Outputs — vLLM](https://docs.vllm.ai/en/v0.8.4/features/structured_outputs.html?utm_source=openai)
21. [GitHub - mlc-ai/xgrammar: Fast, Flexible and Portable Structured Generation](https://github.com/mlc-ai/xgrammar?utm_source=openai)
22. [Quantization Overview — torchao 0.13 documentation](https://docs.pytorch.org/ao/stable/quantization_overview.html?utm_source=openai)
23. [Quantization Overview — torchao 0.12 documentation](https://docs.pytorch.org/ao/stable/quantization.html?utm_source=openai)
24. [torchao.quantization.quant\_api — torchao 0.13 documentation](https://docs.pytorch.org/ao/stable/_modules/torchao/quantization/quant_api.html?utm_source=openai)
25. [GitHub - dropbox/gemlite: Fast low-bit matmul kernels in Triton](https://github.com/dropbox/gemlite?utm_source=openai)
26. [Hexcute: A Tile-based Programming Language with Automatic Layout and Task-Mapping Synthesis](https://arxiv.org/abs/2504.16214?utm_source=openai)
27. [Tutorial | PSAL at POSTECH](https://www.psal.postech.ac.kr/pytorchsim-tutorial?utm_source=openai)
28. [GitHub - databricks/megablocks](https://github.com/databricks/megablocks?utm_source=openai)
29. [透過 Triton 持久化快取感知分組 GEMM 核心加速 MoE – PyTorch - PyTorch 框架](https://pytorch.com.tw/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/?utm_source=openai)
30. [通过局部性感知核设计加速MoE模型推理——PyTorch - PyTorch 框架](https://pytorch.ac.cn/blog/accelerating-moe-model/?utm_source=openai)
31. [Group GEMM — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html?utm_source=openai)
32. [GitHub - ByteDance-Seed/Triton-distributed: Distributed Compiler based on Triton for Parallel Systems](https://github.com/ByteDance-Seed/Triton-distributed?utm_source=openai)
