## Executive Summary (as of **Dec 15, 2025**)

After a web-based novelty check against **recent papers (2024–2025)** and **active open-source implementations**, here’s the short conclusion:

- **Best single bet for a top-tier (OSDI/MLSys) “compiler/systems” submission:**  
  **(3) Mosaic** — *Layout-aware cross-kernel fusion using TileLang symbolic propagation*  
  Rationale: there is a real and widely-felt “middle-end gap” (layout discontinuity across custom kernels) and I did **not** find an existing, widely-adopted system that already provides *graph-level* symbolic **layout unification across independently-authored custom kernels** in the TileLang ecosystem. TileLang’s IR already exposes exactly the primitives you need, making this both **novel** and **feasible**. 

- **Strong runner-up (also potentially Green):**  
  **(4) Triton‑V** — *Virtualized tiling for paged/ragged tensors*  
  Rationale: there is substantial prior work on ragged tensors and paged KV-cache kernels, but **not** a clean “Triton-first-class ragged/paged tensor type + indirection-aware compiler pipelining” solution that’s already established; and upstream Triton still has active performance gap discussions for paged attention. 

- **Most likely already “solved enough” to be risky as a novelty claim:**  
  **(5) Forge** — *fused FSM kernels for constrained generation* is heavily overlapped by **XGrammar**, which is already integrated into **vLLM, SGLang, TensorRT‑LLM**, and explicitly targets near‑zero overhead structured generation. 

---

## Comparative Table (Gap Analysis + Verdict)

| # | Candidate | Claimed Gap (one-liner) | Reality (existing systems & evidence) | Verdict | What would make it publishable anyway |
|---:|---|---|---|---|---|
| 1 | **FractalGen** | DAG compilers can’t optimize the *whole autoregressive loop*; need scan/ETDG + persistent kernel | **Mirage Persistent Kernel (MPK)** already compiles LLM inference into a single **persistent megakernel**, reducing latency (reported 1.2×–6.7×) and is open-source.  Also **TileRT** targets ultra-low-latency inference via compiler-driven tile-level runtime.  | **Yellow → Red** | You must differentiate strongly from MPK: e.g., *automatic lifting* of arbitrary PyTorch `generate()` loops into scan IR + formal correctness under dynamic termination + integration into PyTorch compile. |
| 2 | **Spec‑Graph** | No compiler IR for *tree-structured speculation*; runtime-managed trees incur overhead | **SpecInfer** already does tree-based speculative inference+verification (ASPLOS’24) with artifact.  **TensorRT‑LLM** supports speculative decoding modes incl. **Medusa tree**, **EAGLE**, **ReDrafter**, some inside-engine.  vLLM has speculative infra but *tree-attention* was closed “not planned”.  | **Yellow** | Turn it into a *compiler contribution* that demonstrably generalizes beyond SpecInfer/TensorRT: dynamic-tree IR + device-side tree exec plan + allocator specialization + measurable wins over vendor runtimes. |
| 3 | **Mosaic** | Cross-kernel fusion blocked by **layout discontinuity** (swizzles/fragments), compilers treat kernels as black boxes | TileLang exists and exposes layouts/fragments (great substrate).  But I did **not** find an established “global layout solver across kernel boundaries” tool. Also, XLA users report layout mismatches can hurt fusion perf, showing the problem is real.  | **Green** | Make it concrete: implement global constraint solver + AST stitching + correctness + show end-to-end wins on real LLM subgraphs across NVIDIA+AMD. |
| 4 | **Triton‑V** | Triton lacks first-class ragged/paged tensors; PagedAttention in Triton requires manual indirection + scheduling | PagedAttention is established (SOSP’23 / arXiv) and used in vLLM.  Triton has **ragged TMA** support now, but that’s not a general paged KV abstraction.  Triton community still discusses paged attention perf gaps.  FlashInfer provides paged+ragged attention kernels but as a library.  | **Green / Yellow** | The win must be “compiler automation” + portability: show Triton‑V makes paged/ragged kernels easy *and* matches hand-tuned kernels. Differentiate from ragged TMA and from library-only solutions. |
| 5 | **Forge** | Constrained decoding still incurs host-device ping-pong; fuse FSM+mask+sample on GPU | **XGrammar** already targets efficient structured generation (CFG/regex/JSON), claims near-zero overhead, and is integrated into major engines incl. vLLM.  SGLang itself already introduced compressed FSMs (NeurIPS’24).  | **Red** | Only viable if you beat XGrammar materially (e.g., true in-kernel masked softmax+sampling with provable lower latency on GPU, large vocab, batched multi-tenant). Otherwise it’s incremental. |
| 6 | **BitStream** | Inductor can’t fuse mixed-precision due to packed layouts; need layout-aware type promotion + fused dequant | TorchAO already provides **INT4 weight-only** configs and specialized kernels, including explicit layout types (`TensorCoreTiledLayout`) and int4mm operators.  Hexcute (2025) explicitly targets mixed-type operators with **type-inference-based layout synthesis**.  PyTorchSim exists for cycle-accurate NPU simulation.  | **Yellow → Red** | Novelty requires proving Inductor still cannot do X, and your pass enables *new fusions across multiple ops* (not just a better int4 GEMM). Evaluate vs torchao/bitblas/hexcute. |
| 7 | **Auto‑Jagged** | Dynamic MoE jagged tensors rely on “hero kernels”; compiler should synthesize block-sparse Triton automatically | MegaBlocks already provides MoE training via block-sparse reformulation.  FlashInfer has SegmentGEMM (ragged) and MoE kernels.  There are also emerging megakernel MoE libs (e.g., Alpha‑MoE).  | **Yellow** | Make it a *compiler* paper: automatic lowering from nested IR to Triton block pointers + fuse routing+compute+recombine + show parity with MegaBlocks and ease-of-programming wins. |

---

## Detailed Analysis (per candidate)

### 1) **FractalGen** — “Compiling autoregressive loops via nested scans (ETDG/FractalTensor)”

#### (1) Claimed Gap
- Traditional compilers represent models as static DAGs and typically only compile a single decode step, so they miss cross-iteration optimizations.
- This causes a kernel-launch-heavy decode pipeline; FractalGen proposes lowering the entire generation loop to a high-order `scan`/ETDG and generating a **persistent kernel**.

#### (2) Reality (Evidence)
- **FractalTensor is real and already formalizes ETDG + map/reduce/scan** and is published at **SOSP 2024**. 
- However, the “compile the *whole LLM inference* into a single persistent megakernel” direction is already aggressively pursued:
  - **Mirage Persistent Kernel (MPK)** explicitly markets itself as: *“compiling LLM inference into a single megakernel”* with a persistent kernel API and a decode-step counter (`step`) stored on device. 
  - **TileRT** is another 2025-era system focusing on ultra-low-latency generation with compiler-driven tile-level tasking and overlap. 
- Even the broader “move a time loop into a persistent kernel” concept is established in GPU systems literature (e.g., PERKS). 

#### (3) Verdict: **Yellow → Red**
- If your primary novelty claim is *“whole-loop compilation eliminates launch overhead via persistent kernels”*, MPK makes that claim much harder to defend in 2025.   
- If you pivot to the *unique* FractalTensor angle (automatic scan lifting + correctness + generalized dynamic bounds + integration with PyTorch `generate()`), it can still be Yellow, but you must show this is **strictly more automatic and general** than MPK’s API-based graph construction.

#### “How to salvage it” (concrete differentiators)
1. **Automatic loop lifting**: demonstrate compiling an unmodified HuggingFace `generate()` (including stopping criteria) into your scan IR.
2. **Dynamic termination correctness**: prove semantics under EOS termination and variable-length batches (this is where compilers usually break).
3. **Memory virtualization synthesis**: show the compiler can choose between contiguous/paged/cascaded KV layouts automatically (this starts to overlap Triton‑V / eLLM-style “virtual tensor” ideas). 

---

### 2) **Spec‑Graph** — “Compiler IR for tree-structured speculation”

#### (1) Claimed Gap
- Tree-based speculative decoding is powerful but runtime-managed implementations pay CPU orchestration + kernel launch overhead.
- Spec‑Graph proposes an IR with first-class **token trees** (fork/verify primitives), enabling fusion and a tree-local arena allocator.

#### (2) Reality (Evidence)
- **SpecInfer** already exists (ASPLOS 2024) and is explicitly *tree-based speculative inference and verification*, with a public artifact repository. 
- **TensorRT‑LLM** has a dedicated speculative decoding section supporting multiple strategies including **Medusa tree**, **EAGLE**, and **ReDrafter**, with some logic performed inside the engine for certain modes. 
- **FlashInfer** documents kernels that explicitly account for speculative decoding token dimension (e.g., `q_seq_len` in `xqa`). 
- **vLLM** is actively building speculative decoding infrastructure (a “speculators” library), but a request for “Tree-Attention support” was closed as “not planned,” suggesting a gap in that ecosystem specifically. 

#### (3) Verdict: **Yellow**
- The *problem* (tree-structured speculation) is not only known; it has at least one flagship system (SpecInfer) and vendor/runtime support (TensorRT‑LLM).   
- The *compiler IR angle* could still be novel if you truly provide:
  - a reusable IR + compiler passes for arbitrary speculation trees,
  - device-side tree execution planning,
  - and evidence that existing systems don’t generalize or are too brittle.

#### What would push it toward Green
1. **Generalization beyond fixed tree shapes** (Medusa-style) and beyond SpecInfer’s design constraints.
2. **A real compilation pipeline** (MLIR dialect or TVM Relax extension) and automatic lowering to one/few kernels.
3. **Allocator/arena design that beats paging for short-lived trees**, with measurements showing paging overhead is nontrivial for these microkernels. (This directly challenges vLLM/PagedAttention assumptions.) 

---

### 3) **Mosaic** — “Layout-aware cross-kernel fusion via symbolic tile propagation (TileLang)”

#### (1) Claimed Gap
- Even if individual kernels are optimized (FlashAttention, quantized GEMM), **composition** is inefficient because layout mismatches force spills to global memory.
- Compilers treat custom kernels as opaque; Mosaic proposes global layout inference/constraint solving to stitch kernels into megakernels that keep intermediates in registers/shared memory.

#### (2) Reality (Evidence)
- **TileLang is real** (arXiv 2025) and open-source; it is explicitly designed to expose layout/scheduling and achieve high performance across devices. 
- There are many works on *kernel fusion* and *compute/comm overlap*, e.g., **FLUX** (2024) and **TileLink** (2025), but those are primarily about overlapping communication and compute, not “layout discontinuity between independently-authored kernels.” 
- The fact that layout mismatches matter is acknowledged in existing compiler ecosystems: an XLA discussion notes severe regressions when fusion kernels must deal with different I/O layouts. 
- Importantly: I did **not** find a widely-adopted open-source implementation that already does **graph-level symbolic layout unification across kernel boundaries** in TileLang.

#### (3) Verdict: **Green**
This is the clearest “middle-end gap” among your candidates:
- Too low-level for graph compilers today (they don’t model MMA fragment layouts, TMA swizzles, etc).
- Too high-level for kernel DSL authors to solve repeatedly (they end up rewriting megakernels manually).
- TileLang provides a rare opportunity: the compiler can *see* enough about layout to actually do the inference/propagation.

#### What reviewers will demand (and how to satisfy them)
1. **Formal layout constraint model**: define compatibility precisely, include bank-conflict and warpgroup semantics.
2. **End-to-end wins**: show real model blocks (e.g., GEMM→activation→norm→attention epilogues) and demonstrate reduced HBM traffic.
3. **Failure modes**: show how you handle layout conflicts (on-chip re-layout vs fallback), and quantify overhead.

---

### 4) **Triton‑V** — “Virtualized tiling for ragged tensors / PagedAttention”

#### (1) Claimed Gap
- Ragged/paged memory patterns (PagedAttention, MoE routing) break Triton’s “dense tiling” assumptions.
- Implementing paged attention in Triton today is manual and error-prone; Triton‑V proposes first-class ragged tensors and compiler passes like indirection-aware pipelining.

#### (2) Reality (Evidence)
- **PagedAttention/vLLM** is established (SOSP’23 / arXiv 2023). 
- **FlashInfer** provides high-performance **paged KV-cache** attention and explicitly documents ragged tensors and page-table-style layouts (e.g., “ragged query/output” + paged KV cache).   
  This means the kernels exist, but as **hand-built libraries**, not as compiler automation.
- **Upstream Triton has started adding “ragged TMA support”** (bounds-checked ragged descriptors) in recent releases.   
  This partially contradicts any claim that Triton ignores raggedness entirely—but it’s also **not** a full “paged/ragged tensor type system + indirection-aware scheduling.”
- There is ongoing attention-work in Triton community with explicit performance gaps around paged attention. 
- There is related “ragged compiler” history:
  - TVM ecosystem: **CoRa** (TVMCon 2021 talk) for ragged tensors with minimal padding. 
  - IREE has explicit **ragged shape casting** in `TensorExt`. 
- Separately, new LLM-serving memory work proposes “virtual tensor abstraction” ideas (eLLM), which is conceptually adjacent to “software MMU / virtualization.” 

#### (3) Verdict: **Green / Yellow**
- **Green** if you focus on: “Triton compiler can automatically generate high-performance paged/ragged kernels from a high-level abstraction,” and you show parity with hand-tuned kernels across vendors.   
- **Yellow** if Triton’s new ragged TMA work ends up covering a big subset of what you claim (you’ll need to scope carefully). 

#### How to sharpen novelty
1. Make it about **indirect addressing** (page table translation + prefetch/hoist) rather than generic ragged bounds.
2. Show “compiler pass that hoists page-table loads” yields measurable wins where hand-written Triton fails (this is plausible given the open RFC/issue discussion). 
3. Provide a migration path: “write PagedAttention in 20 lines of Triton‑V vs 200 lines of Triton.”

---

### 5) **Forge** — “GPU-fused FSM kernels for constrained generation”

#### (1) Claimed Gap
- Existing constrained decoding requires CPU FSM traversal / masking and causes host-device synchronization.

#### (2) Reality (Evidence)
- **SGLang** (NeurIPS 2024) already includes “compressed finite state machines” for faster structured decoding. 
- More importantly, **XGrammar** (MLSys 2024) is an open-source structured generation engine supporting CFG/regex/JSON and claims extremely low overhead; and it is already integrated into **vLLM** (default guided decoding backend), **SGLang**, and **TensorRT‑LLM**.   
  This directly attacks Forge’s “this is still a big bottleneck” premise in 2025.

#### (3) Verdict: **Red**
Forge’s pitch is too close to what XGrammar has already operationalized (and deployed widely). 

#### Only viable reframing
- Treat XGrammar as baseline and propose **GPU-resident, fused masked softmax+sampling** that *provably reduces tail latency* in multi-tenant serving (e.g., eliminate CPU involvement entirely), and show wins on large vocab and tight latency SLOs.

---

### 6) **BitStream** — “Layout-aware type promotion for mixed-precision fusion (Inductor) + PyTorchSim”

#### (1) Claimed Gap
- General-purpose compilers can’t fuse quantization boundaries due to packed formats; BitStream adds layout-aware type promotion, producing fused Triton kernels with register dequant and evaluates via PyTorchSim.

#### (2) Reality (Evidence)
- **TorchAO** already provides INT4 weight-only quantization configs and kernels; it explicitly exposes layout concepts (e.g., `TensorCoreTiledLayout`) and targets a dedicated int4mm kernel (`_weight_int4pack_mm`).   
  That doesn’t mean “fusion across subsequent ops” is solved, but it does mean the basic compiler/type story is already evolving in PyTorch’s ecosystem.
- **Hexcute (2025)** directly targets mixed-type operators and proposes automatic layout and task mapping synthesis via type inference—very close in spirit to “layout-aware type promotion.” 
- **PyTorchSim** is real and published (MICRO 2025), with a public GitHub and tutorial material; it’s credible as an evaluation vehicle. 

#### (3) Verdict: **Yellow → Red**
- If BitStream is “we fused dequant into matmul”: **Red** (that is the baseline strategy of every performant quantized linear kernel stack).  
- If BitStream is “we add a new IR/type system to Inductor enabling *cross-op* mixed-precision fusion with layout constraints and demonstrably better graph-level fusion decisions than torchao+inductor today”: **Yellow**, but you must prove that gap exists and is large.

#### What would make it publishable
1. A concrete demonstration that Inductor cannot fuse across quantized matmul boundaries today in cases that matter.
2. A general “packing lattice” that supports multiple packing formats and can propagate across ops.
3. Evidence that your pass enables new end-to-end fusions, not just a faster GEMM kernel (Hexcute/TileLang already cover kernel-level aspects). 

---

### 7) **Auto‑Jagged** — “Compile nested parallelism to block-sparse Triton kernels for dynamic MoEs”

#### (1) Claimed Gap
- Jagged tensors and dynamic MoE routing force padding or hero kernels; compilers can’t generate fused block-sparse kernels automatically.

#### (2) Reality (Evidence)
- **MegaBlocks** is already a dedicated MoE training library using block-sparse reformulations (dMoE) and supports grouped GEMM, integrated with Megatron‑LM. 
- **FlashInfer** explicitly supports variable-length / ragged GEMM via `SegmentGEMMWrapper` and integrates many LLM-specific kernels. 
- Industry is also pushing persistent megakernel MoE inference approaches (e.g., Aleph Alpha’s “Alpha‑MoE” megakernel library). 

#### (3) Verdict: **Yellow**
The *kernels* exist; the “compiler synthesizes them from high-level nested parallelism” story is still plausible as novel, but you must:
- show generality beyond grouped GEMM,
- show fusion of routing + compute + recombination,
- and show competitive performance with MegaBlocks/vendor kernels.

#### How to push it toward Green
1. Demonstrate automatic lowering of **routing tables + ragged memory** into Triton block pointers without manual pointer arithmetic.
2. Provide a **fusion analysis across routing boundaries** that libraries don’t provide.
3. Evaluate on both MoE **training** (MegaBlocks comparison) and MoE **inference** with real serving patterns. 

---

## Final Recommendation (pick one)

### Pick **(3) Mosaic** as the highest-value top-tier bet.

**Why Mosaic is the best “guaranteed novelty” relative to the 2024–2025 landscape:**

1. **The problem is real and painful:** layout mismatches across kernels can kill fusion benefits; even mainstream compiler users complain about layout-driven fusion regressions.   
2. **The solution space is not saturated:** while we have many kernel DSLs (TileLang, Triton) and many megakernel efforts (MPK), I did not find a widely adopted framework that does *graph-level symbolic layout constraint solving across independently-authored custom kernels* in the TileLang ecosystem.   
3. **Feasibility is unusually high:** TileLang already exposes layouts, fragments, pipelining, and a compiler stack on TVM—exactly the hooks needed to implement Mosaic-style inter-kernel reasoning.   
4. **High upside for MLSys/OSDI reviewers:** it’s a true “middle-end” contribution—bridging the gulf between graph compilers and hand-written hero kernels.

### Runner-up if you want a *Triton-first* paper:
**(4) Triton‑V** — but you must carefully scope around Triton’s rapidly evolving ragged/TMA features and position it as *paged indirection + compiler scheduling*, not just “raggedness.” 

---

## Practical “next step” checklist for Mosaic (to maximize acceptance probability)

1. **Define the layout constraint language** (what does it mean for producer+consumer to be compatible in regs/shared).
2. **Implement a solver + conflict resolution** (on-chip re-layout vs fallback kernel).
3. **Demonstrate 2–3 killer fusion case studies**:
   - GEMM → activation → norm
   - Quantized GEMM → dequant epilogue → bias/act
   - Attention epilogue fusion (e.g., bias+RoPE+mask)  
4. **Compare against**:
   - Unfused TileLang
   - Triton monolithic kernel (manual)
   - MPK-style megakernel baselines (if applicable)   
5. **Show portability**: at least NVIDIA + AMD, since TileLang explicitly targets both.
