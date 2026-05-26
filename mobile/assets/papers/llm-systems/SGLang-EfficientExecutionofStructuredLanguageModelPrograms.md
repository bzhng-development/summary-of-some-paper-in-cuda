# SGLang: Efficient Execution of Structured Language Model Programs

**ArXiv:** [2312.07104](https://arxiv.org/abs/2312.07104)

## 🎯 Pitch

SGLang is a Python-embedded language plus co-designed runtime that accelerates multi-call, control-flow-heavy LM programs by reusing computation across calls (RadixAttention for KV-cache reuse) and by decoding structured outputs faster (compressed finite state machines). This matters because many real-world LLM applications—agents, few-shot evaluation, tree search, JSON/RAG pipelines—perform many dependent LLM calls with shared prefixes or strict output formats, and SGLang delivers up to 6.4× higher throughput and substantial latency reductions by eliminating redundant work and enabling scalable, reliable execution.

---

## 1. Executive Summary

SGLang introduces a system for efficient programming and execution of structured language model programs, consisting of a Python-embedded frontend language and a co-designed runtime that together systematically exploit multi-call structure in LM programs. Evaluated across diverse workloads—including agent control, few-shot benchmarks, JSON decoding, and multi-turn chat—on models from Llama-7B through Mixtral-8x7B and LLaVA-v1.5-7B, SGLang achieves up to 6.4× higher throughput compared to Guidance, vLLM, and LMQL. The runtime employs two novel optimizations: **RadixAttention**, which maintains an LRU cache of KV cache tensors in a radix tree to enable automatic prefix reuse across multiple generation calls (reusing system prompts, few-shot examples, and chat histories without manual configuration), and **compressed finite state machines**, which accelerate constrained decoding by merging singular-transition edges to decode multiple tokens in a single forward pass (decoding the constant sequence `{"summary": "` in one step rather than token-by-token). A third optimization, **API speculative execution**, reduces API costs for black-box models by having early generation calls continue past their stop conditions, then matching and reusing the extra generated tokens for later primitives. The paper also establishes a compute-optimal scheduling policy via **cache-aware scheduling**, proving that longest-shared-prefix-first ordering achieves optimal cache hit rate for offline batches—an insight that translates to 96% of the optimal hit rate on average across benchmarks, with a 52.4% production cache hit rate for LLaVA-NeXT-34B and 74.1% for Vicuna-33B in the Chatbot Arena deployment.

## 2. Context and Motivation

### The Core Problem: LM Programs Are Inefficient to Program and Execute

Large language models have evolved far beyond single-turn chat. Modern applications demand that LLMs engage in multi-step reasoning, interact with external tools, process structured inputs and outputs, and coordinate multiple dependent generation calls. The paper identifies a fundamental shift: we are moving from *chatting with LLMs* to *programming with LLMs* — using programs to schedule and control the generation processes of LLMs. These "Language Model Programs" (LM Programs) encompass advanced prompting techniques (few-shot learning, self-consistency, tree-of-thought, skeleton-of-thought), agentic workflows (ReAct agents, generative agents), retrieval-augmented generation pipelines, and any application that chains multiple LLM calls interspersed with control flow.

The paper identifies two common properties that define LM programs:

> "**(1) LM programs typically contain multiple LLM calls interspersed with control flow.** This is needed to complete complex tasks and improve overall quality. **(2) LM programs receive structured inputs and produce structured outputs.** This is needed to enable the composition of LM programs and to integrate LM programs into existing software systems." (Section 1)

The problem this paper tackles is deceptively concrete: despite the widespread use of such programs, current systems for expressing and executing them are **inefficient on two fronts** — programming them is unnecessarily tedious and error-prone, and executing them wastes substantial computation and memory.

### Why This Problem Matters

**Programming complexity is a real barrier to innovation.** The paper gives a concrete, quantified example: an equivalent program using an OpenAI API-like interface would take **2.1× as many lines of code** compared to the SGLang implementation of the same multi-dimensional essay judge (Figure 2). This isn't just an aesthetic concern. Developing an LM program requires:

- Extensive and brittle string manipulation to construct prompts, extract outputs, and maintain state across calls
- Experimental tuning of prompts where small changes cascade through dependent calls
- Brittle output parsing that breaks when model outputs deviate slightly from expected formats
- Manual handling of multiple input modalities (text, images, video)
- Manual implementation of parallelism mechanisms when multiple independent branches can proceed simultaneously

The paper argues that this complexity "significantly reduces the readability of even simple programs" (Section 1), which in turn slows experimentation, makes programs harder to debug, and limits who can build sophisticated LM applications. When building an agent that evaluates an essay across multiple dimensions in parallel, then merges judgments, generated a summary, and returns structured JSON — a program that naturally requires forking, parallel generation, and constrained output — the programmer should focus on the evaluation logic, not on managing asynchronous API calls and string parsing.

**Execution inefficiency wastes real compute and money.** The paper points to a specific and pervasive example: the Key-Value (KV) cache. During LLM inference, the forward pass produces intermediate tensors (the KV cache) that are essential for generating subsequent tokens. This computation depends only on prefix tokens. Therefore, when multiple generation calls share a common prefix — system prompts reused across requests, few-shot examples shared across questions, chat history carried through multi-turn conversations — the KV cache for that prefix could theoretically be computed once and reused. In practice, the paper observes that:

> "During typical batch executions of LM programs, numerous opportunities exist to reuse the KV cache across multiple different LLM calls that share a common prefix. However, current systems lack effective mechanisms to facilitate this reuse, resulting in unnecessary computations and wasted memory." (Section 1)

Figure 9 in the appendix illustrates four common sharing patterns — few-shot learning, self-consistency, multi-turn chat, and tree-of-thought — none of which existing systems can automatically handle. The waste is not theoretical. Every few-shot MMLU evaluation recomputes the KV cache for the identical five-shot examples across all test questions. Every multi-turn conversation re-encodes the entire conversation history for each new turn. Every tree-of-thought exploration recomputes shared search history prefixes. This redundant computation increases latency, reduces throughput (by consuming GPU memory that could be used for larger batch sizes), and costs money when using API-based models (which charge per input token).

A second source of execution inefficiency is constrained decoding for structured outputs. When an LM program requires JSON output following a specific schema, existing systems enforce the constraint token-by-token: at each decoding step, they mask invalid tokens based on a finite state machine. However, when the constraint requires a deterministic sequence of tokens — such as the literal string `{"summary": "` — decoding token-by-token requires multiple forward passes even though there is only one valid path. The paper identifies that:

> "existing systems can only decode one token at a time because the lack of integration between the FSM and the model runner in existing systems prevents multi-token processing, resulting in slow decoding." (Section 4)

### Where Prior Approaches Fall Short

The paper categorizes the existing landscape and identifies specific gaps.

**Inference engines (vLLM, TGI, TensorRT-LLM) are workload-agnostic.** State-of-the-art inference engines have been optimized for general-purpose serving — they treat each request as an independent unit and optimize for aggregate throughput without knowledge of the multi-call structure in LM programs. The paper acknowledges these systems' strengths:

> "This makes these systems general and robust but also results in significant inefficiencies for any given workload." (Section 1)

The crucial limitation is KV cache management. vLLM introduced some basic prefix sharing (system prompt reuse), but as the paper notes, it "does not cover multi-level tree-structured sharing or LRU caching" (Section 7). ChunkedAttention explores partial reuse patterns. PromptCache proposes modular reuse beyond exact prefixes but at the cost of potential accuracy drops of up to 43%. None of these systems provide automatic, systematic KV cache reuse that handles the full range of sharing patterns (irregular tree structures, dynamic forks, multi-level sharing) without manual configuration.

**LM programming frameworks (Guidance, LMQL) lack efficient runtimes.** These systems focus on the programming model — making it easier to express LM programs with primitives for generation, selection, and constraints. However, they rely on unoptimized backends:

- Guidance, in the version tested (v0.1.8), uses llama.cpp as its backend and "lacks batching and parallelism support" (Section 6.2)
- LMQL uses Hugging Face Transformers with "slow token-level processing and an unoptimized backend" (Section 6.2), making it impractical for high-throughput scenarios

The paper provides a direct comparison in Table 1, showing that while LMQL and Guidance provide language primitives (`extend`, `gen`, `select`) and support some backends (HF Transformers, llama.cpp, OpenAI), they lack the co-designed runtime that enables the novel optimizations SGLang introduces. SGLang's own SGLang Runtime (SRT) is the key differentiator — it's not just another frontend language but a complete system where the frontend and runtime are designed together to exploit multi-call structure.

**High-level frameworks (LangChain, DSPy) abstract away control but not execution efficiency.** These frameworks operate at a higher level, providing predefined templates or auto-optimized prompts. They can express complex agent workflows, but they ultimately delegate to lower-level inference engines. The paper shows that SGLang can serve as a backend for DSPy (Section 6), accelerating its execution — demonstrating that the efficiency problem exists at the runtime layer, not the abstraction layer.

**Constrained decoding approaches are single-token.** When enforcing output formats (JSON schemas, regular expressions), existing systems convert the regex to a finite state machine and use it to mask invalid tokens during decoding (Willard and Louf, 2023). This approach is correct but inherently single-token: at each step, the FSM determines which tokens are valid, the model samples one token, the FSM advances, and the process repeats. When the constraint forces a deterministic multi-token path, this serial approach wastes forward passes. The paper argues this inefficiency stems from a lack of integration between the FSM logic and the model runner — the FSM operates at the token mask level and cannot coordinate with the inference engine to batch multiple tokens into a single forward pass.

### How This Paper Positions Itself

The paper's central thesis is that **the multi-call structure intrinsic to LM programs should be systematically exploited for efficient execution**. Rather than treating the programming model and the runtime as independent concerns, SGLang co-designs them so that structural information from the frontend (shared prefixes, parallelism opportunities, output constraints) flows to the runtime in a form it can directly optimize.

This is not simply "building a better inference engine." The paper explicitly frames its contribution as a **system-level solution** that spans both the programming interface and the execution engine:

> "The core idea is to systematically exploit the multi-call structure in LM programs for efficient execution. As shown in Fig. 1, it has two parts: a front-end language and a back-end runtime. The front-end simplifies the programming of LM programs, and the runtime accelerates their execution. The two parts can work together for better performance but can also function independently." (Section 1)

The frontend contributes to efficiency in ways that a standalone inference engine cannot achieve. For example, when the `fork` primitive creates parallel branches in a program, the interpreter sends the shared prefix to the runtime as a "Frontend Hint" before sending the branch-specific continuations. This hint "simplifies runtime scheduling and matching, exemplifying the benefits of frontend-runtime co-design" (Section 3). A generic inference engine receiving multiple independent requests has no way to know they share a common prefix — that structural information was lost when the program was decomposed into individual API calls. By maintaining the program structure and communicating it to the runtime, SGLang enables optimizations that are impossible for workload-agnostic systems.

The paper's positioning relative to existing work is nuanced:

- **Versus inference engines**: SGLang doesn't compete on single-request latency; it competes on multi-call program throughput. The speedups come from structural reuse that inference engines cannot see.
- **Versus programming frameworks**: SGLang doesn't claim a more expressive language; it claims a more efficient execution of the same expressiveness. The primitives are similar to Guidance and LMQL (Table 1), but the runtime is fundamentally different.
- **Versus high-level frameworks**: SGLang is complementary, not competitive. It can serve as an efficient backend for systems like DSPy, accelerating their execution without changing their programming model.

The paper also explicitly connects to broader trends in LLM deployment. The observation that SGLang's RadixAttention achieved a 52.4% cache hit rate for LLaVA-NeXT-34B and 74.1% for Vicuna-33B in production (Chatbot Arena deployment, Section 6.2) demonstrates that the structural sharing patterns the system exploits are not artificial benchmarks — they arise naturally in real-world deployments from common system messages, frequently reused example images, and multi-turn chat histories.

The theoretical grounding for RadixAttention's scheduling policy provides a further positioning claim: Theorem 3.1 proves that for offline batches, longest-shared-prefix-first ordering achieves the optimal cache hit rate, with a cache size at least equal to the maximum request length. This transforms KV cache reuse from an empirical heuristic into a principled scheduling problem with provable guarantees — a level of rigor uncommon in LLM serving systems and a direct response to the ad-hoc reuse mechanisms in prior work.

In essence, the paper positions SGLang as filling a gap that exists precisely because the field has treated the "programming" and "executing" of LM programs as separate concerns. By reunifying them, SGLang enables optimizations that neither improved programming models alone nor improved inference engines alone could achieve. The 6.4× throughput improvement is the empirical substantiation of this architectural argument.

## 3. Technical Approach

### 3.1 Reader Orientation

SGLang is a complete system—a Python-embedded domain-specific language paired with a co-designed runtime engine—for writing and efficiently executing programs that orchestrate multiple LLM calls with control flow and structured outputs. The system solves the dual problem of *programming complexity* (brittle string manipulation, manual parallelism, ad-hoc output parsing) and *execution inefficiency* (redundant KV cache computation, serial constrained decoding) by having the frontend language expose the multi-call structure of LM programs to the runtime, which then exploits that structure for aggressive, automatic optimization.

### 3.2 Big-Picture Architecture (Diagram in Words)

The SGLang system, as depicted in Figure 1, has two major components that can work together or independently:

1. **SGLang Frontend Language** — an embedded DSL in Python that provides primitives (`extend`, `gen`, `select`, `fork`, `join`, `image`, `video`) for constructing prompts, calling LLMs, specifying output constraints via regular expressions, and controlling intra-program parallelism. Programs written in this language are executed by an **interpreter** that manages prompt state as an asynchronous stream, submitting primitive operations for non-blocking execution and handling synchronization when results are fetched. A **compiler** mode (discussed in Appendix D) traces programs into computational graphs for additional static optimizations.

2. **SGLang Runtime (SRT)** — the backend execution engine that receives prompts and primitives from the frontend interpreter and applies three novel optimizations:
   - **RadixAttention** (Section 3): Maintains an LRU cache of KV cache tensors organized in a radix tree, enabling automatic prefix reuse across generation calls without manual configuration. Includes a cache-aware scheduling policy that orders requests to maximize cache hit rate.
   - **Compressed Finite State Machine** (Section 4): Accelerates constrained decoding by analyzing the FSM derived from a regular expression constraint, compressing adjacent singular-transition edges, and decoding multiple deterministic tokens in a single forward pass.
   - **API Speculative Execution** (Section 5): For black-box API models (e.g., GPT-4), has early generation calls continue past their stop conditions, then matches and reuses the extra generated tokens for later primitives to reduce API call count and input token costs.

Information flows as follows: a user writes a Python program using SGLang primitives → the interpreter manages prompt state as an asynchronous stream, submitting `extend`, `gen`, `select`, and `fork` operations → for open-weight models, the SGLang Runtime receives these requests, performs prefix matching against the RadixAttention tree, executes generation with optional compressed-FSM constrained decoding, and returns results → the interpreter handles synchronization (blocking on fetched results) and control flow (conditionals, loops). The frontend also sends "hints" to the runtime (e.g., fork structure) that enable better scheduling decisions—an explicit co-design benefit.

### 3.3 Roadmap for the Deep Dive

- **First, the SGLang programming model and execution modes**, because the runtime optimizations are meaningless without understanding the program structures they exploit. This covers the language primitives, the interpreter's asynchronous stream model, and the compiler mode.

- **Second, RadixAttention and KV cache reuse**, because it is the primary source of throughput improvement and the most architecturally novel component. This covers the radix tree data structure, the LRU eviction policy with reference counting, the cache-aware scheduling algorithm, the theoretical optimality result (Theorem 3.1), and the distributed extension.

- **Third, the compressed finite state machine for constrained decoding**, because it addresses the second major inefficiency (serial token-by-token constrained generation) and introduces a clever compiler-style optimization (edge compression) to the decoding process.

- **Fourth, API speculative execution**, because it extends the system's benefits to black-box API models where the runtime cannot modify the inference process itself.

- **Fifth, the compiler mode and code movement optimization**, because it demonstrates the architectural extensibility of the system and an intriguing use of LLMs for compiler optimization.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems paper** whose core idea is that the multi-call structure intrinsic to LM programs—shared prefixes, parallel branches, constrained outputs—can be systematically exposed by a frontend language and automatically exploited by a co-designed runtime for significant efficiency gains.

---

#### The SGLang Programming Model and Execution Modes

The SGLang frontend is a domain-specific language embedded in Python. The key design insight is that prompt construction and LLM interaction should be expressed as operations on a **prompt state** managed as an **asynchronous stream**, rather than as explicit string concatenation and synchronous API calls. This design serves two purposes: it simplifies the programming model (the user writes what looks like sequential code, but the interpreter handles asynchrony), and it preserves the structural information (shared prefixes, fork/join relationships, output constraints) that the runtime needs for optimization.

**Prompt state as an asynchronous stream.** A SGLang program operates on a prompt state object (conventionally named `s`). Primitives are appended to this state using the `+=` operator, which submits the operation to an underlying stream executor running in a background thread:

```python
s += system("You are a helpful assistant.")
s += user("Hello!")
s += assistant(gen("reply"))
```

The critical mechanism is that `+=` is **non-blocking**—it submits the primitive for asynchronous execution and returns immediately, allowing the Python program to continue executing. This is explicitly analogized to launching CUDA kernels asynchronously (Section 2). The stream executor in the background thread manages the actual sequencing and execution of these operations. Fetching a generation result via `s["variable_name"]` **blocks** until the result is ready, ensuring correct synchronization at program points where the output is needed for control flow or further prompt construction. This approach means that the user's Python code can express complex workflows—branching, merging, loops over generations—without manually managing threads, callbacks, or async/await patterns.

**Language primitives.** The paper defines a small set of composable primitives (Section 2, with the running example in Figure 2):

- **`extend` (or `+=` with a string)**: Appends a literal string to the prompt. This is the basic mechanism for constructing the prompt context, including system messages, user inputs, and structural text between generation calls.

- **`gen(name, stop=None, regex=None)`**: Calls the model to generate text, storing the result in a variable accessible via `s[name]`. The optional `stop` argument specifies stop strings (the generation halts when any stop string is encountered). The optional `regex` argument constrains the output to match a regular expression—this is the mechanism that triggers the compressed FSM optimization in the runtime. The generated text is appended to the prompt state.

- **`select(name, choices)`**: Calls the model to choose the highest-probability option from a list of choices. This is implemented by computing the log probabilities of each choice token sequence and selecting the one with the highest score. Unlike `gen`, it does not append the choice text to the prompt state—it only stores the selected choice in `s[name]`. This primitive is used for classification-style decisions (e.g., "Is the essay related? yes/no" in Figure 2).

- **`fork(n)`**: Creates `n` parallel copies of the current prompt state, returning a list of `n` fork objects. Each fork inherits the full prompt history up to the fork point. Operations appended to a fork do not affect other forks. This enables parallel exploration of multiple branches, such as evaluating an essay along multiple dimensions simultaneously (Figure 2) or exploring multiple reasoning paths in tree-of-thought.

- **`join(forks)`**: Merges multiple fork objects back into the parent prompt state. The paper does not specify the exact mechanics of join (whether it concatenates, interleaves, or simply allows the parent to access fork results), but the running example shows that fork results are typically accessed via `f["variable_name"]` and then manually incorporated into the parent state using string concatenation.

- **`image(path)` and `video(path)`**: Accept image and video file paths for multi-modal models. These primitives handle the encoding of visual inputs into the format expected by the model.

The primitives are designed to be **composable with Python control flow and libraries**. The running example in Figure 2 demonstrates this clearly: after the `select` primitive determines whether the essay is related, the program uses a standard Python `if` statement to conditionally return early. The parallel forks are created with a Python `for` loop. The fork results are merged with a Python list comprehension and `"\n".join()`. This design means SGLang does not need to invent a new control flow language—it leverages Python's existing constructs, reducing the learning curve and enabling the use of Python's ecosystem (libraries, debugging tools, type checkers).

**Execution modes: interpreter vs. compiler.** The paper describes two ways to execute SGLang programs (Section 2, "Execution modes"):

- **Interpreter mode (default):** The prompt is treated as an asynchronous stream. Primitives are submitted to the stream executor for non-blocking execution. Each prompt is managed by a stream executor in a background thread. This mode supports all language features, including data-dependent control flow (since Python executes normally, making decisions based on fetched generation results). The interpreter mode is used for all experiments in the main body of the paper.

- **Compiler mode (Appendix D):** SGLang programs can be traced and compiled into computational graphs. The program is executed with abstract arguments, and the primitive operations are recorded as nodes in a directed acyclic graph (DAG) with edges representing dependencies. There are two types of dependencies: **intra-stream** (operations within a single prompt state must execute in order) and **inter-stream** (one stream fetching a variable from another stream creates synchronization). Figure 14b shows the graph for the skeleton-of-thought program (Figure 14a), with three streams corresponding to three function calls. The compiler mode enables graph rewriting optimizations (e.g., code movement for better prefix sharing, as explored in Appendix D.2), reduced runtime overhead (no reinterpretation of Python), and program serialization. However, this mode is limited to programs without data-dependent control flow—a limitation the paper acknowledges and plans to address.

**A critical co-design mechanism: Frontend Hints.** Section 3 mentions a specific example of frontend-runtime co-design. When the interpreter executes a `fork` primitive, it does not simply send the full prompts of each fork to the runtime as independent requests. Instead, it first sends the **shared prefix** (the prompt content up to the fork point) as a hint to the runtime, ensuring that the prefix is correctly inserted into the radix tree. It then sends the fork-specific continuations. This hint "simplifies runtime scheduling and matching" (Section 3) because the runtime knows, before processing the fork-specific parts, that the prefix exists in the tree and can be shared. Without this hint, the runtime would receive multiple requests with overlapping prefixes and would have to infer the sharing from the prompt contents alone—a more complex and error-prone process. This is a concrete instantiation of the paper's thesis that structural information from the frontend enables optimizations that are difficult or impossible for a workload-agnostic runtime.

---

#### RadixAttention: Automatic KV Cache Reuse with a Radix Tree LRU Cache

RadixAttention is the central runtime optimization and the primary source of SGLang's throughput improvements. It addresses a specific inefficiency: in multi-call LM programs, many generation calls share common prompt prefixes (system prompts, few-shot examples, chat history, search history), but state-of-the-art inference engines discard the KV cache after each request completes, forcing redundant recomputation. RadixAttention treats the KV cache as a traditional cache—retaining computed KV tensors across requests, managing them with an LRU eviction policy, and organizing them in a radix tree for efficient prefix matching, insertion, and eviction.

**Background: What the KV cache is and why reusing it matters.** During autoregressive LLM inference, each token in the input sequence produces intermediate tensors (key and value vectors in the self-attention mechanism). These tensors—the KV cache—are required for computing attention scores when generating subsequent tokens. Critically, the KV cache for a token depends **only on that token and all previous tokens in the sequence**, not on future tokens. This means that if two sequences share a common prefix (e.g., the same system prompt and first user message), the KV cache for that prefix can be computed once and reused for both sequences, avoiding redundant computation.

The practical impact is twofold. First, **computation**: prefill (computing the KV cache for input tokens) is a compute-intensive operation that scales quadratically with sequence length in naive implementations and linearly with optimized attention (FlashAttention). Eliminating redundant prefill reduces latency, particularly time-to-first-token. Second, **memory**: KV cache tensors consume significant GPU memory. When multiple requests share a prefix, storing separate copies of the same KV cache wastes memory that could be used for larger batch sizes. By sharing the KV cache, more requests can fit in memory simultaneously, increasing throughput.

The paper identifies four common sharing patterns in Figure 9 (Appendix A.1): (a) few-shot learning, where all questions share the same few-shot examples; (b) self-consistency, where multiple reasoning chains share the same question prompt; (c) multi-turn chat, where each new turn shares the entire conversation history; and (d) tree-of-thought, where sibling branches share the same search history prefix. Existing systems cannot automatically handle all of these patterns. vLLM supports basic prefix sharing (system prompts) but not tree-structured or multi-level sharing. Guidance and LMQL lack batching support entirely. HydraGen and ChunkedAttention focus on CUDA kernel optimizations for shared prefixes but do not implement caching across requests.

**The radix tree data structure.** RadixAttention organizes KV cache tensors in a **radix tree** (also known as a Patricia trie or compressed trie). A radix tree is a space-optimized variant of a trie (prefix tree) where edges can be labeled with sequences of elements (tokens, in this case) rather than single elements. This is important for efficiency: if a request has a long prefix with no branching points, the entire prefix can be stored as a single edge rather than one edge per token.

The tree maps **token sequences to KV cache tensors**. Each node in the tree corresponds to a position in the token sequence, and the edge from a parent to a child is labeled with the substring of tokens that differentiates the child's path from the parent's. The KV cache tensors are stored in a **non-contiguous, paged layout**, where each page corresponds to one token. This paging scheme is compatible with vLLM's PagedAttention (Kwon et al., 2023), which also uses paged KV cache storage for efficient memory management. The radix tree structure itself is stored on the CPU, with "negligible maintenance overhead" (Section 3).

**Dynamic tree operations: insertion, splitting, and matching.** Figure 3 illustrates the dynamic evolution of the radix tree across nine time steps in response to incoming requests. The operations are:

- **Insertion**: When a new request arrives, the runtime searches the tree for the longest matching prefix. The matched portion reuses existing KV cache. The unmatched suffix is inserted as a new branch. In Figure 3 step (2), the entire system prompt, user message, and assistant response are inserted as a single edge to a new node (node "a").

- **Splitting**: When a new request shares only a partial prefix with an existing edge, the edge is **split** at the divergence point. In Figure 3 step (4), a new chat session begins with the same system prompt as an existing session. The existing edge (system prompt + first turn) is split: the shared system prompt becomes its own edge leading to node "b", and the two chat sessions diverge from there (nodes "c" and "d").

- **Matching**: For each incoming request, the runtime searches for the longest prefix match in the tree and reuses the KV cache for that prefix. Only the unmatched suffix needs new KV cache computation.

The key property that enables this is that **KV cache computation is prefix-dependent**: the KV cache for position `k` depends only on tokens `0` through `k`. Therefore, if two sequences match up to position `k`, the KV cache for positions `0` through `k` is identical and can be shared.

**Reference counting and eviction.** GPU memory is finite, so cached KV tensors must be evicted when memory is needed for new requests. RadixAttention uses two mechanisms working together:

- **LRU eviction policy:** The least recently used **leaf** is evicted first. By evicting leaves first, the system preserves common ancestors—if node A is the parent of nodes B and C, and B is evicted as the LRU leaf, C can still reuse node A's KV cache. When C is eventually evicted (becoming a leaf), A can be evicted too. This is a standard property of tree-based LRU caches.

- **Reference counting:** Each node maintains a reference counter indicating how many **currently running** requests are using it. A node is **evictable** only if its reference counter is zero. This prevents eviction of nodes needed by the current batch. In the continuous batching setting (where the model processes multiple requests concurrently, and requests join and leave the batch dynamically), a node remains pinned as long as any active request in the batch uses it.

The paper explicitly notes that RadixAttention does **not** preallocate a fixed-size memory pool for the cache. Instead, "the cached tokens and the currently running requests share the same memory pool" (Section 3). This means the system dynamically allocates memory between cache and active requests based on demand. When many waiting requests need to run, the system will evict cached tokens to make room for a larger batch size. This dynamic allocation is pragmatic—it avoids the problem of tuning cache size, which would depend on workload characteristics—but it means that under high load, the cache hit rate may drop as cached tokens are evicted in favor of batch throughput.

**Cache-aware scheduling.** When many requests are waiting in the queue, the order in which they are executed significantly impacts the cache hit rate. A naive first-come, first-served (FCFS) schedule might alternate between unrelated requests, causing cache thrashing (evicting and recomputing shared prefixes repeatedly). The paper designs a **cache-aware scheduling** algorithm that sorts waiting requests by **matched prefix length** and prioritizes those with longer matches. The pseudocode is provided in Algorithm 1 (Appendix A.2).

The algorithm operates as follows in the continuous batching setting:

1. **Extract all waiting requests** from the queue.
2. **Match each request** against the radix tree to determine its prefix match length (number of cached tokens).
3. **Sort requests** in descending order of matched prefix length. This is the "longest-shared-prefix-first" order.
4. **Select requests for the next batch** by iterating through the sorted list, adding requests until the available memory (evictable cache size plus free pool memory) is exhausted. For each added request, the reference counter of its matched prefix nodes is incremented.
5. **Merge the selected requests** into the current running batch.
6. **Allocate memory and evict** if necessary: if the needed size exceeds available memory, evict LRU leaves until sufficient memory is freed.
7. **Run the batch.**
8. **Process finished requests:** decrement reference counters on their prefix nodes, and insert their full token sequences into the radix tree for future reuse.

The key insight is that executing requests that share long prefixes consecutively maximizes the probability that the shared prefix remains in the cache (not evicted) throughout the batch. This is formalized in Theorem 3.1.

**Theorem 3.1 and its significance.** The paper proves:

> "For a batch of requests, we can achieve an optimal cache hit rate by visiting the radix tree of the requests in the depth-first search order, with a cache size ≥ the maximum request length. The longest-shared-prefix-first order is equivalent to a depth-first search order." (Section 3)

The proof (Appendix A.3) establishes a lower bound on the computational complexity of the KV cache for a set of requests:

$$C \geq \sum_{e \in \text{edges}(T)} |e|$$

where $C$ is the total computational cost of the KV cache for the requests, $T$ is the radix tree built from the requests in the batch, $\text{edges}(T)$ is the set of all edges in $T$, and $|e|$ is the size (number of tokens) of the KV cache associated with edge $e$.

**What this computes:** The lower bound says that, at minimum, every edge in the radix tree must have its KV cache computed once. Since edges represent unique token sequences (after factoring out shared prefixes), there is no way to avoid computing each unique token sequence at least once.

**Why this form:** The radix tree compresses shared prefixes into shared edges. If two requests share a prefix of $k$ tokens, those $k$ tokens correspond to shared edges in the tree. The sum over edges captures that each unique token subsequence (after sharing) must be computed once. The lower bound is achievable only if shared edges are never recomputed.

The proof then shows that DFS order achieves this lower bound. When visiting the tree in DFS order, the first time an edge is traversed, its KV cache is computed. Because DFS explores the entire subtree rooted at that edge before backtracking, the edge remains in the cache (since it is continuously hit by the subtree requests) and is never recomputed. With a cache size at least as large as the maximum request length (the longest path from root to leaf), no edge on the current DFS path is evicted during the traversal of its subtree. The proof also shows by induction that longest-shared-prefix-first order is equivalent to DFS order on the radix tree: after processing a node, the next node with the longest shared prefix is the one with the lowest common ancestor on the current cached path, which is exactly the next node in DFS.

This theorem is significant because it transforms KV cache reuse from an empirical heuristic into a **principled scheduling problem with a provably optimal policy** (for the offline case). It provides a theoretical foundation for why the cache-aware scheduling algorithm works, and it gives a clear target (DFS order) that the online scheduler approximates. The paper notes that in the online case (where requests arrive dynamically, not in a fixed batch), the DFS order is disrupted, but the longest-shared-prefix schedule still approximates DFS behavior on the newly added portion of the tree (Appendix A.3).

**Distributed RadixAttention (data parallelism).** When running SGLang with multiple replica workers (data parallelism, where each worker is an independent instance of the model), each worker maintains its own sub-tree of the KV cache. The router maintains a **meta-tree**—a trie that tracks all sub-trees and the devices they reside on. When a new batch of requests arrives, the router performs prefix matching on the meta-tree and dispatches requests to workers based on **affinity** (the length of the shared prefix with each worker). After requests are processed, both the router and workers update their trees independently. If a worker evicts a node, it commits the eviction to a queue; the router processes this queue during low-activity periods to update the meta-tree. The paper reports that with four workers on MMLU, this design achieves linear scaling and optimal cache hit rate "with minimal overhead from this weakly consistent distributed cache design" (Appendix A.4). This is a practical engineering contribution: maintaining perfect consistency between router and workers would require synchronous communication on every eviction, which would be prohibitively expensive.

**Frontend Hints in RadixAttention.** As described earlier, the frontend interpreter sends fork prefixes as hints to the runtime. This is more than a convenience—it ensures the radix tree is correctly structured before the fork-specific prompts arrive, preventing race conditions where two fork requests might each try to insert the shared prefix independently, creating duplicate subtrees or requiring complex merging logic. The hint mechanism is a concrete example of how the frontend-runtime co-design enables correctness and simplicity that would be harder to achieve in a decoupled architecture.

---

#### Compressed Finite State Machine for Fast Constrained Decoding

The second major runtime optimization addresses constrained decoding: generating text that conforms to a specified regular expression (e.g., a JSON schema). Constrained decoding is essential for LM programs that need to produce structured outputs for downstream processing or integration with software systems. The paper identifies a specific inefficiency in existing approaches and proposes a compiler-style optimization—compressing the finite state machine—to accelerate the process.

**Background: How constrained decoding works.** When a regular expression constraint is specified (e.g., the JSON schema `\{"summary": "[\w\d\s]+\.", "grade": "[ABCD][+-]?"\}` from Figure 2), the system converts the regex into a **Finite State Machine (FSM)**. An FSM is a directed graph where:

- **States** represent positions in the regex matching process (e.g., "we have matched the opening brace and are now expecting the key `summary`")
- **Edges** represent valid transitions between states, labeled with characters or strings that can advance the match
- An **initial state** is the starting point (before any characters are matched)
- **Final states** represent complete matches of the regex

During autoregressive decoding, the LLM generates one token at a time. At each step, the current FSM state determines which tokens are **allowed** (i.e., which tokens, when decoded to their string representations, would advance the FSM along a valid transition). Tokens that would lead to invalid transitions are masked (their log probabilities are set to negative infinity), ensuring the model can only sample valid next tokens. This is the standard approach described in Willard and Louf (2023) and implemented in Guidance and LMQL.

**The inefficiency: single-token decoding on deterministic paths.** The problem arises when the FSM has a sequence of states where each state has exactly **one** outgoing transition (a "singular transition edge"). In such cases, there is only one valid next token at each step. For example, the literal string `{"summary": "` in the JSON schema corresponds to multiple tokens (e.g., `{`, `"`, `summary`, `"`, `:`, `"`), but each token in this sequence is deterministic given the constraint. Yet existing systems decode these tokens one at a time, requiring a separate forward pass for each token, even though there is no choice involved—the model is forced to output the same token regardless.

The paper states: "the lack of integration between the FSM and the model runner in existing systems prevents multi-token processing, resulting in slow decoding" (Section 4). The FSM operates at the token-mask level: it provides a mask of allowed tokens for the current step, but it does not communicate to the model runner that the current state is on a deterministic path where multiple tokens could be decoded together.

**Building the compressed FSM.** SGLang's solution is to perform a preprocessing step on the FSM before using it for decoding:

1. **Build a character-level FSM**: Start with the original FSM derived from the regex, where edges are labeled with characters or short strings.

2. **Identify singular transition edges**: An edge is a **singular transition edge** if:
   - Its source node has exactly one outgoing edge (no branching), AND
   - That edge has exactly one acceptable character/string (no alternatives like `[0-9]`).

3. **Compress adjacent singular edges**: A **compressed edge** is formed by merging a sequence of consecutively adjacent edges `(e_0, e_1, ..., e_k)` where `e_1` through `e_k` are singular transition edges, into a single edge. The text of the compressed edge is the concatenation of the texts of the original edges.

4. **Recursively apply compression**: Starting from the initial state, the algorithm recursively merges singular transition edges into their preceding edges until no further compression is possible. The result is a **Compressed FSM** where deterministic multi-character (and thus multi-token) paths are represented as single edges.

Figure 4 illustrates this with the regex `{"summary": "`. The original FSM (Figure 4a) has separate states and edges for each character: `0` → `{` → `1` → `"` → `2` → `s` → `3` → ..., requiring multiple decoding steps (Figure 4c). The compressed FSM (Figure 4b) merges the entire deterministic sequence into a single edge from state `0` to a state that expects the summary text, allowing the entire sequence to be decoded in one step (Figure 4d).

**Decoding with the compressed FSM: Jump-Forward.** At runtime, when the model is in a state connected to a compressed edge, the runtime can perform a **Jump-Forward**: instead of decoding one token at a time, it feeds the entire text of the compressed edge as a multi-token sequence in a single forward pass. This bypasses multiple iterative decode steps and their associated forward passes.

The mechanics of Jump-Forward involve careful handling of tokenization:

- The compressed edge stores its text as a string (e.g., `{"summary": "`).
- When the model reaches a state with a compressed outgoing edge, the runtime needs to convert this string into tokens for the LLM's forward pass.
- However, tokenization is not a simple character-by-character mapping. The paper gives a concrete example: the compressed text `{"summary": "` should tokenize as `{`, `"`, `summary`, `"`, `:`, `"`, `_` (underscore representing space), not as `{"`, `summa`, `ry`, `":`, `_"` or other arbitrary partitions. The LLM's tokenizer has a specific vocabulary and tokenization algorithm; arbitrarily splitting the compressed string would produce tokens the model was not trained on, distorting the probability distribution.

**Retokenization to handle tokenization artifacts.** To solve this, SGLang performs **retokenization**: it takes the original tokenized prefix (all tokens up to the current state) plus the text of the compressed edge, and re-tokenizes the entire sequence with the model's tokenizer. The previously computed prefix tokens are used as-is for KV cache purposes; only the new tokens from the compressed edge are extracted and fed to the model. The paper states this "only brings minor retokenization overhead" (Appendix B.2).

**Probability distortion: a limitation.** The paper acknowledges a subtle limitation of the compressed FSM approach: it can **distort probability distributions** in cases where the constraint includes multiple alternative paths of different lengths. The example given (Appendix B.3) involves a regex like `Excellent|Above Average|Fair|Below Average`. If the model is at a state that can transition to any of these options, and a compressed edge aggregates them, the model's probability over the compressed token sequences may not accurately reflect the intended distribution over the choices. This is because the LLM's tokenizer may split the choice strings into different numbers of tokens, and the sum of probabilities of all token sequences that result in a given choice is not directly computed during constrained decoding.

The workaround suggested is to include the choices directly in the prompt (prefill), so the model is aware of them and can align its token generation accordingly. However, the paper acknowledges this doesn't solve the fundamental issue and highlights it as a direction for future research.

---

#### API Speculative Execution for Black-Box Models

The previous optimizations (RadixAttention, compressed FSM) require modifying the model inference process and are therefore only applicable to open-weight models running on SGLang's own runtime. To extend benefits to API-access-only models (e.g., OpenAI's GPT-4, Anthropic's Claude), the paper introduces **API speculative execution**.

**The problem: redundant input tokens in multi-call programs.** When a SGLang program makes multiple generation calls to an API model, each call is a separate API request. Consider a program that extracts multiple fields from a text:

```python
s += context
s += "name:" + gen("name", stop="\n")
s += "job:" + gen("job", stop="\n")
s += "age:" + gen("age", stop="\n")
```

Naively, this generates three separate API calls. Each call sends the `context` as part of the input prompt, and the user pays for those input tokens three times. This is wasteful when the context is large and the extraction pattern is predictable.

**The mechanism: continue past stop conditions on early calls.** API speculative execution works as follows:

1. **On the first generation call** (e.g., `gen("name", stop="\n")`), instead of stopping precisely at the newline, the system instructs the API (via its parameters) to continue generating a few more tokens past the stop condition.

2. **The interpreter retains the extra generated tokens** (beyond the stop string) rather than discarding them.

3. **On subsequent generation calls** (e.g., `gen("job", stop="\n")`), the interpreter checks whether the extra tokens from the previous call already contain the text that the subsequent call would generate. If the extra tokens match the expected pattern (e.g., the model continued with `job: Software Engineer\n` after generating the name), the interpreter **reuses** those tokens instead of making a new API call.

4. **The user saves the input token cost** and API latency for the subsequent calls.

The paper reports a concrete result: on a Wikipedia page extraction task (extracting three fields using GPT-3.5), API speculative execution reduces input token costs by "about threefold" (Section 6.2). This is because the first call's input cost covers the context once, and if the speculation succeeds, the second and third calls are avoided entirely.

**Accuracy of speculation.** The effectiveness of this technique depends on the model reliably producing the expected multi-field output format in a single generation. The paper notes that "with careful prompt engineering, the model can correctly match the template with high accuracy" (Section 5). Few-shot prompting (providing examples of the desired multi-field output format) improves accuracy. However, if the speculation fails (the model's extra output does not match the expected pattern for subsequent fields), the system falls back to making the individual API calls. The cost in the failure case is the extra output tokens generated during speculation, which is typically small compared to the input token savings in the success case.

This optimization is less architecturally novel than RadixAttention or the compressed FSM—it is essentially a prompt engineering trick combined with reuse of already-generated output—but it demonstrates SGLang's practical approach to optimizing all deployment scenarios, not just open-weight models.

---

#### Compiler Mode and GPT-4-Assisted Code Movement Optimization

While not the main focus of the paper, the compiler mode (Appendix D) introduces an architectural extension and an intriguing optimization technique that leverages LLMs themselves for compiler optimization.

**Intermediate representation (IR).** SGLang defines an IR for representing SGLang programs as computational graphs. The graph nodes correspond to primitive operations: `ConstantText` (literal strings), `Argument` (function parameters), `Gen`, `Select`, `Variable` (fetched results), `Fork`, `GetForkItem`, and `Join`. Edges represent dependencies. Intra-stream dependencies (sequential `+=` operations within a single prompt state) enforce execution order. Inter-stream dependencies (one stream fetching a variable from another) enforce synchronization.

Figure 14 shows this concretely: the skeleton-of-thought program (Figure 14a) compiles into a graph (Figure 14b) with three streams. Stream 1 (the main prompt) generates two tips, then forks into Stream 2 and Stream 3 (each expanding one tip), then merges the results back into Stream 1 for the final summary.

**Graph construction via tracing.** The graph is constructed by running the program with abstract arguments (placeholders instead of real values) and recording the sequence of primitive invocations and their dependencies. This tracing approach is limited to programs without data-dependent control flow—a `gen` whose result determines which `if` branch is taken cannot be traced statically. The paper notes this limitation and plans to address it in future work.

**Case study: code movement for improving prefix sharing.** The paper explores a specific compiler optimization enabled by the IR: reordering nodes to increase the length of shareable constant prefixes. The key observation is that the order of prompt components affects KV cache reusability. For example:

- Prompt A: "Here is a question: {question}. Please act as a math expert and solve it."
- Prompt B: "Please act as a math expert and solve it. Here is a question: {question}."

If many questions share the same role instruction ("Please act as a math expert..."), moving that instruction to the front of all prompts creates a longer shareable prefix compared to having the question-specific text first.

Traditional program analysis cannot perform this optimization because the text strings are opaque natural language—the compiler doesn't know that "Please act as..." and "Here is a question..." are semantically independent and can be reordered without changing meaning. The paper takes an unconventional approach: **prompting GPT-4 to perform the optimization**. The process:

1. A few-shot prompt is constructed with several examples showing SGLang IR graphs and their semantically equivalent reordered versions.
2. GPT-4 is given a new SGLang IR graph and asked to reorder nodes to maximize shareable prefix length.
3. The output is validated by manual inspection.

On a test set of 15 prompt templates (with 5 used as few-shot examples), GPT-4 successfully reordered 12 out of 15 without altering semantics, achieving an average increase of **60 tokens in shareable prefix length**. Failures occurred when GPT-4 incorrectly understood the semantic dependencies between prompt components, sometimes moving all constants to the front even when the original order mattered (e.g., instructions that reference specific preceding content).

This case study is presented as exploratory—"More work is needed to make these kinds of optimizations reliable in the future" (Appendix D.2)—but it demonstrates an interesting direction: using LLMs as compiler optimization passes for natural language programs. The optimization is classified as "aggressive" because it does not strictly preserve the original computation (reordering changes the prompt, which could affect model behavior), distinguishing it from traditional compiler optimizations that must be semantics-preserving in a formal sense.

## 4. Key Insights and Innovations

### Innovation 1: KV Cache Reuse as a General-Purpose Caching Problem Rather Than a Request-Lifecycle Artifact

The field's default treatment of the KV cache prior to this work was fundamentally tied to the request lifecycle: compute the cache during prefill, use it during decode, discard it when the request completes. This made intuitive sense for independent API calls—there was no obvious reason to keep stale tensors around. Even systems that explored some reuse (vLLM's prefix sharing, PromptCache's modular reuse) treated it as a special-case optimization: manually configure this system prompt to be shared, or accept accuracy degradation for non-exact matches. The KV cache was a side effect of inference, not a first-class resource to be managed.

RadixAttention reframes the problem entirely. By treating the KV cache as a **traditional cache**—with an LRU eviction policy, a tree-structured index for efficient prefix matching, reference counting for active-request safety, and dynamic memory sharing with active requests—it elevates KV cache reuse from an ad-hoc optimization to a principled systems problem. The insight is not the specific data structure (radix trees are well-known) but the recognition that **the multi-call structure of LM programs naturally creates cache-friendly access patterns** that can be exploited with standard caching techniques if the runtime preserves the cache across requests.

This is a conceptual shift with practical consequences. In a traditional "discard-after-request" system, sharing can only happen within a single batch—requests that arrive at different times cannot benefit from each other's computation. RadixAttention's persistent cache means that even sequential, non-overlapping requests benefit from reuse: a chat session that sends one message now and another in five minutes still reuses the conversation history's KV cache. The 52.4% and 74.1% cache hit rates observed in the Chatbot Arena production deployment (Section 6.2) validate that this persistence matters in practice, not just in benchmark micro-batches.

The theoretical contribution—Theorem 3.1's proof that longest-shared-prefix-first (DFS order) achieves optimal cache hit rate for offline batches with a cache size exceeding the maximum request length—further distinguishes this from prior ad-hoc approaches. This transforms scheduling from a "try some heuristics" problem into one with a **provably optimal policy** in the offline setting, providing both an upper bound on achievable efficiency and a concrete target for online scheduling algorithms. Prior work (vLLM, ChunkedAttention, HydraGen) treated scheduling as an implementation detail; SGLang treats it as an optimization problem with theoretical guarantees.

The significance goes beyond the 6.4× throughput gains. It establishes that **inference-time caching for LLMs is not just about memory management but about scheduling**—the order in which requests are processed determines the cache hit rate, which in turn determines throughput. This insight parallels the evolution of operating system page replacement algorithms from FIFO to LRU to working-set models, and suggests a rich design space for future work on cache-aware scheduling policies that balance fairness (preventing starvation under longest-prefix-first) with efficiency.

### Innovation 2: Co-Designing the Programming Model and Runtime for Structural Information Flow

Most LLM systems treat the programming interface and the execution engine as separable concerns. Write your application in LangChain, DSPy, or raw API calls; the inference engine (vLLM, TGI) serves requests blindly. This separation is clean architecturally but loses information: when a DSPy program forks three reasoning branches and merges them, the inference engine sees four independent requests with no knowledge of their shared prefix or fork/join structure. The programmer's intent—"these branches share context up to this point"—is lost in translation to API calls.

SGLang's most distinctive architectural contribution is not any single optimization but the **systematic co-design** that makes structural information flow from the frontend language to the runtime optimizer. The "Frontend Hint" mechanism exemplify this: when the interpreter executes `fork(n)`, it explicitly sends the shared prefix to the runtime before the branch-specific continuations, ensuring the radix tree is properly structured. Without this hint, the runtime would have to infer sharing from raw token sequences—possible but more complex and error-prone. With it, the fork structure becomes an explicit signal that the runtime can act on.

This co-design philosophy manifests throughout the system:
- The `regex` argument on `gen` communicates output constraints to the runtime, which activates the compressed FSM path—not as a separate post-processing step but integrated into the generation loop.
- The asynchronous stream execution model (`+=` is non-blocking) enables the interpreter to continue executing Python control flow while generations run, expressing parallelism that the runtime can exploit.
- The compiler mode's IR makes prompt structure statically analyzable, enabling optimizations like the GPT-4-assisted code movement for prefix sharing (Appendix D.2) that would be impossible with opaque string-based APIs.

The comparison to Guidance and LMQL (Table 1) is instructive here. Those systems provide similar language primitives (`extend`, `gen`, `select`) but lack a co-designed runtime; they delegate to generic backends (HF Transformers, llama.cpp, OpenAI API). SGLang's SGLang Runtime (SRT) is not just "another backend"—it's a backend that **understands the primitives**. When Guidance submits a `gen` call to llama.cpp, the backend sees a completion request. When SGLang submits a `gen` call to SRT, the runtime knows it's part of a larger program structure, can consult the radix tree for prefix matches, and can activate compressed FSM decoding if a regex is specified. The same primitive, but the runtime sees different optimization opportunities because the structural information survived the interface boundary.

This is more than an engineering convenience. It represents a **design principle for LLM systems**: the interface between the programming model and the execution engine should preserve as much of the program's structure as possible, because that structure encodes optimization opportunities. The 6.4× improvement over vLLM (Figure 5) is not just "SGLang has RadixAttention and vLLM doesn't"—vLLM has some prefix sharing. Rather, vLLM cannot see the fork/join structure, cannot receive frontend hints, and cannot activate specialized constrained decoding paths based on program-level annotations. The gain comes from **information preservation across the interface**, not just algorithmic improvements below it.

### Innovation 3: Compiler-Style Edge Compression as an Optimization for Constrained Decoding

Constrained decoding—generating text that matches a regular expression—has been approached from the language side (Guidance, LMQL, Outlines) as a token-masking problem: build an FSM from the regex, use the current FSM state to mask invalid tokens, generate one token, advance the FSM, repeat. This is correct and general but inherently **serial** on deterministic paths. If the constraint forces the literal string `{"summary": "`, the model must make a separate forward pass for `{`, then `"`, then `summary`, then `"`, then `:`, then `"`—six passes for a sequence with zero degrees of freedom.

What makes SGLang's approach distinctive is not that it identifies this inefficiency (the waste is obvious once pointed out) but that it applies a **compiler optimization technique—edge compression on a state machine—to the decoding process**. The operation is conceptually identical to what a regex compiler does when converting a regex to a deterministic FSM: merge sequences of states with single transitions. But existing constrained decoding systems kept the FSM and the inference engine separate—the FSM provided token masks, the engine sampled tokens, and neither considered whether multi-token advance was possible. The compressed FSM bridges this gap by analyzing the FSM's structure and communicating to the inference engine that a particular transition can be taken in a single step.

This is fundamentally an **integration insight**, not an algorithmic one. The individual pieces (FSM-based constrained decoding, multi-token forward passes) existed independently. The innovation is recognizing that **the FSM and the model runner must be co-designed** to exploit deterministic paths. The paper explicitly frames this as a limitation of prior systems: "the lack of integration between the FSM and the model runner in existing systems prevents multi-token processing" (Section 4). The 1.6× throughput improvement on JSON decoding benchmarks comes from this integration, not from any single component's superiority.

The probability distortion problem (Appendix B.3) suggests this innovation is both practical and incomplete. On simple deterministic paths (`{"summary": "`), the optimization is unambiguously beneficial and semantics-preserving. On branching paths with alternatives of different token lengths (`Excellent|Above Average|Fair`), the compressed FSM can aggregate choices in ways that don't accurately reflect the model's token-level probability distribution. This is a diagnostic finding: it reveals that **tokenization and constrained decoding interact in non-obvious ways**, and that optimizations that are safe at the character level may not be safe at the token level. This boundary condition is itself a contribution—it defines where the optimization applies and where more sophisticated techniques (e.g., computing exact sequence probabilities over compressed paths) would be needed.

### Innovation 4: Difficulty-Agnostic, Structure-Driven Optimization as an Alternative to Adaptive Computation

The dominant narrative in LLM optimization circa 2023–2024 has been **adaptive computation**: spend more compute where it's needed (difficult problems, uncertain steps) and less where it isn't (easy problems, confident predictions). This manifests in techniques like speculative decoding, early exiting, mixture-of-experts routing, and—most relevant to this paper—the compute-optimal test-time scaling work that SGLang is compared against in some contexts. The assumption is that efficiency comes from **per-sample adaptivity**: understanding the difficulty of each input and allocating resources accordingly.

SGLang pursues a fundamentally different optimization axis: **structural regularity exploitation**. Instead of asking "how hard is this problem?" and adapting compute per-sample, it asks "what parts of this program's execution are deterministic or shared?" and eliminates redundant work across samples. The key insight is that LM programs—by virtue of their multi-call structure—contain substantial **structural redundancy** that is independent of the semantic difficulty of the task. A 5-shot MMLU evaluation has a 500-token shared prefix regardless of whether the question is easy or hard. A multi-turn chat reuses the entire conversation history regardless of the complexity of the current turn's response. A tree-of-thought exploration shares search history prefixes regardless of whether the current branch leads to a correct answer.

This distinction matters because it suggests these two optimization strategies are **complementary, not competing**. Adaptive computation techniques address variability in the *semantic* difficulty of individual samples; SGLang's structural optimizations address redundancy in the *syntactic* structure of multi-call programs. A system could, in principle, apply both: use RadixAttention to eliminate redundant prefix computation and then use speculative decoding or early exiting to accelerate the remaining unique computation. The paper does not explore this combination, but the conceptual framework it establishes makes the complementarity clear.

The significance of this distinction extends to how one thinks about optimizing LLM systems. If the dominant inefficiency in LM programs is structural redundancy (which the paper's cache hit rate measurements suggest—50% to 99% across benchmarks, Appendix Figure 13), then the most impactful optimizations are those that identify and exploit program structure. This suggests a research direction orthogonal to the current focus on adaptive computation: **program analysis for LM programs**—techniques that statically or dynamically analyze multi-call LM programs to identify sharing opportunities, parallelizable branches, and constraint-satisfying decoding shortcuts. The compiler mode and GPT-4-assisted code movement (Appendix D) are early steps in this direction, but the paper makes clear that richer program analyses (scheduling, memory planning, auto-parallelization) remain open problems.

### Innovation 5: The Radix Tree as a Unifying Abstraction for Diverse KV Cache Reuse Patterns

Prior work on KV cache reuse addressed specific sharing patterns in isolation. vLLM handled system prompt sharing within a batch. ChunkedAttention handled prefix-aware chunking for long sequences. PromptCache proposed modular reuse of arbitrary substrings at the cost of accuracy. Each approach solved a specific case, and none provided a general mechanism that could handle the full diversity of sharing patterns that arise in LM programs—few-shot examples, multi-turn histories, tree-of-thought branches, self-consistency samples, and arbitrary combinations thereof (Figure 9).

The radix tree abstraction unifies these cases elegantly. Any sharing pattern, no matter how irregular, can be represented as paths in the tree. The tree operations—insertion, splitting, matching, eviction—are the same regardless of whether the sharing is a simple system prompt, a multi-level tree-of-thought exploration, or a batch of few-shot queries with overlapping examples. The dynamic splitting mechanism (Figure 3, step 4) automatically handles partial matches: when a new chat session shares only the system prompt with an existing session, the tree splits at the divergence point, creating shared ancestors and session-specific subtrees. No manual configuration, no per-use-case engineering.

What makes this an innovation rather than an obvious data structure choice is the **reference counting + LRU eviction integration**. In a system where cached KV tensors may be simultaneously used by running requests, eviction cannot simply follow LRU order—a node that is the LRU leaf might be needed by a currently executing batch. The reference counting mechanism (only evict nodes with zero references) ensures correctness under continuous batching, while the LRU-on-leaves policy ensures that shared ancestors are preserved as long as any descendant is active. This combination enables the radix tree to function correctly in a dynamic serving environment, not just in a static caching scenario.

The distributed extension (Appendix A.4) further demonstrates the abstraction's generality. By maintaining worker-local radix trees and a router-level meta-tree, the system extends prefix-aware caching to data-parallel deployments with minimal consistency overhead. The router dispatches requests based on prefix affinity (which worker has the relevant cached prefix), and workers independently manage their local trees. This "weakly consistent" design—where evictions are communicated asynchronously—is pragmatic and achieves linear scaling on MMLU with four workers. It suggests that the radix tree abstraction is not tied to single-GPU deployments but generalizes to distributed settings.

The theoretical result (Theorem 3.1) anchors this abstraction in a formal guarantee: depth-first traversal of the radix tree achieves optimal cache hit rate for offline batches. This transforms the radix tree from a convenient data structure into the **provably correct representation for the offline KV cache reuse problem**. The proof is straightforward (each edge in the tree must be computed at least once; DFS computes each exactly once), but its presence elevates the system design from empirical engineering to principled optimization. It also provides a clear metric for evaluating online scheduling policies: how close do they come to DFS-order optimality? The paper's report that cache-aware scheduling approaches 96% of the optimal hit rate on average (Figure 13) uses this theorem's bound as the reference point.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The evaluation spans twelve distinct workloads rather than a single benchmark dataset. These include: 5-shot MMLU [14] and 20-shot HellaSwag [61] for few-shot learning benchmarks; ReAct agent traces [57] and generative agent traces [36] extracted and replayed from the original papers; Tree-of-thought [56] on GSM-8K problems; Skeleton-of-thought [33] for tip generation; LLM Judge with branch-solve-merge prompting [40]; JSON decoding with a regex-specified schema; multi-turn chat with 4 turns (short output: 4–8 tokens; long output: 256–512 tokens, with input per turn randomly sampled between 256–512 tokens); DSPy retrieval-augmented generation (RAG) pipeline [20] using its official example; and multi-modal benchmarks llava-bench-in-the-wild and ActivityNet for image and video models respectively. For multi-modal baselines, the paper uses the model authors' original Hugging Face Transformers implementations since other baseline systems "are not well supported."

- **Base model(s).** The primary experiments use Llama-2 models at 7B and 70B scales [49], the sparse mixture-of-experts Mixtral-8x7B [17], multi-modal LLaVA-v1.5-7B for images [27], LLaVA-NeXT-34B for video [62], and OpenAI's GPT-3.5 for API model experiments. All open-weight models use float16 precision. The paper argues these models span a representative range of contemporary LLM capabilities and architectures (dense, MoE, multi-modal). The parameter range (7B to 70B) tests whether optimizations generalize across model scales.

- **Metrics.** Two performance metrics are reported: **throughput** and **latency**. Throughput is measured as the maximum number of program instances executed per second (programs per second, p/s), obtained by "running a sufficiently large batch of program instances." Latency is measured by executing "a single program at a time without batching" and reporting the average latency across multiple instances. The paper normalizes both metrics relative to SGLang's performance in bar charts (Figures 5, 6, 7, 12), making SGLang always 1.0 and baselines fractions thereof. For multi-modal experiments (Table 2), throughput is reported in images per second and frames per second. An additional metric, **cache hit rate**, is defined as `number of cached prompt tokens / number of prompt tokens` (Section 3) and is reported per-benchmark in Figure 13 (Appendix).

- **Baselines.** Three primary baselines are compared:
  - **Guidance** [13]: v0.1.8 with the llama.cpp backend. The paper notes this version "lacks batching and parallelism support."
  - **vLLM** [23]: v0.2.5 with its default API server. The paper explicitly states that "RadixAttention has been partially integrated as an optional experimental feature into the latest version of vLLM; therefore, we used an earlier version for comparison" (footnote, Section 6.1).
  - **LMQL** [4]: v0.7.3 with the Hugging Face Transformers backend. The paper notes LMQL suffers from "slow token-level processing and an unoptimized backend."
  
  For multi-modal experiments (LLaVA models), the baseline is "the model authors' original implementation in Hugging Face Transformers" since other baseline systems do not support these models well (Section 6.2).

- **Generation budget / compute accounting.** The paper does not use a generation budget in the sense of a fixed token or sample count per benchmark. Instead, throughput comparisons are made by running all systems on the same workload (same set of program instances) and measuring how many programs per second each system can process. This implicitly gives each system access to the same computational resources (GPU type and count) but allows them to use those resources differently (different batch sizes, different memory management). All systems compute the same results—"we do not turn on optimizations that will change the computation results so that all systems compute the same results" (Section 6.1). Hardware is explicitly controlled: 7B models run on a single NVIDIA A10G GPU (24GB); larger models run on multiple A10G GPUs with tensor parallelism; some additional experiments use A100 (80GB) GPUs.

- **Cross-validation / statistical protocol.** No cross-validation or statistical significance testing is reported. The paper does not report error bars, confidence intervals, or variance estimates on throughput and latency measurements. The production deployment numbers (52.4% and 74.1% cache hit rates) are reported as observed values over one month of operation without uncertainty quantification. The 20-shot GPT-4 code movement optimization experiment uses 5 of 20 templates as few-shot training examples and 15 as test cases, with manual inspection to verify correctness—this is the closest the paper comes to a train/test split, and it applies only to that specific ablation.

### Main Quantitative Results

#### End-to-End Throughput and Latency on Open-Weight Models

**Headline result:** SGLang improves throughput by up to **6.4×** and reduces latency by up to **3.7×** compared to vLLM, Guidance, and LMQL across a diverse set of LM program workloads on Llama-7B (Figures 5 and 6). The magnitude and source of improvement vary substantially by benchmark.

**Figure 5 (throughput, normalized)** shows SGLang at 1.0 on all twelve benchmarks, with vLLM achieving between approximately 0.25 and 0.95 depending on the workload. The largest throughput advantages for SGLang appear on MMLU (vLLM at ~0.15 of SGLang's throughput), multi-turn chat (short) (vLLM at ~0.15), and DSPy RAG Pipeline (vLLM at ~0.25). The smallest advantage is on multi-turn chat (long), where vLLM achieves approximately 0.95 of SGLang's throughput—nearly identical. Guidance and LMQL are present on only the first seven benchmarks (those they support) and typically achieve between 0.05 and 0.3 of SGLang's throughput; on the last five benchmarks they are excluded "due to slow performance and missing functionalities" (Section 6.2). LMQL's poor performance is attributed to "slow token-level processing and an unoptimized backend," while Guidance "lacks batching and parallelism support."

**Figure 6 (latency, normalized)** shows a similar pattern: SGLang at 1.0 (lower is better, so baselines have values greater than 1.0). The largest latency reductions are on JSON Decoding (vLLM at ~3.7× higher latency) and MMLU (vLLM at ~3.5× higher latency). On multi-turn chat (long), vLLM's latency is nearly identical to SGLang's (~1.0×).

**Benchmark-specific explanations for the speedups** (Section 6.2):

- **MMLU (5-shot):** SGLang reuses the KV cache of the five-shot examples via RadixAttention. This benefits both throughput (shared KV cache reduces memory usage, enabling larger batch sizes) and latency (reduced prefill computation, lowering time-to-first-token).

- **HellaSwag (20-shot):** SGLang achieves "two-level sharing"—reusing both the few-shot examples and the common question prefix for multiple-choice options. This creates deeper sharing trees than the single-level sharing on MMLU.

- **ReAct Agents and Generative Agents:** SGLang reuses the KV cache of the agent template and previous calls. Since agent traces involve multiple turns of observation, reasoning, and action, the conversation history accumulates, creating growing shared prefixes that RadixAttention exploits.

- **Tree-of-Thought and Skeleton-of-Thought:** SGLang parallelizes generation calls within a single program (using `fork`) and reuses the KV cache "as much as possible" across branches. The tree-structured search in Tree-of-Thought creates exactly the kind of irregular sharing pattern that RadixAttention's radix tree handles automatically.

- **JSON Decoding:** SGLang accelerates decoding specifically through the compressed finite state machine, which decodes multiple tokens at once on deterministic constraint paths.

- **Multi-turn chat:** SGLang reuses the KV cache of chat history. The speedup is pronounced for short outputs because "KV cache reuse mostly helps reduce the prefix time." For long outputs, "there is not much sharing between different chat sessions and the decoding time dominates," explaining why multi-turn chat (long) shows almost no speedup—the shared prefix (4 turns of conversation history) is small relative to the 256–512 tokens generated per turn, and decoding dominates overall latency.

- **DSPy RAG Pipeline:** SGLang reuses the KV cache of the common context example shared across retrieval-augmented queries.

**Cache hit rates achieved (Figure 13, Appendix):** The paper reports achieved and optimal cache hit rates across all benchmarks. Achieved hit rates range from approximately 50% to 99%. The cache-aware scheduling policy "approaches 96% of the optimal hit rate on average," meaning the gap between achieved and theoretically optimal (DFS-order) hit rates is small. This is critical evidence that the scheduling algorithm is effective: it is not just that the workload has sharing opportunities, but that SGLang's scheduler successfully exploits them near the theoretical limit.

#### Results on Larger Models with Tensor Parallelism

**Headline result:** The speedup patterns observed on Llama-7B generalize to larger models. Figure 7 shows normalized throughput on Mixtral-8x7B (an 8-expert sparse mixture-of-experts model) with tensor parallelism across 8 A10G GPUs. SGLang achieves 1.0 throughput on all benchmarks; vLLM achieves between approximately 0.2 and 0.9 depending on workload. The relative ordering of benchmarks is similar to the Llama-7B results: MMLU and multi-turn chat (short) show the largest advantages; multi-turn chat (long) shows the smallest. Figure 12 (Appendix) shows similar results for Llama-2-70B on 4 A100 GPUs. Guidance and LMQL are omitted from these experiments because "they lack efficient implementations of tensor parallelism."

The significance of these results is that **RadixAttention generalizes across model architectures** (dense Llama vs. sparse MoE Mixtral) **and scales** (7B to 70B parameters). This is non-trivial: a caching optimization that works for a 7B model might not work for a larger model if memory constraints change the eviction dynamics or if the cache hit rate drops due to different batch size requirements. The paper provides evidence that neither issue occurs in practice.

#### Results on Multi-Modal Models

**Headline result:** SGLang provides throughput up to **6× higher** than the Hugging Face Transformers baseline on multi-modal benchmarks (Table 2). On LLaVA-v1.5-7B (image) running on llava-bench-in-the-wild, SGLang achieves **1.15 images/second** vs. the original implementation's **0.18 images/second** (a 6.4× improvement). On LLaVA-NeXT-34B (video) running on ActivityNet, SGLang achieves **0.10 frames/second** vs. **0.02 frames/second** (a 5× improvement).

The mechanism for multi-modal sharing differs slightly from text-only sharing. For RadixAttention, the system "compute[s] the hash of the input images and use[s] it as the key in the radix tree, allowing us to reuse the KV cache of the image tokens from the same image" (Section 6.2). On llava-bench-in-the-wild, "there are multiple questions about the same image," and SGLang reuses the KV cache in this case—multiple questions about a shared image benefit from prefix sharing of the image encoding tokens.

#### Production Deployment Results

**Headline result:** In the Chatbot Arena production deployment serving open-weight models with a single SGLang worker per model, the observed RadixAttention cache hit rate over one month was **52.4% for LLaVA-NeXT-34B** and **74.1% for Vicuna-33B** (Section 6.2). Cache hits originated from "common system messages, frequently reused example images, and multi-turn chat histories." This reduced first-token latency by an average of **1.7× for Vicuna-33B**.

This is a crucial result because it demonstrates that the cache hit rates observed in controlled benchmarks (50–99%) are not artifacts of synthetic workloads—they manifest in real-world production traffic where request patterns are less predictable. The 52.4% for LLaVA-NeXT-34B (a vision-language model) vs. 74.1% for Vicuna-33B (a text-only model) suggests that image-based queries have less prefix sharing than text-based conversations, which aligns with the intuition that multi-turn chat histories (text) are more repetitive than diverse image uploads. The 1.7× latency reduction translates the abstract cache hit rate metric into a user-facing benefit.

#### API Speculative Execution Results

**Headline result:** On a Wikipedia page extraction task using GPT-3.5, API speculative execution reduces input token costs by "about threefold" (Section 6.2). The task extracts three fields from a Wikipedia page. With naive multi-call execution, each of the three `gen` calls sends the context as part of its input prompt, incurring the input token cost three times. API speculative execution has the first call continue past its stop condition to generate all three fields in one go (if speculation succeeds), incurring the input cost only once.

The paper does not report accuracy numbers for the speculation (i.e., what fraction of the time the model correctly generates all three fields in the speculation vs. requiring fallback to individual calls), nor does it report latency improvements. The cost reduction figure alone is reported. This makes the result suggestive rather than definitive—the practical benefit depends on speculation accuracy, which is not quantified.

### Ablation Studies and Robustness Checks

**Cache hit rate vs. performance (Figure 8a, 8b):** The paper studies the relationship between cache hit rate and performance metrics on the tree-of-thought benchmark by "partially disabling matched tokens at runtime"—essentially simulating varying cache hit rates. Figure 8a shows that as cache hit rate increases from approximately 10% to 99%, batch size increases from roughly 20 to 45, and throughput increases from approximately 20 to 40 tokens/second. Figure 8b shows that first-token latency drops from approximately 400 seconds to roughly 200 seconds, and total latency drops from approximately 1,200 seconds to roughly 300 seconds. The relationship between cache hit rate and all four metrics is monotonic but non-linear—the steepest gains occur at the low end of the cache hit rate range (10% to 50%), with diminishing returns beyond 60–70%.

**RadixAttention component ablation (Figure 8c):** The paper tests six degraded configurations against the full optimization baseline ("Full Optimization" at normalized throughput 1.0) across four benchmarks (LLM Judge, Tree of Thought, MMLU, Multi-Turn Chat (short)):

- **No Cache:** Disabling the KV cache entirely (i.e., recomputing KV cache for every request). Throughput drops to approximately 0.1–0.3 across benchmarks. This quantifies the baseline cost of no reuse: roughly a 3–10× throughput penalty.

- **No Tree Structure:** Using a "simple table-based cache instead of a tree-structured cache." Throughput is approximately 0.4–0.7. The tree structure matters because a flat table cannot efficiently represent partial prefix matches—when a new request shares only part of an existing prefix, a table either duplicates storage or fails to recognize the match, while the radix tree splits edges (as in Figure 3, step 4).

- **FCFS Schedule:** Using first-come, first-served scheduling instead of cache-aware scheduling. Throughput is approximately 0.6–0.85. The penalty is larger on benchmarks with more diverse request patterns (where FCFS causes cache thrashing) and smaller on benchmarks where most requests naturally share prefixes (e.g., MMLU, where all questions share few-shot examples regardless of scheduling order).

- **Random Schedule:** Using random request ordering. Throughput is approximately 0.6–0.8, similar to or slightly worse than FCFS. This confirms that scheduling order matters but that random is not substantially worse than FCFS—both are suboptimal, but the gap between random and cache-aware is meaningful.

- **No Frontend Parallelism:** Disabling the interpreter's ability to execute `fork` branches in parallel. Throughput drops to approximately 0.5–0.7 on Tree of Thought (which uses heavy parallelism) and 0.8–0.9 on LLM Judge and MMLU (where parallelism is less central to the workload).

- **No Frontend Hint:** Disabling the interpreter's hint mechanism that sends fork prefixes to the runtime before fork-specific continuations. Throughput drops to approximately 0.7–0.85. This is a non-obvious but important ablation: it shows that the frontend-runtime co-design (the hint mechanism) provides measurable throughput benefits beyond what the runtime could achieve with prefix matching alone. Without hints, the runtime must infer prefix sharing from raw prompt sequences, which is less reliable and can cause race conditions in tree insertion.

The key finding is that **all components are necessary for best performance**—no single ablation accounts for the full gap between "Full Optimization" and "No Cache." This suggests that RadixAttention's gains come from the combination of tree structure, caching policy, scheduling, frontend hints, and parallelism, not from any single component.

**Overhead of RadixAttention (no-cache-reuse benchmark):** On the ShareGPT dataset, which has "no KV cache reuse opportunities," running 100 requests takes 74.3 seconds. The time spent managing RadixAttention data structures (tree operations, matching, LRU tracking) is **0.2 seconds, less than 0.3%** overhead. This is a crucial robustness check: RadixAttention imposes negligible cost when there is nothing to share. This justifies the paper's claim that "we can turn on RadixAttention by default" (Section 6.3)—there is no need to disable it for workloads without sharing opportunities because the overhead is minimal. The low overhead is attributed to the linear complexity of tree operations and the fact that the tree structure is stored on the CPU.

**Compressed finite state machine ablation:** On the JSON decoding benchmark, the compressed FSM increases throughput by **1.6×** compared to token-by-token constrained decoding. Additionally, preprocessing the FSM (compressing edges) and reusing it across a batch of requests is essential: "redoing the preprocessing for each request makes the throughput 2.4× lower" (Section 6.3). This quantifies a practical engineering consideration—the compressed FSM must be constructed once per regex constraint and shared across all requests using that constraint, not rebuilt per-request. The 2.4× penalty for per-request preprocessing is larger than the 1.6× gain from compression, meaning that without reuse, the compressed FSM would be a net negative. This is because building and compressing the FSM involves analyzing the regex, constructing the state machine graph, identifying singular edges, and recursively merging them—operations whose cost is amortized over many decoding requests.

**Compiler mode: GPT-4 code movement optimization (Appendix D.2):** On 15 test prompt templates (with 5 used as few-shot examples), GPT-4 successfully reorders graph nodes to increase shareable prefix length in **12 out of 15 cases**, achieving an average increase of **60 tokens in shareable prefix length**. The 3 failures occurred when "GPT-4 incorrectly understood the semantic dependencies between prompt components," sometimes "put[ting] all constants upfront even when such ordering changes the original semantics" (Appendix D.2). The paper acknowledges this optimization is "too aggressive and puts all constants upfront even when such ordering changes the original semantics" and notes that "more work is needed to make these kinds of optimizations reliable." This is presented as an exploratory case study, not a production-ready optimization. The key insight is that GPT-4 can perform a form of compiler optimization (reordering for better prefix sharing) that is intractable for traditional program analysis because the dependencies are semantic (natural language meaning) rather than syntactic (data flow).

### Critical Assessment

#### Claim 1: SGLang achieves up to 6.4× higher throughput compared to state-of-the-art inference systems on various workloads.

**What was tested:** The throughput comparison in Figure 5 covers twelve diverse workloads on Llama-7B with a single A10G GPU. The 6.4× figure appears to correspond to the MMLU benchmark (where vLLM achieves approximately 0.15 of SGLang's throughput, implying SGLang is roughly 6.7× faster). Similar large gains appear on ReAct Agents, Generative Agents, Tree of Thought, Skeleton of Thought, JSON Decoding, multi-turn chat (short), and DSPy RAG Pipeline—all showing vLLM at 0.15–0.30 of SGLang's throughput (3.3–6.7× improvement).

**What was not tested:** The comparison is against specific versions of baselines (vLLM v0.2.5, Guidance v0.1.8, LMQL v0.7.3) that the paper explicitly notes are missing features that would close the gap. The vLLM comparison is against a version "before RadixAttention was partially integrated" (footnote, Section 6.1)—the current vLLM includes some of the optimizations SGLang introduces. Guidance and LMQL are tested with backends (llama.cpp, HF Transformers) that the paper acknowledges are not optimized for throughput. A fairer comparison might test SGLang against the latest versions of these systems, or against vLLM with its experimental prefix caching enabled, or against Guidance with a vLLM backend. The paper does not run these comparisons.

**Conditional nature of the claim:** The 6.4× figure is the maximum, not the typical. On multi-turn chat (long), SGLang and vLLM are nearly identical (vLLM at ~0.95 of SGLang's throughput). On HellaSwag, vLLM is at ~0.4 (2.5× improvement). The paper is transparent about this variation but the headline "up to 6.4×" should be understood as workload-dependent, with the largest gains on workloads with heavy prefix sharing (few-shot, multi-turn, tree-structured search) and minimal gains on workloads dominated by unique, long-form generation.

**Missing metrics:** The paper reports only normalized throughput and latency in bar charts. Absolute throughput numbers (programs per second, tokens per second) are not reported, making it impossible to assess whether the absolute performance is practically useful (e.g., can it serve real-time requests?) or to compare against systems not in the normalized bar charts. Memory usage is not reported, so the tradeoff between cache memory consumption and throughput improvement is invisible. The paper mentions that "GPU memory is quickly filled by the KV cache" as motivation for LRU eviction but never quantifies memory utilization under different cache configurations.

#### Claim 2: RadixAttention enables automatic KV cache reuse across multiple generation calls.

**What was tested:** The cache hit rate measurements in Figure 13 show achieved hit rates of 50–99% across benchmarks, with cache-aware scheduling reaching 96% of the optimal hit rate on average. The ablation study (Figure 8c) demonstrates that removing the cache, removing the tree structure, or using suboptimal scheduling all degrade performance. The production deployment data (52.4% and 74.1% hit rates) show the reuse persists in real-world traffic.

**What was demonstrated vs. what was claimed:** "Automatic" is the key word here. The paper claims RadixAttention requires no manual configuration—no specifying which prefixes to share, no per-workload tuning. The experiments support this: the same system, without per-benchmark configuration, achieves high cache hit rates across twelve diverse workloads. The radix tree's dynamic splitting and LRU eviction handle the different sharing patterns automatically. This is genuinely demonstrated, not just asserted. However, the "automatic" claim has a boundary: it applies to exact prefix matches. Semantic or fuzzy matching (e.g., "You are a helpful assistant" vs. "You are a helpful AI assistant") would not be automatically handled. The paper acknowledges this as future work (Section 8).

**Missing ablation:** The paper does not compare against a system with manual prefix sharing configuration. For example, on MMLU, one could manually configure vLLM to share the few-shot examples across requests. How much of SGLang's advantage comes from "automatic" vs. simply "having sharing at all"? This ablation would isolate the value of automation from the value of sharing. Without it, we cannot tell whether SGLang's advantage on MMLU is because vLLM cannot share prefixes at all (in the tested version) or because manual configuration is possible but not used.

#### Claim 3: The compressed finite state machine accelerates constrained decoding by decoding multiple tokens at once.

**What was tested:** The 1.6× throughput improvement on the JSON decoding benchmark is a clean, direct test. The ablation showing that per-request FSM preprocessing reduces throughput by 2.4× further validates that the optimization (compression + reuse) is necessary, not just the idea of FSM-guided decoding.

**What was not tested:** The paper does not evaluate the compressed FSM on a range of regex complexities. The JSON schema in Figure 2 is relatively simple (fixed keys, constrained value types). Would the 1.6× improvement hold for more complex schemas with nested objects, arrays, optional fields, or enumeration alternatives? The probability distortion issue (Appendix B.3) acknowledges that compressed FSMs can misrepresent token probabilities when alternatives have different lengths—does this cause accuracy degradation in practice? The paper does not measure output correctness or schema compliance rates, only throughput. A schema that is decoded faster but produces invalid JSON more often would not be a net win. This is a significant omission for a system that emphasizes "structured outputs" as a core use case.

#### Claim 4: Cache-aware scheduling with longest-shared-prefix-first ordering achieves near-optimal cache hit rates.

**What was tested:** Figure 13 shows achieved hit rates approaching 96% of optimal on average, with the optimal computed from Theorem 3.1. Figure 8c shows that FCFS and random scheduling both underperform cache-aware scheduling.

**What was not tested:** Theorem 3.1 assumes offline scheduling (all requests known in advance) with a cache size at least equal to the maximum request length. The online case is approximated but not guaranteed. The paper does not measure how close the online scheduler comes to DFS-order optimality under dynamic arrival patterns, nor does it study the degradation as cache size decreases below the maximum request length. The 96% figure is an average across benchmarks, but the distribution matters: are there benchmarks where the online scheduler falls significantly below optimal? The paper does not address starvation—longest-shared-prefix-first can indefinitely delay requests that share no prefix with the current cache contents. The paper acknowledges this as future work (Section 8) but does not measure how often starvation occurs in practice.

#### Claim 5: SGLang's programming model simplifies LM program development (2.1× fewer lines of code).

**What was tested:** The paper provides one example (the multi-dimensional essay judge in Figure 2) and claims the equivalent OpenAI API-like program would take 2.1× as many lines of code.

**What was not tested:** This is a single anecdote, not a systematic evaluation. There is no study comparing SGLang to Guidance, LMQL, or raw API code across multiple tasks with multiple programmers. Code length is a weak proxy for programming simplicity—a program could be shorter but harder to understand, or longer but more explicit and debuggable. The paper does not report programmer time, error rates, or qualitative feedback from users. The 2.1× figure is suggestive but not demonstrated as a general property of the language.

#### Overall Assessment

The experimental evaluation strongly supports the paper's central thesis—that exploiting multi-call structure in LM programs can yield substantial throughput improvements. The evidence is most robust for the RadixAttention optimization, where the ablation study, cache hit rate measurements across diverse workloads, production deployment data, and theoretical grounding (Theorem 3.1) form a coherent and convincing case. The compressed FSM optimization is demonstrated on a single benchmark with a 1.6× improvement, which is solid but somewhat narrow. The API speculative execution is demonstrated anecdotally. The programming model's simplicity is asserted rather than systematically evaluated.

The most significant gap is the comparison against updated baselines—the paper benchmarks against versions of vLLM, Guidance, and LMQL that predate SGLang's innovations and lack features that would close the performance gap. This is understandable for a systems paper (the baseline systems improved over time), but it means the "up to 6.4×" figure should be interpreted as the advantage over late-2023 systems, not over current (mid-2024) alternatives. The production deployment data (52.4% and 74.1% cache hit rates) partially addresses this concern by demonstrating real-world impact independent of baseline comparisons.

A second gap is the absence of accuracy or quality metrics for the structured output optimization. The compressed FSM improves throughput by 1.6×, but if it introduces output validity errors (due to the probability distortion issue), the throughput gain is misleading. Similarly, the paper does not measure whether RadixAttention's cache evictions ever cause correctness issues (e.g., evicting and later recomputing a prefix produces different KV cache due to floating-point nondeterminism—though this is unlikely with deterministic attention implementations, it is not addressed).

A third gap is the single-GPU focus of most experiments. While tensor parallelism results are reported for larger models (Figures 7, 12) and data parallelism is discussed in Appendix A.4, the paper does not systematically evaluate SGLang at the scale where LLM serving systems are typically deployed (dozens of GPUs, thousands of concurrent requests). The linear scaling claim for data parallelism on MMLU with four workers is encouraging but minimal.

## 6. Limitations and Trade-offs

### 1. The $6.4\times$ Throughput Claim Benchmarks Against Outdated Baselines Missing Current Optimizations

**The assumption or constraint.** The paper's headline throughput comparison (Figure 5) benchmarks SGLang against vLLM v0.2.5, Guidance v0.1.8, and LMQL v0.7.3. The authors explicitly acknowledge that these are not the latest versions and that key optimizations have since been added to the baselines. Regarding vLLM, a footnote in Section 6.1 states:

> "RadixAttention has been partially integrated as an optional experimental feature into the latest version of vLLM; therefore, we used an earlier version for comparison."

For Guidance and LMQL, the backends tested (llama.cpp and Hugging Face Transformers, respectively) are acknowledged to have fundamental performance limitations—Guidance "lacks batching and parallelism support" and LMQL uses "an unoptimized backend" (Section 6.2). The comparison is therefore not SGLang vs. state-of-the-art at time of publication, but SGLang vs. intentionally weaker configurations of prior systems.

**The consequence.** A practitioner deciding whether to deploy SGLang today cannot use the $6.4\times$ figure to estimate advantage over current vLLM, which already incorporates some prefix-sharing optimizations. The true throughput advantage over contemporary systems is unknown from this paper's evaluation. On workloads where SGLang's advantage is small (e.g., multi-turn chat (long), where vLLM achieves ~0.95 of SGLang's throughput), the gap may have closed entirely with vLLM's updated prefix caching. The $6.4\times$ headline number may overstate the practical benefit for users who have access to current versions of the baseline systems.

**What evidence exists in the paper.** The normalized throughput bars in Figure 5 show vLLM at approximately 0.15 of SGLang's throughput on MMLU (the benchmark where the $6.4\times$ figure likely originates), 0.15 on multi-turn chat (short), and 0.25 on DSPy RAG. However, the footnote explicitly disclaims these numbers as being against a version without prefix caching. Figure 7 shows similar patterns on Mixtral-8x7B. No comparison against vLLM with prefix caching enabled is provided. No comparison against any backend (e.g., vLLM) for Guidance or LMQL is tested.

**Mitigation status.** The paper partially mitigates this concern through the production deployment data (Section 6.2): the 52.4% and 74.1% cache hit rates observed in Chatbot Arena over one month demonstrate real-world benefit independent of baseline comparisons. However, this production data does not provide a throughput comparison against alternative systems—it only shows that SGLang's caching works in production, not how much better it works than alternatives. The authors do not frame this as a limitation to be addressed; rather, the baseline choice is treated as a necessary consequence of the field's rapid evolution. No plans to benchmark against updated baselines are mentioned in Section 8.

---

### 2. KV Cache Eviction Under Memory Pressure Creates a Throughput-Correctness Tradeoff That Is Not Evaluated

**The assumption or constraint.** RadixAttention uses an LRU eviction policy to manage GPU memory, dynamically sharing memory between cached KV tensors and active requests. The paper states in Section 3:

> "We do not preallocate a fixed-size memory pool as a cache. Instead, we let the cached tokens and the currently running requests share the same memory pool. Therefore, the system dynamically allocates memory for cache and running requests. When enough waiting requests run, the system will evict all cached tokens in favor of a larger batch size."

This design means that under high load (many waiting requests), the cache is sacrificed for batch throughput. The reference counting mechanism (Section 3) prevents eviction of nodes actively used by the current batch, but **there is no mechanism to prevent eviction of a prefix that will be needed by a request that is currently in the queue but has not yet entered the batch**. A request that shares a long prefix with cached data may have that prefix evicted while it waits, forcing recomputation when it eventually runs.

**The consequence.** This creates an uncharacterized tradeoff between throughput and cache effectiveness. The paper's cache hit rate measurements (50–99% across benchmarks, Figure 13) were obtained under conditions where the waiting queue dynamics did not cause destructive eviction. Under real production load with many concurrent users, cached prefixes may be evicted before requests that could benefit from them are scheduled, reducing the effective cache hit rate below what the benchmarks report. The paper's cache-aware scheduling algorithm sorts waiting requests by prefix match length (Algorithm 1), which partially addresses this by prioritizing requests with cached prefixes. However, the algorithm selects requests **up to available memory**—if the batch is memory-constrained and many waiting requests share a prefix, only the first few will benefit; the rest may see their shared prefix evicted by the batch's own memory consumption before they run. The paper does not measure or bound this degradation.

**What evidence exists in the paper.** The only evidence is indirect. The production deployment data (Section 6.2) reports 52.4% and 74.1% cache hit rates, but these are "observed over one month" with no information about load levels, queue depths, or eviction rates during that period. The benchmarks (Figures 5–7) run "a sufficiently large batch" but do not vary batch size or memory pressure to measure cache hit rate degradation. The "No Cache" ablation (Figure 8c) shows the extreme end of this spectrum (zero caching), but the intermediate regime—where caching is active but evictions occur due to memory pressure—is not explored. The paper's overhead measurement on ShareGPT (0.3% overhead, Section 6.3) confirms RadixAttention is cheap in the no-sharing case but does not measure overhead or effectiveness in the high-contention, high-eviction case.

**Mitigation status.** Not addressed. The paper acknowledges that the LRU eviction policy exists and describes its mechanics, but does not evaluate its behavior under memory pressure or propose mechanisms to bound eviction-related cache hit rate degradation. Future work on "extending RadixAttention to operate across multiple levels of the memory hierarchy (e.g., DRAM, Disk)" (Section 8) could mitigate this by allowing evicted KV tensors to be offloaded rather than discarded, but this is only a suggestion. The starvation problem in cache-aware scheduling—where requests with no cached prefix are indefinitely delayed—is explicitly acknowledged but similarly deferred to future work.

---

### 3. The Compressed Finite State Machine Can Distort Token Probability Distributions, Potentially Producing Invalid or Lower-Quality Outputs

**The assumption or constraint.** The compressed FSM optimization (Section 4) works by merging adjacent singular-transition edges in the FSM and decoding multiple tokens at once on deterministic paths. This is safe when the path truly has no alternatives—the deterministic literal string `{"summary": "` in Figure 2. However, when the constraint includes **enumeration alternatives of different token lengths**, the compressed FSM can aggregate paths in ways that do not accurately reflect the LLM's token-level probability distribution. The paper explicitly acknowledges this in Appendix B.3:

> "The challenge caused by the gap between strings and tokens also brings the problem of skewed probability distribution... This occurs because the LLM doesn't recognize the specific range of choices, leading to inappropriate token sequences. Computing accurate probabilities for each choice requires summing the probabilities of all token sequences that result in each choice, which complicates decoding and adds overhead."

The toy example given is a regex like `Excellent|Above Average|Fair|Below Average`. If these alternatives are aggregated on a compressed edge, the model's per-token probabilities may not map correctly to the probability of selecting each full alternative.

**The consequence.** The compressed FSM can produce outputs that, while technically matching the regex, do not reflect the model's true probability distribution over the valid outputs. In the worst case, this could mean selecting an output that the model considers unlikely or even semantically incorrect. The practical consequence depends on the application: for JSON key names (fixed strings like `"summary"`), the distortion is irrelevant because there is only one valid path. For constrained generation where the regex expresses genuine semantic alternatives (e.g., grade choices `[ABCD][+-]?` or Likert scale responses), the distortion could affect output quality. The paper does not measure this—the JSON decoding benchmark (Figure 5) reports only throughput, not output validity or quality. A 1.6× throughput improvement (Section 6.3) that produces incorrect JSON or semantically wrong selections would not be a net gain.

**What evidence exists in the paper.** None. The compressed FSM's effectiveness is measured solely through throughput (1.6× improvement on JSON decoding). There is no measurement of output correctness, schema compliance rate, or probability fidelity. The probability distortion issue is acknowledged in Appendix B.3 as a conceptual limitation, not an experimentally characterized one. The workaround suggested—"include the choices or the regex directly in the prefill prompt, guiding the LLM to be aware of its choices"—is described but not tested.

**Mitigation status.** Partially acknowledged but not resolved. The paper states: "this approach doesn't solve the underlying issue of distorted probabilities, highlighting the need for further research to improve the compressed FSM's accuracy" (Appendix B.3). Section 8 does not list this as a future direction, suggesting the authors consider it a known limitation of the technique rather than a priority for near-term improvement. A practitioner using SGLang's `regex` argument on constraints with meaningful alternatives should be aware that the compressed FSM's probability distortion is an uncharacterized risk.

---

### 4. Effortless Programming Simplicity Is Asserted with a Single Anecdote Rather Than Systematically Demonstrated

**The assumption or constraint.** One of SGLang's two core value propositions is that its frontend language simplifies LM program development. The paper claims in Section 1 that "SGLang greatly simplifies this program, as an equivalent program using an OpenAI API-like interface would take 2.1× as many lines of code due to manual string manipulation and parallelism control." This claim is based on a single program—the multi-dimensional essay judge in Figure 2.

**The consequence.** A practitioner evaluating whether to adopt SGLang for its programming model (as opposed to its runtime performance, which is separately demonstrated) has insufficient evidence to assess the claimed simplicity benefit. Code length is a weak proxy for programming simplicity—a shorter program might be more cryptic (using dense primitives whose semantics must be learned) or harder to debug (asynchronous execution with blocking on fetch obfuscates control flow). There is no evidence about: how long it takes programmers to learn SGLang, how frequently they make errors, how debuggable the programs are when something goes wrong, or how the programming experience compares to alternatives like Guidance (which also provides Python-embedded primitives) or DSPy (which provides higher-level abstractions). The $2.1\times$ figure for a single program tells the reader nothing about the distribution of code reduction across different program types or complexity levels.

**What evidence exists in the paper.** Exactly one program comparison (Figure 2), with no user study, no multi-task comparison, and no comparison against Guidance's or LMQL's programming models for the same task. Table 1 compares language primitives across systems at the feature level, but this is a feature checklist, not a usability evaluation. The claim that the language is "flexible and composable" (Section 2) is supported by the running example but not systematically tested.

**Mitigation status.** Not addressed as a limitation. The paper treats code reduction as self-evidently valuable and does not acknowledge the need for usability evaluation. This is understandable given the paper's primary contribution is systems optimization (where the evaluation is appropriately quantitative), but it means the programming model's benefits should be treated as a design contribution demonstrated by example, not an empirically validated claim. The integration with DSPy (Section 6) demonstrates SGLang's utility as an accelerator for an existing programming model, but this speaks to runtime efficiency, not programming simplicity.

---

### 5. The Difficulty Estimation Cost for Optimal Scheduling Is Unaccounted for in the Online Case

**The assumption or constraint.** The cache-aware scheduling algorithm in Algorithm 1 achieves near-optimal cache hit rates (96% of optimal on average, Figure 13) by sorting waiting requests by matched prefix length. Theorem 3.1 proves optimality for offline batches. However, in the online serving case, the scheduling decision—which requests to batch together and in what order—depends on knowing each request's prefix match length **before running the batch**. This information is obtained by matching each waiting request against the radix tree (Algorithm 1, steps 1–3). Matching itself has a cost: the runtime must search the radix tree for the longest matching prefix, which involves tree traversal proportional to the depth of the match. The paper states the tree is stored on the CPU with "negligible maintenance overhead" (Section 3) and measures 0.3% overhead for tree operations in the ShareGPT no-sharing case (Section 6.3), but this overhead is measured on a workload **without sharing opportunities**. On workloads with deep radix trees (many sharing patterns, long prefixes), the matching cost could be higher.

**The consequence.** For workloads with very deep sharing trees or extremely high request rates, the cost of prefix matching for cache-aware scheduling could become non-negligible, particularly since matching must happen for all waiting requests before batch selection (steps 2–4 of Algorithm 1). If matching cost scales with tree depth and queue length, there exists a throughput regime where the scheduling overhead erodes the gains from improved cache hit rate. The paper does not identify this regime or bound it. A practitioner deploying SGLang at very high request rates might need to measure matching overhead in their specific workload, but the paper provides no guidance for doing so.

**What evidence exists in the paper.** The 0.3% overhead on ShareGPT (Section 6.3) provides a lower bound—on a workload with no sharing, the tree operations are cheap. But this is the easiest case for the radix tree: minimal tree depth, no splitting, few nodes. The overhead on MMLU (deep sharing: five-shot examples shared across all questions, producing a tree with many requests branching from the same few-shot prefix) is not separately reported. The production deployment data (Section 6.2) reports cache hit rates but not scheduling overhead. The paper does not provide a scaling analysis of matching cost as a function of tree depth, number of concurrent requests, or request arrival rate.

**Mitigation status.** Not addressed. The paper treats the radix tree's CPU-side operations as inherently negligible and does not suggest that matching overhead could become a bottleneck. Section 8's future directions do not mention optimizing matching efficiency. The 0.3% figure is presented as sufficient evidence that RadixAttention can be "turned on by default" (Section 6.3), but this conclusion is based on a single, sharing-free workload and may not generalize to workloads that actually benefit from RadixAttention (which are, by construction, workloads with deep sharing trees).

---

### 6. The System Is Evaluated on a Single Family of Open-Weight Models and a Narrow Set of Multi-Modal Architectures

**The assumption or constraint.** All throughput and latency experiments (Figures 5–7, 12) use Llama-2 (7B, 70B) and Mixtral-8x7B models. The multi-modal experiments (Table 2) use LLaVA-v1.5-7B and LLaVA-NeXT-34B. The API experiment uses GPT-3.5. This covers dense transformer models, mixture-of-experts models, and two vision-language architectures. The paper states in Section 6.1 that Llama-2 models are "representative of the capabilities of many contemporary LLMs," and the generalization to Mixtral and LLaVA is intended to demonstrate broader applicability.

**The consequence.** Several aspects of SGLang's optimizations could be model-dependent in ways the evaluation does not expose. RadixAttention's effectiveness depends on the KV cache memory footprint per token, which varies with model dimension (hidden size, number of layers, number of attention heads). Larger models (beyond 70B) or models with different architectures (e.g., non-transformer architectures, models with different attention mechanisms like grouped-query attention or multi-query attention) might exhibit different memory pressure dynamics, affecting eviction rates and cache hit rates. The compressed FSM optimization depends on tokenizer behavior—how the tokenizer maps between characters and tokens, which varies substantially across model families (e.g., GPT-series tokenizers vs. Llama tokenizers vs. PaLM tokenizers). A regex constraint that compresses well under one tokenizer might produce different compression ratios or probability distortion effects under another.

The paper does not test on any encoder-decoder architecture (T5, BART), any models with multi-query attention (where KV cache size is reduced by sharing key-value heads), or any models from families other than Llama/Mixtral/LLaVA. The production deployment data (Section 6.2) includes Vicuna-33B (a Llama fine-tune) and LLaVA-NeXT-34B, which are within the Llama family.

**What evidence exists in the paper.** The evaluation covers three open-weight model families (Llama-2, Mixtral, LLaVA) and one API model (GPT-3.5), spanning parameter counts from 7B to 70B and architectures from dense to MoE to vision-language. The trends are consistent across these models (Figures 5, 7, 12 show similar patterns). However, all open-weight models tested are decoder-only transformer architectures from the Llama lineage. The paper does not test on models with substantially different tokenizers (GPT-2/3/4 tokenizers differ from Llama's SentencePiece tokenizer), different attention patterns (multi-query, grouped-query), or different architectural paradigms (encoder-decoder, state-space models).

**Mitigation status.** Acknowledged implicitly by scope but not addressed as a limitation. The paper does not claim to have tested on all model architectures and does not suggest that results would differ on other architectures. Section 8's future directions do not mention evaluation on additional model families. A practitioner using SGLang with a non-Llama model should be aware that the throughput improvements, cache hit rates, and compressed FSM compression ratios have not been validated outside the Llama model family and its derivatives.

## 7. Implications and Future Directions

- **How this work changes the landscape**
  - It reframes LLM serving not just as isolated completion requests but as execution of **structured programs**, where:
    - shared prompt prefixes are a cacheable resource, and
    - constrained decoding is a compiler/runtime problem rather than purely a prompting trick (Figure 1; Sections 3–4).
  - It demonstrates that co-designing a programming interface with a runtime can unlock optimizations that are difficult to retrofit into generic API servers (Section 3 frontend hints; Figure 8(c) co-design ablations).

- **Follow-up research directions enabled/suggested (explicitly listed in Section 8 and implied by limitations)**
  - **Fairness-aware cache scheduling:** resolve starvation while maintaining cache locality (Section 3; Section 8 mentions fixing starvation; also references fair scheduling work [42] in Section 3).
  - **Memory-hierarchy-aware caching:** extend RadixAttention beyond GPU memory to DRAM/disk tiers (Section 8).
  - **Semantic/fuzzy prefix matching:** reuse beyond exact token prefixes (Section 8).
  - **More reliable compiler optimizations:** improve static scheduling/memory planning; handle data-dependent control flow in compiler mode (Section 8; Appendix D.1).
  - **Constrained decoding correctness:** address distorted probability under compression (Appendix B.3).

- **Practical applications / downstream use cases (supported by evaluated workloads)**
  - **Agents and tool-using workflows:** `ReAct` and generative agents traces run faster due to reuse of templates and prior calls (Section 6.2).
  - **Reasoning/search prompting:** `Tree-of-thought` and `Skeleton-of-thought` benefit from parallel subcalls plus reuse (Section 6.2).
  - **Evaluation pipelines:** LLM judges with branch-solve-merge map naturally to `fork` and benefit from parallelism + reuse (Figure 2; Section 6.2).
  - **Structured outputs at scale:** JSON/regex constrained decoding benefits from compressed FSM throughput improvements (Section 6.3; Figure 4).
  - **RAG systems:** shared context examples enable reuse in DSPy pipelines (Section 6.2).
  - **Multimodal Q/A over the same media:** hashing images to reuse image-token KV cache (Section 6.2; Table 2).

- **Evidence of deployment relevance**
  - The system is deployed in `Chatbot Arena`, reporting observed cache hit rates (52.4% for `LLaVA-Next-34B`, 74.1% for `Vicuna-33B`) and a **1.7× first-token latency reduction** for `Vicuna-33B` (Section 6.2, “Production deployment”). This supports the claim that prefix reuse occurs in real traffic via common system messages, reused images, and chat histories (Section 6.2).

> Section 6.2 “Production deployment” provides these concrete observed cache hit rates and the reported first-token latency reduction, indicating RadixAttention’s reuse opportunities persist outside synthetic benchmarks.
