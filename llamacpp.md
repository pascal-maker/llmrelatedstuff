---
type: concept
status: active
area: AI research
tags:
  - llama-cpp
  - llm
  - inference
  - ggml
  - metal
---

# Llama.cpp Easily Explained

## The big picture

`llama.cpp` takes text, lets a model predict one new token, adds that token to the text, and repeats.

```text
your prompt
-> tokens
-> model calculations
-> scores for possible next tokens
-> choose one token
-> repeat
```

On a Mac, the complete path is roughly:

```text
HTTP request
-> llama-server
-> tokenizer
-> llama_decode()
-> Qwen describes its neural network
-> GGML builds a computation graph
-> backend scheduler
-> Metal
-> Apple GPU
-> logits
-> sampler
-> next token
```

## The three main responsibilities

It helps to separate three parts:

- `llama_context` manages the current inference run.
- `llama_model` knows how this model's neural network is constructed.
- GGML executes the required calculations on available hardware.

Inside `llama_context::decode()`, the context asks the model to build a graph:

```cpp
auto * gf = model.build_graph(gparams);
```

This graph is later sent to the backend scheduler:

```cpp
ggml_backend_sched_graph_compute_async(sched.get(), gf);
```

The first call describes the work. The second begins executing that work.

## Why each model builds its own graph

Qwen, Llama, Mamba, and DeepSeek do not all use exactly the same architecture. Therefore, `llama_model::build_graph()` delegates to the model-specific implementation:

```text
llama_context::decode()
-> llama_model::build_graph()
-> build_arch_graph()
```

For Qwen, this architecture-specific code lives in `src/models/qwen.cpp`.

The easiest mental model is:

```text
Qwen describes what calculations are needed.
GGML turns that description into executable work.
Metal knows how to run that work on the Apple GPU.
```

Qwen's C++ code is not manually calculating every number immediately. Calls such as `ggml_add()` usually add operations to a graph that will be executed later.

## Step 1: Turn token IDs into vectors

The tokenizer first changes text into integer IDs:

```text
"Belgium has a capital"
-> [18342, 521, 264, 6721]
```

The model cannot do useful neural-network calculations directly on those IDs. The embedding layer looks up a vector for each token:

```text
token ID
-> embedding lookup
-> vector of numbers
```

In the Qwen graph, this begins with something like:

```cpp
inpL = build_inp_embd(model.tok_embd);
```

The vectors become the input to the Transformer layers.

## Step 2: Run the Transformer layers

Each Qwen layer performs roughly the same sequence:

```text
input
-> RMSNorm
-> create Q, K, and V
-> apply RoPE to Q and K
-> attention
-> add the original input
-> RMSNorm
-> feed-forward network
-> add the previous representation
```

### Attention

Q, K, and V stand for:

- Q: Query
- K: Key
- V: Value

They let the current token decide which earlier tokens matter and what information to take from them.

RoPE adds position information to Q and K. This helps the model distinguish between tokens appearing earlier and later in the sequence.

### Residual connections

After attention, llama.cpp adds the layer's earlier input back to the result:

```cpp
ggml_add(ctx0, cur, inpSA);
```

Conceptually:

```text
new attention result + original input
```

This is called a residual connection. It helps information flow through many layers.

### Feed-forward network

After another normalization, the representation passes through the feed-forward network. Qwen uses large learned matrices such as `ffn_up`, `ffn_gate`, and `ffn_down`, together with a SiLU activation.

Simplified:

```text
attention result
-> normalize
-> expand and transform
-> apply activation/gating
-> reduce again
-> add residual
```

The output of one layer becomes the input of the next layer. This repeats through all Transformer layers.

## Step 3: Produce logits

After the final Transformer layer, the model applies a final normalization and the language-model head.

```text
last hidden representation
-> final RMSNorm
-> language-model head
-> logits
```

Logits are scores for every possible next token. They are not probabilities yet.

For example:

```text
Brussels    12.7
Antwerp      5.1
Belgium      4.2
Paris        2.3
banana      -4.8
```

A higher score means the model currently considers that token more suitable.

The graph is built toward the final logits tensor:

```cpp
res->t_logits = cur;
ggml_build_forward_expand(gf, cur);
```

At this point, llama.cpp has described the computation, but the hardware still needs to execute it.

## Step 4: Let GGML and Metal execute the graph

The GGML backend scheduler decides which backend should execute each graph operation.

```text
GGML graph
-> scheduler chooses a backend
-> backend implements the operation
-> hardware performs the calculation
```

On a Mac, many supported operations can go through Metal to the Apple GPU. Large matrix multiplications are especially important because the model constantly multiplies hidden representations by weight matrices for attention, the feed-forward network, and the final output.

```text
GGML matrix-multiplication node
-> Metal chooses an appropriate implementation
-> GPU command is encoded
-> Apple GPU executes it
```

Not every operation must run on the GPU. llama.cpp can divide work between CPU and GPU depending on model placement, supported operations, buffers, and configuration.

## Unified memory on Apple Silicon

Apple Silicon uses one shared memory pool for the CPU and GPU:

```text
             unified memory
            /              \
          CPU              GPU
```

This makes large local models practical, but everything competes for the same memory:

- macOS and other applications
- model weights
- KV cache
- compute buffers
- temporary tensors
- Metal resources

Therefore, a 17 GB GGUF file needs more than 17 GB of total memory while running.

## Step 5: Reuse earlier work with the KV cache

Text generation is autoregressive: the model produces one token at a time. Every new token must attend to earlier tokens.

Without a cache, the model would repeatedly recreate the K and V values for all earlier tokens. The KV cache stores those values so they can be reused.

```text
earlier token 1 -> K1 and V1
earlier token 2 -> K2 and V2
earlier token 3 -> K3 and V3
```

For the current token:

```text
current Q
-> attends to cached K and V
-> attention result
```

The new token's K and V are then added to the cache for future steps.

llama.cpp stores K and V for every Transformer layer. Its cache code provides operations such as `get_k()`, `get_v()`, `cpy_k()`, and `cpy_v()` to read existing cache state and store new state.

## Prefill versus decode

Suppose a prompt contains 1,000 tokens.

During prefill, llama.cpp processes those prompt tokens and fills the KV cache:

```text
1,000 prompt tokens
-> Transformer
-> K and V for each layer
-> populated KV cache
-> first logits
```

During decode, it generates one new token at a time:

```text
one new token
-> calculate its Q, K, and V
-> use cached K and V
-> produce logits
-> choose a token
-> store new K and V
-> repeat
```

This is why logs report prompt-processing speed and token-generation speed separately.

## Why a long context uses so much memory

The KV cache must hold information for tokens across many layers. A larger context therefore requires more cache capacity.

Going from 32k to 262k context is approximately eight times as many token positions:

```text
262,144 / 32,768 = 8
```

The exact memory use also depends on the architecture, number of layers, K/V dimensions, cache datatype, number of sequences, and attention method. But context length is a major reason memory use grows.

This is why a large model plus a very long context can overwhelm a 32 GB Mac even when the GGUF file itself appears to fit.

## Step 6: Choose one token with the sampler

After the GPU produces logits, the sampler turns all those scores into one token.

llama.cpp can combine several sampler stages:

```text
logits
-> top-k
-> top-p
-> temperature
-> final selection
-> one token
```

### Temperature

Temperature changes how strongly the sampler favors the highest-scoring tokens.

```text
lower temperature
-> sharper distribution
-> more predictable choices

higher temperature
-> flatter distribution
-> more varied choices
```

Applications normally handle temperature zero as a special deterministic or greedy case.

### Top-k

Top-k keeps only a fixed number of the highest-scoring candidates.

```text
150,000 candidates
-> keep the best 40
-> discard the rest
```

### Top-p

Top-p keeps the most likely candidates until their combined probability reaches a chosen threshold.

For `top_p = 0.9`, the sampler keeps a group whose cumulative probability reaches roughly 90%.

Unlike top-k, the number of retained tokens changes depending on how confident the model is.

## Step 7: Feed the chosen token back into the model

The model does not generate a whole sentence in one GPU call. It repeats the inference loop one token at a time:

```text
prompt
-> prefill and populate KV cache
-> logits
-> sample "Brussels"
-> decode "Brussels"
-> update KV cache
-> new logits
-> sample " is"
-> decode " is"
-> update KV cache
-> repeat
```

Generation ends when llama.cpp reaches an end-of-sequence token, a token limit, a stop sequence, or a cancellation.

## The easiest mental model

The entire system can be remembered as four jobs:

```text
MODEL CODE
describes the neural network

GGML
represents the calculations as a graph

BACKENDS SUCH AS METAL
execute those calculations on hardware

SAMPLER
turns the final scores into one next token
```

The KV cache connects the repeated generation steps by remembering useful attention information from earlier tokens.

So the shortest possible summary is:

```text
text
-> token IDs
-> vectors
-> Transformer layers
-> GGML graph
-> Apple GPU
-> logits
-> sampler
-> one token
-> save K and V
-> repeat
```

## Related notes

- [[LLMs]]
- [[AI Chip Architectures - Jacob Peake]]
