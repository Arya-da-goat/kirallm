# Kira — Personal LLM From Scratch

Kira is a **real decoder-only Transformer language model implemented in PyTorch**. It is not an API wrapper and it does not require OpenAI, Gemini, Claude, or a Hugging Face hosted inference service.

The repository contains the neural architecture, tokenizer, training loop, checkpointing, local inference, and a small starter corpus. **No model weights are bundled**, because useful LLM weights are large and a model must be trained on your own machine to become Kira.

## What is actually implemented?

- Byte-level BPE tokenizer with a reversible UTF-8 round trip
- Token embeddings
- Rotary positional embeddings (RoPE)
- RMSNorm
- Pre-norm Transformer blocks
- Causal multi-head self-attention
- Grouped-query attention (GQA)
- Optional query/key normalization
- SwiGLU feed-forward networks
- Residual connections
- Tied token embedding / LM head
- Stable residual initialization
- Next-token cross-entropy objective
- AdamW with decoupled weight decay
- Parameter-grouped optimizer (norms/1-D parameters are not decayed)
- Linear warmup + cosine learning-rate decay
- Gradient accumulation
- Gradient clipping
- BF16/FP16 mixed precision on CUDA
- Train/validation split
- Validation loss + perplexity
- Best/final/periodic checkpoints
- Resume training
- Top-k, top-p, temperature, and repetition-penalty sampling
- Local terminal chat
- Local FastAPI inference server on `127.0.0.1:8765`
- No cloud AI inference dependency

## Important reality check

A Transformer architecture is only the **brain structure**. The model becomes capable through training on a large, high-quality corpus.

The included `data/train.txt` is deliberately tiny so the repository stays small. It is useful for checking that the complete pipeline works, but it is **not enough to produce a ChatGPT-level model**.

To make Kira genuinely capable, you need a large corpus that you are legally allowed to train on and enough compute to optimize the model. More parameters alone do not guarantee better results: data quality, token count, training stability, evaluation, and instruction tuning all matter.

## Recommended development path

### Stage 1 — prove the pipeline

Use the included corpus and a small configuration. Confirm:

```text
corpus → tokenizer → token IDs → Transformer → loss → gradients → checkpoint → generation
```

### Stage 2 — pretraining

Replace `data/train.txt` with a large, clean corpus. Keep validation data separated from training data.

### Stage 3 — instruction tuning

Create conversational examples and train Kira to follow instructions and maintain a consistent assistant style.

### Stage 4 — evaluation

Measure validation loss and create a separate test set that the model never sees during training.

### Stage 5 — scale

Only after the complete pipeline is stable should you increase context length, model width, layers, data, and training steps.

## Windows setup

Use Python 3.11+.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For an NVIDIA GPU, install the appropriate PyTorch build for your CUDA setup using the official PyTorch installation instructions, then install the remaining requirements.

## Train the tokenizer

From the project root:

```powershell
python train_tokenizer.py --text data/train.txt --vocab-size 16000 --out tokenizer/model
```

The tokenizer is trained from the corpus instead of downloading a pretrained tokenizer.

## Train Kira

```powershell
python train.py --config config.yaml
```

Useful options:

```powershell
python train.py --config config.yaml --resume checkpoints/kira-final.pt
```

The trainer prints loss, learning rate, gradient norm, validation loss, and validation perplexity. It saves:

```text
checkpoints/
├── kira-best.pt
├── kira-final.pt
└── kira-step-XXXX.pt
```

## Chat locally

```powershell
python chat.py --checkpoint checkpoints/kira-best.pt
```

There is no external AI call. The checkpoint is loaded into PyTorch on your CPU/GPU.

## Run the local server

```powershell
python serve.py
```

The API listens only on:

```text
http://127.0.0.1:8765
```

Health check:

```text
GET /health
```

Generation:

```text
POST /generate
```

Example JSON:

```json
{
  "prompt": "User: Explain gravity simply.\nKira:",
  "max_new_tokens": 160,
  "temperature": 0.75,
  "top_k": 50,
  "top_p": 0.92,
  "repetition_penalty": 1.05
}
```

## Create instruction data

Kira can consume simple JSONL examples:

```json
{"instruction":"Explain gravity simply.","response":"Gravity is the attraction between objects that have mass."}
{"instruction":"What is Python?","response":"Python is a general-purpose programming language."}
```

Convert them to training text:

```powershell
python tools/make_dataset.py instructions.jsonl data/train.txt
```

You can also use chat-style records:

```json
{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hello! I'm Kira."}]}
```

## Model architecture

```text
Token IDs
   │
   ▼
Token Embedding
   │
   ▼
┌───────────────────────────────┐
│ Transformer Block             │
│                               │
│ RMSNorm                       │
│    ↓                          │
│ GQA Self-Attention + RoPE     │
│    ↓                          │
│ Residual                      │
│    ↓                          │
│ RMSNorm                       │
│    ↓                          │
│ SwiGLU Feed-Forward Network   │
│    ↓                          │
│ Residual                      │
└───────────────────────────────┘
          × N layers
   │
   ▼
RMSNorm
   │
   ▼
Tied LM Head
   │
   ▼
Next-token logits
```

## Scaling profiles

The default config is a serious small-model starting point. Exact memory requirements depend on sequence length, batch size, optimizer states, precision, and activation memory.

### Small / learning

```yaml
max_seq_len: 512
dim: 384
n_layers: 6
n_heads: 6
n_kv_heads: 2
ffn_dim: 1024
```

### Medium

```yaml
max_seq_len: 1024
dim: 768
n_layers: 12
n_heads: 12
n_kv_heads: 4
ffn_dim: 2048
```

### Larger

Only attempt a larger configuration after checking available VRAM/RAM and training throughput. A useful next step is increasing data and training tokens before blindly increasing parameters.

## Self-check

The repository includes a CPU-friendly architectural test:

```powershell
python tests.py
```

It checks tokenizer round-tripping, Transformer output shape, finite loss, and gradient flow.

## Keeping Kira truly yours

This project intentionally separates the **model architecture** from the **data**. You can train the model on a corpus you have permission to use and then instruction-tune it for Kira's personality and behavior.

The final model weights are created by your training run and stored locally in `checkpoints/`.

Do not train on private conversations, personal information, or copyrighted material unless you have the necessary rights or permission.

## Connect your Kira web UI

`integration/kira-client.js` contains two small browser helpers:

```js
import { askKira, chatWithKira } from './kira-client.js';

const answer = await askKira('Explain self-attention simply.');

const reply = await chatWithKira([
  { role: 'user', content: 'Hello Kira' }
]);
```

Start the local server first with `python serve.py`. The server is restricted to localhost for the listening address; CORS is enabled so a local browser UI can call it.

## New data + instruction pipeline

The project now includes a safer corpus workflow and supervised instruction tuning.

### Prepare a corpus

```powershell
python tools/prepare_corpus.py path\to\txt\folder --out data/corpus
python tools/inspect_dataset.py data/corpus/train.txt
```

The preparation script normalizes line endings/whitespace, removes exact duplicate documents, and creates a separate validation corpus. Keeping validation documents separate prevents the old random-chunk split from leaking neighboring text into validation.

### Train the tokenizer on the training split

```powershell
python train_tokenizer.py --text data/corpus/train.txt --vocab-size 16000 --out tokenizer/model
```

### Pretrain

Set `training.data_file` to `data/corpus/train.txt` and `training.validation_file` to `data/corpus/val.txt` in your config, then:

```powershell
python train.py --config config.yaml
```

### Instruction tuning

Create `data/instructions.jsonl` using either:

```json
{"instruction":"Explain gravity simply.","response":"Gravity is the attraction between objects with mass."}
```

or chat messages:

```json
{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hello! I'm Kira."}]}
```

Then fine-tune a pretrained Kira checkpoint locally:

```powershell
python -m training.sft --config config.yaml --data data/instructions.jsonl --tokenizer tokenizer/model --checkpoint checkpoints/kira-best.pt --steps 2000
```

The SFT trainer masks the prompt portion of the sequence, so the loss focuses on producing assistant responses.

## What makes this a real LLM

The model is not a database of answers. During pretraining it minimizes next-token cross-entropy with backpropagation through learned embeddings, RoPE attention, Transformer blocks, and the output projection. The weights in the checkpoint are the learned parameters of Kira.

For a strong model, **data quality and token count are just as important as architecture**. Do not expect the tiny included corpus to produce useful general intelligence; it only exists to make the software runnable and inspectable.
