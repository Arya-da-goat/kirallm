# Kira development roadmap

## Phase A — data quality
1. Build a legally usable corpus.
2. Normalize and exact-deduplicate documents.
3. Keep validation documents separate from training documents.
4. Inspect token/character statistics before training.

## Phase B — base pretraining
Train the decoder-only Transformer on next-token prediction. Monitor training and validation loss, perplexity, gradient norms, throughput and checkpoint recovery.

## Phase C — instruction tuning
Use `training/sft.py` with high-quality instruction/chat examples. The loss is masked on the user prompt so the model is optimized mainly for assistant responses.

## Phase D — evaluation
Maintain a held-out set of questions covering math, coding, factual QA, instruction following and conversation. Never train on the evaluation set.

## Phase E — tools and product
Connect the local model to Kira's UI. Keep web search, calculators and file/image tools outside the neural model so they can be updated independently.

## Phase F — scaling
Only after the complete pipeline is stable, increase parameters, context length and token count according to available hardware.
