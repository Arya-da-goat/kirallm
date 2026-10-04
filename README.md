# Training data

Keep legally usable, high-quality text in this directory. Do not assume that text found online is automatically licensed for model training.

Recommended layout:

- `corpus/` — cleaned pretraining text (`train.txt`, `val.txt`)
- `instructions.jsonl` — instruction/chat examples for supervised fine-tuning

For pretraining, prepare a corpus with:

```powershell
python tools/prepare_corpus.py path\to\your\txt\folder --out data/corpus
```

Then train the tokenizer on `data/corpus/train.txt` and configure the trainer to use the matching validation file.
