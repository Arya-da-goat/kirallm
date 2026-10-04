"""Lightweight self-checks for the tokenizer and Transformer."""
import torch
from tokenizer.bpe import BPETokenizer
from kira.model import KiraConfig, KiraLM


def main():
    tok = BPETokenizer.train("Hello Kira! This is a tokenizer test. " * 100, vocab_size=512)
    sample = "Hello, Kira! नमस्ते"
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample, (tok.decode(ids), sample)

    cfg = KiraConfig(vocab_size=len(tok.vocab), max_seq_len=32, dim=128, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=352)
    model = KiraLM(cfg)
    x = torch.tensor([ids[:16]], dtype=torch.long)
    y = torch.roll(x, shifts=-1, dims=1)
    logits, loss = model(x, y)
    assert logits.shape == (1, x.shape[1], len(tok.vocab))
    assert torch.isfinite(loss)
    loss.backward()
    print("Kira self-check passed.")


if __name__ == "__main__":
    main()
