# models/hybrid_tts.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class SimpleTextEncoder(nn.Module):
    """Small text encoder for prototyping. Replace with LLaMA tokenizer+model."""
    def __init__(self, vocab_size: int = 32000, d_model: int = 512, max_len: int = 256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=2048),
            num_layers=3
        )
        self.d_model = d_model

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = x.permute(1,0,2)  # transformer expects seq, batch, dim
        out = self.transformer(x)
        out = out.permute(1,0,2)
        # return pooled embedding (mean)
        return out.mean(dim=1)  # batch x d_model

class TokenDecoder(nn.Module):
    """Decoder that predicts discrete speech tokens autoregressively (toy)."""
    def __init__(self, token_vocab_size: int = 4096, d_model: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(token_vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.proj = nn.Linear(d_model, token_vocab_size)
        self.d_model = d_model

    def forward(self, token_input_ids, conditioning_vec):
        # token_input_ids: batch x seq
        token_emb = self.token_emb(token_input_ids)
        # prepend conditioning as initial hidden state
        cond = conditioning_vec.unsqueeze(0)  # 1 x batch x d
        out, _ = self.rnn(token_emb, cond)
        logits = self.proj(out)
        return logits

class HybridTTSModel(nn.Module):
    """Hybrid TTS prototype:
    text -> text_embedding
    speaker_emb -> speaker_projection
    conditioning = concat(project(text_emb), project(speaker_emb))
    decoder generates sequence of discrete speech tokens.
    """
    def __init__(self, text_vocab_size=32000, token_vocab_size=4096, d_model=512):
        super().__init__()
        self.text_encoder = SimpleTextEncoder(vocab_size=text_vocab_size, d_model=d_model)
        self.speaker_proj = nn.Linear(256, d_model)  # expect speaker embedding dim 256 (toy)
        self.text_proj = nn.Linear(d_model, d_model)
        self.decoder = TokenDecoder(token_vocab_size, d_model)
        # token generation helpers
        self.token_vocab_size = token_vocab_size

    def forward(self, input_ids, speaker_emb, token_input_ids):
        # input_ids: batch x text_seq
        text_vec = self.text_encoder(input_ids)
        text_cond = self.text_proj(text_vec)
        sp = self.speaker_proj(speaker_emb)  # batch x d_model
        conditioning = (text_cond + sp) / 2.0  # simple fusion
        logits = self.decoder(token_input_ids, conditioning)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, speaker_emb, max_tokens:int=200, temperature:float=1.0, device='cpu'):
        batch = input_ids.size(0)
        generated = torch.full((batch,1), fill_value=0, dtype=torch.long, device=device)  # start token 0
        text_vec = self.text_encoder(input_ids)
        cond = (self.text_proj(text_vec) + self.speaker_proj(speaker_emb)) / 2.0
        for _ in range(max_tokens):
            logits = self.decoder(generated, cond)  # batch x seq x vocab
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tok], dim=1)
        return generated[:,1:]  # drop initial start token
