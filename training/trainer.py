# training/trainer.py
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List
from models.hybrid_tts import HybridTTSModel
from models.speech_tokenizer import get_speech_tokenizer
from models.speaker_encoder import get_speaker_encoder

class TTSDataset(Dataset):
    def __init__(self, metadata_csv: str, text_tokenizer, speech_tokenizer, speaker_encoder, sample_rate=22050):
        self.df = pd.read_csv(metadata_csv)
        self.text_tokenizer = text_tokenizer  # function/text tokenizer wrapper
        self.speech_tokenizer = speech_tokenizer
        self.speaker_encoder = speaker_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['audio_file']
        text = row['text']
        speaker = row.get('speaker_id', 'spk_0')
        # tokenization
        input_ids = self.text_tokenizer.encode(text)  # pytorch tensor or list
        # get speaker embedding
        spk_emb = self.speaker_encoder.encode_file(audio_path)
        # get target speech tokens
        tokens = self.speech_tokenizer.encode_file(audio_path)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'speaker_emb': torch.tensor(spk_emb, dtype=torch.float),
            'tokens': torch.tensor(tokens, dtype=torch.long)
        }

def collate_fn(batch):
    # simple pad to max length in batch (toy)
    batch_input = [b['input_ids'] for b in batch]
    batch_tokens = [b['tokens'] for b in batch]
    batch_spk = torch.stack([b['speaker_emb'] for b in batch]).float()
    input_ids = torch.nn.utils.rnn.pad_sequence(batch_input, batch_first=True, padding_value=0)
    token_ids = torch.nn.utils.rnn.pad_sequence(batch_tokens, batch_first=True, padding_value=0)
    return input_ids, batch_spk, token_ids

def default_text_tokenizer_factory():
    # toy text tokenizer: split on spaces and map to small vocab
    class ToyTok:
        def __init__(self):
            self.vocab = {"<pad>":0, "<unk>":1}
            self.next_id = 2
        def encode(self, text):
            tokens = []
            for w in text.strip().split():
                if w not in self.vocab:
                    self.vocab[w] = self.next_id
                    self.next_id += 1
                tokens.append(self.vocab[w])
            return tokens
    return ToyTok()

def train_loop(metadata_csv: str, epochs: int = 1, batch_size: int = 2, device: str = "cpu"):
    # Create components
    speech_tokenizer = get_speech_tokenizer(kind="toy")
    speaker_encoder = get_speaker_encoder(kind="toy")
    text_tokenizer = default_text_tokenizer_factory()
    ds = TTSDataset(metadata_csv, text_tokenizer, speech_tokenizer, speaker_encoder)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = HybridTTSModel(text_vocab_size=5000, token_vocab_size=speech_tokenizer.codebook_size, d_model=256)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        model.train()
        for step, (input_ids, spk_emb, target_tokens) in enumerate(dl):
            input_ids = input_ids.to(device)
            spk_emb = spk_emb.to(device)
            target_tokens = target_tokens.to(device)
            # shift right tokens for decoder input
            decoder_in = torch.cat([torch.zeros((target_tokens.size(0),1), dtype=torch.long, device=device), target_tokens[:,:-1]], dim=1)
            logits = model(input_ids, spk_emb, decoder_in)
            # compute CE on flattened tensors
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % 10 == 0:
                print(f"[Epoch {epoch}] Step {step} Loss {loss.item():.4f}")

        # save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/hybrid_tts_epoch{epoch}.pt")
        print(f"Saved checkpoint: checkpoints/hybrid_tts_epoch{epoch}.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="data/splits/train_metadata.csv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    train_loop(args.meta, epochs=args.epochs, batch_size=args.batch, device="cpu")
