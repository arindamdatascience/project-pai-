# inference/realtime_engine.py
import torch
import numpy as np
import soundfile as sf
from typing import Optional
from models.hybrid_tts import HybridTTSModel
from models.speech_tokenizer import get_speech_tokenizer
from models.speaker_encoder import get_speaker_encoder

class RealtimeEngine:
    def __init__(self, model_ckpt: Optional[str] = None, device: str = "cpu", vocoder: Optional[object]=None):
        self.device = device
        self.speech_tokenizer = get_speech_tokenizer("toy")
        self.speaker_encoder = get_speaker_encoder("toy")
        self.model = HybridTTSModel(text_vocab_size=5000, token_vocab_size=self.speech_tokenizer.codebook_size, d_model=256)
        if model_ckpt and os.path.exists(model_ckpt):
            self.model.load_state_dict(torch.load(model_ckpt, map_location=device))
        self.model.to(device)
        self.vocoder = vocoder  # plug BigVGAN/HifiGAN here

    def synthesize(self, text_token_ids, speaker_audio_path: str, max_len: int=200):
        sp_emb = self.speaker_encoder.encode_file(speaker_audio_path)
        sp_emb = torch.tensor(sp_emb).unsqueeze(0).to(self.device)
        input_ids = torch.tensor([text_token_ids], dtype=torch.long).to(self.device)
        tokens = self.model.generate(input_ids, sp_emb, max_tokens=max_len, device=self.device)
        token_list = tokens[0].cpu().numpy().tolist()
        # decode tokens to waveform
        wav = self.speech_tokenizer.decode_tokens(token_list)
        # optionally run through vocoder — if available use that
        if self.vocoder:
            wav = self.vocoder(mel=wav)  # placeholder
        return wav

if __name__ == "__main__":
    import os
    # Demo
    from training.trainer import default_text_tokenizer_factory
    tt = default_text_tokenizer_factory()
    engine = RealtimeEngine(model_ckpt=None, device="cpu")
    # create toy speaker file path
    speaker_wav = "data/processed/demo_spk.wav"
    text = "नमस्ते दोस्तों"
    input_ids = tt.encode(text)
    wav = engine.synthesize(input_ids, speaker_wav, max_len=100)
    sf.write("outputs/demo_out.wav", wav, 22050)
    print("Saved outputs/demo_out.wav")
