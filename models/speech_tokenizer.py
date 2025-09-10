# models/speech_tokenizer.py
import os
import numpy as np
import soundfile as sf
import torchaudio
from typing import List, Tuple, Optional

# NOTE: Replace this toy implementation with XCodec2 / EnCodec when available.
# The toy tokenizer will convert audio -> mel -> k-means-like indices (not production quality).

class ToySpeechTokenizer:
    """Toy speech tokenizer (placeholder).
    Produces integer "tokens" by slicing mel frames and hashing — useful for prototyping.
    Replace with real XCodec2/EnCodec tokenizers for production.
    """

    def __init__(self, sample_rate: int = 22050, n_mels: int = 80, hop_length: int = 256):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        # small pseudo-codebook size
        self.codebook_size = 1024

    def load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        audio, sr = sf.read(path)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio).float(),
                orig_freq=sr, new_freq=self.sample_rate
            ).numpy()
            sr = self.sample_rate
        return audio, sr

    def audio_to_mel(self, audio: np.ndarray):
        import librosa
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=self.n_mels, hop_length=self.hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.T  # frames x n_mels

    def mel_to_tokens(self, mel_frames: np.ndarray) -> List[int]:
        # Simple hash-based quantization for prototyping
        tokens = []
        for frame in mel_frames:
            # reduce frame to int by summing and mod codebook_size
            value = int(np.abs(frame).sum()) % self.codebook_size
            tokens.append(value)
        return tokens

    def tokens_to_mel(self, tokens: List[int], frames_per_token: int = 1):
        # Reconstruct a naive mel by repeating basic patterns (toy)
        mel = np.zeros((len(tokens) * frames_per_token, self.n_mels), dtype=np.float32)
        for i, t in enumerate(tokens):
            base = (t % 10) / 10.0
            mel[i * frames_per_token:(i + 1) * frames_per_token, :] = base
        return mel

    def mel_to_waveform(self, mel):
        # Placeholder: use Griffin-Lim to reconstruct waveform (low-quality)
        import librosa
        mel = mel.T  # back to n_mels x frames
        S = librosa.db_to_power(mel)
        wav = librosa.feature.inverse.mel_to_audio(S, sr=self.sample_rate, hop_length=self.hop_length)
        return wav

    # Public API
    def encode_file(self, wav_path: str) -> List[int]:
        audio, sr = self.load_audio(wav_path)
        mel = self.audio_to_mel(audio)
        return self.mel_to_tokens(mel)

    def encode_audio(self, audio: np.ndarray, sr: int) -> List[int]:
        mel = self.audio_to_mel(audio)
        return self.mel_to_tokens(mel)

    def decode_tokens(self, tokens: List[int]) -> np.ndarray:
        mel = self.tokens_to_mel(tokens)
        wav = self.mel_to_waveform(mel)
        return wav

# convenience factory
def get_speech_tokenizer(kind: str = "toy", **kwargs):
    if kind == "toy":
        return ToySpeechTokenizer(**kwargs)
    # elif kind == "xcodec2":
    #    return XCodec2Wrapper(...)
    else:
        return ToySpeechTokenizer(**kwargs)
