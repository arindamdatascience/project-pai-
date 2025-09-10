# models/speaker_encoder.py
import numpy as np
import soundfile as sf
import librosa
from typing import Optional

class ToySpeakerEncoder:
    """Toy speaker encoder: average MFCCs -> L2-normalized vector.
    Replace with ECAPA-TDNN or other high-quality speaker encoder for production.
    """

    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 20):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.embedding_dim = n_mfcc  # toy

    def load_audio(self, path: str):
        audio, sr = sf.read(path)
        if sr != self.sample_rate:
            audio = librosa.resample(audio.astype(float), sr, self.sample_rate)
        return audio

    def encode_file(self, path: str) -> np.ndarray:
        audio = self.load_audio(path)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        emb = mfcc.mean(axis=1)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)

    def encode_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.sample_rate:
            audio = librosa.resample(audio.astype(float), sr, self.sample_rate)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        emb = mfcc.mean(axis=1)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)

def get_speaker_encoder(kind: str = "toy", **kwargs) -> ToySpeakerEncoder:
    return ToySpeakerEncoder(**kwargs)
