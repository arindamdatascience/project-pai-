import os
import pandas as pd
from datasets import load_dataset

# --------------- CONFIG ---------------
# Change language as needed (e.g., IndicTTS_Tamil, IndicTTS_Hindi, IndicTTS_Bengali, etc.)
HF_DATASET = "SPRINGLab/IndicTTS_Tamil"
OUTPUT_AUDIO_DIR = "data/raw/indictts/tamil/audio"
OUTPUT_META_CSV = "data/splits/indictts_tamil_metadata.csv"

# ------------- SCRIPT -------------------
def download_and_prepare():
    print(f"Loading dataset from Hugging Face: {HF_DATASET}")
    ds = load_dataset(HF_DATASET, split="train")
    
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    print("Downloading audio and building metadata...")

    rows = []
    for idx, item in enumerate(ds):
        # Download audio
        audio = item["audio"]
        text = item["text"]

        fname = f"utt_{idx:05d}.wav"
        path = os.path.join(OUTPUT_AUDIO_DIR, fname)

        with open(path, "wb") as f:
            f.write(audio["array"].astype("float32").tobytes())

        duration = audio["duration"]
        gender = item.get("gender", "unknown")

        rows.append([path, text, gender, duration])

        if idx % 500 == 0:
            print(f"  Processed {idx}/{len(ds)} samples...")

    # Create metadata CSV
    df = pd.DataFrame(rows, columns=["audio_file", "text", "gender", "duration"])
    os.makedirs(os.path.dirname(OUTPUT_META_CSV), exist_ok=True)
    df.to_csv(OUTPUT_META_CSV, index=False)
    print(f"Metadata saved to {OUTPUT_META_CSV}")

if __name__ == "__main__":
    download_and_prepare()
