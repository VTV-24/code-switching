
import csv
import os
import re
import time
from pathlib import Path

import pandas as pd
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "data" / "audio"
METADATA = ROOT / "data" / "metadata.csv"
OUTPUT_DIR = ROOT / "results_cs01"

MODEL_SIZE = "small"
LIMIT = 50

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(METADATA, encoding="utf-8-sig").fillna("")
df = df.head(LIMIT)

print(f"So mau: {len(df)}")
print(f"Nap Whisper model: {MODEL_SIZE}")

# CPU là lựa chọn tương thích rộng để bắt đầu.
# Nếu máy đã cài CUDA/cuDNN phù hợp, có thể đổi sang device="cuda".
model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8",
)

def transcribe(audio_path, prompt=None):
    start = time.perf_counter()

    segments, info = model.transcribe(
        str(audio_path),
        language="vi",
        task="transcribe",
        beam_size=5,
        initial_prompt=prompt if prompt else None,
        condition_on_previous_text=False,
        vad_filter=False,
    )

    text = "".join(seg.text for seg in segments).strip()
    elapsed = time.perf_counter() - start

    return text, elapsed

results = []

for index, row in df.iterrows():
    filename = str(row["audio"])
    audio_path = AUDIO_DIR / filename

    if not audio_path.exists():
        print(f"Bo qua, khong tim thay: {audio_path}")
        continue

    terms = [
        term.strip()
        for term in str(row["terms"]).split(";")
        if term.strip()
    ]

    # T1 dùng danh sách thuật ngữ của chính mẫu.
    # Đây là phép thử oracle, không phải ngữ cảnh tự động.
    prompt = ", ".join(terms) if terms else None

    print(f"\n[{len(results) + 1}/{len(df)}] {filename}")

    for condition, context in [
        ("T0_baseline", None),
        ("T1_context", prompt),
    ]:
        try:
            hypothesis, elapsed = transcribe(audio_path, context)

            results.append({
                "audio": filename,
                "condition": condition,
                "reference": str(row["text"]),
                "terms": ";".join(terms),
                "hypothesis": hypothesis,
                "elapsed_seconds": round(elapsed, 3),
                "context": context or "",
                "model": MODEL_SIZE,
            })

            print(f"  {condition}: {hypothesis}")
            print(f"  Thoi gian: {elapsed:.2f}s")

            # Lưu từng mẫu để không mất kết quả khi dừng giữa chừng.
            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )

        except Exception as exc:
            print(f"  Loi {condition}: {exc}")

print("\nDa hoan thanh.")
print("Ket qua:", OUTPUT_DIR / "predictions.csv")
