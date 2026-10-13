
import csv
import io
import os

import soundfile as sf
from datasets import load_dataset

OUTPUT_DIR = "data"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
CSV_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")
MAX_FILES = 50

os.makedirs(AUDIO_DIR, exist_ok=True)

print("Dang ket noi toi ViMedCSS...", flush=True)

dataset = load_dataset(
    "tensorxt/ViMedCSS",
    split="train",
    streaming=True,
)

count = 0
checked = 0

with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "audio",
            "text",
            "terms",
            "duration_seconds",
            "topic",
        ],
    )
    writer.writeheader()

    for item in dataset:
        checked += 1

        terms = item.get("cs_terms_list")
        if not terms or not str(terms).strip():
            continue

        try:
            duration = float(item.get("duration_seconds") or 0)
        except (TypeError, ValueError):
            continue

        if not 2 <= duration <= 20:
            continue

        audio = item.get("audio")
        if not isinstance(audio, dict):
            continue

        audio_bytes = audio.get("bytes")
        if not audio_bytes and audio.get("path"):
            print(
                f"Bo qua mau {item.get('segment_id')}: "
                "audio khong co bytes",
                flush=True,
            )
            continue

        if not audio_bytes:
            continue

        try:
            # Đọc audio trực tiếp từ dữ liệu bytes
            audio_array, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32",
            )

            audio_id = str(item["segment_id"])
            filename = f"{audio_id}.wav"
            audio_path = os.path.join(AUDIO_DIR, filename)

            sf.write(
                audio_path,
                audio_array,
                sample_rate,
                subtype="PCM_16",
            )

            writer.writerow({
                "audio": filename,
                "text": item.get("segment_text", ""),
                "terms": terms,
                "duration_seconds": duration,
                "topic": item.get("topic", ""),
            })

            f.flush()
            count += 1

            print(
                f"[{count}/{MAX_FILES}] Da luu {filename}",
                flush=True,
            )

        except Exception as e:
            print(
                f"Loi mau {item.get('segment_id')}: {e}",
                flush=True,
            )

        if count >= MAX_FILES:
            break

print(f"\nHoan thanh: {count}/{MAX_FILES} file", flush=True)
print(f"So mau da kiem tra: {checked}", flush=True)
print(f"Audio: {AUDIO_DIR}", flush=True)
print(f"Metadata: {CSV_PATH}", flush=True)
