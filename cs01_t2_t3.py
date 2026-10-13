"""
CS-01: T2 context-length experiment + T3 CTranslate2 score/logit API inspection.
Run from G:\ASR with the project .venv activated:
    python cs01_t2_t3.py --limit 20

Expected metadata: data\metadata.csv
Expected audio directory: data\audio
Expected metadata columns: audio, text, terms (also accepts segment_text/cs_terms_list).
Outputs:
    results_cs01\t2_predictions.csv
    results_cs01\t2_summary.csv
    results_cs01\t3_api_inspection.txt

Important:
- T2 compares no prompt against prompts of 3/10/30 terms.
- For 10/30-term prompts, sample-specific terms are placed first, then terms
  from the corpus-wide vocabulary are added as context distractors. This tests
  both context length and the effect of extra terms; interpret it accordingly.
- This script does NOT claim QNN support or direct token-score modification.
"""
import argparse
import csv
import inspect
import re
import sys
import time
import unicodedata
from pathlib import Path
from collections import Counter

import pandas as pd
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent
META_PATH = ROOT / "data" / "metadata.csv"
AUDIO_DIR = ROOT / "data" / "audio"
OUT_DIR = ROOT / "results_cs01"


def first_existing(row, names):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan":
            return str(value).strip()
    return ""


def split_terms(value):
    if not value:
        return []
    # CSV terms are usually semicolon-separated; tolerate commas and pipes too.
    pieces = re.split(r"[;|,\n]+", str(value))
    result, seen = [], set()
    for piece in pieces:
        term = piece.strip().strip("[]'\" ")
        if not term:
            continue
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            result.append(term)
    return result


def resolve_audio(value):
    p = Path(value)
    candidates = [p] if p.is_absolute() else [ROOT / p, AUDIO_DIR / p, ROOT / "data" / p]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    # If metadata contains a path with stale folder prefixes, try basename.
    return AUDIO_DIR / Path(value).name


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def edit_distance(a, b):
    # Levenshtein distance using a single row of memory.
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def word_error_counts(reference, hypothesis):
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return None, 0, 0
    errors = edit_distance(ref, hyp)
    return errors / len(ref), errors, len(ref)


def char_error_counts(reference, hypothesis):
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(hypothesis).replace(" ", "")
    if not ref:
        return None, 0, 0
    errors = edit_distance(ref, hyp)
    return errors / len(ref), errors, len(ref)


def transcribe(model, audio_path, prompt):
    segments, info = model.transcribe(
        str(audio_path),
        language="vi",
        task="transcribe",
        beam_size=5,
        temperature=0.0,
        initial_prompt=prompt if prompt else None,
        condition_on_previous_text=False,
        vad_filter=False,
        without_timestamps=True,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def inspect_t3_api(output_path):
    lines = []
    lines.append("CS-01 T3 API inspection")
    lines.append(f"Python: {sys.version}")
    try:
        import ctranslate2
        lines.append(f"CTranslate2: {ctranslate2.__version__}")
        gen = ctranslate2.models.Whisper.generate
        lines.append("\nctranslate2.models.Whisper.generate docstring:")
        lines.append(gen.__doc__ or "(no docstring)")
        try:
            lines.append("\ninspect.signature(generate):")
            lines.append(str(inspect.signature(gen)))
        except Exception as exc:
            lines.append(f"\ninspect.signature(generate) unavailable: {type(exc).__name__}: {exc}")
        doc = gen.__doc__ or ""
        lines.append("\nChecks:")
        lines.append(f"return_logits_vocab mentioned: {'return_logits_vocab' in doc}")
        lines.append(f"custom logits processor mentioned in public generate docs: {'logits_processor' in doc.lower() or 'logits processor' in doc.lower()}")
        lines.append(f"return_scores mentioned: {'return_scores' in doc}")
        lines.append(
            "\nInterpretation: return_logits_vocab/return_scores, if exposed by this installed build, "
            "are for returning diagnostic outputs. Their presence alone does not mean the Python API "
            "lets the caller alter token logits during decoding."
        )
    except Exception as exc:
        lines.append(f"Could not inspect CTranslate2: {type(exc).__name__}: {exc}")
    lines.append(
        "\nScope: this is a Windows/CTranslate2 API check only; no Qualcomm/QNN runtime was tested."
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[T3] Đã lưu kiểm tra API: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="Số file bắt đầu thử; mặc định 20")
    parser.add_argument("--model", default="small", help="Tên model faster-whisper; mặc định small")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Mặc định CPU để khớp thử nghiệm trước")
    parser.add_argument("--compute-type", default="int8", help="Mặc định int8")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inspect_t3_api(OUT_DIR / "t3_api_inspection.txt")

    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {META_PATH}. Kiểm tra metadata.csv có đúng ở G:\\ASR\\data\\metadata.csv không."
        )

    df = pd.read_csv(META_PATH, dtype=str).fillna("")
    audio_col = next((c for c in ["audio", "audio_path", "file", "filename"] if c in df.columns), None)
    ref_col = next((c for c in ["text", "segment_text", "reference", "transcript"] if c in df.columns), None)
    terms_col = next((c for c in ["terms", "cs_terms_list", "cs_terms", "keywords"] if c in df.columns), None)

    if not audio_col:
        raise ValueError(f"Không tìm thấy cột audio. Các cột hiện có: {list(df.columns)}")
    if not ref_col:
        raise ValueError(f"Không tìm thấy cột transcript tham chiếu. Các cột hiện có: {list(df.columns)}")
    if not terms_col:
        raise ValueError(f"Không tìm thấy cột terms. Các cột hiện có: {list(df.columns)}")

    rows = []
    for _, row in df.iterrows():
        audio_value = str(row[audio_col]).strip()
        audio_path = resolve_audio(audio_value)
        terms = split_terms(row[terms_col])
        reference = str(row[ref_col]).strip()
        if audio_value and audio_path.exists() and reference and terms:
            rows.append({
                "audio_value": audio_value,
                "audio_path": audio_path,
                "reference": reference,
                "terms": terms,
            })
        if len(rows) >= args.limit:
            break

    if not rows:
        raise RuntimeError(
            "Không tìm được mẫu hợp lệ. Cần có file audio tồn tại, transcript tham chiếu và terms trong metadata.csv."
        )

    # Corpus vocabulary supplies additional terms when a sample has fewer than 10/30 terms.
    vocabulary = []
    seen = set()
    for _, row in df.iterrows():
        if terms_col in row:
            for term in split_terms(row[terms_col]):
                key = term.casefold()
                if key not in seen:
                    seen.add(key)
                    vocabulary.append(term)

    print(f"[T2] Model={args.model}, device={args.device}, compute_type={args.compute_type}")
    print(f"[T2] Số mẫu hợp lệ: {len(rows)}")
    print("[T2] Điều kiện: T0, T2_3, T2_10, T2_30")
    print("[T3] File kiểm tra API đã được tạo trước khi chạy ASR.")

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    conditions = [("T0_baseline", 0), ("T2_3_terms", 3), ("T2_10_terms", 10), ("T2_30_terms", 30)]
    result_rows = []

    total = len(rows) * len(conditions)
    counter = 0
    for sample_idx, sample in enumerate(rows, 1):
        for condition, target_count in conditions:
            counter += 1
            if target_count == 0:
                chosen = []
                prompt = None
            else:
                # Relevant sample terms first, then corpus-wide terms as fillers.
                chosen = list(sample["terms"][:target_count])
                chosen_seen = {x.casefold() for x in chosen}
                for term in vocabulary:
                    if len(chosen) >= target_count:
                        break
                    if term.casefold() not in chosen_seen:
                        chosen.append(term)
                        chosen_seen.add(term.casefold())
                prompt = "Thuật ngữ chuyên ngành có thể xuất hiện: " + ", ".join(chosen)

            start = time.perf_counter()
            try:
                hypothesis = transcribe(model, sample["audio_path"], prompt)
                elapsed = time.perf_counter() - start
                wer, wer_err, wer_n = word_error_counts(sample["reference"], hypothesis)
                cer, cer_err, cer_n = char_error_counts(sample["reference"], hypothesis)
                error = ""
            except Exception as exc:
                elapsed = time.perf_counter() - start
                hypothesis = ""
                wer = cer = None
                wer_err = wer_n = cer_err = cer_n = None
                error = f"{type(exc).__name__}: {exc}"

            result_rows.append({
                "audio": sample["audio_value"],
                "audio_path": str(sample["audio_path"]),
                "condition": condition,
                "requested_context_terms": target_count,
                "actual_context_terms": len(chosen),
                "context_terms": "; ".join(chosen),
                "reference": sample["reference"],
                "hypothesis": hypothesis,
                "WER": wer,
                "WER_errors": wer_err,
                "WER_ref_words": wer_n,
                "CER": cer,
                "CER_errors": cer_err,
                "CER_ref_chars": cer_n,
                "elapsed_seconds": round(elapsed, 3),
                "model": args.model,
                "device": args.device,
                "compute_type": args.compute_type,
                "error": error,
            })
            print(f"[{counter}/{total}] {sample_idx}/{len(rows)} {condition}: {sample['audio_value']} ({elapsed:.1f}s)"
                  + (f" ERROR={error}" if error else ""))

    pred_path = OUT_DIR / "t2_predictions.csv"
    pd.DataFrame(result_rows).to_csv(pred_path, index=False, encoding="utf-8-sig")

    # Aggregate corpus-level WER/CER from total edit counts, plus runtime.
    summary_rows = []
    res_df = pd.DataFrame(result_rows)
    for condition, group in res_df.groupby("condition", sort=False):
        valid_wer = group.dropna(subset=["WER"])
        wer_total_err = valid_wer["WER_errors"].sum() if len(valid_wer) else 0
        wer_total_ref = valid_wer["WER_ref_words"].sum() if len(valid_wer) else 0
        cer_total_err = valid_wer["CER_errors"].sum() if len(valid_wer) else 0
        cer_total_ref = valid_wer["CER_ref_chars"].sum() if len(valid_wer) else 0
        summary_rows.append({
            "condition": condition,
            "samples": len(group),
            "failed_samples": int((group["error"].astype(str) != "").sum()),
            "context_terms_mean": round(group["actual_context_terms"].mean(), 2),
            "WER_corpus": (wer_total_err / wer_total_ref) if wer_total_ref else None,
            "CER_corpus": (cer_total_err / cer_total_ref) if cer_total_ref else None,
            "mean_elapsed_seconds": round(group["elapsed_seconds"].mean(), 3),
            "median_elapsed_seconds": round(group["elapsed_seconds"].median(), 3),
        })
    summary_path = OUT_DIR / "t2_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nHoàn tất T2.")
    print(f"Chi tiết: {pred_path}")
    print(f"Tổng hợp: {summary_path}")
    print(f"T3 API: {OUT_DIR / 't3_api_inspection.txt'}")
    print("\nLưu ý: T2_10/T2_30 có thể thêm thuật ngữ từ toàn bộ corpus làm distractor.")
    print("Nếu metadata không có transcript tham chiếu đáng tin cậy, không dùng WER/CER làm kết luận cuối.")


if __name__ == "__main__":
    main()
