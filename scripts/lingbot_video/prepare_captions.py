"""Convert dataset captions into LingBot-Video structured JSON captions.

The LingBot-Video DiT was trained ONLY on structured JSON captions produced by
the official prompt rewriter. Training with plain natural-language captions
feeds out-of-distribution text to the model and degrades it. Run this script
BEFORE training to rewrite the "text" field of the dataset metadata:

    # needs the official rewriter weights:
    #   REWRITER_BASE_MODEL  -> Qwen3.6-27B VLM (or --base)
    #   REWRITER_ADAPTER     -> step2 LoRA adapter, peft format (or --adapter)
    python scripts/lingbot_video/prepare_captions.py \
        --metadata datasets/my_dataset/metadata.json \
        --data_root datasets/my_dataset \
        --output datasets/my_dataset/metadata_json.json \
        --mode ti2v --duration 3.3

Then point train.sh at the rewritten metadata
(DATASET_META_NAME=datasets/my_dataset/metadata_json.json).

Behaviour:
  - entries whose "text" is already a valid structured JSON caption are kept
    as-is (use --overwrite to re-rewrite them);
  - rewriting is resumed safely: the output file is saved after every sample;
  - for --mode ti2v the video's first frame is fed to the rewriter;
  - the original metadata file is never modified in place.
"""
import argparse
import json
import os
import sys

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, project_root)

from videox_fun.models.lingbot_video_rewriter import is_valid_caption, LingBotVideoRewriter


def load_metadata(path):
    if path.endswith(".csv"):
        import csv
        with open(path, "r") as f:
            return list(csv.DictReader(f))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_metadata(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def extract_first_frame(video_path):
    """Return the first frame of a video as a PIL image (ti2v condition)."""
    from PIL import Image
    try:
        from decord import VideoReader
        vr = VideoReader(video_path)
        return Image.fromarray(vr[0].asnumpy())
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot read first frame of {video_path}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="dataset metadata (.json list or .csv)")
    ap.add_argument("--data_root", default="", help="dataset root used to resolve file_path")
    ap.add_argument("--output", required=True, help="path of the rewritten metadata")
    ap.add_argument("--mode", default="ti2v", choices=["t2v", "ti2v", "t2i"])
    ap.add_argument("--duration", type=float, default=5.0,
                    help="clip duration in seconds fed to the rewriter (match your training clip)")
    ap.add_argument("--base", default=os.environ.get("REWRITER_BASE_MODEL", ""),
                    help="rewriter base VLM path (or set REWRITER_BASE_MODEL)")
    ap.add_argument("--adapter", default=os.environ.get("REWRITER_ADAPTER", ""),
                    help="rewriter step2 LoRA adapter path (or set REWRITER_ADAPTER)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-rewrite captions that are already valid JSON captions")
    args = ap.parse_args()

    entries = load_metadata(args.metadata)
    print(f"loaded {len(entries)} entries from {args.metadata}")

    rewriter = LingBotVideoRewriter(base=args.base, adapter=args.adapter)

    n_done = n_kept = n_fail = 0
    for i, entry in enumerate(entries):
        text = entry.get("text", "")
        if is_valid_caption(text) and not args.overwrite:
            n_kept += 1
            continue
        file_path = entry.get("file_path", "")
        first_frame = None
        if args.mode == "ti2v" and file_path:
            video_path = file_path if os.path.isabs(file_path) else os.path.join(args.data_root, file_path)
            try:
                first_frame = extract_first_frame(video_path)
            except Exception as e:
                print(f"[{i}] WARN: cannot read first frame of {video_path}: {e}")
        try:
            entry["text"] = rewriter.rewrite(text, args.mode, first_frame, args.duration)
            caption = entry["text"]
        except Exception as e:
            caption = None
            print(f"[{i}] ERROR rewriting {file_path}: {e}")
        if caption is None:
            n_fail += 1
            print(f"[{i}] FAILED, keep original text: {file_path}")
        else:
            n_done += 1
        # checkpoint after every sample so long runs can be interrupted
        save_metadata(entries, args.output)
        print(f"[{i + 1}/{len(entries)}] rewritten={n_done} kept={n_kept} failed={n_fail} -> {file_path}")

    save_metadata(entries, args.output)
    print(f"\nDONE: rewritten={n_done}, already-valid={n_kept}, failed={n_fail}\noutput: {args.output}")
    if n_fail:
        print("NOTE: failed entries still carry their original natural-language text; "
              "fix them manually (see videox_fun/pipeline/lingbot_video_caption.py) "
              "or re-run this script.")


if __name__ == "__main__":
    main()
