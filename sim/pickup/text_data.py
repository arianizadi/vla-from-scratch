"""Web text for stage 1 of the from-scratch path (a language model trained from random weights).

Downloads one shard of DCLM-baseline (https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0,
about 61,000 documents, a few hundred MB) and writes it in the layout the `text_untokenized` pipeline
reads: tar shards of <key>.json files holding {"text": ...}, plus a manifest.jsonl.

    python -m sim.pickup.text_data --out data/custom/dclm
"""

import argparse
import io
import json
import os
import tarfile

import zstandard
from huggingface_hub import hf_hub_download

REPO = "mlfoundations/dclm-baseline-1.0"
FILENAME = "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--max_docs", type=int, default=100_000, help="stop after this many documents")
    p.add_argument("--samples_per_shard", type=int, default=4096)
    args = p.parse_args()

    manifest_path = os.path.join(args.out, "manifest.jsonl")
    if os.path.exists(manifest_path):
        print(f"{manifest_path} already exists; nothing to do.")
        return

    path = hf_hub_download(repo_id=REPO, filename=FILENAME, repo_type="dataset")
    docs = []
    with open(path, "rb") as f:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8")
        for line in reader:
            if len(docs) >= args.max_docs:
                break
            docs.append(json.loads(line)["text"])

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for shard, start in enumerate(range(0, len(docs), args.samples_per_shard)):
        stem = f"shard-{shard:04d}"
        batch = docs[start : start + args.samples_per_shard]
        with tarfile.open(os.path.join(args.out, f"{stem}.tar"), "w") as tar:
            for i, text in enumerate(batch):
                payload = json.dumps({"text": text}).encode("utf-8")
                info = tarfile.TarInfo(name=f"{start + i:08d}.json")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        manifest.append({"shard": stem, "num_sequences": len(batch)})
    with open(manifest_path, "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in manifest)
    print(f"wrote {len(docs)} documents in {len(manifest)} shards to {args.out}")


if __name__ == "__main__":
    main()
