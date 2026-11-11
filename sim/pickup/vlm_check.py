"""Check what the stage-2 VLM learned: caption held-out pickup scenes and score the object ordering.

Uses the held-out captions that list objects "from left to right:"; the VLM is prompted with the
caption up to the colon and must complete it (greedy decoding). Exact match needs every object
identified and ordered correctly. Runs in the Foundry environment:

    vla-foundry python -m sim.pickup.vlm_check --exp experiments/pickup/custom_vlm \
        --shards data/custom/sim_captions_heldout
"""

import argparse
import glob
import io
import tarfile

import torch
from PIL import Image

from vla_foundry.data.processor import apply_chat_template
from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.models.vision_language_backbones.simple_vlm_processor import SimpleVLMProcessor
from vla_foundry.params.model_params import ModelParams
from vla_foundry.params.train_experiment_params import load_params_from_yaml

p = argparse.ArgumentParser()
p.add_argument("--exp", required=True)
p.add_argument("--shards", required=True)
p.add_argument("--max", type=int, default=300)
a = p.parse_args()

model = create_model(load_params_from_yaml(ModelParams, f"{a.exp}/config_model.yaml"), load_pretrained=False)
ckpt = sorted(glob.glob(f"{a.exp}/checkpoints/checkpoint_*.pt"), key=lambda s: int(s.split("_")[-1][:-3]))[-1]
load_model_checkpoint(model, ckpt)
model.eval().cuda()
proc = SimpleVLMProcessor()
proc.image_seq_length = 256

MARK = "from left to right:"
n = exact = first = 0
shown = 0
for tar_path in sorted(glob.glob(f"{a.shards}/*.tar")):
    with tarfile.open(tar_path) as tar:
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    for name in sorted(f for f in files if f.endswith(".txt")):
        text = files[name].decode()
        if MARK not in text or n >= a.max:
            continue
        prefix, truth = text.split(MARK)
        image = Image.open(io.BytesIO(files[name[:-4] + ".jpg"])).convert("RGB")
        # same chat-template wrapping as the training pipeline, cut after the colon
        prompt = apply_chat_template(proc, 1, text).split(MARK)[0] + MARK
        inputs = proc([image], [prompt], padding=False)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(
                input_ids=inputs["input_ids"].cuda(),
                pixel_values=inputs["pixel_values"].cuda(),
                attention_mask=inputs["attention_mask"].cuda(),
                max_new_tokens=28,
                top_k=1,
                eos_token_id=proc.tokenizer.eos_token_id,
            )
        pred = proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = pred.split(".")[0].strip()
        truth = truth.strip().rstrip(".")
        n += 1
        exact += pred == truth
        first += pred.split(",")[0].split(" and ")[0] == truth.split(",")[0].split(" and ")[0]
        if shown < 6:
            shown += 1
            print(f"{name}\n  truth: {truth}\n  model: {pred}")
print(f"checkpoint {ckpt}")
print(f"exact ordering match: {exact}/{n} = {exact / n:.0%}; leftmost object correct: {first}/{n} = {first / n:.0%}")
