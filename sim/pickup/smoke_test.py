"""Run the scripted expert for a few episodes and report its success rate.

python -m sim.pickup.smoke_test pick_target 20 [contact_sheet.jpg]
"""

import sys
import time

import numpy as np
from PIL import Image

from .env import PickupEnv
from .expert import ScriptedExpert


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "pick_cube"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    sheet = sys.argv[3] if len(sys.argv) > 3 else None
    env = PickupEnv(task)
    expert = ScriptedExpert(env)
    ok, lens, t0, frames = 0, [], time.time(), []
    for ep in range(n):
        obs = env.reset(seed=ep)
        expert.reset()
        done, info = False, {}
        while not done:
            if ep == 0 and env.steps % 12 == 0:
                frames.append(np.concatenate([obs["front"], obs["wrist"]], axis=0))
            obs, done, info = env.step(expert.act())
        ok += info["success"]
        lens.append(env.steps)
        print(f"episode {ep}: target={env.target} success={info['success']} steps={env.steps}", flush=True)
    print(
        f"expert success {ok}/{n}, mean length {np.mean(lens):.1f} steps, "
        f"{sum(lens) / (time.time() - t0):.0f} env steps/s"
    )
    if sheet:
        Image.fromarray(np.concatenate(frames[:8], axis=1)).save(sheet, quality=85)


if __name__ == "__main__":
    main()
