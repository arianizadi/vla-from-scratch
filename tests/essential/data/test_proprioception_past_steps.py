"""Proprioception must only contain the past and current state, also when no past steps are requested."""

from types import SimpleNamespace

import pytest
import torch

from vla_foundry.data.processor.robotics_processor import RoboticsProcessor


def proprioception(past_steps, sequence_len):
    processor = object.__new__(RoboticsProcessor)  # only the two attributes used by the method under test
    processor.data_params = SimpleNamespace(lowdim_past_timesteps=past_steps)
    processor.normalizer = SimpleNamespace(lowdim_past_timesteps=1)  # what the statistics were computed with
    state = torch.arange(float(sequence_len)).view(1, sequence_len, 1)  # value == index in the window
    batch = {"lowdim": {"state": state, "action": torch.zeros(1, sequence_len, 2)}}
    return processor.add_action_and_proprioception_fields(batch, ["action"], ["state"])["proprioception"]


def test_zero_past_steps_gives_only_the_current_state():
    # The window starts at the current step, so anything beyond index 0 is the future.
    out = proprioception(past_steps=0, sequence_len=15)
    assert out.flatten().tolist() == [0.0]


@pytest.mark.parametrize("past_steps", [1, None])
def test_one_past_step_gives_previous_and_current_state(past_steps):
    # None = not set: fall back to the past steps of the statistics (1).
    out = proprioception(past_steps=past_steps, sequence_len=16)
    assert out.flatten().tolist() == [0.0, 1.0]
