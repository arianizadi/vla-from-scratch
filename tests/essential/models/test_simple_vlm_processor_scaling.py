"""uint8 images must be scaled identically whether they arrive as PIL, numpy or torch tensors."""

import numpy as np
import torch
from PIL import Image

from vla_foundry.models.vision_language_backbones.simple_vlm_processor import SimpleVLMProcessor


def test_uint8_inputs_share_one_value_range():
    processor = SimpleVLMProcessor.__new__(SimpleVLMProcessor)  # skip the tokenizer download
    processor.image_size = 8
    processor._mean = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
    processor._std = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)

    hwc = np.random.default_rng(0).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    from_numpy = processor._process_image(hwc)
    from_pil = processor._process_image(Image.fromarray(hwc))
    from_chw_tensor = processor._process_image(torch.from_numpy(hwc).permute(2, 0, 1))

    assert from_numpy.min() >= -1.0 and from_numpy.max() <= 1.0
    torch.testing.assert_close(from_pil, from_numpy)
    torch.testing.assert_close(from_chw_tensor, from_numpy)
