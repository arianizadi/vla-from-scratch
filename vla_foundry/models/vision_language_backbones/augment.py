"""Image augmentation for the ViT backbone while training (ViTBackboneParams.augment): what the camera sees is varied so
that a policy trained from scratch on demonstrations in one look cannot rely on that look.

Per sample, the same for all of its images (both cameras, both frames of the 1 s memory, so that the two frames of a
camera differ only as the scene did and the change looks like one change of the world):
  - a shift of up to SHIFT pixels each way (the image padded by repeating its border, then cropped back);
  - brightness x U(0.8, 1.2), contrast about the image's mean x U(0.8, 1.2);
  - white balance: each channel x U(0.9, 1.1), a moderate tint that keeps a red flower red and a yellow card yellow;
  - with probability 1/2, a random overlay: a smooth random colour field (a 2 x 2 to 8 x 8 grid of colours drawn
    uniformly, interpolated to the image) blended over the whole image with alpha U(0, 0.25).
Nothing is erased or cut out: small parts (buds, ties, the label) stay visible. The images are in the simple_vlm
processor's range ((x - 0.5) / 0.5 for x in 0..1) on the way in and out.
"""

import torch
import torch.nn.functional as F

SHIFT = 8  # pixels at 224
BRIGHTNESS = (0.8, 1.2)
CONTRAST = (0.8, 1.2)
WHITE_BALANCE = (0.9, 1.1)
OVERLAY_P = 0.5
OVERLAY_ALPHA = 0.25
OVERLAY_GRID = (2, 8)


def _uniform(shape, bounds, like):
    low, high = bounds
    return torch.empty(shape, device=like.device, dtype=torch.float32).uniform_(low, high)


def augment_images(pixel_values: torch.Tensor) -> torch.Tensor:
    """pixel_values [B, N, 3, H, W] (processor range) -> the same shape, augmented per sample."""
    b, n, c, h, w = pixel_values.shape
    x = pixel_values.float() * 0.5 + 0.5
    # shift: one offset per sample for all of its images
    pad = SHIFT * h // 224
    if pad > 0:
        padded = F.pad(x.flatten(0, 1), (pad, pad, pad, pad), mode="replicate").unflatten(0, (b, n))
        dx = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
        dy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
        rows = (
            (dy[:, None] + torch.arange(h, device=x.device)[None]).view(b, 1, 1, h, 1).expand(b, n, c, h, w + 2 * pad)
        )
        x = padded.gather(3, rows)
        cols = (dx[:, None] + torch.arange(w, device=x.device)[None]).view(b, 1, 1, 1, w).expand(b, n, c, h, w)
        x = x.gather(4, cols)
    # photometric: brightness, contrast, white balance
    x = x * _uniform((b, 1, 1, 1, 1), BRIGHTNESS, x)
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    x = (x - mean) * _uniform((b, 1, 1, 1, 1), CONTRAST, x) + mean
    x = x * _uniform((b, 1, c, 1, 1), WHITE_BALANCE, x)
    # overlay: a smooth random colour field, the same for all images of a sample
    cells = int(torch.randint(OVERLAY_GRID[0], OVERLAY_GRID[1] + 1, ()).item())
    field = F.interpolate(
        torch.rand(b, c, cells, cells, device=x.device), size=(h, w), mode="bilinear", align_corners=False
    )
    alpha = _uniform((b, 1, 1, 1, 1), (0.0, OVERLAY_ALPHA), x)
    alpha = alpha * (torch.rand(b, 1, 1, 1, 1, device=x.device) < OVERLAY_P)
    x = (1 - alpha) * x + alpha * field[:, None]
    return ((x.clamp(0, 1) - 0.5) / 0.5).to(pixel_values.dtype)
