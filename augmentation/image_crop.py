"""Crop selected decoded camera views; the native processor resizes afterwards."""
import math
from numbers import Real

import torch


class RandomCameraCrop:
    def __init__(self, fraction=.95, camera_probability=.5):
        for name, value, allow_zero in (('fraction', fraction, False),
                                        ('camera_probability', camera_probability, True)):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value > 1
                    or (value < 0 if allow_zero else value <= 0)):
                raise ValueError(f'Invalid crop {name}: {value}')
        self.fraction = float(fraction)
        self.camera_probability = float(camera_probability)

    def __call__(self, sample, image_keys, *, seed, training=True):
        output = dict(sample)
        trace = dict(applied=False, views={})
        if not training or self.camera_probability == 0 or self.fraction == 1:
            return output, trace
        generator = torch.Generator(device='cpu').manual_seed(seed)
        selected = torch.rand(len(image_keys), generator=generator) < self.camera_probability
        # Replay one relative crop location across selected cameras and time.
        # Mapping U[0,1) to an integer gives every valid crop offset equal odds.
        vertical, horizontal = torch.rand(2, generator=generator).tolist()
        for index, key in enumerate(image_keys):
            if not selected[index]:
                continue
            image = sample[key]
            if (not isinstance(image, torch.Tensor) or image.device.type != 'cpu'
                    or image.ndim not in (3, 4) or image.shape[-3] != 3
                    or min(image.shape[-2:]) < 1
                    or (image.dtype != torch.uint8 and not image.is_floating_point())):
                raise ValueError('Crop expects decoded CPU RGB (C,H,W) or (T,C,H,W)')
            if image.is_floating_point() and (not torch.isfinite(image).all()
                                            or image.min() < 0 or image.max() > 1):
                raise ValueError('Crop floating-point images must be unnormalized RGB in [0,1]')
            height, width = image.shape[-2:]
            crop_height = max(1, int(height*self.fraction))
            crop_width = max(1, int(width*self.fraction))
            top = int(vertical*(height-crop_height+1))
            left = int(horizontal*(width-crop_width+1))
            # Tiny images can round to their full original dimensions.
            if (crop_height, crop_width) == (height, width):
                continue
            output[key] = image[..., top:top+crop_height, left:left+crop_width].clone()
            trace['views'][key] = dict(top=top, left=left, height=crop_height,
                                       width=crop_width, input_height=height, input_width=width)
        trace['applied'] = bool(trace['views'])
        return output, trace
