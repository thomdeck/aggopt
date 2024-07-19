"""
Copyright 2024 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

import numpy as np
import torch
from skimage.segmentation import felzenszwalb, slic, quickshift
from torchvision import transforms
from typing import Union, Tuple
import torch
from PIL import Image


def prepare_input(
    path: str,
    transforms: transforms = None,
    grayscale: bool = False,
    return_numpy=False,
) -> Tuple[np.ndarray, Union[np.ndarray, torch.Tensor]]:
    """Function to prepare the input

    Args:
        path (str): path to image
        grayscale (bool): load image as grayscale or RGB. Defaults to False.
        transfroms (transforms, optional): Torchvision callable transform function. Defaults to None.
        return_numpy (bool, optional): Whether to return prepared input as numpy or tensor. Defaults to False.

    Returns:
        Union[np.ndarray, torch.Tensor]: image of shape (height, width, channel) as numpy array and prepared input
            of shape (1, channel, height, width) after applying transforms
    """
    image = Image.open(path)
    if grayscale:
        image = image.convert("L")
    else:
        image = image.convert("RGB")
    org_image = np.array(image)
    if transforms:
        image = transforms(image)
    if not isinstance(image, torch.Tensor):
        image = np.array(image)
        image = torch.from_numpy(image)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    elif len(image.shape) == 2:
        image = (image.unsqueeze(0)).unsqueeze(0)
    if image.shape[-1] == 3 or image.shape[-1] == 1:
        image = image.permute(0, 3, 1, 2)
    if return_numpy:
        image = image.cpu().numpy()
    return org_image, image


def get_segments(x, algo="slic", n_segments=50, max_dist=5, size=16, kernel_size=5):

    def get_squares(img, size=32):
        h, w = img.shape[:2]
        segments = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                segments[i, j] = (i // size) * size + j // size
        return segments.astype(int)

    if type(x) == torch.Tensor:
        x = x.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    if algo == "slic":
        segments = slic(
            x, n_segments=n_segments, compactness=10, sigma=1, start_label=0
        )
    elif algo == "fz":
        segments = felzenszwalb(x, scale=100, sigma=0.5, min_size=150)
    elif algo == "quickshift":
        segments = quickshift(x, kernel_size=kernel_size, max_dist=max_dist, ratio=0.1)
    elif algo == "square":
        segments = get_squares(x, size=size)
    else:
        raise ValueError("unknown algo")

    return segments
