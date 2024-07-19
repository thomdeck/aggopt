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

import matplotlib.pyplot as plt
import torch
import matplotlib
from typing import Union, Dict, Tuple, List
import numpy as np
import math


def plot_explanations(
    inputs: Union[torch.Tensor, np.ndarray],
    explanations: Dict[str, np.ndarray],
    label: str,
    cmap: matplotlib.colors.ListedColormap = "jet",
    overlay: bool = True,
    figsize: Tuple[int, int] = (15,15),
) -> None:
    """Function to plot the explanations

    Args:
        inputs (Union[torch.Tensor, np.ndarray]): Input image of shape (1, channel, height, width)
            or (channel, height, width) or (height, width, channel) or (1, height, width, channel)
        explanations (Dict[str, np.ndarray]): Explanations dict containing names of explainers and their respective explanations.
        label (str): Target class name that is being explained
        cmap (matplotlib.colors.ListedColormap, optional): Color map for matplotlib. Defaults to "jet".
        overlay (bool, optional): Whether to overlay explanations on top of image. Defaults to True.
        figsize (Tuple[int, int], optional): Fig size for matplotlib. Defaults to (15, 15).
    """
    if torch.is_tensor(inputs):
        inputs = inputs.cpu().numpy()
    inputs = np.squeeze(inputs)

    inputs = denormalize_image(inputs)

    if inputs.shape[0] == 3:
        inputs = np.transpose(inputs, (1,2,0))

    N = len(explanations)
    total_rows = int(math.ceil(N / 4))
    total_col = 5
    fig, axs = plt.subplots(total_rows, total_col, figsize=figsize)
    if len(axs.shape) == 1:
        axs = axs.reshape(1, -1)
    axs[0, 0].imshow(inputs)
    axs[0, 0].set_title(f"{label}")
    axs[0, 0].set_axis_off()
    counter = 0
    temp_explanation = [(key, value) for key, value in explanations.items()]
    for i in range(total_rows):
        if i != 0:
            fig.delaxes(axs[i, 0])

    for i in range(total_rows):
        for j in range(1, total_col):
            if counter >= len(temp_explanation):
                fig.delaxes(axs[i, j])
                counter += 1
                continue
            if overlay:
                axs[i, j].imshow(inputs, cmap=cmap)
                axs[i, j].imshow(temp_explanation[counter][1], alpha=0.7, cmap=cmap)
            else:
                axs[i, j].imshow(temp_explanation[counter][1], cmap=cmap)
            axs[i, j].set_axis_off()
            axs[i, j].set_title(f"{temp_explanation[counter][0]}")
            counter += 1

    plt.tight_layout()
    plt.show()

def plot_agg_explanations(
    inputs: Union[torch.Tensor, np.ndarray],
    explanations: Dict[str, np.ndarray],
    label: str,
    cmap: matplotlib.colors.ListedColormap = "jet",
    overlay: bool = True,
    figsize: Tuple[int, int] = (15,5),
    weights: List[float] = [],
) -> None:
    """Function to plot the explanations

    Args:
        inputs (Union[torch.Tensor, np.ndarray]): Input image of shape (1, channel, height, width)
            or (channel, height, width) or (height, width, channel) or (1, height, width, channel)
        explanations (Dict[str, np.ndarray]): Explanations dict containing names of explainers and their respective explanations.
        label (str): Target class name that is being explained
        cmap (matplotlib.colors.ListedColormap, optional): Color map for matplotlib. Defaults to "jet".
        overlay (bool, optional): Whether to overlay explanations on top of image. Defaults to True.
        figsize (Tuple[int, int], optional): Fig size for matplotlib. Defaults to (25, 8).
        weights (List[float], optional): Weights for the explanations. Defaults to [].
    """
    if torch.is_tensor(inputs):
        inputs = inputs.cpu().numpy()
    inputs = np.squeeze(inputs)

    inputs = denormalize_image(inputs)

    if inputs.shape[0] == 3:
        image = np.transpose(inputs, (1,2,0))

    fig, axs = plt.subplots(1, len(explanations)+1,figsize=figsize)
    axs[0].imshow(image)
    axs[0].set_title(f"{label}")
    axs[0].set_axis_off()

    for i, key in enumerate(explanations.keys()):
        if overlay:
            axs[i+1].imshow(image, cmap=cmap)
            axs[i+1].imshow(explanations[key], alpha=0.7, cmap=cmap)
            axs[i+1].set_axis_off()
        else: # No overlay
            axs[i+1].imshow(explanations[key], cmap=cmap)
            axs[i+1].set_axis_off()
        if (i < len(weights) ) and (weights != []):
            axs[i+1].set_title(f"{key}"+r" ($\omega$: " f'{weights[i]:.2f})')
            axs[i+1].set_axis_off()
        else:
            axs[i+1].set_title(f"{key}")
            axs[i+1].set_axis_off()
            
    plt.tight_layout()
    plt.show()



def denormalize_image(
    image,
    mean=torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1),
    std=torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1),
    **params,
):
    """De-normalize a torch image."""
    if isinstance(image, torch.Tensor):
        return (
            image.view(
                [
                    params.get("nr_channels", 3),
                    params.get("img_size", 224),
                    params.get("img_size", 224),
                ]
            )
            * std
        ) + mean
    elif isinstance(image, np.ndarray):
        std
        return (image * std.numpy()) + mean.numpy()
    else:
        print("Make image either a np.array or torch.Tensor before denormalizing.")
        return image