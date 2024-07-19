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

from typing import Any, Optional, Callable, Union
import numpy as np
import torch
from abc import ABC


class Model(ABC):
    """Wrapper for passed model or callable forward function"""

    def __init__(
        self,
        model: Union[torch.nn.Module, Callable[..., Any]],
        device: Optional[str] = "cpu",
    ) -> None:
        """

        Args:
            model (Union[torch.nn.Module, Callable[..., Any]]): Pytorch model or any other callable function
            device (Optional[str], optional): If passed model is Pytorch model, device can be set to GPU or CPU. Defaults to "cpu".
        """

        if not isinstance(model, torch.nn.Module):
            print(
                "Passed model is not a Pytorch model; gradient computation is not available."
            )
        else:
            self.torch_device = torch.device(device)
            model.to(self.torch_device)
        self.model = model

    def __call__(
        self,
        inputs: Union[torch.Tensor, np.ndarray],
        target: Optional[int] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): Input data of shape (batch_size, channel, height, width)
            target (Optional[int], optional): Target class. If none, all the probabilities will be returned. Defaults to None.

        Returns:
            Union[torch.Tensor, np.ndarray]: Class probabilities
        """

        return self.forward(inputs, target)

    def forward(
        self,
        inputs: Union[torch.Tensor, np.ndarray],
        target: Optional[int] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Forward computation function

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): Input data of shape (batch_size, channel, height, width)
            target (Optional[int], optional): Target class. If none, all the probabilities will be returned. Defaults to None.

        Returns:
            Union[torch.Tensor, np.ndarray]: Class probabilities
        """

        if isinstance(inputs, torch.Tensor) and self.is_torchmodel():
            inputs.to(self.torch_device)
            outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)

        if target is None:
            return outputs

        return outputs[:, target].reshape(inputs.shape[0], -1)

    def backward(
        self,
        inputs: torch.Tensor,
        target: int,
    ) -> torch.Tensor:
        """Gradient computation function

        Args:
            inputs (torch.Tensor): Input data of shape (batch_size, channel, height, width)
            target (int): Target class

        Raises:
            Exception: If gradient cannot be computated.

        Returns:
            torch.Tensor: Gradient with respect to the input.
        """

        if not self.is_torchmodel():
            raise Exception(
                "Passed model is not a Pytorch model; gradient computation is not available."
            )
        with torch.autograd.set_grad_enabled(True):
            inputs.requires_grad = True
            outputs = self.forward(inputs, target)
            grads = torch.autograd.grad(torch.unbind(outputs), inputs)
        return grads[0]

    def is_torchmodel(self) -> bool:
        """function to check if passed model is a Pytorch model

        Returns:
            bool: Returns True if passed model is a Pytorch model else returns False.
        """
        return isinstance(self.model, torch.nn.Module)
