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

from src.base_explainer import explainer
from src.model import Model
import torch
import numpy as np
from typing import Union, Dict
import scipy.ndimage as ndi
from captum.attr import (
    Saliency,
    GuidedBackprop,
    IntegratedGradients,
    NoiseTunnel,
    InputXGradient,
)


class saliency_explainer(explainer):

    def __init__(
        self,
        model: Model,
        smooth: bool = True,
        abs: bool = True,
        normalize: bool = True,
    ) -> None:
        self.model = model
        if self.model.is_torchmodel():
            self.model = self.model.model
        self.smooth = smooth
        self.abs = abs
        self.normalize = normalize
        self.explainer = Saliency(self.model)
        super().__init__(model, smooth, abs, normalize)

    def explain(
        self, inputs: Union[torch.Tensor, np.ndarray], target: int, **kwargs
    ) -> Dict[str, np.ndarray]:
        
        inputs, target = self.preprocess_input(inputs, target)

        explanation = self.explainer.attribute(inputs, target=target, **kwargs)
        explanation = explanation.sum(axis=1).squeeze().cpu().detach().numpy()   

        explanation = self.postprocess_explanation(explanation)

        self.explanation = explanation
        
        return {"saliency": explanation}


class guidedbackprop_explainer(explainer):

    def __init__(
        self,
        model: Model,
        smooth: bool = True,
        abs: bool = True,
        normalize: bool = True,
    ) -> None:
        self.model = model
        if self.model.is_torchmodel():
            self.model = self.model.model
        self.smooth = smooth
        self.abs = abs
        self.normalize = normalize
        self.explainer = GuidedBackprop(self.model)
        super().__init__(model, smooth, abs, normalize)

    def explain(
        self, inputs: Union[torch.Tensor, np.ndarray], target: int, **kwargs
    ) -> Dict[str, np.ndarray]:
        
        inputs, target = self.preprocess_input(inputs, target)

        explanation = self.explainer.attribute(inputs, target=target, **kwargs)
        explanation = explanation.sum(axis=1).squeeze().cpu().detach().numpy()   

        explanation = self.postprocess_explanation(explanation)

        self.explanation = explanation
        return {"Guided BP": explanation}


class intgrad_explainer(explainer):

    def __init__(
        self,
        model: Model,
        smooth: bool = True,
        abs: bool = True,
        normalize: bool = True,
    ) -> None:
        self.model = model
        if self.model.is_torchmodel():
            self.model = self.model.model
        self.smooth = smooth
        self.abs = abs
        self.normalize = normalize
        self.explainer = IntegratedGradients(self.model, multiply_by_inputs=True)
        super().__init__(model, smooth, abs, normalize)

    def explain(
        self, inputs: Union[torch.Tensor, np.ndarray], target: int, **kwargs
    ) -> Dict[str, np.ndarray]:
        
        inputs, target = self.preprocess_input(inputs, target)

        explanation = self.explainer.attribute(inputs, target=target, **kwargs)
        explanation = explanation.sum(axis=1).squeeze().cpu().detach().numpy()            

        explanation = self.postprocess_explanation(explanation)
        self.explanation = explanation

        return {"IntGrad": explanation}


class smoothgrad_explainer(explainer):

    def __init__(
        self,
        model: Model,
        smooth: bool = True,
        abs: bool = True,
        normalize: bool = True,
    ) -> None:
        self.model = model
        if self.model.is_torchmodel():
            self.model = self.model.model
        self.smooth = smooth
        self.abs = abs
        self.normalize = normalize
        self.explainer = NoiseTunnel(Saliency(self.model))
        super().__init__(model, smooth, abs, normalize)

    def explain(
        self, inputs: Union[torch.Tensor, np.ndarray], target: int, **kwargs
    ) -> Dict[str, np.ndarray]:
        
        inputs, target = self.preprocess_input(inputs, target)

        explanation = self.explainer.attribute(inputs, target=target,nt_type='smoothgrad_sq', nt_samples=20, stdevs=0.5, abs=False,**kwargs)
        explanation = explanation.sum(axis=1).squeeze().cpu().detach().numpy()

        explanation = self.postprocess_explanation(explanation)
        self.explanation = explanation

        return {"SmoothGrad": explanation}
    

class vargrad_explainer(explainer):

    def __init__(
        self,
        model: Model,
        smooth: bool = True,
        abs: bool = True,
        normalize: bool = True,
    ) -> None:
        self.model = model
        if self.model.is_torchmodel():
            self.model = self.model.model
        self.smooth = smooth
        self.abs = abs
        self.normalize = normalize
        self.explainer = NoiseTunnel(Saliency(self.model))
        super().__init__(model, smooth, abs, normalize)

    def explain(
        self, inputs: Union[torch.Tensor, np.ndarray], target: int, **kwargs
    ) -> Dict[str, np.ndarray]:

        inputs, target = self.preprocess_input(inputs, target)

        explanation = self.explainer.attribute(inputs, target=target,nt_type='smoothgrad_sq', nt_samples=20, stdevs=0.5, abs=False, **kwargs)
        explanation = explanation.sum(axis=1).squeeze().cpu().detach().numpy()   

        explanation = self.postprocess_explanation(explanation)
        self.explanation = explanation

        return {"VarGrad": explanation}


class inputxgrad_explainer(explainer):

    def __init__(
        self,
        model: Model,
        smooth: bool = True,
        abs: bool = True,
        normalize: bool = True,
    ) -> None:
        self.model = model
        if self.model.is_torchmodel():
            self.model = self.model.model
        self.smooth = smooth
        self.abs = abs
        self.normalize = normalize
        self.explainer = InputXGradient(self.model)
        super().__init__(model, smooth, abs, normalize)

    def explain(
        self, inputs: Union[torch.Tensor, np.ndarray], target: int, **kwargs
    ) -> Dict[str, np.ndarray]:

        inputs, target = self.preprocess_input(inputs, target)

        explanation = self.explainer.attribute(inputs, target=target, **kwargs)
        explanation = explanation.sum(axis=1).squeeze().cpu().detach().numpy()   

        explanation = self.postprocess_explanation(explanation)

        self.explanation = explanation
        return {"InputxGrad": explanation}
