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

from src.model import Model
import numpy as np
import torch
from typing import Union, Dict, List, Tuple
from abc import ABC
import scipy.ndimage as ndi

class explainer(ABC):
    """Abstract explainer class"""

    def __init__(
        self,
        model: Model,
        smooth: bool = True,
        abs: bool = True,
        normalize: bool = True,
    ) -> None:
        """

        Args:
            model (Model): Pytorch model or callable function wrapped in the model class
            smooth (bool, optional): Whether to smooth the explanation. Defaults to True.
            abs (bool, optional): Whether to take absolute value of the explanation. Defaults to True.
            normalize (bool, optional): Whether to normalize the explanation. Defaults to True.
        """
        self.model = model
        self.smooth = smooth
        self.abs = abs
        self.normalize = normalize
        self.explanation = None

    def explain(
        self, inputs: Union[torch.Tensor, np.ndarray], target: int, **kwargs
    ) -> Dict[str, np.ndarray]:
        """Function that explains the target class given input

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): input of shape (1, channel, height, width)
            target (int): Target class

        Returns:
            Dict[str, np.ndarray]: Explanation of shape (height, width)
        """

    def get_explanation(self) -> np.ndarray:
        """Function to get the stored explanation after explain method is called

        Returns:
            np.ndarray: Explanation of shape (height, width)
        """
        return self.explanation
    
    def preprocess_input(self, inputs: Union[torch.Tensor, np.ndarray], target_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function to prepare input for the model"""

        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        inputs.to(self.model.torch_device)
        target = torch.tensor(target_idx).to(dtype=torch.long).to(self.model.torch_device)

        return inputs, target
    
    def normalize_explanation(
        self,
        explanation: Union[torch.Tensor, np.ndarray],
        mode: str = "min-max",
    ) -> Union[torch.Tensor, np.ndarray]:
        """Function to normalize the explanation

        Args:
            explanation (Union[torch.Tensor, np.ndarray]): explanation of shape (1, height, width)
            mode (str, optional): How to normalize the explanation? Either "scale_negative" or "positive_only"
            or "min-max" with absolute value first. Defaults to "min-max".

        Returns:
            Union[torch.Tensor, np.ndarray]: Normalized explanation of shape (1, height, width)
        """

        explanation_type_tensor = torch.is_tensor(explanation)
        
        if explanation_type_tensor:
            explanation = explanation.detach().cpu().numpy()
        
        if mode == "scale_negative":
            e_max = explanation.max()
            e_min = explanation.min()

            if e_max == e_min:
                if explanation_type_tensor:
                    explanation = torch.from_numpy(explanation).to(self.model.torch_device)
                return explanation

            explanation[explanation > 0] /= e_max
            explanation[explanation < 0] /= -e_min

        elif mode == "positive_only":
            # Only keep positive values and normalize [0,1]
            explanation = np.maximum(explanation, 0)
            explanation = (explanation - explanation.min()) / (explanation.max() - explanation.min())
        else:
            explanation = np.abs(explanation)
            explanation = (explanation - explanation.min()) / (explanation.max() - explanation.min())
        
        if explanation_type_tensor:
            return torch.from_numpy(explanation).to(self.model.torch_device)
        
        return explanation
    
    def postprocess_explanation(self, explanation: np.ndarray
    ) -> np.ndarray:
        """Function to postprocess the explanation

        Args:
            explanation (np.ndarray): Explanation of shape (height, width)

        Returns:
            np.ndarray: Postprocessed explanation of shape (height, width)
        """
        if self.normalize:
            explanation = self.normalize_explanation(explanation)
        if self.abs:
            if torch.is_tensor(explanation):
                explanation = explanation.abs()
            else:
                explanation = np.abs(explanation)
        if self.smooth:
            if torch.is_tensor(explanation):
                explanation = explanation.cpu().numpy()

            explanation = ndi.gaussian_filter(explanation, sigma=(1, 1))

            if torch.is_tensor(explanation):
                explanation = torch.from_numpy(explanation).to(self.model.torch_device)
        
        return explanation


class agg_explainer(explainer):
    """Abstract class for an explainer that aggregates multiple explanations"""

    def __init__(
        self,
        model: Model,
        explainers: List[explainer] = None,
        explainers_kwargs: List[Dict] = None,
        smooth: bool = True,
        abs: bool = True,
        normalize: bool = True,
    ) -> None:
        """

        Args:
            model (Model): Pytorch model or callable function wrapped in the model class
            explainers (List[explainer]): List of explainers to combine
            explainers_kwargs (List[Dict]): List of kwargs that is needed to pass while calling explain function
                of above explainers.
            smooth (bool, optional): Whether to smooth the explanation. Defaults to True.
            abs (bool, optional): Whether to take absolute value of the explanation. Defaults to True.
            normalize (bool, optional): Whether to normalize the explanation. Defaults to True.
        """
        self.model = model
        self.smooth = smooth
        self.abs = abs
        self.normalize = normalize
        self.explainers = explainers
        self.explainers_kwargs = explainers_kwargs
        self.weights = None
        super().__init__(model, smooth, abs, normalize)

    def explain(
        self, inputs: Union[torch.Tensor, np.ndarray], target: int, **kwargs
    ) -> Dict[str, np.ndarray]:
        """Function that explains the target class given input

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): input of shape (1, channel, height, width)
            target (int): Target class

        Returns:
            Dict[str, np.ndarray]: Explanation of shape (height, width)
        """
        pass

    def compute_individual_explanations(
        self, inputs: Union[torch.Tensor, np.ndarray], target: int, store: bool = True
    ) -> List[np.ndarray]:
        """Function to compute individual explanations

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): input of shape (1, channel, height, width)
            target (int): Target class

        Returns:
            List[np.ndarray]: List of explanations of shape (height, width)
        """
        explanations_names = []
        explanations_list = []

        inputs, target = self.preprocess_input(inputs, target)

        for i, explainer in enumerate(self.explainers):
            exp = explainer.explain(inputs, target, **self.explainers_kwargs[i])
            for key, value in exp.items():
                explanations_names.append(key)
                explanations_list.append(value)
        
        if store:   
            self.explanations = {name: exp for name, exp in zip(explanations_names, explanations_list)}

        return np.stack(explanations_list, axis=0)
    
    def compute_weights(
            self, explanations: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Function to compute weights for the aggregated explanation

        Args:
            explanations (Union[torch.Tensor, np.ndarray]): Explanations of shape (num_explanations, height, width)

        Returns:
            Dict[str, np.ndarray]: Weights of shape (num_explanations, 1)
        """
        pass

    def get_weights(self) -> np.ndarray:
        """Function to get the stored weights after compute_weights method is called

        Returns:
            np.ndarray: Weights of shape (num_explanations, 1)
        """
        return self.weights
    
  
    
