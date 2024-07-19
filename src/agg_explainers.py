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

from src.base_explainer import agg_explainer
from src.base_explainer import explainer
from src.model import Model
import torch
import numpy as np
from typing import Union, List, Dict, Tuple
import cvxpy as cp



class AGGmean_explainer(agg_explainer):
    """AGG mean explainer from the paper"""

    def __init__(
        self,
        model: Model,
        explainers: List[explainer],
        explainers_kwargs: List[Dict],
        smooth: bool = False,
        abs: bool = False,
        normalize: bool = False,
    ) -> None:
        """

        Args:
            model (Model): Pytorch model or callable function wrapped in the model class
            explainers (List[explainer]): List of explainers to combine
            explainers_kwargs (List[Dict]): List of kwargs that is needed to pass while calling explain function
                of above explainers.
            smooth (bool, optional): Whether to smooth the explanation. Defaults to False.
            abs (bool, optional): Whether to take absolute value of the explanation. Defaults to False.
            normalize (bool, optional): Whether to normalize the explanation. Defaults to False.
        """
  
        super().__init__(model,explainers, explainers_kwargs, smooth, abs, normalize)
        self.name = "AGGmean"

    
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
        explanation = self.compute_individual_explanations(inputs, target)

        weights = self.compute_weights()

        explanation_agg = np.average(explanation, weights = weights, axis=0)
        
        explanation_agg = self.postprocess_explanation(explanation_agg)
        
        self.weights = np.ones(len(self.explainers)) / len(self.explainers)

        return {self.name: explanation_agg}
    
    def compute_weights(self,
    ) -> np.ndarray:
        "Function to compute weights of the explanation"
        self.weights = np.ones(len(self.explainers)) / len(self.explainers)
        return self.weights
            

class AGGvar_explainer(agg_explainer):
    """AGG var explainer from the paper"""

    def __init__(
        self,
        model: Model,
        explainers: List[explainer],
        explainers_kwargs: List[Dict],
        smooth: bool = False,
        abs: bool = False,
        normalize: bool = False,
    ) -> None:
        """

        Args:
            model (Model): Pytorch model or callable function wrapped in the model class
            explainers (List[explainer]): List of explainers to combine
            explainers_kwargs (List[Dict]): List of kwargs that is needed to pass while calling explain function
                of above explainers.
            smooth (bool, optional): Whether to smooth the explanation. Defaults to False.
            abs (bool, optional): Whether to take absolute value of the explanation. Defaults to False.
            normalize (bool, optional): Whether to normalize the explanation. Defaults to False.
        """
  
        super().__init__(model,explainers, explainers_kwargs, smooth, abs, normalize)
        self.name = "AGGvar"

    
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
        explanation = self.compute_individual_explanations(inputs, target)

        weights = self.compute_weights()

        explanation_agg = np.sum(explanation * weights, axis=0)

        explanation_agg = self.postprocess_explanation(explanation_agg)

        return {self.name: explanation_agg}
    
    def compute_weights(self,
    ) -> np.ndarray:
        "Function to compute weights of the explanation"

        explanation = np.stack([self.explanations[key] for key in self.explanations])

        variance = np.var(explanation, axis=0, ddof=1, keepdims=True)

        weights = 1 / (variance + 1 * 0.226) * (1 / len(self.explainers))

        self.weights = np.squeeze(weights, axis=0)

        return weights
    

class AGGrobust_explainer(agg_explainer):
    """AGG robust explainer from the paper https://arxiv.org/pdf/2406.05090"""

    def __init__(
        self,
        model: Model,
        explainers: List[explainer],
        explainers_kwargs: List[Dict],
        mode: str = "uniform",
        noise_lvl: float = 0.1,
        num_samples: int = 50,
        smooth: bool = False,
        abs: bool = False,
        normalize: bool = False,
    ) -> None:
        """

        Args:
            model (Model): Pytorch model or callable function wrapped in the model class
            explainers (List[explainer]): List of explainers to combine
            explainers_kwargs (List[Dict]): List of kwargs that is needed to pass while calling explain function
                of above explainers.
            mode (str, optional): Mode of perturbation to compute average sensitivity. Defaults to "uniform".
            noise_lvl (float, optional): Noise level of perturbation to compute average sensitivity. Defaults to 0.1.
            num_samples (int, optional): Number of samples for perturbations to optimize for average sensitiviy. Defaults to 50.
            smooth (bool, optional): Whether to smooth the explanation. Defaults to False.
            abs (bool, optional): Whether to take absolute value of the explanation. Defaults to False.
            normalize (bool, optional): Whether to normalize the explanation. Defaults to False.
        """
        super().__init__(model,explainers, explainers_kwargs, smooth, abs, normalize)
        self.explainers = explainers
        self.explainers_kwargs = explainers_kwargs
        self.mode = mode
        self.noise_lvl = noise_lvl
        self.num_samples = num_samples
        self.name = "AGGrobust"

    def compute_weights(self, explanations : np.array, gamma2: np.ndarray) -> np.ndarray:
        """Function to compute weights for the aggreagted explanation based on the average sensitivity as generalized L2 metric

        Args:
            explanations (np.ndarray): Explanation of shape (1, height, width, num_explainers)
            gamma2 (np.ndarray): Perturbed Explanation of shape (num_samples, height * width, num_explainers)
                This corresponds to the gamma2 parameter in the paper to obtain Average Sensitivit as L2 metric

        Returns:
            np.ndarray: Weights of shape (num_explainers)
        """
        # get number of explainer methods
        k = explanations.shape[-1]

        #Compute parameter matrix Gamma for considered L2 metric
        Gamma = explanations - gamma2
        Gamma = Gamma.transpose(0, 2, 1) @ Gamma
        Gamma = Gamma.mean(axis=0)
   
        x = cp.Variable(k)
        prob = cp.Problem(
            cp.Minimize((1 / 2) * cp.quad_form(x, Gamma)),
            [x >= 0, np.ones((1, k)) @ x == 1],
        )
        prob.solve()

        weights = np.maximum(x.value,0)
        self.weights = weights

        return weights


    def _perturb_fn_robust(self, inputs: Union[torch.Tensor, np.ndarray] 
    ) -> Union [torch. Tensor, np.ndarray]:

        """Function to perturb samples for robustness optimization

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): Input of shape (1, channel, height, width)

        Returns:
            Union[torch.Tensor, np.ndarray]: Perturbed samples of shape (num_samples, channel, height, width)
        """
        inputs_isnumpy = isinstance(input, np.ndarray)
        if inputs_isnumpy:
            inputs = torch.from_numpy(inputs)
        shape_expanded = [self.num_samples] + list(inputs.shape[1:])
        device = inputs.device
        if self.mode == "uniform":
            noise = (
                torch.zeros(size=shape_expanded)
                .uniform_(-self.noise_lvl, self.noise_lvl)
                .to(device)
            )

        elif self.mode == "gaussian":
            noise = (
                torch.zeros(size=shape_expanded).normal_(0, self.noise_lvl).to(device)
            )
        perturbed_inputs = inputs + noise

        if inputs_isnumpy:
            perturbed_inputs = perturbed_inputs.cpu().numpy()

        return perturbed_inputs


    def explain(self, inputs: Union[torch.Tensor, np.ndarray], target: int, **kwargs) -> Dict[str, np.ndarray]:
        """Function that explains the target class prediction of a given input 

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): input of shape (1, channel, height, width)
            target (int): Target class

        Returns:
            Dict[str, np.ndarray]: Explanation of shape (height, width)
        """
        explanation = self.compute_individual_explanations(inputs, target)

        # perturb inputs
        inputs_pert = self._perturb_fn_robust(inputs)

        # compute explanations for perturbed inputs
        explanations_pert =[self.compute_individual_explanations(x.unsqueeze(0), target, store=False) for x in inputs_pert]
        explanations_pert = np.stack(explanations_pert, axis=0)

        #compute optimal weights
        d = explanation.shape[1]*explanation.shape[2]

        weights = self.compute_weights(
            explanation
            .transpose(1,2,0)
            .reshape(1, d, -1),
            gamma2=explanations_pert
            .reshape(-1,explanations_pert.shape[1], explanations_pert.shape[2] * explanations_pert.shape[3])
            .transpose(0,2,1)
        )

        explanation_agg = np.average(explanation, weights = weights, axis=0)

        explanation_agg = self.postprocess_explanation(explanation_agg)

        return {self.name: explanation_agg}
    


class AGGfaith_explainer(agg_explainer):
    """AGG faith explainer from the paper https://arxiv.org/pdf/2406.05090"""

    def __init__(
        self,
        model: Model,
        explainers: List[explainer],
        explainers_kwargs: List[Dict],
        baseline: Union[torch.Tensor, np.ndarray],
        segments: Union[torch.Tensor, np.ndarray]= None,
        share: int = 0.2,
        num_samples: int = 100,
        batch_size: int = 10,
        smooth: bool = False,
        abs: bool = False,
        normalize: bool = False,
    ) -> None:
        """

        Args:
            model (Model): Pytorch model or callable function wrapped in the model class
            explainers (List[explainer]): List of explainers to combine
            explainers_kwargs (List[Dict]): List of kwargs that is needed to pass while calling explain function
                of above explainers.
            baseline (Union[torch.Tensor, np.ndarray]): Baseline used to perturb of shape (1, channel, height, width)
            segments (Union[torch.Tensor, np.ndarray], optional): Segments used to perturb of shape (1, height, width). Defaults to None.
            share (float, optional): Share of features to perturb. Defaults to 0.2.
            num_samples (int, optional): Number of samples to use. Defaults to 100.
            batch_size (int, optional): Batch size to compute prediction changes due to perturbation. Defaults to 10.
            smooth (bool, optional): Whether to smooth the explanation. Defaults to False.
            abs (bool, optional): Whether to take absolute value of the explanation. Defaults to False.
            normalize (bool, optional): Whether to normalize the explanation. Defaults to False.
        """
        super().__init__(model,explainers, explainers_kwargs, smooth, abs, normalize)
        self.explainers = explainers
        self.explainers_kwargs = explainers_kwargs
        self.baseline = baseline
        self.num_samples = num_samples
        self.segments = segments
        self.share = share
        self.batch_size = batch_size

        self.name = "AGGfaith"

    def compute_weights(self, explanations: np.array, gamma1: np.array,  gamma2: np.ndarray
    ) -> np.ndarray:
        """Function to compute weights for the aggregated explanation based on Infidelity as generalized L2 metric

        Args:
            explanations (np.ndarray): Explanation of shape (1, height*width, num_explainers)
            gamma1 (np.ndarray): Infidelity perturbation (batch_size, num_samples//batch_size, height*width)
                This corresponds to the gamma1 parameter in the paper to obtain Infidelity as L2 metric
            gamma2 (np.ndarray): Prediction change due to perturbation (batch_size, num_samples//batch_size, 1)
                This corresponds to the gamma2 parameter in the paper to obtain Infidelity as L2 metric

        Returns:
            np.ndarray: Weights of shape (num_explainers)
        """
        # get number of explainer methods
        k = explanations.shape[-1]

        Gamma = gamma1 @ explanations

        # normalize Gamma like proposed in the infidelity paper Yeh et al. 2019
        Gamma *= np.mean(gamma2*Gamma, axis=0, keepdims=True) / np.mean(Gamma*Gamma, axis=0, keepdims=True)

        Gamma= Gamma - gamma2
   
        Gamma = Gamma.transpose(0,2,1) @ Gamma
        Gamma = Gamma.mean(axis=0)
   
        x = cp.Variable(k)
        prob = cp.Problem(
            cp.Minimize((1 / 2) * cp.quad_form(x, Gamma)),
            [x >= 0, np.ones((1, k)) @ x == 1],
        )
        prob.solve()

        weights = np.maximum(x.value,0)
        self.weights = weights

        return weights
    
    def _perturb_by_baseline(self, x, segments=None, baseline=None, share=0.2):

        if len(x.shape) == 4:
            x = x.squeeze(0)

        if segments is None:
            segments = np.arange(x.shape[1] * x.shape[2]).reshape(x.shape[1], x.shape[2])

        mask = np.repeat(np.expand_dims(segments, axis=0), 3, axis=0)
        idx = np.unique(mask)

        if baseline is None:
            baseline = torch.zeros_like(x)

        if int(idx.size * (share)) == 0:
            return x
        else:
            choice = np.random.choice(idx, int(idx.size * (share)), replace=False)

        mask = torch.from_numpy(mask).long()

        x_pert = x.clone()

        flat_mask = mask.reshape(-1)

        # Find the flat indices where the segment IDs match the chosen ones
        chosen_indices = np.nonzero(np.isin(flat_mask, choice))[0]

        # Convert flat indices back to the original shape
        chosen_mask = np.unravel_index(chosen_indices, mask.shape)

        # Replace the selected values with the baseline values
        x_pert[chosen_mask] = baseline[chosen_mask]

        return x_pert


    def _perturb_fn_infid(self, inputs: Union[torch.Tensor, np.ndarray] 
    ) -> Union [torch. Tensor, np.ndarray]:

        """Function to compute perturbation

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): Input of shape (1, channel, height, width)

        Returns:
            Union[torch.Tensor, np.ndarray]: Perturbed samples of shape (num_samples, channel, height, width)
        """
        
        inputs_isnumpy = isinstance(input, np.ndarray)
        if inputs_isnumpy:
            inputs = torch.from_numpy(inputs)

        inputs_pert = []
        for _ in range(self.batch_size):

            perturbed_inputs = []

            for _ in range(self.num_samples//self.batch_size):

                perturbed_inputs.append(
                    self._perturb_by_baseline(
                        inputs, self.segments, baseline=self.baseline, share=self.share
                    )
                )

            perturbed_inputs = torch.stack(perturbed_inputs)
            inputs_pert.append(perturbed_inputs)

        inputs_pert = torch.stack(inputs_pert, axis=0)

        if inputs_isnumpy:
            inputs_pert = inputs_pert.cpu().numpy()
        
        return inputs_pert


    def explain(self, inputs: Union[torch.Tensor, np.ndarray], target: int, **kwargs
    ) -> Dict[str, np.ndarray]:
        """Function that explains the target class given input

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): input of shape (1, channel, height, width)
            target (int): Target class

        Returns:
            Dict[str, np.ndarray]: Explanation of shape (height, width)
        """

        explanation = self.compute_individual_explanations(inputs, target)

        # perturb inputs
        inputs_pert = self._perturb_fn_infid(inputs)

        pert = torch.stack(
            [(inputs.detach() != p.detach()).float() for p in inputs_pert], axis=0
        )
        pert = (
            pert.mean(axis=2)
            .reshape(inputs_pert.shape[0], inputs_pert.shape[1], -1)
            .cpu()
            .numpy()
        )  # shape: (batch_size, num_samples/batch_size, height*width)

        pred = self.model(inputs).detach().cpu().numpy()[:, target]
        pred_diff = np.stack(
            [
                pred - self.model(p).detach().cpu().numpy()[:, target]
                for p in inputs_pert
            ],
            axis=0,
        )  # shape: (batch_size, num_samples/batch_size)

        # compute number of features
        d = explanation.shape[1] * explanation.shape[2] 

        weights = self.compute_weights(
            explanation.transpose(1, 2, 0).reshape(1, d, -1),
            gamma1 = pert,
            gamma2 = pred_diff.reshape(inputs_pert.shape[0], -1, 1),
        )

        explanation_agg = np.average(explanation, weights = weights, axis=0)

        explanation_agg = self.postprocess_explanation(explanation_agg)

        return {self.name: explanation_agg}



class AGGopt_explainer(AGGfaith_explainer):
    """AGG opt explainer from the paper https://arxiv.org/pdf/2406.05090"""

    def __init__(
        self,
        model: Model,
        explainers: List[explainer],
        explainers_kwargs: List[Dict],
        baseline: Union[torch.Tensor, np.ndarray],
        segments: Union[torch.Tensor, np.ndarray]= None,
        mode: str = "uniform",
        noise_lvl: float = 0.1,
        num_samples_robust: int = 20,
        lambda_robust: int = 0.5,
        lambda_faith: int = 0.5,
        share: int = 0.25,
        num_samples_faith: int = 64,
        batch_size: int = 16,
        smooth: bool = False,
        abs: bool = False,
        normalize: bool = False,
    ) -> None:
        """

        Args:
            model (Model): Pytorch model or callable function wrapped in the model class
            explainers (List[explainer]): List of explainers to combine
            explainers_kwargs (List[Dict]): List of kwargs that is needed to pass while calling explain function
                of above explainers.
            mode (str, optional): Mode of perturbation to compute average sensitivity. Defaults to "uniform".
            noise_lvl (float, optional): Noise level of perturbation to compute average sensitivity. Defaults to 0.1.
            num_samples_robust (int, optional): Number of samples for perturbations to optimize for average sensitiviy. Defaults to 20.
            baseline (Union[torch.Tensor, np.ndarray]): Baseline used to perturb for infidelity of shape (1, channel, height, width)
            lambda_robust (int, optional): Additonal weightening factor for robustness during weight optimization. Defaults to 0.5.
            lambda_faith (int, optional): Additonal weightening factor for faithfulness during weight optimization. Defaults to 0.5.
            segments (Union[torch.Tensor, np.ndarray], optional): Segments used to perturb for infidelity of shape (1, height, width). Defaults to None.
            share (int, optional): Share of features to perturb. Defaults to 0.2.
            num_samples_faith (int, optional): Number of samples for perturbations to optimize for infidelity. Defaults to 64.
            batch_size (int, optional): Batch size to compute prediction changes due to perturbation. Defaults to 16.
            smooth (bool, optional): Whether to smooth the explanation. Defaults to False.
            abs (bool, optional): Whether to take absolute value of the explanation. Defaults to False.
            normalize (bool, optional): Whether to normalize the explanation. Defaults to False.
        """
        super().__init__(model,explainers, explainers_kwargs,baseline, segments, share, num_samples_faith, batch_size, smooth, abs, normalize)

        self.num_samples_robust = num_samples_robust
        self.mode = mode
        self.noise_lvl = noise_lvl
        self.lambda_robust = lambda_robust
        self.lambda_faith = lambda_faith
        self.name = "AGGopt"


    def _perturb_fn_robust(self, inputs: Union[torch.Tensor, np.ndarray] 
    ) -> Union [torch. Tensor, np.ndarray]:

        """Function to perturb samples for robustness optimization

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): Input of shape (1, channel, height, width)

        Returns:
            Union[torch.Tensor, np.ndarray]: Perturbed samples of shape (num_samples, channel, height, width)
        """
        inputs_isnumpy = isinstance(input, np.ndarray)
        if inputs_isnumpy:
            inputs = torch.from_numpy(inputs)
        shape_expanded = [self.num_samples] + list(inputs.shape[1:])
        device = inputs.device
        if self.mode == "uniform":
            noise = (
                torch.zeros(size=shape_expanded)
                .uniform_(-self.noise_lvl, self.noise_lvl)
                .to(device)
            )

        elif self.mode == "gaussian":
            noise = (
                torch.zeros(size=shape_expanded).normal_(0, self.noise_lvl).to(device)
            )
        perturbed_inputs = inputs + noise

        if inputs_isnumpy:
            perturbed_inputs = perturbed_inputs.cpu().numpy()

        return perturbed_inputs
    
    def compute_weights(self, explanations: np.array, gamma2_robust: np.ndarray, gamma1_faith: np.array,  gamma2_faith: np.ndarray
    ) -> np.ndarray:
        """Function to compute weights for the aggregated explanation based on Infidelity and Robustness as generalized L2 metrics

        Args:
            explanations (np.ndarray): Explanation of shape (1, height*width, num_explainers)
            gamma1 (np.ndarray): Infidelity perturbation (batch_size, num_samples//batch_size, height*width)
                This corresponds to the gamma1 parameter in the paper to obtain Infidelity as L2 metric
            gamma2 (np.ndarray): Prediction change due to perturbation (batch_size, num_samples//batch_size, 1)
                This corresponds to the gamma2 parameter in the paper to obtain Infidelity as L2 metric

        Returns:
            np.ndarray: Weights of shape (num_explainers)
        """
        # get number of explainer methods
        k = explanations.shape[-1]

        Gamma_faith = gamma1_faith @ explanations

        # normalize Gamma like proposed in the infidelity paper Yeh et al. 2019
        Gamma_faith *= np.mean(gamma2_faith*Gamma_faith, axis=0, keepdims=True) / np.mean(Gamma_faith*Gamma_faith, axis=0, keepdims=True)

        Gamma_faith= Gamma_faith - gamma2_faith
   
        Gamma_faith = Gamma_faith.transpose(0,2,1) @ Gamma_faith


        Gamma_robust = explanations - gamma2_robust
        Gamma_robust = Gamma_robust.transpose(0, 2, 1) @ Gamma_robust

        lamb_robust = self.lambda_robust / np.sqrt((Gamma_robust**2).sum())
        lamb_faith = self.lambda_faith / np.sqrt((Gamma_faith**2).sum())
        Gamma = lamb_robust * Gamma_robust.mean(axis=0) + lamb_faith * Gamma_faith.mean(axis=0)

   
        x = cp.Variable(k)
        prob = cp.Problem(
            cp.Minimize((1 / 2) * cp.quad_form(x, Gamma)),
            [x >= 0, np.ones((1, k)) @ x == 1],
        )
        prob.solve()

        weights = np.maximum(x.value,0)
        self.weights = weights

        return weights

           
    def explain(self, inputs: Union[torch.Tensor, np.ndarray], target: int, **kwargs
    ) -> Dict[str, np.ndarray]:
        """Function that explains the target class given input

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): input of shape (1, channel, height, width)
            target (int): Target class

        Returns:
            Dict[str, np.ndarray]: Explanation of shape (height, width)
        """
        explanation = self.compute_individual_explanations(inputs, target)


        # perturb inputs
        inputs_pert_faith = self._perturb_fn_infid(inputs)
        inputs_pert_robust = self._perturb_fn_robust(inputs)

        # compute explanations for perturbed inputs
        explanations_pert = [self.compute_individual_explanations(x.unsqueeze(0), target, store=False) for x in inputs_pert_robust]
        explanations_pert = np.stack(explanations_pert, axis=0)

        pert = torch.stack(
            [(inputs.detach() != p.detach()).float() for p in inputs_pert_faith], axis=0
        )
        pert = (
            pert.mean(axis=2)
            .reshape(inputs_pert_faith.shape[0], inputs_pert_faith.shape[1], -1)
            .cpu()
            .numpy()
        )  # shape: (batch_size, num_samples/batch_size, height*width)

        pred = self.model(inputs).detach().cpu().numpy()[:, target]
        pred_diff = np.stack(
            [
                pred - self.model(p).detach().cpu().numpy()[:, target]
                for p in inputs_pert_faith
            ],
            axis=0,
        )  # shape: (batch_size, num_samples/batch_size)

        # compute number of features
        d = explanation.shape[1] * explanation.shape[2]

        weights = self.compute_weights(
            explanation
            .transpose(1,2,0)
            .reshape(1,d, -1),
            gamma2_robust = explanations_pert
            .reshape(-1, explanations_pert.shape[1], d)
            .transpose(0,2,1),
            gamma1_faith = pert,
            gamma2_faith = pred_diff
            .reshape(inputs_pert_faith.shape[0], -1, 1)
        )

        explanation_agg = np.average(explanation, weights = weights, axis=0)

        explanation_agg = self.postprocess_explanation(explanation_agg)

        self.explanations = self.explanations | {self.name: explanation_agg}

        return {self.name: explanation_agg}
