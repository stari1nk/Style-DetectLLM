import os
from typing import List, Optional
import math

import joblib
import torch
import torch.functional as F
from scipy.stats import norm
import numpy as np
from sklearn.preprocessing import StandardScaler


class EuclideanDetector:
    def __init__(self):
        self.center = []  

    def fit_features(self, X_normal: torch.Tensor, y: torch.Tensor, device: str = "cuda"):
       
        self.center = X_normal.mean(dim=0)
        return None

    def predict_features(self, X: torch.Tensor, batch_size: int = 40960) -> torch.Tensor:
        n = X.shape[0]
        scores = []
        
        for i in range(0, n, batch_size):
            
            X_batch = X[i:i+batch_size]
           
            score_batch = torch.norm(X_batch - self.center, dim=1)
            scores.append(score_batch)
        
        
        return torch.cat(scores)

class DeEuclideanDetector:
    def __init__(self):
        self.center = []  

    def fit_features(self, X_normal: torch.Tensor, y: torch.Tensor, device: str = "cuda"):
        
        center = X_normal.mean(dim=0)
        self.center = center / center.norm()
        return None

    def predict_features(self, X: torch.Tensor, batch_size: int = 40960) -> torch.Tensor:
        n = X.shape[0]
        scores = []
        
        for i in range(0, n, batch_size):
            
            X_batch = X[i:i+batch_size]
            
            proj = X_batch @ self.center
            other_norm = torch.sqrt((X_batch ** 2).sum(dim=-1) - proj ** 2) 
            score = -proj/other_norm
            scores.append(score)
        
       
        return torch.cat(scores)



class PCADetector:
    def __init__(self):
        self.pc = None  

    def fit_features(self, X_normal: torch.Tensor, y: torch.Tensor, device: str = "cuda"):
        
        mean = X_normal.mean(dim=0, keepdim=True)
        X = X_normal - mean

        
        cov = X.T @ X / (X.shape[0] - 1)

        
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        self.pc = eigenvectors[:, -1:]  

    def predict_features(self, X: torch.Tensor, batch_size: int = 40960):
        proj = X @ self.pc
        return -torch.abs(proj).squeeze(1)

class MahalanobisDetector:
    def __init__(self, eps=1e-6):
        self.mean = None
        self.cov_inv = None
        self.eps = eps

    def fit_features(self, X_normal: torch.Tensor, y: torch.Tensor, device: str = "cuda"):
        self.mean = X_normal.mean(dim=0)
        X = X_normal - self.mean

        
        cov = X.T @ X / (X.shape[0] - 1)

        
        cov = cov + self.eps * torch.eye(cov.shape[0], device=X.device)

        
        self.cov_inv = torch.linalg.inv(cov)

    def predict_features(self, X: torch.Tensor):
        diff = X - self.mean
        md = torch.sqrt(torch.sum((diff @ self.cov_inv) * diff, dim=1))
        return md


class OODDetector:
    def __init__(
        self,
        detector_name: str = "Mahalanobis",
        detector_kwargs: Optional[dict] = None,
        scale: Optional[str] = None,
        pca: Optional[int] = None,
        device: str = "cuda"):
        self.detector_kwargs = detector_kwargs or {"model": None}
        self.detector_name = detector_name
        if detector_name == "Eu":
            self.detector = EuclideanDetector()
        elif detector_name == "PCA":
            self.detector = PCADetector()
        elif detector_name == "Ma":
            self.detector = MahalanobisDetector()
        elif detector_name == "DeEu":
            self.detector = DeEuclideanDetector()
        else:
            raise ValueError(f"Unknown detector: {detector_name}")

        self.scale = scale
        self.pca = pca
        self.device = device
        self.multi = False
        self.pca_params = []
        self.scale_params = []
        self.ood_params = []
    
    def standardize(self, X: torch.Tensor, ids: int) -> torch.Tensor:
        """Standardize each feature matrix X according to scale settings.

        Returns transformed X and associated parameters (mean/std) when applicable.
        """
        if self.scale == "Z":
            if ids < len(self.scale_params):
                mean, std = self.scale_params[ids]
            else:
                mean = X.mean(dim=0, keepdim=True)
                std = X.std(dim=0, keepdim=True) + 1e-6
            if ids >= len(self.scale_params):
                self.scale_params.append([mean, std])
            return (X - mean) / std, [mean, std]

        elif self.scale == "L2":
            # L2 normalization to unit norm per sample
            norm = X.norm(dim=1, keepdim=True) + 1e-12
            return X / norm, None

        else:
            return X, None
    
    def pca_transform(self, X: torch.Tensor, ids: int) -> torch.Tensor:
        """Project X into PCA subspace. Stores PCA params if not already.

        Returns projected X and pca params [means, directions].
        """
        if ids < len(self.pca_params):
            means, directions = self.pca_params[ids]
        else:
            # compute per-sample mean if input is batched as (N, D)
            means = X.mean(dim=0, keepdim=True)
            # torch.pca_lowrank is sometimes deprecated; we keep compatibility.
            _, _, directions = torch.pca_lowrank(X - means, q=self.pca)
            self.pca_params.append([means, directions])

        X_centered = X - means
        return X_centered @ directions, [means, directions]

 
    def fit(self, X_train: List[torch.Tensor], y: torch.Tensor):
        if self.detector_name == "Semantic_ED":
            X_for_fit = X_train[0] if isinstance(X_train, list) else X_train
            self.detector.fit_features(X_for_fit, y, device=self.device)
            scores = self.detector.predict_features(X_for_fit)
            means = scores.mean()
            stds = scores.std() + 1e-6
            self.ood_params = [means, stds]
            print(f"Fit complete. OOD score mean: {means:.4f}, std: {stds:.4f}")
            return None

        for i in range(len(X_train)):
            if self.scale is not None:
                X_train[i], standard_param = self.standardize(X_train[i], i)
                self.scale_params.append(standard_param)
            if self.pca is not None:
                X_train[i], pca_param = self.pca_transform(X_train[i], i)
                self.pca_params.append(pca_param)
        if not self.multi:
            X_train = X_train[0]
        if self.detector is not None:
            self.detector.fit_features(X_train, y, device = self.device)
            scores = self.detector.predict_features(X_train)
            means = scores.mean()
            stds = scores.std() + 1e-6
            self.ood_params = [means, stds]
        else:
            ood_means = X_train.mean()
            diffs = X_train - ood_means
            scores = (diffs * diffs).sum(dim=1)
            means = scores.mean()
            stds = scores.std() + 1e-6
            self.ood_params = [means, stds]
        print(f"Fit complete. OOD score mean: {means:.4f}, std: {stds:.4f}")
        return None
    
    def predict(self, X_test: List[torch.Tensor]) -> dict:
        if self.detector_name == "Semantic_ED":
            X_for_pred = X_test[0] if isinstance(X_test, list) else X_test
            scores = self.detector.predict_features(X_for_pred)
            pred_means = scores.mean()
            n = scores.shape[0]
            t_stat = (pred_means - self.ood_params[0]) / (self.ood_params[1] / math.sqrt(n))
            t_stat_np = t_stat.detach().cpu().numpy()
            p_value = 1.0 - norm.cdf(t_stat_np)
            return {
                "p_value": -float(p_value),
                "t_stat": float(t_stat.cpu().item()),
                "means": float(pred_means.cpu().item()) if isinstance(pred_means, torch.Tensor) else float(pred_means),
                "n": n,
                "scores": scores.detach().cpu().numpy(),
            }

        for i in range(len(X_test)):
            if self.scale is not None:
                X_test[i], _ = self.standardize(X_test[i], i)
            if self.pca is not None:
                X_test[i], _ = self.pca_transform(X_test[i], i)

        if not self.multi:
            X_test = X_test[0]

        if self.detector is not None:
            scores = self.detector.predict_features(X_test)
        else:
            X_test_mat = X_test
            ood_means = self.ood_params[0]
            # ensure same dims for mean-based score method
            diffs = X_test_mat - ood_means
            scores = (diffs * diffs).sum(dim=1)

        pred_means = scores.mean()
        n = scores.shape[0]
        t_stat = (pred_means - self.ood_params[0]) / (self.ood_params[1] / math.sqrt(n))

        t_stat_np = t_stat.detach().cpu().numpy()
        p_value = 1.0 - norm.cdf(t_stat_np)

        return {
            "p_value": -float(p_value),
            "t_stat": float(t_stat.cpu().item()),
            "means": float(pred_means.cpu().item()) if isinstance(pred_means, torch.Tensor) else float(pred_means),
            "n": n,
            "scores": scores.detach().cpu().numpy(),
        }

        
    def save(self, filepath: str) -> None:
        """
        Save the fitted detector state (including detector, scale_params, pca_params) to `filepath`.
        Uses joblib for serialization.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        state = {
            "detector": self.detector,
            "detector_name": self.detector_name,
            "scale": self.scale,
            "pca": self.pca,
            "device": self.device,
            "multi": self.multi,
            "scale_params": self.scale_params,
            "pca_params": self.pca_params,
            "detector_kwargs": self.detector_kwargs,
            "ood_params": self.ood_params,
        }
        with open(filepath, "wb") as f:
            joblib.dump(state, f)
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load detector state from `filepath` and return a ready-to-use instance.
        """
        with open(filepath, "rb") as f:
            state = joblib.load(f)
        # Reconstruct instance
        instance = cls(
            detector_name=state["detector_name"],
            detector_kwargs=state["detector_kwargs"],
            scale=state["scale"],
            pca=state["pca"],
            device=state["device"],
        )
        instance.detector = state["detector"]
        instance.multi = state["multi"]
        instance.scale_params = state["scale_params"]
        instance.pca_params = state["pca_params"]
        instance.ood_params = state["ood_params"]
        return instance
    
        


