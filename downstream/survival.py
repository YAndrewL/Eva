# -*- coding: utf-8 -*-
"""
@File   :  survival.py
@Time   :  2025/08/13 08:59
@Author :  Yufan Liu
@Desc   :  Models for survival analysis
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler


class CoxRegression:
    """Cox Proportional Hazards Regression Model using lifelines."""

    def __init__(self, standardize=True):
        self.standardize = standardize
        self.model = None
        self.scaler = StandardScaler() if standardize else None
        self.is_fitted = False

    def fit(self, features, survival_time, survival_status):
        """Fit Cox regression model."""
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        n_samples, n_features = features.shape
        if len(survival_time) != n_samples or len(survival_status) != n_samples:
            raise ValueError("All arrays must have the same number of samples")

        if self.standardize:
            features = self.scaler.fit_transform(features)

        # Create DataFrame for lifelines
        df_data = {}
        for i in range(n_features):
            df_data[f"feature_{i}"] = features[:, i]
        df_data["duration"] = survival_time
        df_data["event"] = survival_status.astype(bool)
        df = pd.DataFrame(df_data)

        self.model = CoxPHFitter(penalizer=0.1)
        self.model.fit(df, duration_col="duration", event_col="event")
        self.is_fitted = True
        return self

    def predict(self, features):
        """Predict risk scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        if self.standardize:
            features = self.scaler.transform(features)

        # Create DataFrame for lifelines prediction
        n_samples, n_features = features.shape
        df_data = {}
        for i in range(n_features):
            df_data[f"feature_{i}"] = features[:, i]
        df = pd.DataFrame(df_data)

        # lifelines uses predict_partial_hazard for risk scores
        return self.model.predict_partial_hazard(df).values.flatten()

    def get_coefficients(self):
        """Get model coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        return self.model.params_.values

    def get_summary(self):
        """Get model summary statistics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        return {
            "concordance_index": self.model.concordance_index_,
            "log_likelihood": self.model.log_likelihood_,
            "aic": self.model.AIC_partial_,
            "coefficients": self.get_coefficients(),
        }

    def get_training_c_index(self):
        """Get C-index on training data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting C-index")
        # lifelines provides training C-index directly
        return self.model.concordance_index_

    def get_test_c_index(self, test_features, test_survival_time, test_survival_status):
        """Calculate C-index on test data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        test_risk_scores = self.predict(test_features)
        # lifelines concordance_index expects event indicator, duration, and risk scores
        # Convert survival_status to boolean for lifelines
        test_survival_status_bool = test_survival_status.astype(bool)
        c_index = concordance_index(test_survival_time, -test_risk_scores, test_survival_status_bool)
        return c_index


class CoxLoss(nn.Module):
    """Partial Likelihood Loss Function for survival analysis.

    Seee MMP implementation: https://github.com/mahmoodlab/MMP/blob/main/src/utils/losses.py
    """

    def __init__(self):
        super(CoxLoss, self).__init__()

    def partial_ll_loss(self, lrisks, survival_times, event_indicators):
        """
        lrisks: log risks, B x 1
        survival_times: time bin, B x 1
        event_indicators: event indicator, B x 1
        """
        num_uncensored = torch.sum(event_indicators, 0)
        if num_uncensored.item() == 0:
            return {"loss": torch.sum(lrisks) * 0}

        survival_times = survival_times.squeeze(1)
        event_indicators = event_indicators.squeeze(1)
        lrisks = lrisks.squeeze(1)

        sindex = torch.argsort(-survival_times)
        survival_times = survival_times[sindex]
        event_indicators = event_indicators[sindex]
        lrisks = lrisks[sindex]

        log_risk_stable = torch.logcumsumexp(lrisks, 0)
        likelihood = lrisks - log_risk_stable
        uncensored_likelihood = likelihood * event_indicators
        logL = -torch.sum(uncensored_likelihood)

        # negative average log-likelihood
        return logL / num_uncensored

    def forward(self, risk_scores, survival_time, survival_status):
        # Convert to the format expected by partial_ll_loss
        lrisks = risk_scores.unsqueeze(1)  # B x 1
        survival_times = survival_time.unsqueeze(1)  # B x 1
        event_indicators = survival_status.unsqueeze(1)  # B x 1
        result = self.partial_ll_loss(lrisks, survival_times, event_indicators)
        return result


class LinearProbingCox(nn.Module):
    """Linear Probing Cox Regression Model."""

    def __init__(self, input_dim):
        super(LinearProbingCox, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Sequential(nn.LayerNorm(input_dim, eps=1e-6), nn.Linear(input_dim, 1))

    def forward(self, x):
        risk_scores = self.linear(x)
        return risk_scores.squeeze(-1)

    def fit(self, features, survival_time, survival_status, learning_rate=0.001, num_epochs=20, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)
        features = features.to(device)
        survival_time = survival_time.to(device)
        survival_status = survival_status.to(device)

        criterion = CoxLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        history = {"train_loss": [], "train_c_index": []}

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            risk_scores = self(features)
            loss = criterion(risk_scores, survival_time, survival_status)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                risk_np = risk_scores.detach().cpu().numpy()
                time_np = survival_time.detach().cpu().numpy()
                status_np = survival_status.detach().cpu().numpy()
                try:
                    # Convert status to boolean for lifelines
                    status_np_bool = status_np.astype(bool)
                    c_index = concordance_index(time_np, -risk_np, status_np_bool)
                except:
                    c_index = 0.5

            history["train_loss"].append(loss.item())
            history["train_c_index"].append(c_index)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, C-index = {c_index:.4f}")

        return history

    def predict(self, features):
        """Predict risk scores."""
        self.eval()
        with torch.no_grad():
            return self(features)

    def get_coefficients(self):
        """Get model coefficients."""
        return self.linear.weight.detach().cpu().numpy().flatten()

    def get_test_c_index(self, test_features, test_survival_time, test_survival_status):
        """Calculate C-index on test data."""
        test_risk_scores = self.predict(test_features)
        risk_np = test_risk_scores.detach().cpu().numpy()
        time_np = test_survival_time.detach().cpu().numpy()
        status_np = test_survival_status.detach().cpu().numpy()

        # Convert status to boolean for lifelines
        status_np_bool = status_np.astype(bool)
        c_index = concordance_index(time_np, -risk_np, status_np_bool)
        return c_index
