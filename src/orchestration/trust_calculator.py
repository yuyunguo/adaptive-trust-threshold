"""
Trust Calculator Engine
Implements T = f(C, E, S) trust scoring for clinical AI agents
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrustScore:
    """Trust score calculation result"""
    trust_score: float
    calibrated_confidence: float
    explainability: float
    semantic_similarity: float
    threshold: float
    passes_threshold: bool
    raw_confidence: float
    timestamp: datetime


class TrustCalculator:
    """
    Trust calculation engine implementing T = f(C, E, S)
    Based on research from MDPI Diagnostics paper results
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize Trust Calculator

        Args:
            weights: Weight dictionary for trust components
        """
        # Default weights from optimized sensitivity analysis (manuscript: α=0.60, β=0.24, γ=0.16)
        self.weights = weights or {
            'confidence': 0.60,      # α: Confidence weight (optimized from 162 configurations)
            'explainability': 0.24,  # β: Explainability weight
            'semantic_similarity': 0.16  # γ: Semantic similarity weight
        }

        # Calibration parameters (to be learned from data)
        self.calibration_params = {
            'platt_a': 1.0,
            'platt_b': 0.0,
            'temperature': 1.0
        }

        # Trust calculation history
        self.trust_history: List[TrustScore] = []

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Validate weights
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Validate that weights are properly configured"""
        required_weights = ['confidence', 'explainability', 'semantic_similarity']

        for weight in required_weights:
            if weight not in self.weights:
                raise ValueError(f"Missing required weight: {weight}")

        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:  # Allow small floating point error
            self.logger.warning(f"Trust weights sum to {weight_sum:.3f}, not 1.0")

    def calculate_trust_score(self,
                            confidence: float,
                            explainability: float,
                            semantic_similarity: float,
                            agent_type: str = "default") -> TrustScore:
        """
        Calculate trust score: T = w_c*C + w_e*E + w_s*S

        Args:
            confidence: Raw model confidence [0,1]
            explainability: explanation quality score [0,1]
            semantic_similarity: similarity to clinical guidelines [0,1]
            agent_type: Type of agent for specialized thresholds

        Returns:
            TrustScore object with all components and threshold check
        """

        # Validate inputs
        self._validate_inputs(confidence, explainability, semantic_similarity)

        # Step 1: Calibrate confidence (Platt scaling)
        calibrated_confidence = self._calibrate_confidence(confidence)

        # Step 2: Calculate weighted trust score
        trust_score = (
            self.weights['confidence'] * calibrated_confidence +
            self.weights['explainability'] * explainability +
            self.weights['semantic_similarity'] * semantic_similarity
        )

        # Step 3: Calculate adaptive threshold θ(E)
        threshold = self._adaptive_threshold(explainability, agent_type)

        # Step 4: Determine if agent passes threshold
        passes_threshold = trust_score >= threshold

        # Create result object
        result = TrustScore(
            trust_score=float(trust_score),
            calibrated_confidence=float(calibrated_confidence),
            explainability=float(explainability),
            semantic_similarity=float(semantic_similarity),
            threshold=float(threshold),
            passes_threshold=bool(passes_threshold),
            raw_confidence=float(confidence),
            timestamp=datetime.now()
        )

        # Store in history
        self.trust_history.append(result)

        # Log result
        self.logger.debug(f"Trust score calculated: {trust_score:.3f} "
                         f"(threshold: {threshold:.3f}, passes: {passes_threshold})")

        return result

    def _validate_inputs(self, confidence: float, explainability: float,
                        semantic_similarity: float) -> None:
        """Validate input parameters"""
        inputs = {
            'confidence': confidence,
            'explainability': explainability,
            'semantic_similarity': semantic_similarity
        }

        for name, value in inputs.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {type(value)}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")

    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """
        Platt scaling for confidence calibration
        σ(af(x) + b) where f(x) is raw confidence
        """
        a = self.calibration_params['platt_a']
        b = self.calibration_params['platt_b']

        # Avoid numerical overflow
        score = a * raw_confidence + b
        score = np.clip(score, -500, 500)

        calibrated = 1.0 / (1.0 + np.exp(-score))

        # Apply temperature scaling if needed
        temperature = self.calibration_params.get('temperature', 1.0)
        if temperature != 1.0:
            calibrated = calibrated ** (1.0 / temperature)

        return float(calibrated)

    def _adaptive_threshold(self, explainability: float, agent_type: str = "default") -> float:
        """
        Dynamic threshold: θ(E) = τ_base + α(1 - E)
        Lower explanation quality requires higher trust
        """

        # Base thresholds by agent type
        base_thresholds = {
            'retrieval': 0.5,      # Data retrieval is more predictable
            'reasoning': 0.6,      # Clinical reasoning needs higher trust
            'treatment': 0.7,      # Treatment recommendations are critical
            'communication': 0.4,   # Communication can be more flexible
            'default': 0.5
        }

        tau_base = base_thresholds.get(agent_type, 0.5)
        alpha = 0.2  # Sensitivity parameter (reduced from 0.3 for stability)

        # Higher threshold when explainability is low
        threshold = tau_base + alpha * (1 - explainability)

        # Ensure threshold stays within reasonable bounds
        return float(np.clip(threshold, 0.3, 0.9))

    def update_calibration(self,
                          predictions: List[float],
                          ground_truth: List[bool],
                          method: str = "platt") -> None:
        """
        Update calibration parameters using training data

        Args:
            predictions: List of raw confidence scores
            ground_truth: List of actual outcomes (True/False)
            method: Calibration method ("platt" or "temperature")
        """

        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        if len(predictions) < 10:
            self.logger.warning("Insufficient data for calibration (need ≥10 samples)")
            return

        try:
            if method == "platt":
                self._update_platt_scaling(predictions, ground_truth)
            elif method == "temperature":
                self._update_temperature_scaling(predictions, ground_truth)
            else:
                raise ValueError(f"Unknown calibration method: {method}")

            self.logger.info(f"Calibration updated using {method} scaling with {len(predictions)} samples")

        except Exception as e:
            self.logger.error(f"Calibration update failed: {e}")

    def _update_platt_scaling(self, predictions: List[float], ground_truth: List[bool]) -> None:
        """Update Platt scaling parameters"""
        try:
            from sklearn.linear_model import LogisticRegression

            X = np.array(predictions).reshape(-1, 1)
            y = np.array(ground_truth, dtype=int)

            # Fit logistic regression for Platt scaling
            lr = LogisticRegression(random_state=42)
            lr.fit(X, y)

            self.calibration_params['platt_a'] = float(lr.coef_[0][0])
            self.calibration_params['platt_b'] = float(lr.intercept_[0])

            self.logger.info(f"Platt scaling updated: a={self.calibration_params['platt_a']:.3f}, "
                           f"b={self.calibration_params['platt_b']:.3f}")

        except ImportError:
            self.logger.error("sklearn not available for Platt scaling")

    def _update_temperature_scaling(self, predictions: List[float], ground_truth: List[bool]) -> None:
        """Update temperature scaling parameter"""
        # Simplified temperature scaling implementation
        # In practice, would use proper optimization

        from scipy.optimize import minimize_scalar

        def temperature_loss(temp: float) -> float:
            """Loss function for temperature scaling"""
            if temp <= 0:
                return float('inf')

            calibrated_preds = [p ** (1.0 / temp) for p in predictions]

            # Calculate Brier score
            brier_score = np.mean([(pred - truth) ** 2
                                  for pred, truth in zip(calibrated_preds, ground_truth)])
            return brier_score

        try:
            result = minimize_scalar(temperature_loss, bounds=(0.1, 5.0), method='bounded')

            if result.success:
                self.calibration_params['temperature'] = float(result.x)
                self.logger.info(f"Temperature scaling updated: T={result.x:.3f}")
            else:
                self.logger.warning("Temperature scaling optimization failed")

        except Exception as e:
            self.logger.error(f"Temperature scaling update failed: {e}")

    def get_calibration_metrics(self,
                              predictions: List[float],
                              ground_truth: List[bool]) -> Dict[str, float]:
        """
        Calculate calibration metrics for evaluation

        Args:
            predictions: Raw confidence scores
            ground_truth: Actual outcomes

        Returns:
            Dictionary of calibration metrics
        """
        if len(predictions) != len(ground_truth) or len(predictions) == 0:
            return {}

        # Apply current calibration
        calibrated_preds = [self._calibrate_confidence(p) for p in predictions]

        # Calculate metrics
        metrics = {}

        # Brier Score (lower is better)
        brier_score = np.mean([(pred - truth) ** 2
                              for pred, truth in zip(calibrated_preds, ground_truth)])
        metrics['brier_score'] = float(brier_score)

        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(calibrated_preds, ground_truth)
        metrics['expected_calibration_error'] = float(ece)

        # Reliability (how close predicted probabilities are to actual frequencies)
        reliability = self._calculate_reliability(calibrated_preds, ground_truth)
        metrics['reliability'] = float(reliability)

        return metrics

    def _calculate_ece(self, predictions: List[float], ground_truth: List[bool], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = [(pred >= bin_lower) and (pred < bin_upper)
                     for pred in predictions]
            prop_in_bin = sum(in_bin) / len(predictions)

            if prop_in_bin > 0:
                # Calculate accuracy and confidence for this bin
                bin_predictions = [predictions[i] for i, in_b in enumerate(in_bin) if in_b]
                bin_truth = [ground_truth[i] for i, in_b in enumerate(in_bin) if in_b]

                accuracy_in_bin = sum(bin_truth) / len(bin_truth)
                avg_confidence_in_bin = sum(bin_predictions) / len(bin_predictions)

                ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _calculate_reliability(self, predictions: List[float], ground_truth: List[bool]) -> float:
        """Calculate reliability score"""
        # Simple correlation-based reliability measure
        if len(predictions) < 2:
            return 0.0

        try:
            correlation = np.corrcoef(predictions, [float(gt) for gt in ground_truth])[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    def get_trust_statistics(self) -> Dict[str, float]:
        """Get statistics about trust score history"""
        if not self.trust_history:
            return {'message': 'No trust scores calculated yet'}

        recent_scores = [ts.trust_score for ts in self.trust_history[-100:]]  # Last 100
        pass_rates = [ts.passes_threshold for ts in self.trust_history[-100:]]

        return {
            'total_calculations': len(self.trust_history),
            'mean_trust_score': float(np.mean(recent_scores)),
            'std_trust_score': float(np.std(recent_scores)),
            'min_trust_score': float(np.min(recent_scores)),
            'max_trust_score': float(np.max(recent_scores)),
            'threshold_pass_rate': float(np.mean(pass_rates)),
            'current_weights': self.weights.copy(),
            'calibration_params': self.calibration_params.copy()
        }

    def reset_calibration(self) -> None:
        """Reset calibration parameters to defaults"""
        self.calibration_params = {
            'platt_a': 1.0,
            'platt_b': 0.0,
            'temperature': 1.0
        }
        self.logger.info("Calibration parameters reset to defaults")

    def export_trust_history(self) -> List[Dict]:
        """Export trust calculation history for analysis"""
        return [
            {
                'trust_score': ts.trust_score,
                'calibrated_confidence': ts.calibrated_confidence,
                'explainability': ts.explainability,
                'semantic_similarity': ts.semantic_similarity,
                'threshold': ts.threshold,
                'passes_threshold': ts.passes_threshold,
                'raw_confidence': ts.raw_confidence,
                'timestamp': ts.timestamp.isoformat()
            }
            for ts in self.trust_history
        ]