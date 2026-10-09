"""
Real-Time Trust Monitoring System
Production-ready trust monitoring with automated alerts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json

from .enhanced_trust_analyzer import ClinicalTrustMetrics


@dataclass
class TrustAlert:
    """Trust monitoring alert"""
    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: str
    model_name: str
    message: str
    current_value: float
    threshold_value: float
    recommended_action: str


@dataclass
class MonitoringBatch:
    """Monitoring batch result"""
    batch_id: int
    timestamp: str
    predictions: Dict[str, Dict]
    trust_scores: Dict[str, float]
    alerts: List[TrustAlert]
    performance_metrics: Dict[str, float]


class RealTimeTrustMonitor:
    """Production-ready trust monitoring system inspired by demo framework"""

    def __init__(self, trust_analysis: Dict[str, ClinicalTrustMetrics], ml_pipeline):
        """
        Initialize Real-Time Trust Monitor

        Args:
            trust_analysis: Trust analysis results
            ml_pipeline: ML pipeline with trained models
        """
        self.trust_analysis = trust_analysis
        self.ml_pipeline = ml_pipeline
        self.monitoring_history = []
        self.logger = logging.getLogger(__name__)

        # Alert threshold configuration
        self.alert_thresholds = {
            'trust_drop': 0.15,  # Alert if trust drops by 15%
            'consistency_threshold': 0.2,  # Alert if model predictions vary by 20%
            'minimum_trust': 0.5,  # Alert if any model falls below 50% trust
            'performance_degradation': 0.1,  # Alert if accuracy drops by 10%
            'prediction_variance': 0.25  # Alert if prediction variance exceeds 25%
        }

        # Monitoring configuration
        self.monitoring_config = {
            'batch_size': 10,
            'alert_cooldown': 300,  # 5 minutes between similar alerts
            'max_alerts_per_hour': 20,
            'auto_response_enabled': True
        }

        self.last_alerts = {}  # Track recent alerts for cooldown
        self.alert_counter = {}  # Track alert frequency

    def simulate_real_time_monitoring(self, n_samples: int = 50,
                                    batch_size: Optional[int] = None) -> Tuple[List[MonitoringBatch], List[TrustAlert]]:
        """Simulate real-time trust monitoring on sample data"""
        self.logger.info(f"Starting real-time trust monitoring simulation for {n_samples} samples")

        if not hasattr(self.ml_pipeline, 'models'):
            self.logger.warning("ML pipeline models not available for monitoring simulation")
            return [], []

        batch_size = batch_size or self.monitoring_config['batch_size']

        # Get sample data for monitoring
        if not hasattr(self.ml_pipeline, 'X_test') or not hasattr(self.ml_pipeline, 'y_test'):
            self.logger.warning("Test data not available - generating synthetic data")
            return self._simulate_synthetic_monitoring(n_samples, batch_size)

        X_sample = self.ml_pipeline.X_test[:n_samples]
        y_sample = self.ml_pipeline.y_test[:n_samples]

        monitoring_results = []
        all_alerts = []

        self.logger.info(f"Processing {len(X_sample)} samples in batches of {batch_size}")

        for i in range(0, len(X_sample), batch_size):
            batch_X = X_sample[i:i+batch_size]
            batch_y = y_sample[i:i+batch_size]

            batch_result = self._process_monitoring_batch(
                batch_X, batch_y, i // batch_size + 1
            )

            monitoring_results.append(batch_result)
            all_alerts.extend(batch_result.alerts)

            # Log batch processing
            self.logger.debug(f"Batch {batch_result.batch_id} processed: "
                            f"{len(batch_result.alerts)} alerts generated")

        self.monitoring_history.extend(monitoring_results)

        # Generate monitoring summary
        self._log_monitoring_summary(monitoring_results, all_alerts)

        return monitoring_results, all_alerts

    def _process_monitoring_batch(self, batch_X, batch_y, batch_id: int) -> MonitoringBatch:
        """Process a single monitoring batch"""

        batch_result = MonitoringBatch(
            batch_id=batch_id,
            timestamp=f"T+{batch_id:02d}",
            predictions={},
            trust_scores={},
            alerts=[],
            performance_metrics={}
        )

        model_predictions = []
        model_confidences = []

        # Process each model
        for model_name, trust_info in self.trust_analysis.items():
            if model_name not in self.ml_pipeline.models:
                continue

            try:
                model = self.ml_pipeline.models[model_name]

                # Get batch predictions
                pred_proba = model.predict_proba(batch_X)[:, 1]  # Probability of positive class
                predictions = (pred_proba > 0.5).astype(int)

                # Calculate batch performance metrics
                batch_accuracy = np.mean(predictions == batch_y)
                avg_confidence = np.mean(pred_proba)

                # Calculate real-time trust score
                base_trust = trust_info.trust_score
                performance_adjustment = (batch_accuracy - 0.5) * 0.2
                real_time_trust = np.clip(base_trust + performance_adjustment, 0.1, 1.0)

                # Store results
                batch_result.predictions[model_name] = {
                    'accuracy': batch_accuracy,
                    'avg_confidence': avg_confidence,
                    'predictions': predictions.tolist(),
                    'probabilities': pred_proba.tolist()
                }

                batch_result.trust_scores[model_name] = real_time_trust
                model_predictions.append(pred_proba)
                model_confidences.append(avg_confidence)

                # Check for trust alerts
                alerts = self._check_trust_alerts(
                    model_name, base_trust, real_time_trust, batch_accuracy, avg_confidence
                )
                batch_result.alerts.extend(alerts)

            except Exception as e:
                self.logger.error(f"Error processing model {model_name}: {e}")
                batch_result.predictions[model_name] = {'error': str(e)}
                batch_result.trust_scores[model_name] = 0.0

        # Check ensemble-level alerts
        if len(model_predictions) > 1:
            ensemble_alerts = self._check_ensemble_alerts(
                model_predictions, model_confidences, batch_id
            )
            batch_result.alerts.extend(ensemble_alerts)

        # Calculate batch-level performance metrics
        batch_result.performance_metrics = self._calculate_batch_metrics(
            batch_result.predictions, batch_y
        )

        return batch_result

    def _check_trust_alerts(self, model_name: str, base_trust: float,
                           current_trust: float, accuracy: float,
                           confidence: float) -> List[TrustAlert]:
        """Check for trust-related alerts"""

        alerts = []
        current_time = datetime.now()

        # Trust drop alert
        trust_drop = base_trust - current_trust
        if trust_drop > self.alert_thresholds['trust_drop']:
            if self._should_generate_alert(model_name, 'TRUST_DROP', current_time):
                severity = 'HIGH' if trust_drop > 0.25 else 'MEDIUM'
                alert = TrustAlert(
                    alert_id=f"{model_name}_TRUST_DROP_{int(current_time.timestamp())}",
                    timestamp=current_time,
                    alert_type='TRUST_DROP',
                    severity=severity,
                    model_name=model_name,
                    message=f"Trust dropped by {trust_drop:.2%}",
                    current_value=current_trust,
                    threshold_value=base_trust - self.alert_thresholds['trust_drop'],
                    recommended_action="Review model performance and consider retraining"
                )
                alerts.append(alert)

        # Low trust alert
        if current_trust < self.alert_thresholds['minimum_trust']:
            if self._should_generate_alert(model_name, 'LOW_TRUST', current_time):
                alert = TrustAlert(
                    alert_id=f"{model_name}_LOW_TRUST_{int(current_time.timestamp())}",
                    timestamp=current_time,
                    alert_type='LOW_TRUST',
                    severity='HIGH',
                    model_name=model_name,
                    message=f"Trust below minimum threshold: {current_trust:.3f}",
                    current_value=current_trust,
                    threshold_value=self.alert_thresholds['minimum_trust'],
                    recommended_action="Switch to backup model or require human verification"
                )
                alerts.append(alert)

        # Performance degradation alert
        if accuracy < 0.6:  # Assuming baseline accuracy should be > 60%
            if self._should_generate_alert(model_name, 'PERFORMANCE_DEGRADATION', current_time):
                alert = TrustAlert(
                    alert_id=f"{model_name}_PERF_DEGRADE_{int(current_time.timestamp())}",
                    timestamp=current_time,
                    alert_type='PERFORMANCE_DEGRADATION',
                    severity='MEDIUM',
                    model_name=model_name,
                    message=f"Accuracy below expected threshold: {accuracy:.3f}",
                    current_value=accuracy,
                    threshold_value=0.6,
                    recommended_action="Investigate data quality and model drift"
                )
                alerts.append(alert)

        return alerts

    def _check_ensemble_alerts(self, model_predictions: List[np.ndarray],
                              model_confidences: List[float],
                              batch_id: int) -> List[TrustAlert]:
        """Check for ensemble-level alerts"""

        alerts = []
        current_time = datetime.now()

        if len(model_predictions) < 2:
            return alerts

        # Check prediction consistency
        prediction_matrix = np.array(model_predictions)
        prediction_variance = np.var(prediction_matrix, axis=0)
        high_variance_count = np.sum(
            prediction_variance > self.alert_thresholds['consistency_threshold']
        )

        if high_variance_count > len(prediction_matrix[0]) * 0.3:  # More than 30% inconsistent
            if self._should_generate_alert('ENSEMBLE', 'INCONSISTENT_PREDICTIONS', current_time):
                alert = TrustAlert(
                    alert_id=f"ENSEMBLE_INCONSISTENT_{batch_id}_{int(current_time.timestamp())}",
                    timestamp=current_time,
                    alert_type='INCONSISTENT_PREDICTIONS',
                    severity='MEDIUM',
                    model_name='ENSEMBLE',
                    message=f"High prediction variance in {high_variance_count}/{len(prediction_matrix[0])} samples",
                    current_value=high_variance_count / len(prediction_matrix[0]),
                    threshold_value=0.3,
                    recommended_action="Flag for expert review and investigate model disagreement"
                )
                alerts.append(alert)

        # Check confidence variance
        confidence_variance = np.var(model_confidences)
        if confidence_variance > self.alert_thresholds['prediction_variance']:
            if self._should_generate_alert('ENSEMBLE', 'CONFIDENCE_VARIANCE', current_time):
                alert = TrustAlert(
                    alert_id=f"ENSEMBLE_CONF_VAR_{batch_id}_{int(current_time.timestamp())}",
                    timestamp=current_time,
                    alert_type='CONFIDENCE_VARIANCE',
                    severity='LOW',
                    model_name='ENSEMBLE',
                    message=f"High confidence variance detected: {confidence_variance:.3f}",
                    current_value=float(confidence_variance),
                    threshold_value=self.alert_thresholds['prediction_variance'],
                    recommended_action="Monitor ensemble stability and consider model recalibration"
                )
                alerts.append(alert)

        return alerts

    def _should_generate_alert(self, model_name: str, alert_type: str,
                              current_time: datetime) -> bool:
        """Check if alert should be generated based on cooldown and frequency limits"""

        alert_key = f"{model_name}_{alert_type}"

        # Check cooldown period
        if alert_key in self.last_alerts:
            time_diff = (current_time - self.last_alerts[alert_key]).total_seconds()
            if time_diff < self.monitoring_config['alert_cooldown']:
                return False

        # Check hourly frequency limit
        hour_key = current_time.strftime('%Y%m%d_%H')
        if hour_key not in self.alert_counter:
            self.alert_counter[hour_key] = 0

        if self.alert_counter[hour_key] >= self.monitoring_config['max_alerts_per_hour']:
            return False

        # Update tracking
        self.last_alerts[alert_key] = current_time
        self.alert_counter[hour_key] += 1

        return True

    def _calculate_batch_metrics(self, predictions: Dict, batch_y) -> Dict[str, float]:
        """Calculate batch-level performance metrics"""

        metrics = {}

        valid_models = [
            name for name, pred in predictions.items()
            if 'error' not in pred and 'accuracy' in pred
        ]

        if valid_models:
            accuracies = [predictions[name]['accuracy'] for name in valid_models]
            confidences = [predictions[name]['avg_confidence'] for name in valid_models]

            metrics = {
                'ensemble_accuracy_mean': np.mean(accuracies),
                'ensemble_accuracy_std': np.std(accuracies),
                'ensemble_confidence_mean': np.mean(confidences),
                'ensemble_confidence_std': np.std(confidences),
                'model_agreement_rate': self._calculate_agreement_rate(predictions, valid_models)
            }

        return metrics

    def _calculate_agreement_rate(self, predictions: Dict, valid_models: List[str]) -> float:
        """Calculate agreement rate between models"""

        if len(valid_models) < 2:
            return 1.0

        model_preds = []
        for model_name in valid_models:
            preds = predictions[model_name]['predictions']
            model_preds.append(preds)

        # Calculate pairwise agreement
        agreements = []
        for i in range(len(model_preds)):
            for j in range(i + 1, len(model_preds)):
                agreement = np.mean(np.array(model_preds[i]) == np.array(model_preds[j]))
                agreements.append(agreement)

        return float(np.mean(agreements)) if agreements else 1.0

    def _simulate_synthetic_monitoring(self, n_samples: int,
                                     batch_size: int) -> Tuple[List[MonitoringBatch], List[TrustAlert]]:
        """Simulate monitoring with synthetic data when real data unavailable"""

        self.logger.info("Generating synthetic monitoring data")

        monitoring_results = []
        all_alerts = []

        np.random.seed(42)

        for batch_id in range(1, (n_samples // batch_size) + 2):
            batch_result = MonitoringBatch(
                batch_id=batch_id,
                timestamp=f"T+{batch_id:02d}",
                predictions={},
                trust_scores={},
                alerts=[],
                performance_metrics={}
            )

            # Generate synthetic results for each model
            for model_name, trust_info in self.trust_analysis.items():
                # Simulate performance with some variance
                base_accuracy = 0.7 + trust_info.trust_score * 0.2
                noise = np.random.normal(0, 0.1)
                batch_accuracy = np.clip(base_accuracy + noise, 0.4, 0.95)

                # Simulate trust score adjustment
                performance_adjustment = (batch_accuracy - 0.7) * 0.2
                real_time_trust = np.clip(trust_info.trust_score + performance_adjustment, 0.1, 1.0)

                batch_result.predictions[model_name] = {
                    'accuracy': batch_accuracy,
                    'avg_confidence': 0.6 + np.random.random() * 0.3,
                    'predictions': np.random.randint(0, 2, batch_size).tolist()
                }

                batch_result.trust_scores[model_name] = real_time_trust

                # Check for alerts (with lower probability for synthetic data)
                if np.random.random() < 0.1:  # 10% chance of alert
                    alert_types = ['TRUST_DROP', 'LOW_TRUST', 'PERFORMANCE_DEGRADATION']
                    alert_type = np.random.choice(alert_types)

                    alert = TrustAlert(
                        alert_id=f"{model_name}_{alert_type}_{batch_id}",
                        timestamp=datetime.now(),
                        alert_type=alert_type,
                        severity=np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                        model_name=model_name,
                        message=f"Synthetic alert: {alert_type}",
                        current_value=real_time_trust,
                        threshold_value=0.5,
                        recommended_action="Investigate synthetic alert"
                    )
                    batch_result.alerts.append(alert)

            monitoring_results.append(batch_result)
            all_alerts.extend(batch_result.alerts)

        return monitoring_results, all_alerts

    def _log_monitoring_summary(self, monitoring_results: List[MonitoringBatch],
                               all_alerts: List[TrustAlert]):
        """Log monitoring session summary"""

        self.logger.info("Real-time monitoring session summary:")
        self.logger.info(f"  • Batches processed: {len(monitoring_results)}")
        self.logger.info(f"  • Total alerts generated: {len(all_alerts)}")

        # Alert breakdown
        alert_types = {}
        for alert in all_alerts:
            alert_types[alert.alert_type] = alert_types.get(alert.alert_type, 0) + 1

        if alert_types:
            self.logger.info("  • Alert breakdown:")
            for alert_type, count in alert_types.items():
                self.logger.info(f"    - {alert_type}: {count}")

        # Trust score stability
        if monitoring_results:
            model_names = list(monitoring_results[0].trust_scores.keys())
            for model_name in model_names:
                trust_scores = [
                    batch.trust_scores.get(model_name, 0)
                    for batch in monitoring_results
                ]
                if trust_scores:
                    avg_trust = np.mean(trust_scores)
                    trust_stability = 1 - np.std(trust_scores)
                    self.logger.info(f"  • {model_name}: avg_trust={avg_trust:.3f}, "
                                   f"stability={trust_stability:.3f}")

    def export_monitoring_report(self, monitoring_results: List[MonitoringBatch],
                               alerts: List[TrustAlert], save_path: str) -> Dict:
        """Export comprehensive monitoring report"""

        report = {
            'monitoring_session': {
                'start_time': datetime.now().isoformat(),
                'total_batches': len(monitoring_results),
                'total_alerts': len(alerts),
                'monitoring_config': self.monitoring_config
            },
            'batch_results': [
                {
                    'batch_id': batch.batch_id,
                    'timestamp': batch.timestamp,
                    'trust_scores': batch.trust_scores,
                    'performance_metrics': batch.performance_metrics,
                    'alert_count': len(batch.alerts)
                }
                for batch in monitoring_results
            ],
            'alerts': [
                {
                    'alert_id': alert.alert_id,
                    'timestamp': alert.timestamp.isoformat(),
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'model': alert.model_name,
                    'message': alert.message,
                    'current_value': alert.current_value,
                    'threshold': alert.threshold_value,
                    'action': alert.recommended_action
                }
                for alert in alerts
            ],
            'summary_analytics': self._generate_summary_analytics(monitoring_results, alerts)
        }

        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Monitoring report exported to {save_path}")
        return report

    def _generate_summary_analytics(self, monitoring_results: List[MonitoringBatch],
                                   alerts: List[TrustAlert]) -> Dict:
        """Generate summary analytics for monitoring session"""

        if not monitoring_results:
            return {}

        # Trust score trends
        model_names = list(monitoring_results[0].trust_scores.keys())
        trust_trends = {}

        for model_name in model_names:
            scores = [batch.trust_scores.get(model_name, 0) for batch in monitoring_results]
            trust_trends[model_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'trend': 'stable' if np.std(scores) < 0.1 else 'variable'
            }

        # Alert analysis
        alert_analysis = {
            'total_alerts': len(alerts),
            'alerts_by_type': {},
            'alerts_by_severity': {},
            'alerts_by_model': {}
        }

        for alert in alerts:
            alert_analysis['alerts_by_type'][alert.alert_type] = \
                alert_analysis['alerts_by_type'].get(alert.alert_type, 0) + 1
            alert_analysis['alerts_by_severity'][alert.severity] = \
                alert_analysis['alerts_by_severity'].get(alert.severity, 0) + 1
            alert_analysis['alerts_by_model'][alert.model_name] = \
                alert_analysis['alerts_by_model'].get(alert.model_name, 0) + 1

        return {
            'trust_trends': trust_trends,
            'alert_analysis': alert_analysis,
            'monitoring_quality': {
                'avg_batch_processing_success': np.mean([
                    len([p for p in batch.predictions.values() if 'error' not in p]) /
                    len(batch.predictions) if batch.predictions else 0
                    for batch in monitoring_results
                ]),
                'system_stability': 'high' if len(alerts) < len(monitoring_results) * 0.1 else 'moderate'
            }
        }