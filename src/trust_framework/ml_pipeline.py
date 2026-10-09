"""
Enhanced Trust-Orchestrated ML Pipeline
Production-ready ML pipeline with comprehensive trust management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import joblib

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.orchestration.trust_calculator import TrustCalculator, TrustScore
from src.trust_framework.enhanced_trust_analyzer import EnhancedTrustAnalyzer, ClinicalTrustMetrics


class EnhancedTrustOrchestrationPipeline:
    """Production-ready ML pipeline with comprehensive trust management"""

    def __init__(self, random_state: int = 42):
        """
        Initialize Enhanced Trust Orchestration Pipeline

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

        # Initialize models
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=random_state,
                learning_rate=0.1
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=random_state,
                max_iter=500,
                early_stopping=True
            )
        }

        # Initialize trust calculator
        self.trust_calculator = TrustCalculator()

        # Storage for results
        self.performance_metrics = {}
        self.trust_scores = {}
        self.feature_importance = {}
        self.model_predictions = {}

        # Data splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Pipeline configuration
        self.pipeline_config = {
            'test_size': 0.2,
            'validation_folds': 5,
            'trust_threshold': 0.6,
            'performance_threshold': 0.7,
            'feature_selection_threshold': 0.01
        }

    def prepare_data(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Prepare data for ML pipeline"""
        self.logger.info(f"Preparing data: {len(features)} samples, {len(features.columns)} features")

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, target,
            test_size=self.pipeline_config['test_size'],
            random_state=self.random_state,
            stratify=target
        )

        self.logger.info(f"Data split complete: Train={len(self.X_train)}, Test={len(self.X_test)}")
        self.logger.info(f"Target distribution - Train: {self.y_train.value_counts().to_dict()}")
        self.logger.info(f"Target distribution - Test: {self.y_test.value_counts().to_dict()}")

    def train_models(self) -> Dict[str, Any]:
        """Train all models with cross-validation"""
        self.logger.info("Training models with cross-validation")

        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        training_results = {}

        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name}")

            try:
                # Cross-validation
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=StratifiedKFold(
                        n_splits=self.pipeline_config['validation_folds'],
                        shuffle=True,
                        random_state=self.random_state
                    ),
                    scoring='roc_auc',
                    n_jobs=-1
                )

                # Train final model
                model.fit(self.X_train, self.y_train)

                # Test set predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]

                # Calculate performance metrics
                metrics = self._calculate_performance_metrics(
                    self.y_test, y_pred, y_pred_proba
                )

                # Add cross-validation results
                metrics['cv_auc_mean'] = cv_scores.mean()
                metrics['cv_auc_std'] = cv_scores.std()

                self.performance_metrics[model_name] = metrics
                self.model_predictions[model_name] = {
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }

                # Extract feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = list(zip(
                        self.X_train.columns,
                        model.feature_importances_
                    ))
                    importance.sort(key=lambda x: x[1], reverse=True)
                    self.feature_importance[model_name] = importance
                elif hasattr(model, 'coef_'):
                    importance = list(zip(
                        self.X_train.columns,
                        abs(model.coef_[0])
                    ))
                    importance.sort(key=lambda x: x[1], reverse=True)
                    self.feature_importance[model_name] = importance

                training_results[model_name] = {
                    'status': 'success',
                    'metrics': metrics,
                    'cv_scores': cv_scores.tolist()
                }

                self.logger.info(f"{model_name} training complete - AUC: {metrics['auc']:.3f}")

            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                training_results[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }

        self.logger.info(f"Model training complete: {len(training_results)} models processed")
        return training_results

    def calculate_trust_scores(self) -> Dict[str, ClinicalTrustMetrics]:
        """Calculate comprehensive trust scores for all models"""
        self.logger.info("Calculating trust scores using enhanced trust analyzer")

        if not self.performance_metrics:
            raise ValueError("Models not trained. Call train_models() first.")

        # Calculate base trust scores from performance
        for model_name, metrics in self.performance_metrics.items():
            # Combine multiple performance metrics for competence
            competence_score = (
                metrics['accuracy'] * 0.3 +
                metrics['precision'] * 0.2 +
                metrics['recall'] * 0.2 +
                metrics['f1_score'] * 0.15 +
                metrics['auc'] * 0.15
            )

            self.trust_scores[model_name] = competence_score

        # Use enhanced trust analyzer for comprehensive analysis
        trust_analyzer = EnhancedTrustAnalyzer(self, None)  # Enhanced data not needed for this calculation
        trust_analysis = trust_analyzer.calculate_comprehensive_trust_scores()

        self.logger.info(f"Trust score calculation complete for {len(trust_analysis)} models")
        return trust_analysis

    def _calculate_performance_metrics(self, y_true, y_pred, y_pred_proba) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'negative_predictive_value': self._calculate_npv(y_true, y_pred)
        }

        return metrics

    def _calculate_specificity(self, y_true, y_pred) -> float:
        """Calculate specificity (true negative rate)"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def _calculate_npv(self, y_true, y_pred) -> float:
        """Calculate negative predictive value"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0

    def select_trusted_models(self, trust_analysis: Dict[str, ClinicalTrustMetrics],
                            min_trust_score: Optional[float] = None) -> List[str]:
        """Select models that meet trust criteria"""

        min_trust = min_trust_score or self.pipeline_config['trust_threshold']

        trusted_models = [
            model_name for model_name, metrics in trust_analysis.items()
            if metrics.trust_score >= min_trust and metrics.passes_threshold
        ]

        self.logger.info(f"Selected {len(trusted_models)} trusted models: {trusted_models}")
        return trusted_models

    def create_ensemble_prediction(self, trusted_models: List[str],
                                 weighting_strategy: str = 'trust_weighted') -> Dict[str, Any]:
        """Create ensemble predictions from trusted models"""
        self.logger.info(f"Creating ensemble prediction from {len(trusted_models)} models")

        if not trusted_models:
            raise ValueError("No trusted models available for ensemble")

        # Get predictions from trusted models
        model_probabilities = []
        model_weights = []

        for model_name in trusted_models:
            if model_name in self.model_predictions:
                probabilities = self.model_predictions[model_name]['probabilities']
                model_probabilities.append(probabilities)

                # Determine weight based on strategy
                if weighting_strategy == 'trust_weighted':
                    weight = self.trust_scores.get(model_name, 1.0)
                elif weighting_strategy == 'performance_weighted':
                    weight = self.performance_metrics[model_name]['auc']
                else:  # equal_weighted
                    weight = 1.0

                model_weights.append(weight)

        if not model_probabilities:
            raise ValueError("No predictions available from trusted models")

        # Normalize weights
        model_weights = np.array(model_weights)
        model_weights = model_weights / model_weights.sum()

        # Calculate weighted ensemble prediction
        ensemble_probabilities = np.average(
            np.array(model_probabilities),
            axis=0,
            weights=model_weights
        )

        ensemble_predictions = (ensemble_probabilities > 0.5).astype(int)

        # Calculate ensemble performance
        ensemble_metrics = self._calculate_performance_metrics(
            self.y_test, ensemble_predictions, ensemble_probabilities
        )

        ensemble_result = {
            'predictions': ensemble_predictions,
            'probabilities': ensemble_probabilities,
            'metrics': ensemble_metrics,
            'component_models': trusted_models,
            'model_weights': dict(zip(trusted_models, model_weights)),
            'weighting_strategy': weighting_strategy
        }

        self.logger.info(f"Ensemble created - AUC: {ensemble_metrics['auc']:.3f}")
        return ensemble_result

    def save_pipeline(self, save_dir: str) -> None:
        """Save complete pipeline to disk"""
        import os

        os.makedirs(save_dir, exist_ok=True)

        # Save trained models
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)

        # Save pipeline state
        pipeline_state = {
            'performance_metrics': self.performance_metrics,
            'trust_scores': self.trust_scores,
            'feature_importance': self.feature_importance,
            'pipeline_config': self.pipeline_config,
            'feature_names': list(self.X_train.columns) if self.X_train is not None else None
        }

        state_path = os.path.join(save_dir, "pipeline_state.joblib")
        joblib.dump(pipeline_state, state_path)

        self.logger.info(f"Pipeline saved to {save_dir}")

    def load_pipeline(self, save_dir: str) -> None:
        """Load complete pipeline from disk"""
        import os

        # Load trained models
        for model_name in self.models.keys():
            model_path = os.path.join(save_dir, f"{model_name}.joblib")
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)

        # Load pipeline state
        state_path = os.path.join(save_dir, "pipeline_state.joblib")
        if os.path.exists(state_path):
            pipeline_state = joblib.load(state_path)
            self.performance_metrics = pipeline_state.get('performance_metrics', {})
            self.trust_scores = pipeline_state.get('trust_scores', {})
            self.feature_importance = pipeline_state.get('feature_importance', {})
            self.pipeline_config.update(pipeline_state.get('pipeline_config', {}))

        self.logger.info(f"Pipeline loaded from {save_dir}")

    def export_pipeline_summary(self, save_path: str) -> Dict[str, Any]:
        """Export comprehensive pipeline summary"""

        summary = {
            'pipeline_overview': {
                'total_models': len(self.models),
                'trained_models': len(self.performance_metrics),
                'trust_calculated': len(self.trust_scores) > 0,
                'data_prepared': self.X_train is not None
            },
            'model_performance': self.performance_metrics,
            'trust_analysis': {
                model_name: score for model_name, score in self.trust_scores.items()
            },
            'feature_analysis': {
                'total_features': len(self.X_train.columns) if self.X_train is not None else 0,
                'top_features_by_model': {
                    model_name: importance[:10] if importance else []
                    for model_name, importance in self.feature_importance.items()
                }
            },
            'deployment_readiness': {
                'high_performance_models': [
                    model_name for model_name, metrics in self.performance_metrics.items()
                    if metrics['auc'] >= self.pipeline_config['performance_threshold']
                ],
                'trusted_models': [
                    model_name for model_name, score in self.trust_scores.items()
                    if score >= self.pipeline_config['trust_threshold']
                ],
                'ensemble_ready': len([
                    score for score in self.trust_scores.values()
                    if score >= self.pipeline_config['trust_threshold']
                ]) >= 2
            },
            'pipeline_metadata': {
                'creation_timestamp': pd.Timestamp.now().isoformat(),
                'configuration': self.pipeline_config,
                'random_state': self.random_state
            }
        }

        # Save summary
        import json
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Pipeline summary exported to {save_path}")
        return summary