"""
Interactive Trust Threshold Configuration
Clinical risk-based threshold management system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

from .enhanced_trust_analyzer import ClinicalTrustMetrics


class InteractiveTrustThresholdConfig:
    """Interactive threshold configuration inspired by demo sensitivity analysis"""

    def __init__(self, trust_analysis: Dict[str, ClinicalTrustMetrics], ml_pipeline):
        """
        Initialize Interactive Trust Threshold Configuration

        Args:
            trust_analysis: Trust analysis results
            ml_pipeline: ML pipeline with trained models
        """
        self.trust_analysis = trust_analysis
        self.ml_pipeline = ml_pipeline
        self.logger = logging.getLogger(__name__)

        # Default threshold configuration
        self.current_thresholds = {
            'competence': 0.5,
            'explainability': 0.5,
            'semantic_similarity': 0.5,
            'overall_trust': 0.6
        }

        # Predefined threshold configurations
        self.threshold_configs = [
            {
                'name': 'Conservative',
                'competence': 0.7,
                'explainability': 0.7,
                'semantic': 0.7,
                'overall': 0.75,
                'description': 'High confidence required for all components'
            },
            {
                'name': 'Moderate',
                'competence': 0.6,
                'explainability': 0.6,
                'semantic': 0.6,
                'overall': 0.65,
                'description': 'Balanced approach for most clinical scenarios'
            },
            {
                'name': 'Permissive',
                'competence': 0.5,
                'explainability': 0.5,
                'semantic': 0.5,
                'overall': 0.55,
                'description': 'Lower barriers with increased human oversight'
            },
            {
                'name': 'Clinical-Optimized',
                'competence': 0.65,
                'explainability': 0.75,
                'semantic': 0.5,
                'overall': 0.7,
                'description': 'Prioritizes explainability for clinical acceptance'
            }
        ]

    def analyze_threshold_sensitivity(self) -> Dict[str, Dict]:
        """Analyze sensitivity to different threshold configurations"""
        self.logger.info("Analyzing trust threshold sensitivity")

        results = {}

        for config in self.threshold_configs:
            passing_models = []

            for model_name, trust_info in self.trust_analysis.items():
                competence = trust_info.competence
                explainability = trust_info.explainability
                semantic = trust_info.semantic_similarity
                trust_score = trust_info.trust_score

                passes = (
                    competence >= config['competence'] and
                    explainability >= config['explainability'] and
                    semantic >= config['semantic'] and
                    trust_score >= config['overall']
                )

                if passes:
                    passing_models.append({
                        'model': model_name,
                        'trust_score': trust_score,
                        'competence': competence,
                        'explainability': explainability,
                        'semantic_similarity': semantic
                    })

            results[config['name']] = {
                'config': config,
                'passing_models': passing_models,
                'pass_rate': len(passing_models) / len(self.trust_analysis),
                'deployment_suitability': self._assess_deployment_suitability(
                    len(passing_models), len(self.trust_analysis)
                )
            }

            self.logger.debug(f"Threshold sensitivity analysis complete for {config['name']}")

        return results

    def recommend_clinical_thresholds(self) -> Dict[str, Dict]:
        """Recommend clinical-specific thresholds based on risk levels"""
        self.logger.info("Generating clinical threshold recommendations")

        # Analyze current model performance for percentile-based thresholds
        competence_scores = [info.competence for info in self.trust_analysis.values()]
        explainability_scores = [info.explainability for info in self.trust_analysis.values()]
        trust_scores = [info.trust_score for info in self.trust_analysis.values()]

        # Clinical risk levels with adaptive thresholds
        risk_levels = {
            'ICU Critical Care': {
                'description': 'Life-threatening decisions requiring highest confidence',
                'competence': max(0.8, np.percentile(competence_scores, 90)),
                'explainability': max(0.8, np.percentile(explainability_scores, 90)),
                'semantic': 0.6,
                'overall': max(0.8, np.percentile(trust_scores, 90)),
                'human_oversight': 'Continuous monitoring with immediate alerts',
                'acceptable_risk': 'Minimal - <1% false positive rate'
            },
            'Emergency Department': {
                'description': 'Time-sensitive decisions with moderate risk tolerance',
                'competence': max(0.7, np.percentile(competence_scores, 75)),
                'explainability': max(0.7, np.percentile(explainability_scores, 75)),
                'semantic': 0.6,
                'overall': max(0.7, np.percentile(trust_scores, 75)),
                'human_oversight': 'Expert validation within 1 hour',
                'acceptable_risk': 'Low - <5% false positive rate'
            },
            'General Ward': {
                'description': 'Routine monitoring with clinical oversight',
                'competence': max(0.6, np.percentile(competence_scores, 60)),
                'explainability': max(0.6, np.percentile(explainability_scores, 60)),
                'semantic': 0.5,
                'overall': max(0.6, np.percentile(trust_scores, 60)),
                'human_oversight': 'Daily clinical review',
                'acceptable_risk': 'Moderate - <10% false positive rate'
            },
            'Screening/Triage': {
                'description': 'Population-level screening with human verification',
                'competence': max(0.5, np.percentile(competence_scores, 50)),
                'explainability': max(0.5, np.percentile(explainability_scores, 50)),
                'semantic': 0.4,
                'overall': max(0.5, np.percentile(trust_scores, 50)),
                'human_oversight': 'Batch review and follow-up',
                'acceptable_risk': 'Higher - <15% false positive rate for sensitivity'
            }
        }

        # Calculate model compatibility for each risk level
        for level_name, thresholds in risk_levels.items():
            passing_models = []
            for model_name, info in self.trust_analysis.items():
                if (info.competence >= thresholds['competence'] and
                    info.explainability >= thresholds['explainability'] and
                    info.semantic_similarity >= thresholds['semantic'] and
                    info.trust_score >= thresholds['overall']):
                    passing_models.append(model_name)

            thresholds['compatible_models'] = passing_models
            thresholds['model_coverage'] = len(passing_models) / len(self.trust_analysis)

        return risk_levels

    def suggest_dynamic_threshold_adjustment(self) -> Dict[str, Any]:
        """Suggest dynamic threshold adjustments based on context"""
        self.logger.info("Analyzing dynamic threshold adjustment opportunities")

        # Calculate ensemble reliability metrics
        trust_scores = [info.trust_score for info in self.trust_analysis.values()]
        avg_trust = np.mean(trust_scores)
        trust_std = np.std(trust_scores)

        adjustment_strategy = {
            'current_ensemble_metrics': {
                'average_trust': avg_trust,
                'trust_variance': trust_std,
                'trust_range': (min(trust_scores), max(trust_scores)),
                'coefficient_of_variation': trust_std / avg_trust if avg_trust > 0 else 0
            }
        }

        # Variance-based recommendations
        if trust_std < 0.1:  # Low variance - consistent models
            adjustment_strategy['consistency_recommendation'] = {
                'type': 'HIGH_CONSISTENCY',
                'action': 'Reduce thresholds by 5-10% for broader model utilization',
                'rationale': 'Models show consistent performance',
                'suggested_ensemble': 'Weighted averaging for improved stability'
            }
        elif trust_std > 0.2:  # High variance - inconsistent models
            adjustment_strategy['consistency_recommendation'] = {
                'type': 'HIGH_VARIANCE',
                'action': 'Increase thresholds by 10-15% for selective model use',
                'rationale': 'Significant performance differences detected',
                'suggested_ensemble': 'Use only top-performing models'
            }
        else:
            adjustment_strategy['consistency_recommendation'] = {
                'type': 'MODERATE_CONSISTENCY',
                'action': 'Current thresholds appropriate',
                'rationale': 'Acceptable model consistency',
                'suggested_ensemble': 'Weighted ensemble based on trust scores'
            }

        # Temporal adjustment strategies
        adjustment_strategy['temporal_strategies'] = {
            'real_time': {
                'threshold_adjustment': 'Lower by 10-15%',
                'oversight': 'Immediate human verification',
                'rationale': 'Time-critical decisions with safety net'
            },
            'batch_processing': {
                'threshold_adjustment': 'Increase by 10-20%',
                'oversight': 'Batch review and validation',
                'rationale': 'Higher confidence for autonomous processing'
            },
            'emergency_situations': {
                'threshold_adjustment': 'Lower explainability, maintain competence',
                'oversight': 'Post-hoc explanation generation',
                'rationale': 'Prioritize accuracy in critical situations'
            }
        }

        return adjustment_strategy

    def _assess_deployment_suitability(self, passing_models: int, total_models: int) -> str:
        """Assess deployment suitability based on passing model rate"""
        pass_rate = passing_models / total_models if total_models > 0 else 0

        if pass_rate >= 0.75:
            return "Excellent - Multiple models available for ensemble"
        elif pass_rate >= 0.5:
            return "Good - Sufficient models for reliable deployment"
        elif pass_rate >= 0.25:
            return "Limited - Consider threshold adjustment or model improvement"
        else:
            return "Insufficient - Significant development needed"

    def export_threshold_configuration(self, save_path: str,
                                     selected_config: Optional[str] = None) -> Dict:
        """Export threshold configuration for deployment"""

        # Generate all analyses
        sensitivity_analysis = self.analyze_threshold_sensitivity()
        clinical_thresholds = self.recommend_clinical_thresholds()
        dynamic_strategy = self.suggest_dynamic_threshold_adjustment()

        # Select configuration (default to Clinical-Optimized)
        if selected_config and selected_config in sensitivity_analysis:
            selected = sensitivity_analysis[selected_config]
        else:
            selected = sensitivity_analysis['Clinical-Optimized']

        configuration = {
            'selected_configuration': {
                'name': selected_config or 'Clinical-Optimized',
                'thresholds': selected['config'],
                'passing_models': selected['passing_models'],
                'deployment_readiness': selected['deployment_suitability']
            },
            'clinical_risk_levels': clinical_thresholds,
            'dynamic_adjustment_strategy': dynamic_strategy,
            'sensitivity_analysis': sensitivity_analysis,
            'deployment_timestamp': pd.Timestamp.now().isoformat(),
            'model_compatibility_matrix': self._generate_compatibility_matrix()
        }

        # Save configuration
        import json
        with open(save_path, 'w') as f:
            json.dump(configuration, f, indent=2, default=str)

        self.logger.info(f"Threshold configuration exported to {save_path}")
        return configuration

    def _generate_compatibility_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Generate model compatibility matrix across different threshold configurations"""

        compatibility_matrix = {}

        for config in self.threshold_configs:
            compatibility_matrix[config['name']] = {}

            for model_name, trust_info in self.trust_analysis.items():
                passes = (
                    trust_info.competence >= config['competence'] and
                    trust_info.explainability >= config['explainability'] and
                    trust_info.semantic_similarity >= config['semantic'] and
                    trust_info.trust_score >= config['overall']
                )
                compatibility_matrix[config['name']][model_name] = passes

        return compatibility_matrix