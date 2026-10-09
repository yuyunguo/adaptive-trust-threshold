"""
Clinical Explainability Dashboard
Provides clinical decision support and model interpretability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

from .enhanced_trust_analyzer import ClinicalTrustMetrics


class ClinicalExplainabilityDashboard:
    """Enhanced clinical explanations inspired by demo framework"""

    def __init__(self, ml_pipeline, trust_analysis: Dict[str, ClinicalTrustMetrics],
                 enhanced_data):
        """
        Initialize Clinical Explainability Dashboard

        Args:
            ml_pipeline: ML pipeline with trained models
            trust_analysis: Trust analysis results
            enhanced_data: Enhanced clinical dataset
        """
        self.ml_pipeline = ml_pipeline
        self.trust_analysis = trust_analysis
        self.data = enhanced_data
        self.logger = logging.getLogger(__name__)

    def generate_clinical_explanations(self) -> Dict[str, Dict]:
        """Generate clinical explanations for model predictions"""
        self.logger.info("Generating clinical model explanations")

        explanations = {}

        for model_name, trust_info in self.trust_analysis.items():
            # Clinical interpretability assessment
            interpretability = trust_info.explainability
            trust_score = trust_info.trust_score

            # Determine clinical use recommendation
            if interpretability >= 0.8:
                clinical_use = "✅ Suitable for direct clinical decision support"
                explanation_type = "Feature coefficients and clinical rules available"
            elif interpretability >= 0.6:
                clinical_use = "⚠️ Suitable with expert supervision"
                explanation_type = "Feature importance rankings available"
            else:
                clinical_use = "❌ Research use only - requires interpretation layer"
                explanation_type = "Black box - use SHAP/LIME for explanations"

            # Trust-based recommendation
            if trust_score >= 0.7:
                recommendation = "🟢 HIGH CONFIDENCE - Deploy in clinical workflow"
            elif trust_score >= 0.5:
                recommendation = "🟡 MODERATE CONFIDENCE - Use with clinical oversight"
            else:
                recommendation = "🔴 LOW CONFIDENCE - Additional validation needed"

            explanations[model_name] = {
                'clinical_use': clinical_use,
                'explanation_type': explanation_type,
                'recommendation': recommendation,
                'interpretability_score': interpretability,
                'trust_score': trust_score,
                'clinical_deployment_ready': trust_score >= 0.6 and interpretability >= 0.6
            }

            self.logger.debug(f"Clinical explanation generated for {model_name}")

        return explanations

    def clinical_feature_importance_summary(self) -> Dict[str, List[Tuple[str, float]]]:
        """Summarize clinical feature importance across models"""
        self.logger.info("Analyzing clinical feature importance")

        feature_analysis = {}

        if hasattr(self.ml_pipeline, 'feature_importance'):
            # Clinical feature categories
            clinical_domains = {
                'Vital Signs': ['hr_', 'sbp_', 'dbp_', 'temp_', 'resp_', 'spo2_'],
                'Demographics': ['age', 'gender'],
                'ICU Metrics': ['icu_', 'los', 'emergency', 'intensive'],
                'Laboratory': ['lab_', 'abnormal'],
                'Clinical Complexity': ['complexity', 'shock', 'pulse']
            }

            for model_name in self.trust_analysis.keys():
                if model_name in self.ml_pipeline.feature_importance:
                    model_features = self.ml_pipeline.feature_importance[model_name]

                    domain_importance = {domain: [] for domain in clinical_domains}

                    for feature, importance in model_features[:15]:  # Top 15 features
                        assigned = False
                        for domain, keywords in clinical_domains.items():
                            if any(keyword in feature.lower() for keyword in keywords):
                                domain_importance[domain].append((feature, importance))
                                assigned = True
                                break
                        if not assigned:
                            domain_importance['Other'] = domain_importance.get('Other', [])
                            domain_importance['Other'].append((feature, importance))

                    feature_analysis[model_name] = domain_importance

        return feature_analysis

    def generate_trust_orchestration_summary(self) -> Dict[str, Any]:
        """Generate trust orchestration recommendations"""
        self.logger.info("Generating trust orchestration summary")

        # Model ranking by trust score
        sorted_models = sorted(
            self.trust_analysis.items(),
            key=lambda x: x[1].trust_score,
            reverse=True
        )

        # Ensemble recommendation
        high_trust_models = [
            name for name, info in self.trust_analysis.items()
            if info.trust_score >= 0.6
        ]

        # Clinical deployment readiness
        trust_scores = [info.trust_score for info in self.trust_analysis.values()]
        avg_trust = np.mean(trust_scores)
        pass_rate = sum(
            1 for info in self.trust_analysis.values()
            if info.passes_threshold
        ) / len(self.trust_analysis)

        # Determine deployment readiness
        if avg_trust >= 0.7 and pass_rate >= 0.75:
            deployment_status = "✅ READY for clinical pilot deployment"
        elif avg_trust >= 0.6 and pass_rate >= 0.5:
            deployment_status = "⚠️ CONDITIONAL deployment with clinical oversight"
        else:
            deployment_status = "❌ NOT READY - requires additional development"

        return {
            'model_ranking': [
                {
                    'model': name.replace('_', ' ').title(),
                    'trust_score': info.trust_score,
                    'passes_threshold': info.passes_threshold,
                    'clinical_suitability': info.clinical_suitability
                }
                for name, info in sorted_models
            ],
            'ensemble_recommendations': {
                'high_trust_models': [m.replace('_', ' ').title() for m in high_trust_models],
                'ensemble_suitable': len(high_trust_models) >= 2
            },
            'deployment_metrics': {
                'average_trust': avg_trust,
                'threshold_pass_rate': pass_rate,
                'deployment_status': deployment_status
            }
        }

    def export_clinical_report(self, save_path: str) -> Dict[str, Any]:
        """Export comprehensive clinical explainability report"""

        # Generate all components
        explanations = self.generate_clinical_explanations()
        feature_analysis = self.clinical_feature_importance_summary()
        orchestration_summary = self.generate_trust_orchestration_summary()

        # Create comprehensive report
        clinical_report = {
            'summary': {
                'total_models': len(self.trust_analysis),
                'deployment_ready_models': sum(
                    1 for exp in explanations.values()
                    if exp['clinical_deployment_ready']
                ),
                'average_trust_score': orchestration_summary['deployment_metrics']['average_trust'],
                'generated_timestamp': pd.Timestamp.now().isoformat()
            },
            'model_explanations': explanations,
            'feature_importance_by_domain': feature_analysis,
            'trust_orchestration': orchestration_summary,
            'clinical_recommendations': self._generate_clinical_recommendations(
                explanations, orchestration_summary
            )
        }

        # Save as JSON for programmatic access
        import json
        with open(save_path, 'w') as f:
            json.dump(clinical_report, f, indent=2, default=str)

        self.logger.info(f"Clinical explainability report exported to {save_path}")
        return clinical_report

    def _generate_clinical_recommendations(self, explanations: Dict,
                                         orchestration: Dict) -> List[Dict[str, str]]:
        """Generate specific clinical deployment recommendations"""

        recommendations = []

        # High-trust model recommendations
        high_trust_models = [
            model for model, exp in explanations.items()
            if exp['trust_score'] >= 0.7
        ]

        if high_trust_models:
            recommendations.append({
                'type': 'PRIMARY_DEPLOYMENT',
                'priority': 'HIGH',
                'recommendation': f"Deploy {', '.join([m.replace('_', ' ').title() for m in high_trust_models])} "
                                 f"for direct clinical decision support",
                'rationale': "High trust scores (≥0.7) with adequate explainability"
            })

        # Moderate trust recommendations
        moderate_trust_models = [
            model for model, exp in explanations.items()
            if 0.5 <= exp['trust_score'] < 0.7
        ]

        if moderate_trust_models:
            recommendations.append({
                'type': 'SUPERVISED_DEPLOYMENT',
                'priority': 'MEDIUM',
                'recommendation': f"Deploy {', '.join([m.replace('_', ' ').title() for m in moderate_trust_models])} "
                                 f"with mandatory clinical oversight",
                'rationale': "Moderate trust scores require human verification"
            })

        # Ensemble recommendations
        if orchestration['ensemble_recommendations']['ensemble_suitable']:
            recommendations.append({
                'type': 'ENSEMBLE_STRATEGY',
                'priority': 'HIGH',
                'recommendation': "Implement ensemble approach using top-performing models",
                'rationale': "Multiple high-trust models available for improved reliability"
            })

        # Training/improvement recommendations
        low_trust_models = [
            model for model, exp in explanations.items()
            if exp['trust_score'] < 0.5
        ]

        if low_trust_models:
            recommendations.append({
                'type': 'MODEL_IMPROVEMENT',
                'priority': 'MEDIUM',
                'recommendation': f"Retrain or improve {', '.join([m.replace('_', ' ').title() for m in low_trust_models])}",
                'rationale': "Trust scores below clinical deployment threshold"
            })

        return recommendations