"""
Main Orchestration Engine
Trust-Orchestrated Intelligence System Controller
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# Import our components
from ..agents.data_retrieval_agent import DataRetrievalAgent
from ..agents.clinical_reasoning_agent import ClinicalReasoningAgent
from .trust_calculator import TrustCalculator


@dataclass
class OrchestrationResult:
    """Result of orchestrated clinical decision"""
    subject_id: int
    hadm_id: Optional[int]
    status: str
    decision: Dict
    audit_trail: Dict
    total_utility: float
    execution_time: float
    timestamp: datetime


class TrustOrchestratedIntelligence:
    """
    Main orchestration engine implementing:
    max_{P} ∑_{i=1}^{n} T_i · R(A_i) subject to T_i ≥ τ_i
    """

    def __init__(self, db_connection: Any, config: Dict = None):
        """
        Initialize Trust-Orchestrated Intelligence system

        Args:
            db_connection: Database connection for MIMIC data
            config: System configuration dictionary
        """
        self.db_connection = db_connection
        self.config = config or self._default_config()

        # Initialize trust calculator
        self.trust_calculator = TrustCalculator(
            weights=self.config.get('trust_weights', {})
        )

        # Initialize agents
        self.agents = {
            'retrieval': DataRetrievalAgent(db_connection,
                                          config.get('retrieval_config', {})),
            'reasoning': ClinicalReasoningAgent()
            # Note: Treatment and Communication agents to be added later
        }

        # Orchestration history
        self.decision_history: List[OrchestrationResult] = []

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Trust-Orchestrated Intelligence system initialized")

    def _default_config(self) -> Dict:
        """Default system configuration"""
        return {
            'trust_weights': {
                'confidence': 0.4,
                'explainability': 0.3,
                'semantic_similarity': 0.3
            },
            'min_thresholds': {
                'retrieval': 0.5,
                'reasoning': 0.6,
                'treatment': 0.7,
                'communication': 0.4
            },
            'timeout_seconds': 60,
            'require_all_agents': False,
            'max_retry_attempts': 2,
            'enable_audit_logging': True
        }

    def orchestrate_clinical_decision(self,
                                   subject_id: int,
                                   hadm_id: Optional[int] = None,
                                   clinical_context: str = "",
                                   requested_agents: List[str] = None) -> OrchestrationResult:
        """
        Main orchestration method for clinical decision making

        Args:
            subject_id: Patient identifier
            hadm_id: Hospital admission ID
            clinical_context: Additional clinical context
            requested_agents: Specific agents to use (default: all available)

        Returns:
            OrchestrationResult with decision and audit trail
        """

        start_time = datetime.now()

        self.logger.info(f"Starting orchestrated decision for patient {subject_id}")

        # Determine which agents to execute
        agents_to_run = requested_agents or list(self.agents.keys())

        # Step 1: Execute agents and collect outputs
        agent_outputs = {}
        trust_scores = {}
        relevance_scores = {}
        execution_errors = {}

        try:
            # Execute Agent A₁: Data Retrieval
            if 'retrieval' in agents_to_run:
                agent_outputs['retrieval'], trust_scores['retrieval'], relevance_scores['retrieval'] = \
                    self._execute_retrieval_agent(subject_id, hadm_id)

            # Execute Agent A₂: Clinical Reasoning (depends on retrieval)
            if 'reasoning' in agents_to_run and 'retrieval' in agent_outputs:
                if trust_scores.get('retrieval', {}).get('passes_threshold', False):
                    agent_outputs['reasoning'], trust_scores['reasoning'], relevance_scores['reasoning'] = \
                        self._execute_reasoning_agent(agent_outputs['retrieval']['data'], clinical_context)
                else:
                    self.logger.warning("Skipping reasoning agent - retrieval failed trust threshold")

            # TODO: Execute Agent A₃: Treatment Recommendations
            # TODO: Execute Agent A₄: Communication

            # Step 2: Apply trust thresholds and filter agents
            valid_agents = self._filter_agents_by_trust(trust_scores)

            # Step 3: Calculate orchestration utility
            total_utility = self._calculate_total_utility(trust_scores, relevance_scores, valid_agents)

            # Step 4: Synthesize final decision
            final_decision = self._synthesize_decision(agent_outputs, trust_scores, valid_agents)

            # Step 5: Create audit trail
            audit_trail = self._create_audit_trail(
                agent_outputs, trust_scores, relevance_scores,
                valid_agents, total_utility, execution_errors
            )

            status = 'success' if valid_agents else 'insufficient_trust'

        except Exception as e:
            self.logger.error(f"Orchestration failed for patient {subject_id}: {e}")
            return self._create_error_result(str(e), subject_id, hadm_id, start_time)

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Create result object
        result = OrchestrationResult(
            subject_id=subject_id,
            hadm_id=hadm_id,
            status=status,
            decision=final_decision,
            audit_trail=audit_trail,
            total_utility=total_utility,
            execution_time=execution_time,
            timestamp=start_time
        )

        # Record decision
        self.decision_history.append(result)

        self.logger.info(f"Orchestration completed for patient {subject_id} "
                        f"in {execution_time:.2f}s with status: {status}")

        return result

    def _execute_retrieval_agent(self, subject_id: int, hadm_id: Optional[int]) -> Tuple[Dict, Dict, float]:
        """Execute data retrieval agent with trust calculation"""

        try:
            # Execute retrieval
            patient_data, retrieval_metrics = self.agents['retrieval'].retrieve_patient_data(
                subject_id, hadm_id
            )

            # Calculate trust score for retrieval
            trust_result = self.trust_calculator.calculate_trust_score(
                confidence=retrieval_metrics.precision_at_10,
                explainability=0.9,  # Data retrieval is highly explainable
                semantic_similarity=0.8,  # Structured data has high semantic clarity
                agent_type='retrieval'
            )

            # Get relevance score
            relevance_score = self.agents['retrieval'].get_relevance_score()

            # Prepare output
            output = {
                'data': patient_data,
                'metrics': retrieval_metrics,
                'performance_summary': self.agents['retrieval'].get_performance_summary()
            }

            trust_dict = {
                'trust_score': trust_result.trust_score,
                'calibrated_confidence': trust_result.calibrated_confidence,
                'explainability': trust_result.explainability,
                'semantic_similarity': trust_result.semantic_similarity,
                'threshold': trust_result.threshold,
                'passes_threshold': trust_result.passes_threshold,
                'raw_confidence': trust_result.raw_confidence
            }

            return output, trust_dict, relevance_score

        except Exception as e:
            self.logger.error(f"Retrieval agent execution failed: {e}")
            raise

    def _execute_reasoning_agent(self, patient_data: Dict, clinical_context: str) -> Tuple[Dict, Dict, float]:
        """Execute clinical reasoning agent with trust calculation"""

        try:
            # Execute reasoning
            prediction, trust_metrics = self.agents['reasoning'].generate_clinical_prediction(
                patient_data, clinical_context
            )

            # Calculate trust score for reasoning
            trust_result = self.trust_calculator.calculate_trust_score(
                confidence=trust_metrics.confidence,
                explainability=trust_metrics.explainability,
                semantic_similarity=trust_metrics.semantic_similarity,
                agent_type='reasoning'
            )

            # Get relevance score
            relevance_score = self.agents['reasoning'].get_relevance_score()

            # Prepare output
            output = {
                'prediction': prediction,
                'raw_trust_metrics': trust_metrics,
                'performance_summary': self.agents['reasoning'].get_performance_summary()
            }

            trust_dict = {
                'trust_score': trust_result.trust_score,
                'calibrated_confidence': trust_result.calibrated_confidence,
                'explainability': trust_result.explainability,
                'semantic_similarity': trust_result.semantic_similarity,
                'threshold': trust_result.threshold,
                'passes_threshold': trust_result.passes_threshold,
                'raw_confidence': trust_result.raw_confidence
            }

            return output, trust_dict, relevance_score

        except Exception as e:
            self.logger.error(f"Reasoning agent execution failed: {e}")
            raise

    def _filter_agents_by_trust(self, trust_scores: Dict) -> List[str]:
        """Filter agents that meet trust thresholds: T_i ≥ τ_i"""

        valid_agents = []

        for agent_name, trust_metrics in trust_scores.items():
            threshold = self.config['min_thresholds'].get(agent_name, 0.5)

            if trust_metrics.get('passes_threshold', False):
                valid_agents.append(agent_name)
                self.logger.info(f"Agent {agent_name} passed trust threshold "
                               f"({trust_metrics['trust_score']:.3f} >= {threshold})")
            else:
                self.logger.warning(f"Agent {agent_name} failed trust threshold "
                                  f"({trust_metrics.get('trust_score', 0):.3f} < {threshold})")

        return valid_agents

    def _calculate_total_utility(self,
                               trust_scores: Dict,
                               relevance_scores: Dict,
                               valid_agents: List[str]) -> float:
        """Calculate orchestration objective: ∑ T_i · R(A_i)"""

        total_utility = 0.0

        for agent_name in valid_agents:
            if agent_name in trust_scores and agent_name in relevance_scores:
                trust_score = trust_scores[agent_name]['trust_score']
                relevance_score = relevance_scores[agent_name]
                contribution = trust_score * relevance_score
                total_utility += contribution

                self.logger.debug(f"Agent {agent_name} contribution: "
                                f"T={trust_score:.3f} × R={relevance_score:.3f} = {contribution:.3f}")

        return float(total_utility)

    def _synthesize_decision(self,
                           agent_outputs: Dict,
                           trust_scores: Dict,
                           valid_agents: List[str]) -> Dict:
        """Synthesize final clinical decision from valid agent outputs"""

        if not valid_agents:
            return {
                'status': 'insufficient_trust',
                'message': 'No agents met trust thresholds',
                'recommendation': 'Manual review required',
                'confidence': 0.0,
                'risk_level': 'high'
            }

        # Initialize decision structure
        decision = {
            'status': 'success',
            'confidence': 0.0,
            'primary_recommendation': 'Assessment in progress',
            'supporting_evidence': [],
            'risk_factors': [],
            'next_actions': [],
            'data_quality': {},
            'clinical_summary': {}
        }

        # Incorporate retrieval agent results
        if 'retrieval' in valid_agents and 'retrieval' in agent_outputs:
            retrieval_data = agent_outputs['retrieval']['data']
            retrieval_metrics = agent_outputs['retrieval']['metrics']

            decision['data_quality'] = {
                'total_records': retrieval_metrics.total_records,
                'completeness_score': retrieval_metrics.completeness_score,
                'data_types_available': list(retrieval_data.keys()),
                'retrieval_latency': retrieval_metrics.latency_seconds
            }

        # Incorporate reasoning agent results
        if 'reasoning' in valid_agents and 'reasoning' in agent_outputs:
            prediction = agent_outputs['reasoning']['prediction']

            decision['primary_recommendation'] = prediction.primary_diagnosis
            decision['icd_code'] = prediction.icd_code
            decision['differential_diagnoses'] = prediction.differential_diagnoses
            decision['clinical_features_used'] = prediction.clinical_features_used

            # Base confidence on prediction confidence
            decision['confidence'] = prediction.confidence

            # Weight confidence by trust score
            if 'reasoning' in trust_scores:
                trust_weight = trust_scores['reasoning']['trust_score']
                decision['trust_weighted_confidence'] = decision['confidence'] * trust_weight

        # Calculate overall decision confidence based on all valid agents
        agent_contributions = []
        for agent_name in valid_agents:
            if agent_name in trust_scores:
                agent_contributions.append(trust_scores[agent_name]['trust_score'])

        decision['overall_confidence'] = float(np.mean(agent_contributions)) if agent_contributions else 0.0

        # Determine risk level
        if decision['overall_confidence'] >= 0.8:
            decision['risk_level'] = 'low'
        elif decision['overall_confidence'] >= 0.6:
            decision['risk_level'] = 'moderate'
        else:
            decision['risk_level'] = 'high'

        # Add next actions based on confidence and risk
        if decision['risk_level'] == 'high':
            decision['next_actions'].append('Immediate physician review required')
        elif decision['risk_level'] == 'moderate':
            decision['next_actions'].append('Consider additional diagnostic tests')

        decision['agents_used'] = valid_agents
        decision['total_agents_available'] = len(self.agents)

        return decision

    def _create_audit_trail(self,
                          agent_outputs: Dict,
                          trust_scores: Dict,
                          relevance_scores: Dict,
                          valid_agents: List[str],
                          total_utility: float,
                          execution_errors: Dict) -> Dict:
        """Create comprehensive audit trail for transparency"""

        return {
            'orchestration_metadata': {
                'total_utility': total_utility,
                'optimization_objective': "max ∑ T_i · R(A_i) subject to T_i ≥ τ_i",
                'agents_executed': list(agent_outputs.keys()),
                'agents_passed_trust': valid_agents,
                'execution_errors': execution_errors
            },
            'trust_analysis': {
                agent: {
                    'trust_score': metrics['trust_score'],
                    'threshold': metrics['threshold'],
                    'passed': metrics['passes_threshold'],
                    'components': {
                        'confidence': metrics['calibrated_confidence'],
                        'explainability': metrics['explainability'],
                        'semantic_similarity': metrics['semantic_similarity']
                    },
                    'raw_confidence': metrics['raw_confidence']
                }
                for agent, metrics in trust_scores.items()
            },
            'relevance_scores': relevance_scores,
            'agent_performance': {
                agent: output.get('performance_summary', {})
                for agent, output in agent_outputs.items()
            },
            'configuration_used': {
                'trust_weights': self.trust_calculator.weights,
                'trust_thresholds': self.config['min_thresholds'],
                'system_version': '1.0.0'
            },
            'quality_metrics': {
                'trust_threshold_pass_rate': len(valid_agents) / max(1, len(trust_scores)),
                'average_trust_score': float(np.mean([ts['trust_score'] for ts in trust_scores.values()])),
                'system_confidence': total_utility / max(1, len(valid_agents))
            }
        }

    def _create_error_result(self, error_msg: str, subject_id: int,
                           hadm_id: Optional[int], start_time: datetime) -> OrchestrationResult:
        """Create standardized error result"""

        return OrchestrationResult(
            subject_id=subject_id,
            hadm_id=hadm_id,
            status='error',
            decision={
                'status': 'failed',
                'error_message': error_msg,
                'recommendation': 'System error - manual review required',
                'confidence': 0.0,
                'risk_level': 'high'
            },
            audit_trail={
                'error': error_msg,
                'timestamp': start_time,
                'system_state': 'error'
            },
            total_utility=0.0,
            execution_time=(datetime.now() - start_time).total_seconds(),
            timestamp=start_time
        )

    def get_performance_summary(self) -> Dict:
        """Get comprehensive orchestration performance summary"""

        if not self.decision_history:
            return {'message': 'No decisions recorded'}

        recent_decisions = self.decision_history[-100:]  # Last 100 decisions

        # Calculate success metrics
        success_rate = sum(1 for d in recent_decisions
                          if d.status == 'success') / len(recent_decisions)

        avg_utility = np.mean([d.total_utility for d in recent_decisions])
        avg_execution_time = np.mean([d.execution_time for d in recent_decisions])

        # Trust metrics
        trust_stats = self.trust_calculator.get_trust_statistics()

        # Agent performance
        agent_performance = {}
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'get_performance_summary'):
                agent_performance[agent_name] = agent.get_performance_summary()

        return {
            'system_metrics': {
                'total_decisions': len(self.decision_history),
                'recent_success_rate': float(success_rate),
                'average_utility': float(avg_utility),
                'average_execution_time_seconds': float(avg_execution_time),
                'system_uptime': 'active'
            },
            'trust_metrics': trust_stats,
            'agent_performance': agent_performance,
            'configuration': {
                'trust_weights': self.trust_calculator.weights,
                'thresholds': self.config['min_thresholds'],
                'agents_available': list(self.agents.keys())
            }
        }

    def update_trust_calibration(self, validation_data: List[Dict]) -> None:
        """
        Update trust calibration using validation data

        Args:
            validation_data: List of dicts with 'predictions' and 'ground_truth'
        """

        if not validation_data:
            self.logger.warning("No validation data provided for trust calibration")
            return

        try:
            # Extract predictions and ground truth
            predictions = []
            ground_truth = []

            for record in validation_data:
                if 'prediction_confidence' in record and 'actual_outcome' in record:
                    predictions.append(record['prediction_confidence'])
                    ground_truth.append(bool(record['actual_outcome']))

            if len(predictions) >= 10:  # Minimum data requirement
                self.trust_calculator.update_calibration(predictions, ground_truth)
                self.logger.info(f"Trust calibration updated with {len(predictions)} samples")
            else:
                self.logger.warning(f"Insufficient validation data for calibration: {len(predictions)} samples")

        except Exception as e:
            self.logger.error(f"Trust calibration update failed: {e}")

    def export_decision_history(self) -> List[Dict]:
        """Export decision history for analysis"""

        return [
            {
                'subject_id': result.subject_id,
                'hadm_id': result.hadm_id,
                'status': result.status,
                'decision': result.decision,
                'total_utility': result.total_utility,
                'execution_time': result.execution_time,
                'timestamp': result.timestamp.isoformat(),
                # Include key audit trail metrics
                'trust_scores': result.audit_trail.get('trust_analysis', {}),
                'agents_used': result.decision.get('agents_used', [])
            }
            for result in self.decision_history
        ]