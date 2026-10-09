"""
Orchestration package for Trust-Orchestrated Intelligence
Trust calculation and agent coordination
"""

from .trust_calculator import TrustCalculator
from .orchestrator import TrustOrchestratedIntelligence

__all__ = [
    'TrustCalculator',
    'TrustOrchestratedIntelligence'
]