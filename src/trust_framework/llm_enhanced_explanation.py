"""
LLM-Enhanced Explanation Generator using Ollama/qwen3
Enhances clinical explanations with richer context and medical terminology
"""

import subprocess
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class LLMEnhancedExplanationGenerator:
    """Generate enhanced clinical explanations using Ollama/qwen3 LLM"""
    
    def __init__(self, model_name: str = "qwen3:latest", use_llm: bool = True):
        """
        Initialize LLM-enhanced explanation generator
        
        Args:
            model_name: Ollama model name (default: qwen3:latest)
            use_llm: Whether to use LLM (False = fallback to template-based)
        """
        self.model_name = model_name
        self.use_llm = use_llm
        self.llm_available = False
        
        if use_llm:
            self.llm_available = self._check_ollama_availability()
            if not self.llm_available:
                logger.warning("⚠️  Ollama not available, using template-based explanations")
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is available and model exists"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Check if model is in list
                if self.model_name.split(':')[0] in result.stdout.lower():
                    logger.info(f"✅ Ollama available with {self.model_name}")
                    return True
                else:
                    logger.warning(f"⚠️  Model {self.model_name} not found in Ollama")
                    return False
            return False
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"⚠️  Ollama check failed: {e}")
            return False
    
    def _call_ollama(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Call Ollama API with prompt"""
        if not self.llm_available:
            return None
        
        try:
            # Use ollama run - it accepts prompt as argument
            result = subprocess.run(
                ['ollama', 'run', self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=8  # Short timeout - if LLM is slow, use fallback (base explanation is still good)
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Clean up response - remove common prefixes and thinking patterns
                # qwen3 often includes "Thinking..." or similar
                lines = response.split('\n')
                cleaned_lines = []
                skip_thinking = True
                
                for line in lines:
                    line = line.strip()
                    # Skip thinking/processing lines
                    if skip_thinking and ('thinking' in line.lower() or 'let me' in line.lower() or 
                                         line.lower().startswith('okay') or line.lower().startswith('the user')):
                        continue
                    skip_thinking = False  # Once we pass thinking, keep everything
                    
                    # Skip empty lines and prompt echoes
                    if line and not line.startswith('Generate') and not line.startswith('Context:'):
                        cleaned_lines.append(line)
                
                response = ' '.join(cleaned_lines).strip()
                
                # Remove prompt if it was echoed
                if prompt[:30] in response:
                    response = response.split(prompt, 1)[-1].strip()
                
                # If response is too short or empty, return None to use fallback
                if len(response) < 15:
                    return None
                    
                return response
            else:
                logger.warning(f"⚠️  Ollama call failed: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  Ollama call timed out (model may be slow, using fallback)")
            return None
        except Exception as e:
            logger.warning(f"⚠️  Ollama call error: {e}, using fallback")
            return None
    
    def enhance_explanation(self, 
                          base_explanation: str,
                          prediction_data: Dict[str, Any],
                          features: Dict[str, Any]) -> str:
        """
        Enhance explanation with LLM-generated clinical context
        
        Args:
            base_explanation: Template-based explanation
            prediction_data: Patient prediction data
            features: Clinical features dict
            
        Returns:
            Enhanced explanation with richer clinical context
        """
        if not self.use_llm or not self.llm_available:
            return base_explanation
        
        # Extract key clinical information
        diagnosis = prediction_data.get('diagnosis', 'Unknown condition')
        confidence = prediction_data.get('confidence', 0.5)
        
        # Extract enhanced features
        gcs_values = features.get('gcs_total', [])
        map_values = features.get('map', [])
        urine_values = features.get('urine_output', [])
        
        gcs_mean = sum(gcs_values) / len(gcs_values) if gcs_values else None
        map_mean = sum(map_values) / len(map_values) if map_values else None
        urine_total = sum(urine_values) if urine_values else None
        
        # Build prompt for LLM - more concise for faster response
        clinical_measurements = []
        if gcs_mean is not None:
            clinical_measurements.append(f"GCS: {gcs_mean:.1f}")
        if map_mean is not None:
            clinical_measurements.append(f"MAP: {map_mean:.1f} mmHg")
        if urine_total is not None:
            clinical_measurements.append(f"Urine output: {urine_total:.0f} mL/24h")
        
        measurements_str = ", ".join(clinical_measurements) if clinical_measurements else "Standard ICU monitoring"
        
        # Simplified prompt for faster response
        prompt = f"""Write 2 sentences about ICU patient: {diagnosis} (risk: {confidence:.0%}). Include: {measurements_str}. Use medical terms."""
        
        # Call LLM
        llm_response = self._call_ollama(prompt, max_tokens=150)
        
        if llm_response:
            # Clean up response
            explanation = llm_response.strip()
            # Remove any prompt artifacts
            if "Explanation:" in explanation:
                explanation = explanation.split("Explanation:")[-1].strip()
            
            logger.debug(f"✅ LLM-enhanced explanation generated")
            return explanation
        else:
            # Fallback to base explanation
            logger.debug("⚠️  LLM call failed, using base explanation")
            return base_explanation
    
    def enhance_semantic_analysis(self, explanation: str) -> Dict[str, float]:
        """
        Use LLM to analyze semantic similarity components
        
        Args:
            explanation: Clinical explanation text
            
        Returns:
            Dictionary with semantic similarity component scores
        """
        if not self.use_llm or not self.llm_available:
            return {}
        
        prompt = f"""Analyze this clinical explanation for semantic quality. Rate each component 0.0-1.0:

Explanation: "{explanation}"

Rate:
1. Guideline alignment (0.0-1.0): How well does it align with clinical guidelines?
2. Medical knowledge consistency (0.0-1.0): How consistent is it with established medical knowledge?
3. Clinical reasoning quality (0.0-1.0): How well does it demonstrate clinical reasoning?
4. Terminology appropriateness (0.0-1.0): How appropriate is the medical terminology?

Respond in JSON format: {{"guideline": 0.0, "knowledge": 0.0, "reasoning": 0.0, "terminology": 0.0}}"""
        
        llm_response = self._call_ollama(prompt, max_tokens=100)
        
        if llm_response:
            try:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^}]+\}', llm_response)
                if json_match:
                    scores = json.loads(json_match.group())
                    return scores
            except (json.JSONDecodeError, AttributeError):
                logger.warning("⚠️  Could not parse LLM semantic analysis response")
        
        return {}


def create_llm_enhanced_generator(use_llm: bool = True) -> LLMEnhancedExplanationGenerator:
    """Factory function to create LLM-enhanced explanation generator"""
    return LLMEnhancedExplanationGenerator(use_llm=use_llm)

