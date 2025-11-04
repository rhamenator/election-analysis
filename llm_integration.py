"""
LLM Integration Module

Optional module for integrating Large Language Models to provide intelligent
analysis of anomaly detection results, audit prioritization recommendations,
and natural language explanations of statistical findings.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings

# Optional LLM provider imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    warnings.warn("OpenAI not available. Install openai package for GPT integration.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    warnings.warn("Anthropic not available. Install anthropic package for Claude integration.")

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    warnings.warn("Cohere not available. Install cohere package for Command-R+ integration.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyAnalysis:
    """Data class for structured anomaly analysis results."""
    precinct_id: str
    county: str
    precinct: str
    anomaly_types: List[str]
    severity_score: float
    statistical_evidence: Dict[str, float]
    geographic_context: Dict[str, Any]
    audit_priority: str
    explanation: str


class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, config: Dict):
        """Initialize with provider configuration."""
        self.config = config
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response from LLM provider."""
        raise NotImplementedError("Subclasses must implement generate_response")


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider implementation."""
    
    def __init__(self, config: Dict, api_key: str):
        """Initialize OpenAI provider."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install openai package.")
        
        super().__init__(config)
        self.client = openai.OpenAI(api_key=api_key)
        self.model = config.get('model', 'gpt-4')
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using OpenAI GPT."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert election data analyst specializing in statistical anomaly detection and audit prioritization."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error generating response: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, config: Dict, api_key: str):
        """Initialize Anthropic provider."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic not available. Install anthropic package.")
        
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = config.get('model', 'claude-3-sonnet-20240229')
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using Anthropic Claude."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.3,
                system="You are an expert election data analyst specializing in statistical anomaly detection and audit prioritization.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Error generating response: {str(e)}"


class CohereProvider(LLMProvider):
    """Cohere Command-R+ provider implementation."""
    
    def __init__(self, config: Dict, api_key: str):
        """Initialize Cohere provider."""
        if not COHERE_AVAILABLE:
            raise ImportError("Cohere not available. Install cohere package.")
        
        super().__init__(config)
        self.client = cohere.Client(api_key=api_key)
        self.model = config.get('model', 'command-r-plus')
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using Cohere Command-R+."""
        try:
            response = self.client.chat(
                model=self.model,
                message=prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                preamble="You are an expert election data analyst specializing in statistical anomaly detection and audit prioritization."
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            return f"Error generating response: {str(e)}"


class AnomalyReasoningAgent:
    """
    Intelligent agent for analyzing election anomalies and providing audit recommendations.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the reasoning agent."""
        self.config = self._load_config(config_path)
        self.llm_provider = None
        self.enabled = self.config.get('enabled', False)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load LLM configuration."""
        try:
            with open(config_path, 'r') as file:
                full_config = yaml.safe_load(file)
            return full_config.get('llm', {})
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. LLM integration disabled.")
            return {'enabled': False}
    
    def initialize_provider(self, provider: str, api_key: str) -> bool:
        """
        Initialize LLM provider.
        
        Args:
            provider: Provider name ('openai', 'anthropic', 'cohere')
            api_key: API key for the provider
            
        Returns:
            True if successful, False otherwise
        """
        try:
            providers_config = self.config.get('providers', {})
            
            if provider == 'openai' and OPENAI_AVAILABLE:
                self.llm_provider = OpenAIProvider(providers_config.get('openai', {}), api_key)
            elif provider == 'anthropic' and ANTHROPIC_AVAILABLE:
                self.llm_provider = AnthropicProvider(providers_config.get('anthropic', {}), api_key)
            elif provider == 'cohere' and COHERE_AVAILABLE:
                self.llm_provider = CohereProvider(providers_config.get('cohere', {}), api_key)
            else:
                logger.error(f"Provider {provider} not available or not supported")
                return False
            
            self.enabled = True
            logger.info(f"LLM provider {provider} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider {provider}: {e}")
            return False
    
    def analyze_anomalies(self, df: pd.DataFrame, 
                         top_n: int = 20) -> List[AnomalyAnalysis]:
        """
        Analyze top anomalies and create structured analysis.
        
        Args:
            df: DataFrame with anomaly detection results
            top_n: Number of top anomalies to analyze
            
        Returns:
            List of structured anomaly analyses
        """
        if not self.enabled:
            logger.warning("LLM integration not enabled")
            return []
        
        logger.info(f"Analyzing top {top_n} anomalies")
        
        # Find composite anomaly score column
        score_columns = [col for col in df.columns if 'Composite' in col and 'Score' in col]
        if not score_columns:
            score_columns = [col for col in df.columns if 'Score' in col and 'Anomaly' in col]
        
        if not score_columns:
            logger.error("No anomaly score columns found")
            return []
        
        primary_score = score_columns[0]
        
        # Get top anomalies
        top_anomalies = df.nlargest(top_n, primary_score)
        
        analyses = []
        
        for _, row in top_anomalies.iterrows():
            analysis = self._analyze_single_anomaly(row, df)
            if analysis:
                analyses.append(analysis)
        
        return analyses
    
    def _analyze_single_anomaly(self, anomaly_row: pd.Series, 
                               full_df: pd.DataFrame) -> Optional[AnomalyAnalysis]:
        """Analyze a single anomalous precinct."""
        try:
            # Extract key information
            precinct_id = anomaly_row.get('Precinct_ID', f"{anomaly_row['County']}_{anomaly_row['Precinct']}")
            county = anomaly_row.get('County', 'Unknown')
            precinct = anomaly_row.get('Precinct', 'Unknown')
            
            # Identify anomaly types
            anomaly_types = []
            anomaly_flags = [col for col in anomaly_row.index if 'Anomaly_Flag' in col]
            
            for flag_col in anomaly_flags:
                if anomaly_row[flag_col] == 1:
                    method = flag_col.replace('_Anomaly_Flag', '').replace('_', ' ')
                    anomaly_types.append(method)
            
            # Calculate severity score
            score_columns = [col for col in anomaly_row.index if 'Score' in col and 'Anomaly' in col]
            if score_columns:
                severity_score = float(anomaly_row[score_columns[0]])
            else:
                severity_score = 0.5
            
            # Statistical evidence
            statistical_evidence = {}
            for col in ['Turnout_Percent', 'Harris_Share', 'Vote_Share_Deviation']:
                if col in anomaly_row.index:
                    statistical_evidence[col] = float(anomaly_row[col])
            
            # Geographic context
            geographic_context = self._get_geographic_context(anomaly_row, full_df)
            
            # Determine audit priority
            audit_priority = self._calculate_audit_priority(severity_score, anomaly_types)
            
            # Generate explanation
            explanation = self._generate_explanation(
                anomaly_row, anomaly_types, statistical_evidence, geographic_context
            )
            
            return AnomalyAnalysis(
                precinct_id=precinct_id,
                county=county,
                precinct=precinct,
                anomaly_types=anomaly_types,
                severity_score=severity_score,
                statistical_evidence=statistical_evidence,
                geographic_context=geographic_context,
                audit_priority=audit_priority,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing anomaly: {e}")
            return None
    
    def _get_geographic_context(self, anomaly_row: pd.Series, 
                               full_df: pd.DataFrame) -> Dict[str, Any]:
        """Get geographic context for the anomaly."""
        context = {}
        
        # County statistics
        county = anomaly_row.get('County')
        if county:
            county_data = full_df[full_df['County'] == county]
            
            context['county_precincts'] = len(county_data)
            context['county_avg_turnout'] = float(county_data['Turnout_Percent'].mean())
            
            if 'Harris_Share' in county_data.columns:
                context['county_avg_vote_share'] = float(county_data['Harris_Share'].mean())
        
        # Nearby precincts (if coordinates available)
        if all(col in anomaly_row.index for col in ['Lat', 'Lon']):
            lat, lon = anomaly_row['Lat'], anomaly_row['Lon']
            
            # Find precincts within ~5km (approximate degrees)
            nearby = full_df[
                (abs(full_df['Lat'] - lat) < 0.05) & 
                (abs(full_df['Lon'] - lon) < 0.05)
            ]
            
            context['nearby_precincts'] = len(nearby)
            if len(nearby) > 1:
                context['nearby_avg_turnout'] = float(nearby['Turnout_Percent'].mean())
        
        return context
    
    def _calculate_audit_priority(self, severity_score: float, 
                                 anomaly_types: List[str]) -> str:
        """Calculate audit priority level."""
        # Base priority on severity score
        if severity_score > 0.8:
            base_priority = "HIGH"
        elif severity_score > 0.5:
            base_priority = "MEDIUM"
        else:
            base_priority = "LOW"
        
        # Adjust based on anomaly types
        high_priority_methods = ['Shpilkin', 'Statistical', 'ML Composite']
        
        if any(method in ' '.join(anomaly_types) for method in high_priority_methods):
            if base_priority == "MEDIUM":
                base_priority = "HIGH"
            elif base_priority == "LOW":
                base_priority = "MEDIUM"
        
        return base_priority
    
    def _generate_explanation(self, anomaly_row: pd.Series,
                             anomaly_types: List[str],
                             statistical_evidence: Dict[str, float],
                             geographic_context: Dict[str, Any]) -> str:
        """Generate natural language explanation using LLM."""
        if not self.llm_provider:
            return "LLM provider not initialized"
        
        # Build context for LLM
        prompt = self._build_explanation_prompt(
            anomaly_row, anomaly_types, statistical_evidence, geographic_context
        )
        
        return self.llm_provider.generate_response(prompt, max_tokens=300)
    
    def _build_explanation_prompt(self, anomaly_row: pd.Series,
                                 anomaly_types: List[str],
                                 statistical_evidence: Dict[str, float],
                                 geographic_context: Dict[str, Any]) -> str:
        """Build prompt for LLM explanation."""
        prompt = f"""
        Analyze this election precinct anomaly and provide a concise explanation:
        
        PRECINCT INFO:
        - County: {anomaly_row.get('County', 'Unknown')}
        - Precinct: {anomaly_row.get('Precinct', 'Unknown')}
        - Total Votes: {anomaly_row.get('Total_Votes', 'N/A')}
        
        ANOMALY DETECTION RESULTS:
        - Detected by methods: {', '.join(anomaly_types)}
        - Turnout: {statistical_evidence.get('Turnout_Percent', 'N/A'):.1f}%
        - Vote Share: {statistical_evidence.get('Harris_Share', 'N/A'):.1%}
        - Statistical Deviation: {statistical_evidence.get('Vote_Share_Deviation', 'N/A'):.3f}
        
        GEOGRAPHIC CONTEXT:
        - County average turnout: {geographic_context.get('county_avg_turnout', 'N/A'):.1f}%
        - County average vote share: {geographic_context.get('county_avg_vote_share', 'N/A'):.1%}
        - Nearby precincts: {geographic_context.get('nearby_precincts', 'N/A')}
        
        Provide a brief explanation of:
        1. What makes this precinct anomalous
        2. The statistical significance of the deviation
        3. Potential causes or areas for investigation
        
        Keep response under 200 words and focus on actionable insights.
        """
        
        return prompt
    
    def generate_executive_summary(self, analyses: List[AnomalyAnalysis]) -> str:
        """Generate executive summary of all anomaly analyses."""
        if not self.enabled or not analyses:
            return "No anomaly analyses available"
        
        prompt = self._build_summary_prompt(analyses)
        return self.llm_provider.generate_response(prompt, max_tokens=800)
    
    def _build_summary_prompt(self, analyses: List[AnomalyAnalysis]) -> str:
        """Build prompt for executive summary."""
        # Aggregate statistics
        total_anomalies = len(analyses)
        high_priority = sum(1 for a in analyses if a.audit_priority == "HIGH")
        counties = set(a.county for a in analyses)
        
        # Most common anomaly types
        all_types = []
        for analysis in analyses:
            all_types.extend(analysis.anomaly_types)
        
        type_counts = {}
        for atype in all_types:
            type_counts[atype] = type_counts.get(atype, 0) + 1
        
        top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        prompt = f"""
        Generate an executive summary for election anomaly detection analysis:
        
        ANALYSIS OVERVIEW:
        - Total anomalous precincts identified: {total_anomalies}
        - High priority for audit: {high_priority}
        - Counties affected: {len(counties)} ({', '.join(list(counties)[:5])})
        
        MOST COMMON ANOMALY TYPES:
        {chr(10).join(f"- {atype}: {count} precincts" for atype, count in top_types)}
        
        TOP 5 HIGHEST PRIORITY PRECINCTS:
        {chr(10).join(f"- {a.county} {a.precinct}: {a.audit_priority} priority (Score: {a.severity_score:.3f})" for a in analyses[:5])}
        
        SAMPLE EXPLANATIONS:
        {chr(10).join(f"• {a.county} {a.precinct}: {a.explanation[:100]}..." for a in analyses[:3])}
        
        Generate an executive summary that includes:
        1. Overall assessment of data integrity
        2. Geographic patterns in anomalies
        3. Recommended audit priorities and next steps
        4. Statistical significance assessment
        5. Confidence level in findings
        
        Keep response under 500 words and provide actionable recommendations.
        """
        
        return prompt
    
    def get_audit_recommendations(self, analyses: List[AnomalyAnalysis]) -> Dict[str, List[str]]:
        """Generate structured audit recommendations."""
        recommendations = {
            'immediate_action': [],
            'secondary_review': [],
            'monitoring': []
        }
        
        for analysis in analyses:
            precinct_desc = f"{analysis.county} - {analysis.precinct}"
            
            if analysis.audit_priority == "HIGH":
                recommendations['immediate_action'].append(precinct_desc)
            elif analysis.audit_priority == "MEDIUM":
                recommendations['secondary_review'].append(precinct_desc)
            else:
                recommendations['monitoring'].append(precinct_desc)
        
        return recommendations


def main():
    """Example usage of LLM integration."""
    # Create sample anomaly data
    sample_data = pd.DataFrame({
        'County': ['Adams County', 'Baker County', 'Clark County'],
        'Precinct': ['Precinct 1A', 'Precinct 2B', 'Precinct 3C'],
        'Lat': [40.1, 40.2, 40.3],
        'Lon': [-75.1, -75.2, -75.3],
        'Turnout_Percent': [95.2, 87.3, 78.1],
        'Harris_Share': [0.75, 0.82, 0.68],
        'Total_Votes': [1200, 980, 1450],
        'Shpilkin_Anomaly_Flag': [1, 1, 0],
        'ML_Anomaly_Flag': [1, 0, 1],
        'Composite_Anomaly_Score': [0.85, 0.72, 0.61],
        'Vote_Share_Deviation': [0.15, 0.12, 0.08]
    })
    
    # Initialize reasoning agent
    agent = AnomalyReasoningAgent()
    
    print("LLM Reasoning Agent initialized")
    print(f"Enabled: {agent.enabled}")
    
    # Example without API key (will show structure)
    if not agent.enabled:
        print("\nTo use LLM features, add API key:")
        print("agent.initialize_provider('openai', 'your-api-key')")
        print("analyses = agent.analyze_anomalies(data)")
        print("summary = agent.generate_executive_summary(analyses)")


if __name__ == "__main__":
    main()