"""
Enhanced Bid Proposal Automation System
A comprehensive multi-agent system for automating bid proposal creation
"""

import json
import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================

class ProposalConfig:
    """Configuration settings for the proposal system"""
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Alternative
    MODEL_NAME = "llama-3.3-70b-versatile"
    OUTPUT_DIR = Path("proposals")
    TEMPLATE_DIR = Path("templates")
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 300

# ==================== Enhanced Pydantic Models ====================

class Analysis(BaseModel):
    """Bid analysis results with enhanced validation"""
    key_requirements: List[str] = Field(min_items=1)
    evaluation_criteria: List[str] = Field(min_items=1)
    market_insights: List[str] = Field(default_factory=list)
    capability_assessment: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    
    @validator('key_requirements', 'evaluation_criteria', each_item=True)
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Empty requirement or criteria")
        return v.strip()

class OutlineSection(BaseModel):
    """Individual section of the proposal outline"""
    title: str
    subsections: List[str]
    estimated_pages: int = Field(ge=1, le=20)

class Outline(BaseModel):
    """Enhanced proposal outline with flexible sections"""
    executive_summary: List[str]
    company_background: List[str]
    technical_approach: List[str]
    management_approach: List[str]
    past_performance: List[str]
    pricing: List[str]
    risk_management: Optional[List[str]] = Field(default_factory=list)
    implementation_timeline: Optional[List[str]] = Field(default_factory=list)
    compliance_matrix: Optional[List[str]] = Field(default_factory=list)

class TechnicalContent(BaseModel):
    """Detailed technical proposal content"""
    technical_approach: str = Field(min_length=100)
    methodologies: str = Field(min_length=50)
    innovation: str = Field(min_length=50)
    technical_compliance: str = Field(min_length=50)
    technical_risks: Optional[str] = None
    technical_resources: Optional[str] = None
    quality_assurance_plan: Optional[str] = None

class BusinessContent(BaseModel):
    """Business and management content"""
    executive_summary: str = Field(min_length=100)
    company_background: str = Field(min_length=100)
    management_approach: str = Field(min_length=100)
    past_performance: str = Field(min_length=100)
    competitive_advantages: Optional[str] = None
    client_references: Optional[str] = None

class CostBreakdown(BaseModel):
    """Detailed cost structure"""
    direct_costs: str
    indirect_costs: str
    profit_margin: str
    labor_costs: Optional[str] = None
    material_costs: Optional[str] = None
    overhead_allocation: Optional[str] = None

class PricingStrategy(BaseModel):
    """Comprehensive pricing strategy"""
    cost_breakdown: CostBreakdown
    pricing_approach: str
    competitive_analysis: str
    value_justification: str
    payment_terms: Optional[str] = None
    discount_options: Optional[str] = None

class QualityAssessment(BaseModel):
    """Quality assurance assessment with scoring"""
    completeness: str
    completeness_score: int = Field(ge=0, le=100)
    coherence: str
    coherence_score: int = Field(ge=0, le=100)
    compliance: str
    compliance_score: int = Field(ge=0, le=100)
    persuasiveness: str
    persuasiveness_score: int = Field(ge=0, le=100)
    overall_score: int = Field(ge=0, le=100)
    suggested_improvements: List[str]
    critical_issues: Optional[List[str]] = Field(default_factory=list)

# ==================== Agent Factory ====================

class AgentFactory:
    """Factory for creating specialized agents"""
    
    @staticmethod
    def create_model(use_openai: bool = False):
        """Create the appropriate LLM model"""
        if use_openai and ProposalConfig.OPENAI_API_KEY:
            client = AsyncOpenAI(api_key=ProposalConfig.OPENAI_API_KEY)
            return OpenAIChatCompletionsModel(
                model="gpt-4-turbo-preview",
                openai_client=client
            )
        else:
            client = AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=ProposalConfig.GROQ_API_KEY
            )
            return OpenAIChatCompletionsModel(
                model=ProposalConfig.MODEL_NAME,
                openai_client=client
            )
    
    @staticmethod
    def create_bid_analyzer(model=None):
        """Create bid analyzer agent"""
        if not model:
            model = AgentFactory.create_model()
        
        return Agent(
            name='Bid Analyzer',
            instructions="""You are a seasoned expert in analyzing bid requests with deep industry knowledge.
            
Analyze the bid request and provide:
1. Key requirements (technical, business, compliance)
2. Evaluation criteria with weightings if available
3. Market insights and competitive landscape
4. Assessment of our capabilities vs requirements
5. Risk factors and mitigation strategies
6. Opportunities for differentiation

Format as valid JSON matching the Analysis schema exactly.""",
            model=model,
            output_type=Analysis
        )
    
    @staticmethod
    def create_proposal_outliner(model=None):
        """Create proposal outliner agent"""
        if not model:
            model = AgentFactory.create_model()
        
        return Agent(
            name='Proposal Outliner',
            instructions="""You are an expert in creating winning proposal structures.
            
Create a comprehensive outline that:
1. Addresses all requirements systematically
2. Follows best practices for proposal organization
3. Includes win themes throughout
4. Ensures logical flow and readability
5. Incorporates compliance matrix references

Include all standard sections plus any specific sections needed for this bid.
Format as valid JSON matching the Outline schema.""",
            model=model,
            output_type=Outline
        )
    
    @staticmethod
    def create_technical_writer(model=None):
        """Create technical writer agent"""
        if not model:
            model = AgentFactory.create_model()
        
        return Agent(
            name='Technical Writer',
            instructions="""You are an expert technical writer specializing in complex proposals.
            
Write detailed technical content that:
1. Clearly explains technical approach and methodologies
2. Demonstrates understanding of requirements
3. Highlights innovative solutions and best practices
4. Addresses technical risks and mitigation
5. Shows compliance with technical specifications
6. Uses appropriate technical depth for evaluators

Be specific, use examples, and avoid generic statements.
Format as valid JSON matching the TechnicalContent schema.""",
            model=model,
            output_type=TechnicalContent
        )
    
    @staticmethod
    def create_business_writer(model=None):
        """Create business writer agent"""
        if not model:
            model = AgentFactory.create_model()
        
        return Agent(
            name='Business Writer',
            instructions="""You are an expert business writer focused on persuasive proposal content.
            
Write compelling business content that:
1. Presents a strong executive summary with key discriminators
2. Showcases company strengths and relevant experience
3. Demonstrates effective management approaches
4. Highlights past performance with quantifiable results
5. Emphasizes competitive advantages
6. Builds confidence in our capabilities

Use persuasive language while maintaining professionalism.
Format as valid JSON matching the BusinessContent schema.""",
            model=model,
            output_type=BusinessContent
        )
    
    @staticmethod
    def create_pricing_strategist(model=None):
        """Create pricing strategist agent"""
        if not model:
            model = AgentFactory.create_model()
        
        return Agent(
            name='Pricing Strategist',
            instructions="""You are an expert in competitive pricing strategies for complex bids.
            
Develop a pricing strategy that:
1. Provides detailed cost breakdown with justification
2. Explains pricing methodology (fixed, T&M, hybrid)
3. Analyzes competitive positioning
4. Demonstrates value for money
5. Includes payment terms and options
6. Balances competitiveness with profitability

Consider total cost of ownership and lifecycle costs.
Format as valid JSON matching the PricingStrategy schema.""",
            model=model,
            output_type=PricingStrategy
        )
    
    @staticmethod
    def create_quality_assurance(model=None):
        """Create quality assurance agent"""
        if not model:
            model = AgentFactory.create_model()
        
        return Agent(
            name='Quality Assurance',
            instructions="""You are a meticulous quality assurance expert for proposals.
            
Perform comprehensive quality check:
1. Completeness: All sections present and fully addressed (score 0-100)
2. Coherence: Logical flow and consistency (score 0-100)
3. Compliance: Meeting all requirements (score 0-100)
4. Persuasiveness: Compelling value proposition (score 0-100)
5. Overall assessment and score
6. Critical issues requiring immediate attention
7. Specific improvement suggestions

Be thorough and provide actionable feedback.
Format as valid JSON matching the QualityAssessment schema.""",
            model=model,
            output_type=QualityAssessment
        )

# ==================== Proposal Orchestrator ====================

class ProposalOrchestrator:
    """Main orchestrator for the proposal generation workflow"""
    
    def __init__(self, use_async: bool = False):
        self.use_async = use_async
        self.model = AgentFactory.create_model()
        self.agents = self._initialize_agents()
        self.results = {}
        
        # Create output directory
        ProposalConfig.OUTPUT_DIR.mkdir(exist_ok=True)
    
    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize all agents"""
        return {
            'analyzer': AgentFactory.create_bid_analyzer(self.model),
            'outliner': AgentFactory.create_proposal_outliner(self.model),
            'technical_writer': AgentFactory.create_technical_writer(self.model),
            'business_writer': AgentFactory.create_business_writer(self.model),
            'pricing_strategist': AgentFactory.create_pricing_strategist(self.model),
            'quality_assurance': AgentFactory.create_quality_assurance(self.model)
        }
    
    async def run_agent_async(self, agent: Agent, input_data: str) -> Any:
        """Run an agent asynchronously with error handling"""
        try:
            result = await Runner.run(agent, input_data)
            return result.final_output
        except Exception as e:
            logger.error(f"Error running agent {agent.name}: {e}")
            raise
    
    def run_agent_sync(self, agent: Agent, input_data: str) -> Any:
        """Run an agent synchronously with error handling"""
        try:
            result = Runner.run_sync(agent, input_data)
            return result.final_output
        except Exception as e:
            logger.error(f"Error running agent {agent.name}: {e}")
            raise
    
    async def generate_proposal_async(
        self,
        bid_request: str,
        company_info: str,
        specifications: Optional[str] = None,
        project_details: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate complete proposal asynchronously"""
        
        logger.info("Starting proposal generation (async)")
        
        # Step 1: Analyze bid request
        logger.info("Step 1: Analyzing bid request")
        analysis = await self.run_agent_async(
            self.agents['analyzer'],
            f"Bid Request:\n{bid_request}"
        )
        self.results['analysis'] = analysis
        
        # Step 2: Create outline
        logger.info("Step 2: Creating proposal outline")
        outline = await self.run_agent_async(
            self.agents['outliner'],
            f"Requirements Analysis:\n{json.dumps(analysis.model_dump(), indent=2)}"
        )
        self.results['outline'] = outline
        
        # Step 3: Generate content sections in parallel
        logger.info("Step 3: Generating content sections")
        
        technical_task = self.run_agent_async(
            self.agents['technical_writer'],
            f"Outline:\n{json.dumps(outline.model_dump(), indent=2)}\n\n"
            f"Specifications:\n{specifications or 'Use general best practices'}"
        )
        
        business_task = self.run_agent_async(
            self.agents['business_writer'],
            f"Outline:\n{json.dumps(outline.model_dump(), indent=2)}\n\n"
            f"Company Info:\n{company_info}"
        )
        
        pricing_task = self.run_agent_async(
            self.agents['pricing_strategist'],
            f"Project Details:\n{project_details or 'Standard project'}\n\n"
            f"Requirements:\n{json.dumps(analysis.model_dump(), indent=2)}"
        )
        
        # Wait for all content generation
        technical, business, pricing = await asyncio.gather(
            technical_task, business_task, pricing_task
        )
        
        self.results['technical'] = technical
        self.results['business'] = business
        self.results['pricing'] = pricing
        
        # Step 4: Quality assurance
        logger.info("Step 4: Performing quality assurance")
        proposal_content = self._assemble_proposal()
        quality = await self.run_agent_async(
            self.agents['quality_assurance'],
            f"Complete Proposal:\n{proposal_content}"
        )
        self.results['quality'] = quality
        
        # Step 5: Save results
        self._save_results()
        
        logger.info("Proposal generation completed")
        return self.results
    
    def generate_proposal_sync(
        self,
        bid_request: str,
        company_info: str,
        specifications: Optional[str] = None,
        project_details: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate complete proposal synchronously"""
        
        logger.info("Starting proposal generation (sync)")
        
        # Step 1: Analyze bid request
        logger.info("Step 1: Analyzing bid request")
        analysis = self.run_agent_sync(
            self.agents['analyzer'],
            f"Bid Request:\n{bid_request}"
        )
        self.results['analysis'] = analysis
        
        # Step 2: Create outline
        logger.info("Step 2: Creating proposal outline")
        outline = self.run_agent_sync(
            self.agents['outliner'],
            f"Requirements Analysis:\n{json.dumps(analysis.model_dump(), indent=2)}"
        )
        self.results['outline'] = outline
        
        # Step 3: Generate content sections
        logger.info("Step 3: Generating technical content")
        technical = self.run_agent_sync(
            self.agents['technical_writer'],
            f"Outline:\n{json.dumps(outline.model_dump(), indent=2)}\n\n"
            f"Specifications:\n{specifications or 'Use general best practices'}"
        )
        self.results['technical'] = technical
        
        logger.info("Step 3: Generating business content")
        business = self.run_agent_sync(
            self.agents['business_writer'],
            f"Outline:\n{json.dumps(outline.model_dump(), indent=2)}\n\n"
            f"Company Info:\n{company_info}"
        )
        self.results['business'] = business
        
        logger.info("Step 3: Generating pricing strategy")
        pricing = self.run_agent_sync(
            self.agents['pricing_strategist'],
            f"Project Details:\n{project_details or 'Standard project'}\n\n"
            f"Requirements:\n{json.dumps(analysis.model_dump(), indent=2)}"
        )
        self.results['pricing'] = pricing
        
        # Step 4: Quality assurance
        logger.info("Step 4: Performing quality assurance")
        proposal_content = self._assemble_proposal()
        quality = self.run_agent_sync(
            self.agents['quality_assurance'],
            f"Complete Proposal:\n{proposal_content}"
        )
        self.results['quality'] = quality
        
        # Step 5: Save results
        self._save_results()
        
        logger.info("Proposal generation completed")
        return self.results
    
    def _assemble_proposal(self) -> str:
        """Assemble all sections into a complete proposal"""
        sections = []
        
        # Add each section with proper formatting
        if 'analysis' in self.results:
            sections.append(f"=== ANALYSIS ===\n{json.dumps(self.results['analysis'].model_dump(), indent=2)}")
        
        if 'outline' in self.results:
            sections.append(f"=== OUTLINE ===\n{json.dumps(self.results['outline'].model_dump(), indent=2)}")
        
        if 'technical' in self.results:
            sections.append(f"=== TECHNICAL CONTENT ===\n{json.dumps(self.results['technical'].model_dump(), indent=2)}")
        
        if 'business' in self.results:
            sections.append(f"=== BUSINESS CONTENT ===\n{json.dumps(self.results['business'].model_dump(), indent=2)}")
        
        if 'pricing' in self.results:
            sections.append(f"=== PRICING STRATEGY ===\n{json.dumps(self.results['pricing'].model_dump(), indent=2)}")
        
        return "\n\n".join(sections)
    
    def _save_results(self):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        proposal_dir = ProposalConfig.OUTPUT_DIR / f"proposal_{timestamp}"
        proposal_dir.mkdir(exist_ok=True)
        
        # Save each component
        for component_name, component_data in self.results.items():
            filepath = proposal_dir / f"{component_name}.json"
            with open(filepath, 'w') as f:
                json.dump(component_data.model_dump(), f, indent=2)
        
        # Save complete proposal
        complete_proposal = self._assemble_proposal()
        with open(proposal_dir / "complete_proposal.txt", 'w') as f:
            f.write(complete_proposal)
        
        # Generate executive summary
        if 'quality' in self.results:
            summary_path = proposal_dir / "executive_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(self._generate_executive_summary())
        
        logger.info(f"Results saved to {proposal_dir}")
    
    def _generate_executive_summary(self) -> str:
        """Generate an executive summary of the proposal"""
        quality = self.results.get('quality')
        if not quality:
            return "No quality assessment available"
        
        summary = f"""PROPOSAL EXECUTIVE SUMMARY
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

QUALITY SCORES:
- Completeness: {quality.completeness_score}/100
- Coherence: {quality.coherence_score}/100
- Compliance: {quality.compliance_score}/100
- Persuasiveness: {quality.persuasiveness_score}/100
- Overall: {quality.overall_score}/100

ASSESSMENT:
{quality.completeness}

KEY IMPROVEMENTS NEEDED:
"""
        for improvement in quality.suggested_improvements[:5]:
            summary += f"- {improvement}\n"
        
        if quality.critical_issues:
            summary += "\nCRITICAL ISSUES:\n"
            for issue in quality.critical_issues:
                summary += f"- {issue}\n"
        
        return summary

# ==================== Main Execution ====================

def main():
    """Main execution function with example usage"""
    
    # Example inputs - replace with your actual data
    bid_request = """
    Request for Proposal: Cloud Migration Services
    
    We are seeking a qualified vendor to provide comprehensive cloud migration services 
    for our enterprise applications. The project involves:
    - Migration of 50+ applications to AWS
    - Data migration of 100TB+
    - Zero downtime requirements for critical systems
    - Complete migration within 12 months
    - Training for internal teams
    - 24/7 support during and after migration
    
    Evaluation Criteria:
    - Technical approach (40%)
    - Experience and past performance (30%)
    - Pricing (20%)
    - Management approach (10%)
    """
    
    company_info = """
    TechCorp Solutions - Leading Cloud Migration Specialists
    
    - 15+ years in cloud services
    - AWS Advanced Consulting Partner
    - 500+ successful migrations completed
    - Team of 200+ certified cloud engineers
    - ISO 27001 and SOC 2 certified
    - Proprietary migration tools and methodologies
    - 24/7 global support centers
    """
    
    specifications = """
    Technical Requirements:
    - AWS Well-Architected Framework compliance
    - Kubernetes orchestration for containerized apps
    - Terraform for Infrastructure as Code
    - CI/CD pipeline integration
    - Comprehensive security and compliance measures
    - Automated testing and validation
    - Performance optimization
    """
    
    project_details = """
    Project Budget: $5-10 million
    Timeline: 12 months
    Team Size: 20-30 engineers
    Location: Hybrid (on-site and remote)
    """
    
    # Create orchestrator
    orchestrator = ProposalOrchestrator(use_async=False)
    
    try:
        # Generate proposal
        results = orchestrator.generate_proposal_sync(
            bid_request=bid_request,
            company_info=company_info,
            specifications=specifications,
            project_details=project_details
        )
        
        # Display summary
        if 'quality' in results:
            quality = results['quality']
            print("\n" + "="*50)
            print("PROPOSAL GENERATION COMPLETE")
            print("="*50)
            print(f"Overall Score: {quality.overall_score}/100")
            print(f"Completeness: {quality.completeness_score}/100")
            print(f"Coherence: {quality.coherence_score}/100")
            print(f"Compliance: {quality.compliance_score}/100")
            print(f"Persuasiveness: {quality.persuasiveness_score}/100")
            
            if quality.critical_issues:
                print("\nCritical Issues:")
                for issue in quality.critical_issues:
                    print(f"  - {issue}")
            
            print("\nTop Improvements Needed:")
            for improvement in quality.suggested_improvements[:3]:
                print(f"  - {improvement}")
        
    except Exception as e:
        logger.error(f"Failed to generate proposal: {e}")
        raise

# Async main for async execution
async def main_async():
    """Async main execution function"""
    
    # Use same example data as sync version
    bid_request = "Your bid request here..."
    company_info = "Your company info here..."
    specifications = "Your specifications here..."
    project_details = "Your project details here..."
    
    orchestrator = ProposalOrchestrator(use_async=True)
    
    try:
        results = await orchestrator.generate_proposal_async(
            bid_request=bid_request,
            company_info=company_info,
            specifications=specifications,
            project_details=project_details
        )
        
        # Display results (same as sync version)
        
    except Exception as e:
        logger.error(f"Failed to generate proposal: {e}")
        raise

if __name__ == "__main__":
    # Run synchronously
    main()
    
    # Or run asynchronously
    # asyncio.run(main_async())
