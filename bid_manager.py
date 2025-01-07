import json
import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from groq import Groq

# Create a custom LLM class to integrate with CrewAI
class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
    def generate(self, prompt):
        """Centralized method for all LLM interactions"""
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        return chat_completion.choices[0].message.content

# Initialize the LLM
groq_llm = GroqLLM()

# Analysis Tools
def analyze_bid_request(bid_request):
    """
    Analyzes a bid request and provides a structured summary of key requirements,
    evaluation criteria, market insights, and a capability assessment.

    Args:
        bid_request (str): The bid request details.

    Returns:
        dict: A JSON object containing the analysis, or an error message if parsing fails.
    """
    prompt = f"""
    You are an expert bid analyzer. Analyze the following bid request and provide a summary 
    of key requirements, evaluation criteria, and relevant market insights. Additionally, 
    assess our company's capabilities against these requirements.

    Bid Request:
    {bid_request}

    Format your response as valid JSON with the following structure:
    {{
        "key_requirements": ["Requirement 1", "Requirement 2", ...],
        "evaluation_criteria": ["Criterion 1", "Criterion 2", ...],
        "market_insights": ["Insight 1", "Insight 2", ...],
        "capability_assessment": ["Assessment 1", "Assessment 2", ...]
    }}
    """
    try:
        response = groq_llm.generate(prompt)
        parsed_response = json.loads(response)
        if not all(key in parsed_response for key in ["key_requirements", "evaluation_criteria", "market_insights", "capability_assessment"]):
            raise ValueError("Incomplete response structure")
        return parsed_response
    except json.JSONDecodeError:
        return {"error": "Failed to parse the analysis. The response is not valid JSON. Please try again."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def create_proposal_outline(requirements):
    """
    Generates a detailed outline for a bid proposal based on the provided requirements.

    Args:
        requirements (str): The bid requirements.

    Returns:
        dict: A JSON object containing the proposal outline, or an error message if parsing fails.
    """
    prompt = f"""
    You are an expert proposal outliner. Based on the following requirements, create a 
    comprehensive and detailed outline for a bid proposal. The outline should include 
    main sections and relevant subsections, tailored to address the requirements effectively.

    Requirements:
    {requirements}

    Format your response as valid JSON with the following structure:
    {{
        "executive_summary": ["Subsection 1", "Subsection 2", ...],
        "company_background": ["Subsection 1", "Subsection 2", ...],
        "technical_approach": ["Subsection 1", "Subsection 2", ...],
        "management_approach": ["Subsection 1", "Subsection 2", ...],
        "past_performance": ["Subsection 1", "Subsection 2", ...],
        "pricing": ["Subsection 1", "Subsection 2", ...]
    }}
    """
    try:
        response = groq_llm.generate(prompt)
        parsed_response = json.loads(response)
        # Validate that all required keys are present
        expected_keys = [
            "executive_summary", 
            "company_background", 
            "technical_approach", 
            "management_approach", 
            "past_performance", 
            "pricing"
        ]
        if not all(key in parsed_response for key in expected_keys):
            raise ValueError("Incomplete response structure")
        return parsed_response
    except json.JSONDecodeError:
        return {"error": "Failed to parse the outline. The response is not valid JSON. Please try again."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def write_technical_content(outline, specifications):
    """
    Generates technical content for a bid proposal based on the provided outline and specifications.

    Args:
        outline (str): The proposal outline.
        specifications (str): The technical specifications or requirements.

    Returns:
        dict: A JSON object containing the technical content, or an error message if parsing fails.
    """
    prompt = f"""
    You are a technical writing expert. Based on the following outline and technical specifications, 
    write detailed and clear technical content for a bid proposal. Focus on explaining our approach, 
    methodologies, innovations, and how we meet or exceed the technical requirements.

    Outline:
    {outline}

    Technical Specifications:
    {specifications}

    Format your response as valid JSON with the following structure:
    {{
        "technical_approach": "Description of how the technical requirements are addressed.",
        "methodologies": "Detailed explanation of the methodologies to be used.",
        "innovation": "Highlights of innovative approaches or technologies proposed.",
        "technical_compliance": "Specific points demonstrating compliance with technical requirements."
    }}
    """
    try:
        response = groq_llm.generate(prompt)
        parsed_response = json.loads(response)
        # Validate that all required keys are present
        expected_keys = [
            "technical_approach", 
            "methodologies", 
            "innovation", 
            "technical_compliance"
        ]
        if not all(key in parsed_response for key in expected_keys):
            raise ValueError("Incomplete response structure")
        return parsed_response
    except json.JSONDecodeError:
        return {"error": "Failed to parse the technical content. The response is not valid JSON. Please try again."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def write_business_content(outline, company_info):
    """
    Generates business content for a bid proposal based on the provided outline and company information.

    Args:
        outline (str): The proposal outline.
        company_info (str): Information about the company, including strengths, experience, and value proposition.

    Returns:
        dict: A JSON object containing the business content, or an error message if parsing fails.
    """
    prompt = f"""
    You are a business content expert. Based on the following outline and company information, 
    write detailed and compelling business content for a bid proposal. Highlight our company's strengths, 
    experience, and unique value proposition.

    Outline:
    {outline}

    Company Information:
    {company_info}

    Format your response as valid JSON with the following structure:
    {{
        "executive_summary": "Overview of the company's approach and key value propositions.",
        "company_background": "Detailed description of the company's history, experience, and key strengths.",
        "management_approach": "Explanation of how the company manages projects, resources, and risks.",
        "past_performance": "Summary of relevant past performance and successful projects."
    }}
    """
    try:
        response = groq_llm.generate(prompt)
        parsed_response = json.loads(response)
        # Validate that all required keys are present
        expected_keys = [
            "executive_summary", 
            "company_background", 
            "management_approach", 
            "past_performance"
        ]
        if not all(key in parsed_response for key in expected_keys):
            raise ValueError("Incomplete response structure")
        return parsed_response
    except json.JSONDecodeError:
        return {"error": "Failed to parse the business content. The response is not valid JSON. Please try again."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def develop_pricing_strategy(project_details):
    """
    Develops a comprehensive pricing strategy for a bid based on the provided project details.

    Args:
        project_details (str): Information about the project, including costs, market rates, and competition.

    Returns:
        dict: A JSON object containing the pricing strategy, or an error message if parsing fails.
    """
    prompt = f"""
    You are a pricing strategist. Based on the following project details, develop a comprehensive pricing strategy 
    that considers costs, market rates, competition, and the value proposition. Your strategy should balance 
    profitability with competitiveness and align with the project's requirements.

    Project Details:
    {project_details}

    Format your response as valid JSON with the following structure:
    {{
        "cost_breakdown": {{
            "direct_costs": "Description of direct costs involved.",
            "indirect_costs": "Description of indirect costs or overheads.",
            "profit_margin": "Suggested profit margin."
        }},
        "pricing_approach": "Explanation of the overall pricing model or approach.",
        "competitive_analysis": "Analysis of the competition's pricing and positioning.",
        "value_justification": "Rationale for how the proposed pricing aligns with the value provided."
    }}
    """
    try:
        response = groq_llm.generate(prompt)
        parsed_response = json.loads(response)
        # Validate that all required keys are present
        expected_keys = [
            "cost_breakdown", 
            "pricing_approach", 
            "competitive_analysis", 
            "value_justification"
        ]
        if not all(key in parsed_response for key in expected_keys):
            raise ValueError("Incomplete response structure")
        return parsed_response
    except json.JSONDecodeError:
        return {"error": "Failed to parse the pricing strategy. The response is not valid JSON. Please try again."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
        
def quality_check(proposal):
    """
    Performs a quality check on a bid proposal, evaluating completeness, coherence,
    compliance with requirements, and overall persuasiveness.

    Args:
        proposal (str): The bid proposal content to be assessed.

    Returns:
        dict: A JSON object containing the quality assessment, or an error message if parsing fails.
    """
    prompt = f"""
    You are a quality assurance expert. Perform a thorough quality check on the following bid proposal, 
    evaluating it for completeness, coherence, compliance with requirements, and overall persuasiveness. 
    Provide your assessment with suggestions for improvements.

    Proposal:
    {proposal}

    Format your response as valid JSON with the following structure:
    {{
        "completeness": "Assessment of whether all required sections are present and fully addressed.",
        "coherence": "Evaluation of the logical flow and consistency of the proposal.",
        "compliance": "Check for compliance with all specified requirements.",
        "persuasiveness": "Assessment of how convincing the proposal is for the intended audience.",
        "suggested_improvements": [
            "List of areas where improvements can be made for each of the above categories."
        ]
    }}
    """
    try:
        response = groq_llm.generate(prompt)
        parsed_response = json.loads(response)
        # Validate that all required keys are present
        expected_keys = [
            "completeness", 
            "coherence", 
            "compliance", 
            "persuasiveness", 
            "suggested_improvements"
        ]
        if not all(key in parsed_response for key in expected_keys):
            raise ValueError("Incomplete response structure")
        return parsed_response
    except json.JSONDecodeError:
        return {"error": "Failed to parse the quality assessment. The response is not valid JSON. Please try again."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Define Agents
agents = {
    'bid_analyzer': Agent(
        role='Bid Analyzer',
        goal='Efficiently analyze bid requests to extract key requirements, evaluate opportunities, and provide actionable market insights to support strategic decision-making.',
        backstory=(
            "You are a seasoned expert with years of experience in analyzing bid requests, "
            "identifying market trends, and delivering comprehensive evaluations. "
            "Your deep understanding of bidding strategies and industry dynamics allows you "
            "to uncover hidden opportunities and offer insights that drive success."
        ),
        verbose=True,
        allow_delegation=False,
        llm=groq_llm,
        tools=[
            Tool(
                name="Analyze Bid",
                func=analyze_bid_request,
                description=(
                    "Analyzes the details of a bid request, extracts key requirements, "
                    "assesses compliance needs, and generates market insights, including "
                    "trends, potential competitors, and areas of opportunity."
                )
            )
        ]
    ),
    'proposal_outliner': Agent(
        role='Proposal Outliner',
        goal='Design clear, compelling, and tailored proposal structures that address client needs and highlight strategic value.',
        backstory=(
            "You are a highly skilled professional in creating effective and persuasive proposal outlines. "
            "With a deep understanding of client requirements and strategic communication, you craft outlines "
            "that emphasize key points, align with objectives, and set the foundation for impactful proposals."
        ),
        verbose=True,
        allow_delegation=False,
        llm=groq_llm,
        tools=[
            Tool(
                name="Create Outline",
                func=create_proposal_outline,
                description=(
                    "Generates a detailed and structured proposal outline based on provided requirements, "
                    "ensuring alignment with client objectives and emphasizing value propositions."
                )
            )
        ]
    ),
    'technical_writer': Agent(
        role='Technical Writer',
        goal='Craft clear, concise, and highly detailed technical sections for proposals, ensuring accuracy and alignment with client requirements.',
        backstory=(
            "You are an expert in technical writing with extensive experience in creating precise and well-structured "
            "technical content for proposals. Your strong understanding of technical concepts and ability to translate "
            "complex information into clear, client-focused documentation makes you a key contributor to winning proposals."
        ),
        verbose=True,
        allow_delegation=False,
        llm=groq_llm,
        tools=[
            Tool(
                name="Write Technical Content",
                func=write_technical_content,
                description=(
                    "Generates detailed and accurate technical sections for proposals, incorporating technical requirements, "
                    "specifications, and solutions while maintaining clarity and a client-centered focus."
                )
            )
        ]
    ),
    'business_writer': Agent(
        role='Business Writer',
        goal='Craft persuasive, client-focused business sections of proposals that clearly communicate value propositions and align with strategic objectives.',
        backstory=(
            "You are an accomplished business writer with expertise in creating compelling and professional content for proposals. "
            "Your deep understanding of business strategies, market dynamics, and client needs enables you to articulate clear, persuasive, "
            "and results-oriented narratives that resonate with stakeholders."
        ),
        verbose=True,
        allow_delegation=False,
        llm=groq_llm,
        tools=[
            Tool(
                name="Write Business Content",
                func=write_business_content,
                description=(
                    "Generates persuasive and well-structured business sections of a proposal, including executive summaries, "
                    "value propositions, and strategic overviews, tailored to meet client objectives and highlight key benefits."
                )
            )
        ]
    ),
    'pricing_strategist': Agent(
        role='Pricing Strategist',
        goal='Design competitive and profitable pricing strategies that align with market conditions, client budgets, and business objectives.',
        backstory=(
            "You are a seasoned expert in pricing strategy development with a strong background in analyzing market trends, "
            "assessing client needs, and maximizing profitability. Your deep understanding of competitive positioning, cost structures, "
            "and value-based pricing enables you to create strategies that win bids while maintaining financial sustainability."
        ),
        verbose=True,
        allow_delegation=False,
        llm=groq_llm,
        tools=[
            Tool(
                name="Develop Pricing",
                func=develop_pricing_strategy,
                description=(
                    "Creates a tailored pricing strategy for proposals, factoring in market trends, competition, cost analysis, "
                    "and client requirements to ensure competitiveness and profitability."
                )
            )
        ]
    ),
    'quality_assurance': Agent(
        role='Quality Assurance',
        goal='Ensure proposals meet the highest standards of quality, accuracy, and compliance with client requirements and industry best practices.',
        backstory=(
            "You are a meticulous and detail-oriented expert in quality assurance, with a keen eye for identifying errors, "
            "ensuring consistency, and improving overall proposal quality. Your deep understanding of compliance requirements, "
            "style guidelines, and effective communication ensures proposals are polished, professional, and client-ready."
        ),
        verbose=True,
        allow_delegation=False,
        llm=groq_llm,
        tools=[
            Tool(
                name="Quality Check",
                func=quality_check,
                description=(
                    "Conducts a thorough quality check of the proposal, verifying accuracy, consistency, compliance with requirements, "
                    "and alignment with style and formatting guidelines. Provides recommendations for improvements where necessary."
                )
            )
        ]
    )
}

# Define Tasks
tasks = [
    Task(
        description="Analyze the bid request to extract key requirements, evaluate compliance needs, and provide actionable market insights.",
        agent=agents['bid_analyzer']
    ),
    Task(
        description="Create a clear and structured proposal outline that aligns with the bid analysis and client objectives.",
        agent=agents['proposal_outliner']
    ),
    Task(
        description="Write detailed and accurate technical sections of the proposal, ensuring they meet the bid requirements and showcase technical expertise.",
        agent=agents['technical_writer']
    ),
    Task(
        description="Craft persuasive and client-focused business sections of the proposal, highlighting value propositions and strategic benefits.",
        agent=agents['business_writer']
    ),
    Task(
        description="Develop a competitive and profitable pricing strategy that aligns with market trends and client budgets.",
        agent=agents['pricing_strategist']
    ),
    Task(
        description="Perform a thorough quality check on the complete proposal, ensuring it meets standards for accuracy, consistency, and compliance.",
        agent=agents['quality_assurance']
    )
]


# Create and Run Crew
bid_manager_crew = Crew(
    agents=list(agents.values()),
    tasks=tasks,
    verbose=2
)

# Execute the bid management process
result = bid_manager_crew.kickoff()
print(result)