from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain.llms import Groq

# Initialize Groq
groq = Groq(api_key="your-groq-api-key")

# Define tools (these are placeholder functions)
def analyze_bid_request(bid_request):
    prompt = f"""
    Analyze the following bid request and provide a summary of key requirements,
    evaluation criteria, and relevant market insights. Also, assess our company's
    capabilities against these requirements.

    Bid Request:
    {bid_request}

    Please provide your analysis in the following JSON format:
    {{
        "key_requirements": [],
        "evaluation_criteria": [],
        "market_insights": [],
        "capability_assessment": []
    }}
    """
    
    response = groq.generate(prompt)
    
    try:
        analysis = json.loads(response)
        return analysis
    except json.JSONDecodeError:
        return {"error": "Failed to parse the analysis. Please try again."}

def create_proposal_outline(requirements):
    prompt = f"""
    Based on the following requirements, create a detailed outline for a bid proposal.
    The outline should include main sections and subsections.

    Requirements:
    {requirements}

    Please provide your outline in the following JSON format:
    {{
        "executive_summary": [],
        "company_background": [],
        "technical_approach": [],
        "management_approach": [],
        "past_performance": [],
        "pricing": []
    }}
    """
    
    response = groq.generate(prompt)
    
    try:
        outline = json.loads(response)
        return outline
    except json.JSONDecodeError:
        return {"error": "Failed to parse the outline. Please try again."}

def write_technical_content(outline, specifications):
    prompt = f"""
    Write the technical content for a bid proposal based on the following outline
    and technical specifications. Focus on clearly explaining our approach,
    methodologies, and how we meet or exceed the technical requirements.

    Outline:
    {outline}

    Technical Specifications:
    {specifications}

    Please provide your content in the following JSON format:
    {{
        "technical_approach": "",
        "methodologies": "",
        "innovation": "",
        "technical_compliance": ""
    }}
    """
    
    response = groq.generate(prompt)
    
    try:
        technical_content = json.loads(response)
        return technical_content
    except json.JSONDecodeError:
        return {"error": "Failed to parse the technical content. Please try again."}

def write_business_content(outline, company_info):
    prompt = f"""
    Write the business content for a bid proposal based on the following outline
    and company information. Focus on highlighting our company's strengths,
    experience, and unique value proposition.

    Outline:
    {outline}

    Company Information:
    {company_info}

    Please provide your content in the following JSON format:
    {{
        "executive_summary": "",
        "company_background": "",
        "management_approach": "",
        "past_performance": ""
    }}
    """
    
    response = groq.generate(prompt)
    
    try:
        business_content = json.loads(response)
        return business_content
    except json.JSONDecodeError:
        return {"error": "Failed to parse the business content. Please try again."}

def develop_pricing_strategy(project_details):
    prompt = f"""
    Develop a pricing strategy for the bid based on the following project details.
    Consider costs, market rates, competition, and value proposition.

    Project Details:
    {project_details}

    Please provide your strategy in the following JSON format:
    {{
        "cost_breakdown": {{}},
        "pricing_approach": "",
        "competitive_analysis": "",
        "value_justification": ""
    }}
    """
    
    response = groq.generate(prompt)
    
    try:
        pricing_strategy = json.loads(response)
        return pricing_strategy
    except json.JSONDecodeError:
        return {"error": "Failed to parse the pricing strategy. Please try again."}

def quality_check(proposal):
    prompt = f"""
    Perform a quality check on the following proposal. Evaluate for completeness,
    coherence, compliance with requirements, and overall persuasiveness.

    Proposal:
    {proposal}

    Please provide your assessment in the following JSON format:
    {{
        "completeness": "",
        "coherence": "",
        "compliance": "",
        "persuasiveness": "",
        "suggested_improvements": []
    }}
    """
    
    response = groq.generate(prompt)
    
    try:
        quality_assessment = json.loads(response)
        return quality_assessment
    except json.JSONDecodeError:
        return {"error": "Failed to parse the quality assessment. Please try again."}

# Create agents
bid_analyzer = Agent(
    role='Bid Analyzer',
    goal='Analyze bid requests and gather market insights',
    backstory="You are an expert in analyzing bid requests and market trends.",
    verbose=True,
    allow_delegation=False,
    llm=groq,
    tools=[
        Tool(
            name="Analyze Bid",
            func=analyze_bid_request,
            description="Analyzes a bid request and provides a summary of requirements and market insights"
        )
    ]
)

proposal_outliner = Agent(
    role='Proposal Outliner',
    goal='Create a compelling proposal structure',
    backstory="You are skilled at creating effective proposal outlines.",
    verbose=True,
    allow_delegation=False,
    llm=groq,
    tools=[
        Tool(
            name="Create Outline",
            func=create_proposal_outline,
            description="Creates a proposal outline based on requirements"
        )
    ]
)

technical_writer = Agent(
    role='Technical Writer',
    goal='Write technical sections of the proposal',
    backstory="You are an expert in writing technical content for proposals.",
    verbose=True,
    allow_delegation=False,
    llm=groq,
    tools=[
        Tool(
            name="Write Technical Content",
            func=write_technical_content,
            description="Writes technical sections of the proposal"
        )
    ]
)

business_writer = Agent(
    role='Business Writer',
    goal='Write business sections of the proposal',
    backstory="You are skilled at writing compelling business content.",
    verbose=True,
    allow_delegation=False,
    llm=groq,
    tools=[
        Tool(
            name="Write Business Content",
            func=write_business_content,
            description="Writes business sections of the proposal"
        )
    ]
)

pricing_strategist = Agent(
    role='Pricing Strategist',
    goal='Develop competitive pricing strategy',
    backstory="You are an expert in developing pricing strategies for bids.",
    verbose=True,
    allow_delegation=False,
    llm=groq,
    tools=[
        Tool(
            name="Develop Pricing",
            func=develop_pricing_strategy,
            description="Develops a pricing strategy for the proposal"
        )
    ]
)

quality_assurance = Agent(
    role='Quality Assurance',
    goal='Ensure proposal quality and compliance',
    backstory="You are meticulous in reviewing and improving proposal quality.",
    verbose=True,
    allow_delegation=False,
    llm=groq,
    tools=[
        Tool(
            name="Quality Check",
            func=quality_check,
            description="Performs a quality check on the proposal"
        )
    ]
)

# Define tasks
task1 = Task(
    description="Analyze the bid request and provide a summary of requirements and market insights",
    agent=bid_analyzer
)

task2 = Task(
    description="Create a proposal outline based on the bid analysis",
    agent=proposal_outliner
)

task3 = Task(
    description="Write the technical sections of the proposal",
    agent=technical_writer
)

task4 = Task(
    description="Write the business sections of the proposal",
    agent=business_writer
)

task5 = Task(
    description="Develop a pricing strategy for the proposal",
    agent=pricing_strategist
)

task6 = Task(
    description="Perform a quality check on the complete proposal",
    agent=quality_assurance
)

# Create the crew
bid_manager_crew = Crew(
    agents=[bid_analyzer, proposal_outliner, technical_writer, business_writer, pricing_strategist, quality_assurance],
    tasks=[task1, task2, task3, task4, task5, task6],
    verbose=2
)

# Run the crew
result = bid_manager_crew.kickoff()

print(result)
