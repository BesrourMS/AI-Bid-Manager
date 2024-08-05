# AI-Powered Bid Manager System

## Overview

This project implements an AI-powered Bid Manager system using the CrewAI framework and Groq for inference. The system automates the process of analyzing bid requests, creating proposal outlines, writing technical and business content, developing pricing strategies, and performing quality assurance on bid proposals.

## Features

- Multi-agent system with specialized roles:
  - Bid Analyzer
  - Proposal Outliner
  - Technical Writer
  - Business Writer
  - Pricing Strategist
  - Quality Assurance
- Automated bid request analysis
- AI-generated proposal outlines
- AI-assisted content creation for technical and business sections
- Automated pricing strategy development
- Quality assurance checks on completed proposals

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ai-bid-manager.git
   cd ai-bid-manager
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Groq API key:
   - Add your Groq API key

## Usage

1. Prepare your bid request data in a JSON format.

2. Run the main script:
   ```
   python bid_manager.py --input bid_request.json
   ```

3. The system will process the bid request and generate a complete proposal, which will be saved as `final_proposal.json`.

## Configuration

You can customize the behavior of each agent by modifying their roles, goals, and backstories in the `bid_manager.py` file.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) for the multi-agent framework
- [Groq](https://groq.com/) for the AI inference engine

## Disclaimer

This system is designed to assist in the bid management process but should not replace human oversight. Always review and verify the output before submitting any proposals.
