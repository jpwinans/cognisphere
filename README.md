# CogniSphere (WIP)

CogniSphere is knowledge acquisition for AI Agents. It's a Python-based system designed to programmatically index text from various sources, including YouTube transcripts, Wikipedia, scientific journals, and books. It focuses on logical modules, document input, preprocessing, appropriate indexing, data structuring like knowledge graphs, semantic search, and vector embedding. CogniSphere performs textual analysis, compares themes, concepts, and ideas, and generates novel hypotheses, questions, inferred conclusions, and theories for an AI agent to ask relevant questions and gain new knowledge.

## System Overview

The system consists of several components and services that work together to index text from various sources, perform textual analysis, and generate novel hypotheses, questions, inferred conclusions, and theories. The system is designed using microservices architecture to ensure scalability, maintainability, and flexibility.

## Modules and Components

The system is divided into the following logical modules and components:

### A. Data Collection and Ingestion

- Web Scrapers: For collecting data from Wikipedia, scientific journals, and books.
- YouTube Transcript Extractor: For extracting transcripts from YouTube videos.
- Data Ingestion Service: For ingesting the collected data into the system.

### B. Text Processing and Analysis Service

- Preprocessing: Cleaning, tokenizing, and normalizing the text data.
- Named Entity Recognition (NER): Identifying and classifying entities in the text.
- Indexing: Indexing the preprocessed text using appropriate indexing techniques.
- Knowledge Graph: Creating and maintaining a knowledge graph based on the indexed data.
- Vector Embedding: Generating vector embeddings of the text using AI and ML techniques.
- Semantic Search: Searching and retrieving relevant information from the indexed data.
- Textual Analysis: Comparing themes, concepts, and ideas in the text.
- Hypothesis Generation: Generating novel hypotheses, questions, inferred conclusions, and theories.

### C. AI Agent Interaction

- AI Agent Interface: For enabling the AI agent to interact with the system and ask relevant questions.

## Getting Started

These instructions will help you set up the CogniSphere project on your local machine for development and testing purposes.

### Prerequisites

- Python 3.8 or higher
- pip
- virtualenv (optional)

### Installation

1. Clone the repository:

```
git clone https://github.com/jpwinans/cognisphere.git
```

2. Create a virtual environment (optional):

```
cd CogniSphere
python -m venv venv
source venv/bin/activate # for Linux and macOS
venv\Scripts\activate # for Windows
```

3. Install the required packages:

```
pip install -r requirements.txt
```

4. Add your API keys and other required configurations in the appropriate configuration files.

5. Run the application and follow the instructions.

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

