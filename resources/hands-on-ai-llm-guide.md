# hands-on-ai LLM Guide

This comprehensive guide provides detailed information about the `hands-on-ai` package to help large language models (LLMs) understand how to use and explain it effectively.

## 1. Package Overview and Core Features

`hands-on-ai` is a unified educational toolkit designed to teach students how modern AI systems work by building and interacting with them directly. The package includes three primary modules:

| Module | Purpose | CLI Name |
|--------|---------|----------|
| **chat** | Simple chatbot with system prompts and personalities | `chat` |
| **rag** | Retrieval-Augmented Generation for document Q&A | `rag` |
| **agent** | ReAct-style reasoning with tool use | `agent` |

Each module is:
- Self-contained with its own functionality
- Accessible via a single package installation
- Designed for progressive learning from simple to complex AI concepts

The package aims to make AI learning accessible for students and educators through hands-on experimentation with key AI paradigms.

## 2. Installation Instructions

### Basic Installation

```bash
# Install from PyPI
pip install hands-on-ai

# Or directly from GitHub
pip install git+https://github.com/teaching-repositories/hands-on-ai.git
```

### Prerequisites

- Python 3.8 or higher
- For local LLM usage: [Ollama](https://ollama.ai/) or similar local LLM server

### Development Installation

```bash
# Install with development dependencies
pip install hands-on-ai[dev]
```

## 3. Complete API Reference

### Configuration Module

The central configuration system manages settings across all modules.

```python
from hands_on_ai.config import load_config, save_config, get_server_url, get_model

# Get configuration values
config = load_config()  # Returns dict with all config
model = get_model()     # Get default LLM model
server = get_server_url()  # Get Ollama server URL

# Save configuration
config["model"] = "llama3"
save_config(config)
```

#### Key Configuration Functions:
- `load_config()`: Load configuration from file or environment variables
- `save_config(config)`: Save configuration to file
- `get_server_url()`: Get the server URL from config
- `get_model()`: Get the default model from config
- `get_embedding_model()`: Get the default embedding model
- `get_chunk_size()`: Get the default chunk size for RAG

Environment variables that override config:
- `AILABKIT_SERVER`: Override server URL
- `AILABKIT_MODEL`: Override default model
- `AILABKIT_EMBEDDING_MODEL`: Override embedding model
- `AILABKIT_LOG=debug`: Enable debug logging

### Chat Module API

The chat module provides simple LLM interactions with various personalities.

```python
from hands_on_ai.chat import get_response, pirate_bot, coder_bot

# Basic chat usage
response = get_response("Tell me about planets")

# Use a personality bot
pirate_response = pirate_bot("Tell me about sailing ships")
```

#### Core Functions:

`get_response(prompt, model=None, system="You are a helpful assistant.", personality="friendly", stream=False, retries=2)`
- `prompt`: Text prompt to send to the model (required)
- `model`: LLM model to use (defaults to config setting)
- `system`: System message defining bot behavior
- `personality`: Used for fallback character during retries
- `stream`: Whether to request streaming output (default False)
- `retries`: Number of times to retry on error

#### Available Personality Bots:
- `friendly_bot`: General helpful assistant
- `sarcastic_bot`: Sarcastic, humorous responses
- `pirate_bot`: Answers in pirate speak
- `shakespeare_bot`: Shakespearean language
- `coder_bot`: Technical programming assistant
- `teacher_bot`: Educational, explanatory style
- `therapist_bot`: Supportive, reflective responses
- And many more...

### RAG Module API

The RAG (Retrieval-Augmented Generation) module enables document-based question answering.

```python
from hands_on_ai.rag import (
    load_text_file, chunk_text, get_embeddings,
    save_index_with_sources, load_index_with_sources,
    get_top_k, copy_sample_docs
)

# Basic RAG workflow
text = load_text_file("document.pdf")
chunks = chunk_text(text, chunk_size=500)
vectors = get_embeddings(chunks)
save_index_with_sources(vectors, chunks, ["document.pdf"] * len(chunks), "index.npz")

# Query the index
results = get_top_k("What is climate change?", "index.npz", k=3)
```

#### Core Functions:

`load_text_file(path)`: 
- Loads text from .txt, .md, .docx, or .pdf files
- Returns extracted text content

`chunk_text(text, chunk_size=None)`:
- Splits text into chunks for embedding
- Default chunk_size from config if not specified

`get_embeddings(chunks, model=None)`:
- Generates vector embeddings for text chunks
- Uses default embedding model from config if not specified

`save_index_with_sources(vectors, chunks, sources, path)`:
- Saves RAG index with source tracking
- Preserves relationship between chunks and source documents

`load_index_with_sources(path)`:
- Loads previously saved RAG index
- Returns (vectors, chunks, sources) tuple

`get_top_k(query, index_path, k=3, return_scores=False)`:
- Retrieves top k similar chunks for a query
- Optionally returns similarity scores

`get_sample_docs_path()`, `list_sample_docs()`, `copy_sample_docs()`:
- Helper functions for working with sample documents

### Agent Module API

The agent module implements ReAct-style agents with tool use capabilities.

```python
from hands_on_ai.agent import run_agent, register_tool, list_tools
from hands_on_ai.agent import calculator, dictionary, converter

# Run a pre-configured calculator agent
result = calculator("Calculate 5 + 3 * 2")

# Custom agent with tool registration
def weather_tool(location):
    return f"Weather forecast for {location}: Sunny, 25°C"

register_tool("weather", "Get weather for a location", weather_tool)
response = run_agent("What's the weather in London?")
```

#### Core Functions:

`run_agent(prompt, model=None, max_iterations=5, verbose=False)`:
- Runs the agent with the given prompt
- Uses ReAct format for reasoning and tool use
- Returns final response after tool interactions
- Parameters:
  - `prompt`: User question or instruction
  - `model`: LLM model to use (default from config)
  - `max_iterations`: Maximum tool use iterations
  - `verbose`: Print intermediate steps

`register_tool(name, description, function)`:
- Registers a new tool with the agent
- Parameters:
  - `name`: Tool name
  - `description`: Tool description
  - `function`: Python function that implements the tool

`list_tools()`:
- Returns list of all registered tools

#### Built-in Agents:
- `calculator`: Simple calculations
- `dictionary`: Word definitions and information
- `converter`: Unit conversions

## 4. Usage Examples

### Command-Line Interface (CLI) Examples

Each module includes a CLI for easy access to functionality.

#### Main CLI

```bash
# List available modules
handsonai list

# Configure the toolkit
handsonai config set model llama3

# Check environment and setup
handsonai doctor
```

#### Chat CLI

```bash
# Simple question with default bot
chat ask "What is photosynthesis?"

# Use a specific personality
chat ask "Explain quantum physics" --personality teacher_bot

# Start interactive chat session
chat interactive

# Launch web interface
chat web
```

#### RAG CLI

```bash
# Copy sample documents to current directory
rag index --samples

# Index documents in a directory
rag index ./docs --output_file index.npz

# Ask questions about indexed documents
rag ask "What is TCP/IP?" --index_file index.npz

# Interactive RAG session
rag interactive --index_file index.npz

# Web interface for RAG
rag web --index_file index.npz
```

#### Agent CLI

```bash
# List available tools
agent tools

# Ask a question that requires tools
agent ask "Calculate 25 * 4 and convert to binary"

# Interactive agent session
agent interactive

# Launch agent web interface
agent web
```

### Python API Examples

#### Basic Chat Example

```python
from hands_on_ai.chat import get_response, pirate_bot

# Simple chat with default model
response = get_response("What is the capital of France?")
print(response)  # Paris

# Chat with a personality
pirate_response = pirate_bot("Tell me about sailing ships")
print(pirate_response)  # Arr matey! Sailing ships be magnificent vessels...
```

#### RAG Example Workflow

```python
from pathlib import Path
from hands_on_ai.rag import (
    load_text_file, chunk_text, get_embeddings,
    save_index_with_sources, get_top_k
)
from hands_on_ai.chat import get_response

# 1. Load and prepare documents
documents = ["report.pdf", "whitepaper.txt", "notes.md"]
all_chunks = []
all_sources = []

for doc in documents:
    text = load_text_file(Path(doc))
    chunks = chunk_text(text, chunk_size=300)
    all_chunks.extend(chunks)
    all_sources.extend([doc] * len(chunks))

# 2. Create embeddings and index
vectors = get_embeddings(all_chunks)
save_index_with_sources(vectors, all_chunks, all_sources, "my_index.npz")

# 3. Retrieve relevant context for a query
query = "What are the main findings?"
context_chunks = get_top_k(query, "my_index.npz", k=3)

# 4. Format context for the LLM
context_text = "\n\n".join([chunk for chunk, source in context_chunks])
prompt = f"""Answer based on this context:

{context_text}

Question: {query}
"""

# 5. Get response from LLM with context
answer = get_response(prompt, system="You are a helpful research assistant.")
print(answer)
```

#### Custom Agent with Tools

```python
from hands_on_ai.agent import run_agent, register_tool

# Define a custom tool
def reverse_text(text):
    return text[::-1]

# Register the tool with the agent
register_tool(
    "reverse", 
    "Reverse the input text", 
    reverse_text
)

# Define a translator tool
def translate(text):
    # This would use a real translation API in practice
    return f"Translated: {text} (to Spanish)"

register_tool(
    "translate",
    "Translate text to Spanish",
    translate
)

# Run the agent with a question that needs tools
result = run_agent(
    "Please reverse the text 'hello world' and then translate it",
    verbose=True
)

print(f"Final result: {result}")
```

## 5. Architecture Details

### Package Structure

```
hands_on_ai/
├── chat/           # Simple prompt/response chatbot
│   ├── __init__.py
│   ├── bots.py
│   ├── cli.py
│   ├── commands/
│   ├── get_response.py
│   └── personalities/
├── rag/            # Retrieval-Augmented Generation
│   ├── __init__.py
│   ├── cli.py
│   ├── commands/
│   ├── data/
│   └── utils.py
├── agent/          # ReAct-style agents with tools
│   ├── __init__.py
│   ├── agents/
│   ├── cli.py
│   ├── commands/
│   ├── core.py
│   ├── prompts.py
│   └── tools/
├── config.py       # Shared configuration
├── cli.py          # Main CLI entry point
└── utils/          # Shared utilities
```

### Design Principles

1. **Modular Design**: Each module (chat, rag, agent) is self-contained but follows consistent patterns
2. **Progressive Complexity**: Modules increase in complexity, building upon concepts:
   - Chat: Simple LLM interaction
   - RAG: LLM + document context
   - Agent: LLM + reasoning + tools
3. **Educational Focus**: Clear, readable code with comments designed for learning
4. **Configuration System**: Shared settings with multiple override options
5. **Consistent CLI Structure**: Similar command patterns across modules

### Workflow Patterns

1. **Chat Pattern**:
   - User provides prompt
   - System adds context/personality
   - LLM generates response

2. **RAG Pattern**:
   - Documents → Chunking → Embeddings → Index
   - Query → Retrieve context → Augment prompt → LLM response

3. **Agent Pattern** (ReAct):
   - User query → LLM reasoning
   - Tool selection → Tool execution
   - Observation → Further reasoning
   - Final answer

## 6. Common Use Cases and Project Ideas

### Educational Use Cases

1. **Classroom Demonstrations**:
   - Show how different system prompts affect LLM responses
   - Demonstrate the impact of adding document context
   - Explain tool use and reasoning in agents

2. **Progressive Assignments**:
   - Start with personality creation in chat module
   - Advance to document-based QA with RAG
   - Culminate with custom tool creation for agents

3. **Compare/Contrast Learning**:
   - Same prompt across different personalities
   - RAG with/without relevant context
   - Agent with limited vs expanded tool access

### Project Ideas

#### Chat Projects
- Custom expert personality (e.g., scientist, historian)
- Multi-personality chatbot with switchable modes
- Role-playing game character generator

#### RAG Projects
- Personal knowledge base with your notes/documents
- Course material Q&A system
- Technical documentation assistant

#### Agent Projects
- Financial assistant with calculator and conversion tools
- Research agent with web search and summarization
- Code assistant with syntax checking and explanation tools

#### Cross-Module Projects
- Learning coach: combines personality (chat) + materials (RAG) + interactive exercises (agent)
- Academic research assistant: searches documents (RAG) + analyzes data (agent) + explains findings (chat)
- Content creator: generates ideas (chat) + researches topics (RAG) + creates structured content (agent)

## 7. Best Practices and Limitations

### Best Practices

1. **Model Selection**:
   - Use smaller, faster models for development (e.g., llama3)
   - Switch to larger models for better reasoning in complex agent tasks
   - Match model capabilities to task complexity

2. **RAG Optimization**:
   - Keep chunk size between 300-1000 words depending on document type
   - Include deliberate overlap between chunks (10-20%)
   - Test different embedding models for specific domains

3. **Agent Design**:
   - Keep tool descriptions clear and specific
   - Limit agent iterations to 5-10 maximum
   - Implement error handling in custom tools

4. **Configuration Management**:
   - Use environment variables for sensitive settings
   - Create domain-specific config files for different projects
   - Document configuration options for users

### Limitations

1. **LLM Reliability**:
   - Responses vary by model and prompt
   - Hallucinations can occur, especially without RAG
   - Tool selection may not always be optimal

2. **Performance Considerations**:
   - Local LLM inference can be slow on limited hardware
   - Large document collections may require optimization
   - Vector operations scale with document count

3. **Technical Constraints**:
   - CLI is primarily designed for demonstration/learning
   - Web interfaces are basic/educational, not production-grade
   - Limited to English language currently

4. **Dependency on Ollama**:
   - Designed to work with Ollama for local LLM inference
   - May require adaptation for other LLM backends
   - Limited streaming support

### Error Handling

The package implements several error handling strategies:

1. **Retries**: Automatic retries for LLM failures
2. **Fallback Messages**: Personality-specific fallbacks when server errors occur
3. **Configuration Fallbacks**: Defaults when config files can't be read
4. **Verbose Logging**: Optional debug logging with environment variable

## Conclusion

The `hands-on-ai` package provides a comprehensive toolkit for learning and experimenting with modern AI techniques. By offering a progression from simple chatbots to more complex RAG and agent implementations, it enables students and educators to gain hands-on experience with key AI concepts in a structured way.

The modular design, consistent interfaces, and educational focus make it an excellent resource for understanding how LLMs work in different contexts, from basic prompt engineering to more advanced techniques like retrieval-augmented generation and tool-using agents.