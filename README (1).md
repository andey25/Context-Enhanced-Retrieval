# Contextual Document Retrieval System

An AI-powered document search and retrieval system that enhances search
accuracy by adding contextual understanding to document segments.

## Features

-   Smart Document Segmentation: Splits documents into overlapping
    chunks for better context preservation
-   Context Enhancement: Uses LLM to add contextual information to each
    segment
-   Semantic Search: Vector-based similarity search using embeddings
-   Parallel Processing: Efficient batch processing for large document
    collections
-   Multiple Retrieval Methods: Combines exact and approximate matching

## Installation

### Prerequisites

-   Python 3.8+
-   Anthropic API key (for Claude models)

### Setup

``` bash
git clone https://github.com/yourusername/contextual-retrieval.git
cd contextual-retrieval
chmod +x setup.sh
./setup.sh
```

Configure your API key in `config.yaml`:

``` yaml
llm:
  api_key: "your_anthropic_api_key_here"
  model: "claude-3-haiku-20240307"
```

## Usage

### Basic Example

``` python
from document_retrieval import process_document_collection, retrieve_relevant_sources

documents = [
    {"name": "document1", "content": "Your document text here..."},
    {"name": "document2", "content": "Another document text..."}
]

process_document_collection(documents)
query = "What is artificial intelligence?"
results = retrieve_relevant_sources(query, max_sources=3)
print(results)
```

### Advanced Usage

``` python
from document_retrieval import DocumentIndexer, QueryProcessor

indexer = DocumentIndexer()
indexer.store_documents([
    {"name": "research_paper", "content": "Full paper text..."}
])

processor = QueryProcessor()
relevant_docs = indexer.find_similar_documents("research question", 5)
answer = processor.formulate_response("research question", relevant_docs)
print(answer)
```

## Project Structure

    .
    ├── document_retrieval.py
    ├── config.yaml
    ├── setup.sh
    ├── requirements.txt
    ├── data/
    │   ├── raw_documents/
    │   └── processed/
    ├── logs/
    └── chroma_db/

## Configuration

``` yaml
llm:
  api_key: "your_key"
  model: "claude-3-haiku-20240307"

storage:
  path: "./chroma_db"

processing:
  segment_length: 800
  overlap: 100
  embedding_model: "all-MiniLM-L6-v2"
```

## Dependencies

-   sentence-transformers
-   chromadb
-   anthropic
-   PyYAML
-   datasketch
