#!/bin/bash

set -e

echo "Installing dependencies..."

python3 -m pip install --upgrade pip

pip install sentence-transformers
pip install chromadb
pip install anthropic
pip install PyYAML
pip install datasketch
pip install numpy
pip install pypdf
pip install requests

mkdir -p data/raw_documents
mkdir -p data/processed
mkdir -p logs

if [ ! -f config.yaml ]; then
    cat > config.yaml << 'EOF'
llm:
  api_key: "anthropic_api_key"
  model: "claude-3-haiku-20240307"

storage:
  path: "./chroma_db"

processing:
  segment_length: 800
  overlap: 100
  embedding_model: "all-MiniLM-L6-v2"
EOF
    echo "Created config.yaml"
fi

echo "Setup complete!"
echo "Edit config.yaml with your API key"
