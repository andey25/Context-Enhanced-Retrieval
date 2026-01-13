import hashlib
import os
import random
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import requests
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import anthropic

# Load configuration
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)

class TextSegmenter:
    """
    Divides text into manageable segments with overlapping regions
    """
    
    def __init__(self, segment_length: int = 800, overlap: int = 100):
        self.segment_length = segment_length
        self.overlap = overlap
    
    def segment_text(self, text_content: str, source_id: str = "") -> List['TextSegment']:
        """
        Split text into overlapping segments
        """
        segments = []
        text_length = len(text_content)
        start_pos = 0
        
        while start_pos < text_length:
            end_pos = min(start_pos + self.segment_length, text_length)
            segment_content = text_content[start_pos:end_pos]
            
            segments.append(
                TextSegment(
                    content=segment_content,
                    source_id=source_id,
                    position_start=start_pos,
                    position_end=end_pos
                )
            )
            
            start_pos += (self.segment_length - self.overlap)
        
        return segments

@dataclass
class TextSegment:
    """Represents a portion of text with metadata"""
    content: str
    source_id: str
    position_start: int
    position_end: int
    contextual_hint: str = ""

class SemanticEncoder:
    """
    Converts text into numerical representations for similarity comparison
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector representations for multiple texts
        """
        return self.encoder.encode(texts).tolist()

class ContextEnhancer:
    """
    Augments text segments with broader document context
    """
    
    def __init__(self):
        self.llm_client = anthropic.Anthropic(
            api_key=config["llm"]["api_key"]
        )
    
    def enrich_segments(self, full_document: str, segments: List[TextSegment]) -> List[TextSegment]:
        """
        Add contextual information to each segment
        """
        enhanced_segments = []
        worker_count = min(20, len(segments))
        
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_segment = {
                executor.submit(self._add_contextual_hint, full_document, segment): segment
                for segment in segments
            }
            
            for future in future_to_segment:
                segment = future_to_segment[future]
                try:
                    contextual_hint = future.result()
                    segment.contextual_hint = contextual_hint
                    enhanced_segments.append(segment)
                except Exception as e:
                    print(f"Context enhancement failed: {e}")
                    enhanced_segments.append(segment)
        
        return enhanced_segments
    
    def _add_contextual_hint(self, full_text: str, segment: TextSegment) -> str:
        """
        Generate contextual information for a text segment
        """
        prompt = f"""Document Content:
{full_text}

Text Segment:
{segment.content}

Provide brief context about where this segment fits in the overall document.
Focus on document structure and surrounding content.
Keep response under 50 words."""

        response = self.llm_client.messages.create(
            model=config["llm"]["model"],
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

class DocumentIndexer:
    """
    Manages storage and retrieval of document segments
    """
    
    def __init__(self):
        self.encoder = SemanticEncoder()
        self.db_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=config["storage"]["path"]
        ))
        
        # Create or load collection
        self.collection = self.db_client.get_or_create_collection(
            name="document_segments",
            metadata={"hnsw:space": "cosine"}
        )
    
    def store_documents(self, documents: List[Dict[str, Any]]):
        """
        Process and store multiple documents
        """
        segmenter = TextSegmenter()
        enhancer = ContextEnhancer()
        
        for doc_info in documents:
            print(f"Processing: {doc_info['name']}")
            
            # Create segments
            raw_segments = segmenter.segment_text(
                doc_info["content"],
                doc_info["name"]
            )
            
            # Enhance with context
            enhanced_segments = enhancer.enrich_segments(
                doc_info["content"],
                raw_segments
            )
            
            # Prepare data for storage
            self._store_enhanced_segments(enhanced_segments, doc_info["name"])
    
    def _store_enhanced_segments(self, segments: List[TextSegment], doc_name: str):
        """
        Store enhanced segments in vector database
        """
        ids = []
        embeddings = []
        metadata_list = []
        documents = []
        
        for idx, segment in enumerate(segments):
            # Combine context hint with segment content
            full_content = f"{segment.contextual_hint}\n\n{segment.content}"
            
            # Generate unique identifier
            segment_id = f"{doc_name}_{idx}_{self._generate_identifier()}"
            
            ids.append(segment_id)
            documents.append(full_content)
            metadata_list.append({
                "source": doc_name,
                "position": f"{segment.position_start}-{segment.position_end}",
                "content_hash": hashlib.sha256(segment.content.encode()).hexdigest()[:16]
            })
        
        # Batch process embeddings
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_embeddings = self.encoder.encode(batch_docs)
            
            # Store in database
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_docs,
                metadatas=metadata_list[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
    
    def find_similar_documents(self, query_text: str, result_count: int = 3) -> List[str]:
        """
        Find documents relevant to the query
        """
        # Generate query embedding
        query_embedding = self.encoder.encode([query_text])[0]
        
        # Search for similar segments
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=result_count * 30,
            include=["metadatas"]
        )
        
        # Extract unique document sources
        unique_sources = set()
        for metadata_group in results["metadatas"]:
            for metadata in metadata_group:
                unique_sources.add(metadata["source"])
                if len(unique_sources) >= result_count:
                    break
            if len(unique_sources) >= result_count:
                break
        
        return list(unique_sources)
    
    def _generate_identifier(self) -> str:
        """
        Create random identifier for segments
        """
        return ''.join(random.choices(string.ascii_letters + string.digits, k=12))

class QueryProcessor:
    """
    Handles question answering based on retrieved information
    """
    
    def __init__(self):
        self.llm_client = anthropic.Anthropic(
            api_key=config["llm"]["api_key"]
        )
    
    def formulate_response(self, question: str, relevant_segments: List[str]) -> str:
        """
        Generate answer from relevant information
        """
        context_text = "\n---\n".join(relevant_segments)
        
        prompt = f"""Relevant Information:
{context_text}

Question: {question}

Based on the provided information, answer the question concisely.
If information is insufficient, state this clearly."""

        response = self.llm_client.messages.create(
            model=config["llm"]["model"],
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

def process_document_collection(document_set: List[Dict[str, Any]]):
    """
    Main function to index a collection of documents
    """
    indexer = DocumentIndexer()
    indexer.store_documents(document_set)
    print(f"Indexed {len(document_set)} documents")

def retrieve_relevant_sources(query_text: str, max_sources: int = 3) -> List[str]:
    """
    Find documents relevant to a query
    """
    indexer = DocumentIndexer()
    return indexer.find_similar_documents(query_text, max_sources)

# Example usage
if __name__ == "__main__":
    # Sample documents
    sample_docs = [
        {
            "name": "geography_facts",
            "content": "Paris serves as the capital city of France. It is known for its cultural landmarks..."
        },
        {
            "name": "european_capitals", 
            "content": "Berlin functions as Germany's capital. The city has a rich historical background..."
        }
    ]
    
    # Index documents
    process_document_collection(sample_docs)
    
    # Query the index
    relevant_docs = retrieve_relevant_sources("What city is France's capital?")
    print(f"Relevant documents: {relevant_docs}")