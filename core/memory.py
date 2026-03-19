"""
Memory module for the multi-agent research system.
Handles vector database operations and persistent knowledge storage.
"""

import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from pathlib import Path

from .config import config

class ResearchMemory:
    """Memory system for storing research knowledge and agent interactions."""
    
    def __init__(self):
        self.vector_db_path = Path(config.vector_db_path)
        self.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.dimension = 768  # Standard embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Metadata storage
        self.metadata: List[Dict[str, Any]] = []
        self.debate_log: List[Dict[str, Any]] = []
        self.feedback_log: List[Dict[str, Any]] = []
        
        # Load existing data if available
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing memory data from disk."""
        try:
            # Load FAISS index
            if (self.vector_db_path / "index.faiss").exists():
                self.index = faiss.read_index(str(self.vector_db_path / "index.faiss"))
            
            # Load metadata
            if (self.vector_db_path / "metadata.pkl").exists():
                with open(self.vector_db_path / "metadata.pkl", "rb") as f:
                    self.metadata = pickle.load(f)
            
            # Load debate log
            if Path(config.debate_log_path).exists():
                with open(config.debate_log_path, "r") as f:
                    self.debate_log = json.load(f)
            
            # Load feedback log
            if Path(config.feedback_log_path).exists():
                with open(config.feedback_log_path, "r") as f:
                    self.feedback_log = json.load(f)
                    
        except Exception as e:
            print(f"Warning: Could not load existing memory data: {e}")
    
    def save(self):
        """Save all memory data to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.vector_db_path / "index.faiss"))
            
            # Save metadata
            with open(self.vector_db_path / "metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
            
            # Save debate log
            with open(config.debate_log_path, "w") as f:
                json.dump(self.debate_log, f, indent=2)
            
            # Save feedback log
            with open(config.feedback_log_path, "w") as f:
                json.dump(self.feedback_log, f, indent=2)
                
        except Exception as e:
            print(f"Error saving memory data: {e}")
    
    def add_embedding(self, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Add an embedding with metadata to the vector database."""
        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} != {self.dimension}")
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1))
        
        # Store metadata
        metadata["timestamp"] = datetime.now().isoformat()
        metadata["id"] = len(self.metadata)
        self.metadata.append(metadata)
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} != {self.dimension}")
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Return metadata for similar items
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["distance"] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def add_debate_entry(self, topic: str, proposer_argument: str, challenger_argument: str, 
                         moderator_decision: str, score: float):
        """Add a debate entry to the log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "proposer_argument": proposer_argument,
            "challenger_argument": challenger_argument,
            "moderator_decision": moderator_decision,
            "score": score
        }
        self.debate_log.append(entry)
        self.save()
    
    def add_feedback_entry(self, agent_name: str, section: str, score: float, 
                          feedback: str, iteration: int):
        """Add a feedback entry to the log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "section": section,
            "score": score,
            "feedback": feedback,
            "iteration": iteration
        }
        self.feedback_log.append(entry)
        self.save()
    
    def get_recent_debates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent debate entries."""
        return self.debate_log[-limit:]
    
    def get_recent_feedback(self, agent_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent feedback entries, optionally filtered by agent."""
        if agent_name:
            filtered = [entry for entry in self.feedback_log if entry["agent"] == agent_name]
            return filtered[-limit:]
        return self.feedback_log[-limit:]
    
    def get_average_score(self, agent_name: str, recent_n: int = 5) -> float:
        """Get average score for an agent over recent iterations."""
        recent_feedback = self.get_recent_feedback(agent_name, recent_n)
        if not recent_feedback:
            return 0.0
        
        scores = [entry["score"] for entry in recent_feedback]
        return sum(scores) / len(scores)

# Global memory instance
memory = ResearchMemory() 