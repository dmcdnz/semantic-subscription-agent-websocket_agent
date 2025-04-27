#!/usr/bin/env python

"""
Base Agent implementation for containerized agents

This standalone version doesn't require the full semsubscription package.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List
import pickle

# Import torch components for the classification head
try:
    import torch
    from torch import nn
    from sentence_transformers import SentenceTransformer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, classification head functionality will be limited")

logger = logging.getLogger(__name__)

# Define classification head if PyTorch is available
if TORCH_AVAILABLE:
    class InterestClassificationHead(nn.Module):
        """Classification head to determine if a message is relevant to an agent"""
        
        def __init__(self, input_dim, dropout_prob=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout_prob)
            self.linear = nn.Linear(input_dim, 1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, embeddings):
            x = self.dropout(embeddings)
            x = self.linear(x)
            return self.sigmoid(x)

class BaseAgent:
    """
    Base class for containerized agents that don't require the full semsubscription package
    """
    
    def __init__(self, agent_id=None, name=None, description=None, similarity_threshold=0.7):
        """
        Initialize the agent with its parameters
        """
        self.agent_id = agent_id or os.environ.get("AGENT_ID", "unknown")
        self.name = name or os.environ.get("AGENT_NAME", "Unknown Agent")
        self.description = description or "A containerized agent"
        self.similarity_threshold = similarity_threshold
        self.classifier_threshold = 0.5  # Default threshold for interest
        
        # Initialize model properties
        self.embedding_model = None
        self.classification_head = None
        self.use_classifier = False
        self.interest_model = None
        
        # Set up interest models if available
        self.setup_interest_model()
        if TORCH_AVAILABLE:
            self.setup_classifier()
        
        logger.info(f"Initialized agent: {self.name} (ID: {self.agent_id})")
    
    def setup_classifier(self):
        """Load the fine-tuned classifier model for interest determination"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, cannot setup classifier")
            return
        
        # Look for fine-tuned model in the current directory
        model_dir = os.path.join(os.path.dirname(__file__), "fine_tuned_model")
        if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
            logger.warning(f"No fine-tuned model directory found at {model_dir}")
            return
        
        # Check for classification head file
        head_path = os.path.join(model_dir, "classification_head.pt")
        if not os.path.exists(head_path):
            logger.warning(f"No classification head found at {head_path}")
            return
            
        # Load the model
        try:
            logger.info(f"Loading fine-tuned model from {model_dir}")
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Check if this is a sentence transformer directory (has config.json)
            if os.path.exists(os.path.join(model_dir, "config.json")):
                logger.info("Loading SentenceTransformer model from directory")
                self.embedding_model = SentenceTransformer(model_dir, device=device_name)
            else:
                logger.warning(f"No SentenceTransformer model found in {model_dir}")
                return
            
            # Load classification head with explicit device mapping
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.classification_head = InterestClassificationHead(embedding_dim)
            self.classification_head.load_state_dict(torch.load(head_path, map_location=device_name))
            self.classification_head.to(device_name)  # Explicitly move to the right device
            self.classification_head.eval()
            
            # Success - set flag to use classifier
            self.use_classifier = True
            logger.info(f"Successfully loaded classifier for agent {self.name}")
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            self.use_classifier = False
    
    def setup_interest_model(self):
        """Configure the agent's interest model for vector similarity"""
        try:
            # Try to import the InterestModel from the local module
            from interest_model import CustomInterestModel as InterestModel
            # Import the embedding engine from the local module
            from embedding_engine import EmbeddingEngine
            
            # Create the embedding engine and interest model
            embedding_engine = EmbeddingEngine()
            self.interest_model = InterestModel(embedding_engine=embedding_engine)
            
            # Try to load pre-trained interest vectors
            model_dir = os.path.join(os.path.dirname(__file__), "fine_tuned_model")
            interest_vectors_path = os.path.join(model_dir, "interest_model.npz")
            
            if os.path.exists(interest_vectors_path):
                try:
                    logger.info(f"Loading interest vectors from {interest_vectors_path}")
                    self.interest_model.load(interest_vectors_path)
                    logger.info("Successfully loaded interest vectors")
                except Exception as e:
                    logger.error(f"Error loading interest vectors: {e}")
            else:
                logger.warning(f"No interest vectors found at {interest_vectors_path}")
        except ImportError as e:
            logger.warning(f"Could not load interest model: {e}")
    
    def calculate_interest(self, message):
        """
        Calculate agent interest in a message
        
        Args:
            message: Message to calculate interest for
            
        Returns:
            Float interest score between 0 and 1
        """
        # Extract content from message
        if isinstance(message, dict):
            content = message.get('content', '')
        else:
            content = getattr(message, 'content', '')
            
        # Multi-tier interest determination process (same as original system)
        
        # Tier 1: Use fine-tuned classifier if available
        if self.use_classifier and TORCH_AVAILABLE and self.embedding_model and self.classification_head:
            try:
                # Generate embedding
                with torch.no_grad():
                    embedding = self.embedding_model.encode([content], convert_to_tensor=True)
                    # Pass through classification head
                    score = self.classification_head(embedding).item()
                logger.info(f"Classifier interest score: {score}")
                return score
            except Exception as e:
                logger.error(f"Error using classifier: {e}")
                # Continue to next tier
        
        # Tier 2: Use vector similarity if interest model is available
        if self.interest_model:
            try:
                score = self.interest_model.calculate_similarity(content)
                logger.info(f"Vector similarity interest score: {score}")
                return score
            except Exception as e:
                logger.error(f"Error calculating vector similarity: {e}")
                # Continue to fallback
        
        # Tier 3: Keyword matching as fallback
        keywords = self.get_keywords()
        content_lower = content.lower()
        
        # Check for keyword matches
        for keyword in keywords:
            if keyword.lower() in content_lower:
                score = 0.8  # High interest for keyword match
                logger.info(f"Keyword match interest score: {score} (matched '{keyword}')")
                return score
        
        # Final fallback - return minimal interest
        logger.info("No interest determination method succeeded, returning minimal interest")
        return 0.1  # Low default interest
    
    def get_keywords(self):
        """
        Get domain keywords from config
        """
        # Try to load from config.yaml
        try:
            with open("config.yaml", "r") as f:
                import yaml
                config = yaml.safe_load(f)
                return config.get("interest_model", {}).get("keywords", [])
        except Exception as e:
            logger.warning(f"Error loading keywords from config: {e}")
            return ["test", "example", "demo"]
    
    def process_message(self, message):
        """
        Process a message
        
        Args:
            message: Message to process
            
        Returns:
            Optional result dictionary with response
        """
        # Simple echo implementation - override in subclasses
        return {
            "agent": self.name,
            "response": f"Processed message from {self.name}",
            "input": message.get("content", "")
        }
    
    def __str__(self):
        return f"{self.name} ({self.agent_id})"
