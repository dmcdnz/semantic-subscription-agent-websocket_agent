#!/usr/bin/env python

"""
Websocket_agent Agent Implementation

Creates a websocket interface for the eventbus
"""

import json
import logging
import os
import re
import uuid
from typing import Dict, Any, Optional

# For containerized agents, use the local base agent
# This avoids dependencies on the semsubscription module
try:
    # First try to import from semsubscription if available (for local development)
    from semsubscription.agents.EnhancedAgent import EnhancedAgent as BaseAgent
except ImportError:
    try:
        # Fall back to local agent_base for containerized environments
        # Don't use relative import (from .) in templates - it causes errors in containers
        from agent_base import BaseAgent
    except ImportError:
        try:
            # Last resort for Docker environment with current directory
            import sys
            # Add the current directory to the path to find agent_base.py
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from agent_base import BaseAgent
        except ImportError:
            # If all else fails, define a minimal BaseAgent class for compatibility
            class BaseAgent:
                """Minimal implementation of BaseAgent for compatibility"""
                def __init__(self, agent_id=None, name=None, description=None, similarity_threshold=0.7, **kwargs):
                    self.agent_id = agent_id or str(uuid.uuid4())
                    self.name = name or self.__class__.__name__
                    self.description = description or ""
                    self.similarity_threshold = similarity_threshold
                    self.config = kwargs.get('config', {})
                    self.classifier_threshold = 0.5  # Fixed threshold for testing

                def calculate_interest(self, message):
                    """
                    Calculate interest level for a message.
                    
                    This uses the fine-tuned model when available, or falls back to the
                    development implementation when no model is available.
                    
                    Args:
                        message: Message to calculate interest for (dict or object)
                        
                    Returns:
                        float: Interest score between 0.0 and 1.0
                    """
                    # Extract content based on message type (dict or object)
                    content = ""
                    message_id = "unknown"
                    
                    if isinstance(message, dict):
                        content = message.get('content', 'No content in dict')
                        message_id = message.get('id', 'unknown-id')
                    else:
                        content = getattr(message, 'content', 'No content in object')
                        message_id = getattr(message, 'id', 'unknown-id')
                    
                    logging.info(f"Calculating interest for message {message_id}")
                    logging.info(f"Message content: {content[:100]}...")
                    
                    # Use the interest model properly if it exists
                    if hasattr(self, 'interest_model') and self.interest_model:
                        try:
                            # This is the key method to call for the interest model
                            interest_score = self.interest_model.calculate_similarity(content)
                            logging.info(f"Interest model score: {interest_score}")
                            return interest_score
                        except Exception as e:
                            logging.error(f"Error using interest model: {e}")
                            # Continue to fallback implementation
                    
                    # Fallback implementation
                    # Simple implementation: keyword matching for domain relevance
                    keywords = [
                        # Add domain-specific keywords here
                        "question", "answer", "why", "how", "what", "when", "who"
                    ]
                    
                    # Count keyword matches
                    matches = sum(1 for keyword in keywords if keyword.lower() in content.lower())
                    
                    # Calculate interest score based on keyword density
                    if matches > 0:
                        # At least one keyword match - express interest
                        interest_score = min(0.5 + (matches * 0.1), 1.0)  # Scale with matches, cap at 1.0
                    else:
                        # No keywords match, still provide minimal interest
                        interest_score = 0.6  # Default minimal interest
                    
                    logging.info(f"Fallback interest calculation: {interest_score} (based on {matches} keyword matches)")
                    return interest_score
                
                def process_message(self, message):
                    """Process a message and return a test confirmation response"""
                    # Extract message details for better logging
                    content = message.get('content', 'No content') if isinstance(message, dict) else getattr(message, 'content', 'No content')
                    message_id = message.get('id', 'unknown-id') if isinstance(message, dict) else getattr(message, 'id', 'unknown-id')
                    
                    logger.info(f"Test agent processing message {message_id}")
                    logger.info(f"Message content: {content[:100]}...")
                    
                    # Build and return a response
                    response = {
                        "agent": self.__class__.__name__,
                        "status": "success",
                        "message": "This is a test confirmation from the default agent template",
                        "received": content,
                        "processed_at": time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    logger.info(f"Generated response: {str(response)[:200]}...")
                    return response

logger = logging.getLogger(__name__)

# Primary class name without Agent suffix (modern style)
class Websocket_agent(BaseAgent):
    """
    Agent that creates a websocket interface for the eventbus
    """
    
    def __init__(self, agent_id=None, name=None, description=None, similarity_threshold=0.7, **kwargs):
        """
        Initialize the agent with its parameters and setup the classifier
        
        Args:
            agent_id: Optional unique identifier for the agent
            name: Optional name for the agent (defaults to class name)
            description: Optional description of the agent
            similarity_threshold: Threshold for similarity-based interest determination
        """
        # Set default name if not provided
        name = name or "Websocket_agent Agent"
        description = description or "Creates a websocket interface for the eventbus"
        
        # Call parent constructor
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            similarity_threshold=similarity_threshold,
            **kwargs
        )
        
        # Set classifier threshold (since BaseAgent may not have use_classifier parameter)
        self.classifier_threshold = 0.5  # Lower threshold for testing
        
        logger.info(f"{name} agent initialized")
    
    def setup_interest_model(self):
        """
        Set up the agent's interest model, which determines what messages it processes
        This is called automatically during initialization
        """
        # Check for fine-tuned model directory
        # Look for two different types of fine-tuned models:
        # 1. An embedding model (for SentenceTransformer)
        # 2. An interest model (numpy saved file)
        
        model_dir = os.path.join(os.path.dirname(__file__), "fine_tuned_model")
        embedding_model_path = model_dir  # The embedding model would be in the directory itself
        interest_model_path = os.path.join(model_dir, "interest_model.npz")  # The interest vectors
        
        logger.info(f"Looking for fine-tuned models in: {model_dir}")
        
        # List the model directory for debugging
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            files = os.listdir(model_dir)
            logger.info(f"Fine-tuned model directory contains: {files}")
            
            try:
                # Import necessary components for fine-tuned model
                try:
                    # First try importing from semsubscription
                    from semsubscription.vector_db.embedding import EmbeddingEngine, InterestModel
                except ImportError:
                    # Fall back to local implementation for containerized environments
                    from .interest_model import CustomInterestModel as InterestModel
                    from .embedding_engine import EmbeddingEngine
                
                # Create the interest model first
                if os.path.exists(interest_model_path):
                    logger.info(f"Found pre-calculated interest model: {interest_model_path}")
                    # Create a default embedding engine and then load the saved interest vectors
                    embedding_engine = EmbeddingEngine()  # Using default model
                    self.interest_model = InterestModel(embedding_engine=embedding_engine, model_path=interest_model_path)
                    logger.info(f"Successfully loaded pre-calculated interest model")
                # If we have a custom embedding model, load it
                elif os.path.exists(os.path.join(embedding_model_path, "config.json")):
                    logger.info(f"Found custom sentence transformer model at: {embedding_model_path}")
                    embedding_engine = EmbeddingEngine(model_name=embedding_model_path)
                    logger.info(f"Successfully loaded custom embedding model")
                    self.interest_model = InterestModel(embedding_engine=embedding_engine)
                else:
                    logger.warning(f"No valid fine-tuned model found in {model_dir}")
                    # Fall back to standard setup
                    super().setup_interest_model()
                    return
                    
                # Set threshold for the model
                self.interest_model.threshold = self.similarity_threshold
                
                # Domain-specific keywords can be added here
                # self.interest_model.keywords.extend([
                #     "specific_keyword",
                #     "another_keyword"
                # ])
                
                return  # Exit early, we've set up the model successfully
            except Exception as e:
                logger.error(f"Error setting up fine-tuned model: {e}")
                logger.warning("Falling back to default interest model setup")
        
        # Fall back to standard setup if fine-tuned model doesn't exist or fails
        super().setup_interest_model()
        
        # Add domain-specific customizations to the default model
        # For example, to add keywords that should always be of interest:
        # self.interest_model.keywords.extend([
        #     "specific_keyword",
        #     "another_keyword"
        # ])
    
    def process_message(self, message) -> Optional[Dict[str, Any]]:
        """
        Process domain-specific queries
        
        Args:
            message: The message to process (dict in containerized version)
            
        Returns:
            Response data
        """
        try:
            # Handle both Message objects and dictionary messages (for container compatibility)
            if hasattr(message, 'content'):
                content = message.content
                message_id = getattr(message, 'id', 'unknown')
            else:
                content = message.get('content', '')
                message_id = message.get('id', 'unknown')
                
            query = content.lower()
            
            # Log the message being processed
            logger.info(f"Processing message {message_id} with content: '{content[:50]}...'")
            logger.info(f"Message successfully received via event bus")
            
            # Domain for {domain}
            # Add your domain-specific processing logic here
            
            # Test confirmation response to show the message bus is working
            if True:  # Always provide a response for testing
                return {
                    "agent": self.name,
                    "response": f"Message received by {self.name}! This confirms the event bus is working properly.",
                    "message_id": message_id,
                    "content_preview": content[:100] + ("..." if len(content) > 100 else "")
                }
            
            # Example pattern matching for various domain queries
            # These are for when you customize the agent for your specific domain
            if 'help' in query or 'hello' in query:
                return {
                    "agent": self.name,
                    "response": f"Hello! I'm {self.name}, an agent that {self.description.lower()}. How can I help you?"
                }
            elif '{domain}' in query:
                return {
                    "agent": self.name,
                    "response": f"I detected a {domain} related query: {content}"
                }
            #
            # Example:
            # if "weather" in query and ("forecast" in query or "today" in query):
            #     return {
            #         "agent": self.name,
            #         "query_type": "weather_forecast",
            #         "forecast": "Sunny with a high of 72°F"
            #     }}
            
            # Default response if no pattern matches
            return {
                "agent": self.name,
                "query_type": "general_response",
                "response": f"I received your query in the {domain} domain. This is a placeholder response."
            }
            
        except Exception as e:
            logger.error(f"Error in Websocket_agent Agent processing: {e}")
            return {
                "agent": self.name,
                "error": str(e),
                "query": content if 'content' in locals() else "unknown query"
            }


# Define the class with Agent suffix for backwards compatibility
# This prevents import errors in the container
class Websocket_agentAgent(Websocket_agent):
    """Legacy class name with Agent suffix"""
    pass

# Legacy compatibility for BaseAgent fallback imports
BaseAgent = Websocket_agent

# For standalone testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create the agent
    agent = Websocket_agentAgent()
    print(f"Agent created: {agent.name}")
    
    # Test classifier setup
    print("\nClassifier Status:")
    if hasattr(agent, 'classifier_model') and hasattr(agent, 'classification_head'):
        print(f"  Classifier Model: Loaded successfully")
        print(f"  Classification Head: Loaded successfully")
        print(f"  Classifier Threshold: {agent.classifier_threshold}")
    else:
        print("  Warning: Classifier not fully loaded!")
        if not hasattr(agent, 'classifier_model'):
            print("  - Missing classifier_model")
        if not hasattr(agent, 'classification_head'):
            print("  - Missing classification_head")
    
    # Test with sample messages
    test_messages = [
        "Your test query specific to this agent's domain",
        "A query that should probably not be handled by this agent",
        "Another domain-specific query to test routing"
    ]
    
    for i, test_message in enumerate(test_messages):
        print(f"\nTest {i+1}: '{test_message}'")
        
        # Test interest calculation
        from semsubscription.vector_db.database import Message
        message = Message(content=test_message)
        interest_score = agent.calculate_interest(message)
        
        print(f"Interest Score: {interest_score:.4f} (Threshold: {agent.similarity_threshold} for similarity, {agent.classifier_threshold} for classifier)")
        print(f"Agent would {'process' if interest_score >= max(agent.similarity_threshold, agent.classifier_threshold) else 'ignore'} this message")
        
        # If interested, test processing
        if interest_score >= max(agent.similarity_threshold, agent.classifier_threshold):
            result = agent.process_message(message)
            print("Processing Result:")
            print(json.dumps(result, indent=2))
            
    print("\nAgent testing complete.")

