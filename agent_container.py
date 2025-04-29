import os
import time
import json
import logging
import requests
import yaml

# Import the specific agent class based on config
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    agent_class_name = config.get('agent', {}).get('class_name', 'Test')
    package_name = config.get('agent', {}).get('package_name', 'test_agent')
except Exception as e:
    agent_class_name = 'Test' 
    package_name = 'test_agent'
    print(f"Warning: Could not load config.yaml: {e}")

# Dynamic import of the agent class
module_name = f"agent"
try:
    import importlib
    agent_module = importlib.import_module(module_name)
    # Try multiple class name patterns in order:
    # 1. ExactClassName as in config
    # 2. ClassNameAgent (with Agent suffix)
    # 3. BaseAgent as a fallback
    for class_attempt in [agent_class_name, f"{agent_class_name}Agent", "BaseAgent"]:
        try:
            AgentClass = getattr(agent_module, class_attempt)
            print(f"Successfully imported {class_attempt} from {module_name}")
            break
        except AttributeError:
            continue
    else:
        # If the loop completes without finding a class, try one more approach
        # Sometimes classes might be defined with unexpected capitalization
        for name in dir(agent_module):
            if name.lower() == agent_class_name.lower() or \
               name.lower() == f"{agent_class_name.lower()}agent":
                AgentClass = getattr(agent_module, name)
                print(f"Successfully imported {name} from {module_name} using case-insensitive match")
                break
        else:
            # If all attempts fail, create a minimal agent class
            class MinimalAgent:
                def __init__(self, **kwargs):
                    self.classifier_threshold = 0.5
                def calculate_interest(self, message):
                    return 0.0
                def process_message(self, message):
                    return {"error": "Agent class not properly defined"}
            AgentClass = MinimalAgent
            print(f"Warning: Using minimal agent implementation - no suitable class found in {module_name}")
except Exception as e:
    # Create a minimal agent if everything else fails
    class MinimalAgent:
        def __init__(self, **kwargs):
            self.classifier_threshold = 0.5
        def calculate_interest(self, message):
            return 0.0
        def process_message(self, message):
            return {"error": "Agent class not properly defined"}
    AgentClass = MinimalAgent
    print(f"Warning: Could not import agent module: {e}")

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables
AGENT_ID = os.environ.get("AGENT_ID", "unknown")
AGENT_NAME = os.environ.get("AGENT_NAME", "Unknown Agent")

# Get the main core API URL and any alternates
CORE_API_URL = os.environ.get("CORE_API_URL", "http://localhost:8888")
ALTERNATE_CORE_API_URLS = os.environ.get("ALTERNATE_CORE_API_URLS", "").split(",")

# Get polling interval with default of 5 seconds
POLLING_INTERVAL = float(os.environ.get("POLLING_INTERVAL", "5"))

# Function to try API request with multiple URLs
def try_api_request(endpoint, method="get", **kwargs):
    """
    Try API request with all available URLs, falling back to alternates if the main one fails
    """
    # Start with the main URL
    urls_to_try = [CORE_API_URL] + ALTERNATE_CORE_API_URLS
    
    # Remove any empty URLs
    urls_to_try = [url for url in urls_to_try if url]
    
    last_error = None
    for url in urls_to_try:
        try:
            full_url = f"{url}{endpoint}"
            logger.debug(f"Trying API request to {full_url}")
            
            if method.lower() == "get":
                response = requests.get(full_url, **kwargs)
            elif method.lower() == "post":
                response = requests.post(full_url, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            return response
        except Exception as e:
            logger.debug(f"API request to {url} failed: {e}")
            last_error = e
    
    # If we get here, all URLs failed
    logger.error(f"All API connection attempts failed. Last error: {last_error}")
    raise last_error

def register_with_core_system():
    """Register this agent with the core system"""
    try:
        response = try_api_request(
            "/api/agents/register",
            method="post",
            json={
                "agent_id": AGENT_ID,
                "name": AGENT_NAME,
                "container_id": os.environ.get("HOSTNAME", "unknown"),
                "status": "running"
            }
        )
        if response.status_code == 200:
            logger.info(f"Agent registered successfully with core system")
            return True
        else:
            logger.error(f"Failed to register with core system: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error registering with core system: {e}")
        return False

def process_messages():
    """Poll for messages to process"""
    try:
        # Check for pending messages using the API for containerized agents
        response = try_api_request(
            "/api/messages/pending",
            method="get",
            params={
                "agent_id": AGENT_ID
            }
        )
        
        if response.status_code == 200:
            messages = response.json()
            if messages:
                logger.info(f"Received {len(messages)} messages to process")
                for message in messages:
                    process_message(message)
            # We only need to subscribe once on startup, not after every processing cycle
            # This prevents message duplication from repeated subscriptions
        else:
            logger.error(f"Failed to get pending messages: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error checking for messages: {e}")

def subscribe_to_events():
    """Subscribe agent to the event-driven system"""
    try:
        # Register for new message events with direct message delivery
        response = try_api_request(
            f"/api/agents/{AGENT_ID}/subscribe",
            method="post",
            json={
                "subscription_type": "direct",
                "event_types": ["message.created"]
            }
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully subscribed to events with direct message delivery")
            return True
        elif response.status_code == 404:
            # This is expected during the transition to event-driven architecture
            # Fall back to polling mode silently without warning logs
            logger.debug(f"Event subscription not implemented yet, using polling mode")
            return False  
        else:
            logger.warning(f"Failed to subscribe to events: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        # Simply log at debug level since this endpoint may not exist yet
        logger.debug(f"Error subscribing to events: {e}")
        return False
        
# Function to handle direct event delivery of messages
def handle_event_message(message_data):
    """Process a message received directly from an event"""
    try:
        logger.info(f"Received message event directly: {message_data.get('id', 'unknown-id')}")
        # Process the message directly
        process_message(message_data)
    except Exception as e:
        logger.error(f"Error processing direct event message: {e}")

def process_message(message):
    """Process a single message"""
    try:
        # Initialize the agent (use_classifier is already handled in the agent constructor)
        agent = AgentClass()
        
        # Calculate interest score
        interest_score = agent.calculate_interest(message)
        logger.info(f"Interest score for message {message['id']}: {interest_score}")
        
        # First phase: Register interest with the event-driven system
        # This corresponds to the AgentInterestService in the core system
        interest_response = try_api_request(
            f"/api/messages/{message['id']}/interest",
            method="post",
            json={
                "agent_id": AGENT_ID,
                "name": AGENT_NAME,
                "score": interest_score
            }
        )
        
        # Second phase: If interest score exceeds threshold, process the message
        # This corresponds to the MessageProcessingService in the core system
        if interest_score >= agent.classifier_threshold:
            # Process the message and get the result
            result = agent.process_message(message)
            
            if result:
                # Include agent information in the result
                if isinstance(result, dict) and 'agent_id' not in result:
                    result['agent_id'] = AGENT_ID
                
                # Submit processing result to core system
                process_response = try_api_request(
                    f"/api/messages/{message['id']}/process",
                    method="post",
                    json={
                        "agent_id": AGENT_ID,
                        "result": result
                    }
                )
                
                if process_response.status_code == 200:
                    logger.info(f"Successfully processed message {message['id']}")
                else:
                    logger.warning(f"Error submitting processing result: {process_response.status_code} - {process_response.text}")
            else:
                logger.info(f"Agent returned no result for message {message['id']}")
        else:
            logger.info(f"Interest score {interest_score} below threshold {agent.classifier_threshold}, skipping processing")
            
    except Exception as e:
        logger.error(f"Error processing message {message.get('id', 'unknown')}: {e}")

def main():
    """Main entry point for the agent container"""
    logger.info(f"Starting agent container for {AGENT_NAME} (ID: {AGENT_ID})")
    
    # Register with core system
    if not register_with_core_system():
        logger.warning("Continuing without registration...")
    
    # Subscribe to events once at startup
    # This is critical for the event-driven architecture to work properly
    event_subscription_success = subscribe_to_events()
    if event_subscription_success:
        logger.info("Successfully subscribed to message events - using event-driven delivery")
    else:
        logger.info("Event subscription unavailable - falling back to polling mode")
        
    # Initialize the agent for WebSocket server access
    agent = AgentClass()
    
    # Start the WebSocket server if the agent has the method
    if hasattr(agent, 'start_websocket_server'):
        logger.info("Starting WebSocket server...")
        agent.start_websocket_server()
        logger.info("WebSocket server started successfully")
    else:
        logger.warning("Agent does not have start_websocket_server method, WebSocket functionality will not be available")
    
    # Main processing loop
    try:
        while True:
            process_messages()
            time.sleep(POLLING_INTERVAL)  # Use configurable polling interval
    except KeyboardInterrupt:
        logger.info("Shutting down agent container")

if __name__ == "__main__":
    main()
