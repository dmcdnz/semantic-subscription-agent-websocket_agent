import os
import json
import logging
import asyncio
import uvicorn
import httpx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Cookie, Query, Request
from fastapi.middleware.cors import CORSMiddleware

# Import event handlers
from event_handlers import (
    active_connections,
    handle_new_message,
    handle_message_updated,
    handle_agent_interest,
    handle_agent_response,
    handle_agent_status
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="WebSocket Agent Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_token(token: Optional[str] = Query(None)):
    """
    Extract token from query parameters.
    
    This is a simple auth mechanism that can be enhanced with proper JWT validation.
    """
    if token is None:
        return None
    return token

async def verify_github_token(token: Optional[str] = None):
    """
    Verify GitHub authentication token with the core system.
    
    Args:
        token: GitHub authentication token
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    if not token:
        return False
        
    try:
        # Try to validate the token with the core system
        # URLs to try for validation
        urls = [
            "http://localhost:8888",
            "http://host.docker.internal:8888",
            "http://127.0.0.1:8888"
        ]
        
        for url in urls:
            try:
                response = await httpx.AsyncClient().get(
                    f"{url}/api/auth/verify",
                    headers={"Authorization": f"Bearer {token}"}
                )
                if response.status_code == 200:
                    return True
            except:
                continue
                
        return False
    except Exception as e:
        logger.error(f"Error verifying token: {e}")
        return False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = Depends(get_token)):
    """
    Handle WebSocket connections.
    
    Args:
        websocket: The WebSocket connection
        token: Optional auth token
    """
    # Validate GitHub token with the core system
    is_authenticated = await verify_github_token(token)
    
    if not is_authenticated and token is not None:
        await websocket.close(code=4001, reason="Invalid authentication token")
        logger.warning(f"Rejected WebSocket connection due to invalid token")
        return
    
    # Accept the connection
    await websocket.accept()
    
    # Add to active connections
    active_connections.append(websocket)
    logger.info(f"WebSocket client connected. Total connections: {len(active_connections)}")
    
    # Send welcome message
    await websocket.send_text(json.dumps({
        "type": "connection_established",
        "message": "Connected to Semantic Subscription System WebSocket"
    }))
    
    try:
        # Handle incoming messages
        while True:
            # Wait for message from client
            data = await websocket.receive_text()
            
            # Parse the message
            try:
                message = json.loads(data)
                logger.info(f"Received message from client: {message.get('type', 'unknown')}")
                
                # Echo the message back as acknowledgment
                await websocket.send_text(json.dumps({
                    "type": "message_received",
                    "received": message
                }))
                
                # In a full implementation, you could process client commands here
                # For example, subscribing to specific event types
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
                
    except WebSocketDisconnect:
        # Remove from active connections
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Remaining connections: {len(active_connections)}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Try to remove from active connections
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected due to error. Remaining connections: {len(active_connections)}")

# Mock event bus for testing
class MockEventBus:
    """
    Mock event bus for local testing.
    
    In the real implementation, this would be replaced with the actual event bus.
    """
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, event_type: str, callback):
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Callback function
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        logger.info(f"Subscribed to event type: {event_type}")
    
    async def publish(self, event_type: str, data: Dict[str, Any]):
        """
        Publish an event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        logger.info(f"Publishing event: {event_type}")
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                await callback(data)

# Create mock event bus
mock_event_bus = MockEventBus()

# Subscribe to events
def subscribe_to_events():
    # Subscribe to event types that match the existing SSE implementation
    mock_event_bus.subscribe("message.created", handle_new_message)
    mock_event_bus.subscribe("message.updated", handle_message_updated)
    mock_event_bus.subscribe("message.interest.registered", handle_agent_interest)
    mock_event_bus.subscribe("message.processed", handle_agent_response)
    mock_event_bus.subscribe("agent.status", handle_agent_status)
    
    # For backward compatibility, also subscribe to any alternate event names
    mock_event_bus.subscribe("new_message", handle_new_message)
    mock_event_bus.subscribe("agent_interest", handle_agent_interest)
    mock_event_bus.subscribe("agent_response", handle_agent_response)
    mock_event_bus.subscribe("agent_status", handle_agent_status)

subscribe_to_events()

@app.get("/") 
def read_root():
    """
    Root endpoint for health check.
    """
    return {
        "status": "running",
        "service": "WebSocket Agent",
        "connections": len(active_connections)
    }

@app.post("/test-event/{event_type}")
async def test_event(event_type: str, request: Request):
    """
    Test endpoint to publish events (for development/testing).
    
    Args:
        event_type: Type of event to publish
        request: The request containing event data
    """
    data = await request.json()
    
    # Publish to mock event bus
    await mock_event_bus.publish(event_type, data)
    
    return {
        "status": "published",
        "event_type": event_type
    }

# Run the FastAPI app with Uvicorn when this script is executed
def start_server():
    """
    Start the WebSocket server.
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting WebSocket server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
