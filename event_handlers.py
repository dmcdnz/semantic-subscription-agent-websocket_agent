import asyncio
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Reference to the active WebSocket connections
active_connections: List = []

async def broadcast_message(message: Dict[str, Any]):
    """
    Broadcast a message to all connected WebSocket clients.
    
    Args:
        message: The message to broadcast (will be converted to JSON)
    """
    if not active_connections:
        logger.info("No active connections to broadcast to")
        return
    
    # Convert message to JSON string
    message_str = json.dumps(message)
    
    # Send to all active connections
    send_tasks = []
    for connection in active_connections:
        try:
            send_tasks.append(asyncio.create_task(connection.send_text(message_str)))
        except Exception as e:
            logger.error(f"Error broadcasting to connection: {e}")
    
    # Wait for all send tasks to complete if there are any
    if send_tasks:
        await asyncio.gather(*send_tasks, return_exceptions=True)
    
    logger.info(f"Broadcast message to {len(active_connections)} clients")

async def handle_new_message(message_data: Dict[str, Any]):
    """
    Handle a new message event from the event bus and relay it to WebSocket clients.
    
    Args:
        message_data: Data for the new message
    """
    logger.info(f"Handling new message event: {message_data.get('id', 'unknown')}")
    
    # Create the WebSocket message
    websocket_message = {
        "type": "new_message",
        "timestamp": datetime.now().isoformat(),
        "data": message_data
    }
    
    # Broadcast to all connected clients
    await broadcast_message(websocket_message)

async def handle_message_updated(message_data: Dict[str, Any]):
    """
    Handle a message updated event from the event bus.
    
    Args:
        message_data: Data for the updated message
    """
    logger.info(f"Handling message updated event: {message_data.get('id', 'unknown')}")
    
    # Create the WebSocket message
    websocket_message = {
        "type": "message_updated",
        "timestamp": datetime.now().isoformat(),
        "data": message_data
    }
    
    # Broadcast to all connected clients
    await broadcast_message(websocket_message)

async def handle_agent_interest(interest_data: Dict[str, Any]):
    """
    Handle an agent interest event from the event bus.
    
    Args:
        interest_data: Data for the agent interest
    """
    logger.info(f"Handling agent interest event: {interest_data.get('agent_id', 'unknown')} -> {interest_data.get('message_id', 'unknown')}")
    
    # Create the WebSocket message
    websocket_message = {
        "type": "agent_interest",
        "timestamp": datetime.now().isoformat(),
        "data": interest_data
    }
    
    # Broadcast to all connected clients
    await broadcast_message(websocket_message)

async def handle_agent_response(response_data: Dict[str, Any]):
    """
    Handle an agent response event from the event bus.
    
    Args:
        response_data: Data for the agent response
    """
    logger.info(f"Handling agent response event: {response_data.get('agent', 'unknown')} -> {response_data.get('message_id', 'unknown')}")
    
    # Create the WebSocket message
    websocket_message = {
        "type": "agent_response",
        "timestamp": datetime.now().isoformat(),
        "data": response_data
    }
    
    # Broadcast to all connected clients
    await broadcast_message(websocket_message)

async def handle_agent_status(status_data: Dict[str, Any]):
    """
    Handle an agent status change event from the event bus.
    
    Args:
        status_data: Data for the agent status
    """
    logger.info(f"Handling agent status event: {status_data.get('agent_id', 'unknown')} -> {status_data.get('status', 'unknown')}")
    
    # Create the WebSocket message
    websocket_message = {
        "type": "agent_status",
        "timestamp": datetime.now().isoformat(),
        "data": status_data
    }
    
    # Broadcast to all connected clients
    await broadcast_message(websocket_message)
