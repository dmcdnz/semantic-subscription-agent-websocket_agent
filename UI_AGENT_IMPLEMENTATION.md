# UI Agent Implementation Guide

## Overview

This document outlines the approach for creating a specialized UI agent that bridges the Semantic Subscription System's event-driven message bus with a React frontend via WebSockets. This agent will enable real-time updates in the UI by relaying events from the distributed event bus to connected browser clients.

## Architecture

![UI Agent Architecture](https://mermaid.ink/img/pako:eNqVk11P2zAUhv_KkScktdA2YxANrQPBUKmQTRqTgD0gccidPW2t_MY3QYP-99mJnbQbiNF9StL3eXJ8bJ83TMkQWc6kNVZ4UbOw0kypBjt4Eb7Cjqov7PoOnqWCFh_gcX2YgZdRJaNMK8zAK6ncIEeUsiJmWwrWfGgqQc5g29_mxClgPTdkHiQXZY0OZ5AfOAKDVFx7Lsu5kJUK1kZ5rUMOOkzK0kI9JQ2ZdW6p7HAufZf1h-MJQfvCFuRskjDHhzk5HMXxpUwJxfOzESXUExZKp7RJxWEYnzIUjnI2jlkqJLn0vBfWRgWrv0sjrNnXqFwZNagxBVuhklnQQVDcN3o3GOJF_FJ5Z3Gco3HjIKMHZcHJ_uNwm_AYnCH3nLaGrqfOAw-6CkHOhlwkjHUjOhYIY8N1MWOV6vTjuQzcqNZfVQNz5V38JNjM8XFTAcTb_Kh-jIHwHEYnwXUYFrO2gJQNScXyRKzE4yRWkpjEzGGsdB_lrfP3C5_6FnFRdlVLWl4e-YP4u-FVmWLLF8P6rl26GxoRDaX3-bvE8sFQPrAFa52KX28hPL6XBRvnj77JaODdDZPMrZJcxRZsb8msFkuWC9pMTcv-sJPo5OT8etqk_jQ9Hv_--TZJMskcb5LuOOlJq3YlOFRYMrC9HVk0zLc0-DzvUOuGSz8LZuVWK9t9kl_s7MX-Zpb5n8HM9JJZKxr222RpwzbUgRV_fpCpkT9I5OydyVP_DyW2YwM?type=png)

### Components

1. **Core System** - The existing Semantic Subscription System
2. **Event Bus** - Redis-based or in-memory event bus that agents use to communicate
3. **UI Agent** - New specialized agent that:
   - Subscribes to events on the message bus
   - Provides a WebSocket server
   - Relays events to connected WebSocket clients
4. **React Frontend** - Existing React application that will connect via WebSockets

## Implementation Steps

### 1. Create UI Agent Project Structure

```
ui-agent/
├── Dockerfile
├── requirements.txt
├── ui_agent.py          # Main application with WebSocket server
├── event_handlers.py    # Handler functions for different event types
└── config.yaml          # Configuration for agent name/type
```

### 2. Core Agent Functionality

The UI agent will need to:

- Register with the core system like other agents
- Subscribe to the event bus for various event types
- Manage WebSocket connections from browser clients
- Convert and forward events from the bus to WebSocket clients
- Provide basic authentication for WebSocket connections

### 3. Event Types to Relay

The agent should subscribe to and forward these event types:

- `new_message` - When a new message is created
- `message_updated` - When a message is updated
- `agent_interest` - When an agent registers interest in a message
- `agent_response` - When an agent processes a message
- `agent_status` - When an agent's status changes

### 4. WebSocket Protocol

All messages sent over the WebSocket will follow this JSON format:

```json
{
  "type": "event_type",  // e.g., "new_message", "agent_interest"
  "timestamp": "2025-04-28T11:24:41+12:00",
  "data": { ... }  // Event-specific payload
}
```

### 5. WebSocket Authentication

WebSocket connections should be authenticated using the same session mechanism as the REST API:

1. Client authenticates via GitHub OAuth through the REST API
2. Client receives a session cookie
3. Client includes this cookie when establishing the WebSocket connection
4. UI Agent validates the session cookie before accepting the connection

### 6. React Integration

The React frontend will need a WebSocket context provider that:

1. Establishes and maintains a WebSocket connection
2. Handles reconnection on disconnection
3. Processes incoming messages and updates application state
4. Provides a clean API for components to consume real-time data

## Implementation Details

### UI Agent Code Structure

```python
# ui_agent.py

import os
import json
import logging
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Cookie, Depends, Query
from semsubscription.core.event_bus_factory import get_event_bus
from event_handlers import handle_new_message, handle_message_updated, handle_agent_interest, handle_agent_response

# Initialize FastAPI app for WebSocket server
app = FastAPI()

# Get the event bus
event_bus = get_event_bus()

# Track active connections
active_connections = []

# Subscribe to events
def subscribe_to_events():
    event_bus.subscribe("new_message", handle_new_message)
    event_bus.subscribe("message_updated", handle_message_updated)
    event_bus.subscribe("agent_interest", handle_agent_interest)
    event_bus.subscribe("agent_response", handle_agent_response)
    # Add more subscriptions as needed

# Broadcast to all connected clients
async def broadcast_message(message):
    serialized = json.dumps(message)
    for connection in active_connections:
        await connection.send_text(serialized)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session: str = Cookie(None)):
    # Validate session
    if not session or not validate_session(session):
        await websocket.close(code=1008, reason="Not authenticated")
        return
        
    await websocket.accept()
    active_connections.append(websocket)
    
    # Send initial data if needed
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

# Main function
def main():
    # Register with core system
    register_with_core_system()
    
    # Subscribe to events
    subscribe_to_events()
    
    # Start the WebSocket server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
```

```python
# event_handlers.py

import asyncio
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Reference to the broadcast function
# Will be set by ui_agent.py
broadcast_message = None

def set_broadcast_function(func):
    global broadcast_message
    broadcast_message = func

def handle_new_message(message_data: Dict[str, Any]):
    """Handle new message events"""
    logger.info(f"Received new message event: {message_data.get('id', 'unknown')}")
    
    # Create the event to send over WebSocket
    event = {
        "type": "new_message",
        "timestamp": message_data.get("timestamp", ""),
        "data": message_data
    }
    
    # Broadcast to all clients
    if broadcast_message:
        asyncio.create_task(broadcast_message(event))

# Implement similar handlers for other event types
def handle_message_updated(message_data: Dict[str, Any]):
    # Similar implementation
    pass

def handle_agent_interest(interest_data: Dict[str, Any]):
    # Similar implementation
    pass

def handle_agent_response(response_data: Dict[str, Any]):
    # Similar implementation
    pass
```

### React WebSocket Integration

```javascript
// WebSocketContext.js
import React, { createContext, useContext, useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

const WebSocketContext = createContext();

export function WebSocketProvider({ children }) {
  const [messages, setMessages] = useState([]);
  const [agentInterests, setAgentInterests] = useState({});
  const [agentResponses, setAgentResponses] = useState({});
  const [isConnected, setIsConnected] = useState(false);
  const { isAuthenticated } = useAuth();
  
  useEffect(() => {
    let ws = null;
    
    if (isAuthenticated) {
      // Include credentials for authentication
      ws = new WebSocket('ws://localhost:8000/ws');
      
      ws.onopen = () => {
        setIsConnected(true);
        console.log('WebSocket connected');
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'new_message':
            setMessages(prev => [data.data, ...prev]);
            break;
          case 'message_updated':
            setMessages(prev => 
              prev.map(msg => msg.id === data.data.id ? data.data : msg)
            );
            break;
          case 'agent_interest':
            setAgentInterests(prev => ({
              ...prev,
              [data.data.message_id]: {
                ...prev[data.data.message_id],
                [data.data.agent_id]: data.data.score
              }
            }));
            break;
          case 'agent_response':
            setAgentResponses(prev => ({
              ...prev,
              [data.data.message_id]: [
                ...(prev[data.data.message_id] || []),
                data.data
              ]
            }));
            break;
          default:
            console.log('Unknown event type:', data.type);
        }
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        console.log('WebSocket disconnected');
        
        // Attempt reconnection after delay
        setTimeout(() => {
          if (isAuthenticated) {
            console.log('Attempting to reconnect...');
          }
        }, 3000);
      };
    }
    
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [isAuthenticated]);
  
  return (
    <WebSocketContext.Provider value={{
      messages,
      agentInterests,
      agentResponses,
      isConnected
    }}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  return useContext(WebSocketContext);
}
```

## Dockerization

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for WebSocket connections
EXPOSE 8000

# Run the UI agent
CMD ["python", "ui_agent.py"]
```

### requirements.txt

```
fastapi>=0.95.0
uvicorn>=0.21.0
websockets>=10.4
asyncio>=3.4.3
redis>=4.5.1
python-dotenv>=1.0.0
```

### Docker Compose Integration

Add this to your existing `docker-compose.yml`:

```yaml
services:
  # Existing services...
  
  ui-agent:
    build:
      context: ./ui-agent
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - AGENT_ID=ui-agent
      - AGENT_NAME=UI Agent
      - CORE_API_URL=http://core-system:8888
      - USE_REDIS_EVENT_BUS=true
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - core-system
      - redis
```

## Deployment and Testing

1. Create the UI agent project structure
2. Implement the code based on the templates above
3. Build and run the agent container
4. Integrate the WebSocket context in the React frontend
5. Test end-to-end communication

## Benefits of This Approach

1. **Architectural Consistency** - Follows the same pattern as other agents
2. **Separation of Concerns** - UI agent has one focused responsibility
3. **Real-time Updates** - Eliminates polling for efficient updates
4. **Event-Driven** - Aligns with the system's event-driven architecture
5. **Scalability** - Can be horizontally scaled if needed

## Future Enhancements

1. **Message Filtering** - Allow clients to subscribe to specific event types
2. **Multi-user Isolation** - Ensure events are properly isolated by user
3. **Reconnection Handling** - Implement sophisticated reconnection logic
4. **Event History** - Allow clients to request missed events from a specific timestamp
5. **Compressed Payloads** - Implement compression for large event payloads
