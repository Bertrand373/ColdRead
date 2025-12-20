"""
Coachd WebSocket Handler
Production-ready connection management with proper cleanup
"""

import json
import asyncio
import sys
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import time

from .realtime import RealtimeTranscriber


class ConnectionManager:
    """Manages WebSocket connections with proper lifecycle handling"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.transcribers: Dict[str, RealtimeTranscriber] = {}
        self._cleanup_lock = asyncio.Lock()
        print("[WS] ConnectionManager initialized", flush=True)
        
    async def connect(self, websocket: WebSocket, client_id: str) -> RealtimeTranscriber:
        """Accept and store a WebSocket connection"""
        print(f"[WS] connect() called for {client_id}", flush=True)
        
        await websocket.accept()
        print(f"[WS] WebSocket accepted for {client_id}", flush=True)
        
        self.active_connections[client_id] = websocket
        print(f"[WS] Connection stored for {client_id}", flush=True)
        
        # Create transcriber for this connection
        async def on_transcript(data):
            await self.send_json(client_id, data)
            
        async def on_guidance(data):
            await self.send_json(client_id, data)
        
        print(f"[WS] Creating RealtimeTranscriber for {client_id}", flush=True)
        transcriber = RealtimeTranscriber(on_transcript, on_guidance)
        self.transcribers[client_id] = transcriber
        
        print(f"[WS] SUCCESS: WebSocket fully connected for {client_id}", flush=True)
        return transcriber
        
    async def disconnect(self, client_id: str):
        """Clean up connection with proper resource release"""
        async with self._cleanup_lock:
            print(f"[WS] Disconnecting client: {client_id}", flush=True)
            
            # Stop transcriber first
            if client_id in self.transcribers:
                try:
                    await asyncio.wait_for(
                        self.transcribers[client_id].stop(),
                        timeout=5.0  # Don't hang on cleanup
                    )
                except asyncio.TimeoutError:
                    print(f"[WS] WARNING: Transcriber stop timed out for {client_id}", flush=True)
                except Exception as e:
                    print(f"[WS] Error stopping transcriber for {client_id}: {e}", flush=True)
                finally:
                    del self.transcribers[client_id]
            
            # Close WebSocket connection
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].close()
                except Exception:
                    pass  # Already closed
                finally:
                    del self.active_connections[client_id]
                
            print(f"[WS] Client disconnected: {client_id}", flush=True)
            
    async def send_json(self, client_id: str, data: dict):
        """Send JSON data to a specific client with error handling"""
        if client_id not in self.active_connections:
            return
            
        try:
            await self.active_connections[client_id].send_json(data)
        except Exception as e:
            print(f"[WS] Error sending to {client_id}: {e}", flush=True)
            # Don't disconnect here - let the main loop handle it


# Singleton manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    Handle WebSocket connection for real-time transcription.
    Production-ready with proper error handling and cleanup.
    """
    print(f"[WS] websocket_endpoint() called for {client_id}", flush=True)
    
    transcriber = None
    connection_start = time.time()
    
    try:
        print(f"[WS] Calling manager.connect() for {client_id}", flush=True)
        transcriber = await manager.connect(websocket, client_id)
        print(f"[WS] manager.connect() returned for {client_id}", flush=True)
        
        # Start Deepgram connection with timeout
        print(f"[WS] Starting transcriber for {client_id}", flush=True)
        try:
            started = await asyncio.wait_for(
                transcriber.start(),
                timeout=10.0
            )
            print(f"[WS] transcriber.start() returned: {started}", flush=True)
        except asyncio.TimeoutError:
            print(f"[WS] ERROR: Deepgram start timed out for {client_id}", flush=True)
            await websocket.send_json({
                "type": "error",
                "message": "Transcription service timeout - please try again"
            })
            return
        
        if started:
            await websocket.send_json({"type": "ready"})
            print(f"[WS] Transcription ready for {client_id}", flush=True)
        else:
            await websocket.send_json({
                "type": "error",
                "message": "Could not start transcription service"
            })
            print(f"[WS] ERROR: Failed to start transcription for {client_id}", flush=True)
            return
        
        # Main message loop
        print(f"[WS] Entering message loop for {client_id}", flush=True)
        while True:
            try:
                # Check if transcriber died
                if transcriber and not transcriber.is_running:
                    print(f"[WS] Transcriber stopped, exiting loop for {client_id}", flush=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": "Transcription service disconnected"
                    })
                    break
                
                # Receive with timeout to detect stale connections
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=30.0  # Shorter timeout to detect dead transcriber faster
                )
                
                if "bytes" in message:
                    # Audio data - forward to Deepgram
                    if transcriber and transcriber.is_running:
                        await transcriber.send_audio(message["bytes"])
                    
                elif "text" in message:
                    # JSON command
                    try:
                        data = json.loads(message["text"])
                        
                        if data.get("type") == "context":
                            # Update call context
                            context_data = data.get("data", {})
                            if context_data:
                                transcriber.update_context({
                                    "call_type": context_data.get("call_type", "phone"),
                                    "current_product": context_data.get("product", "whole_life"),
                                    "client_age": context_data.get("age"),
                                    "client_occupation": context_data.get("occupation"),
                                    "client_family": context_data.get("family"),
                                    "client_budget": context_data.get("budget"),
                                    "agency": context_data.get("agency")
                                })
                            print(f"[WS] Context updated for {client_id}", flush=True)
                            
                        elif data.get("type") == "down_close":
                            # Agent clicked down-close button
                            transcriber.apply_down_close()
                            print(f"[WS] Down-close applied for {client_id}", flush=True)
                            
                        elif data.get("type") == "stop":
                            print(f"[WS] Stop requested by {client_id}", flush=True)
                            break
                            
                        elif data.get("type") == "ping":
                            # Keepalive response
                            await websocket.send_json({"type": "pong"})
                            
                    except json.JSONDecodeError as e:
                        print(f"[WS] Invalid JSON from {client_id}: {e}", flush=True)
                        
                elif message.get("type") == "websocket.disconnect":
                    print(f"[WS] Client {client_id} sent disconnect", flush=True)
                    break
                    
            except asyncio.TimeoutError:
                # Check if transcriber is still alive
                if transcriber and not transcriber.is_running:
                    print(f"[WS] Transcriber dead during timeout for {client_id}", flush=True)
                    break
                # Otherwise just continue - waiting for audio/commands
                continue
                
            except WebSocketDisconnect:
                print(f"[WS] WebSocket disconnected: {client_id}", flush=True)
                break
                
            except Exception as e:
                print(f"[WS] Error in message loop for {client_id}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                break
                
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected during setup: {client_id}", flush=True)
    except Exception as e:
        print(f"[WS] WebSocket error for {client_id}: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup
        duration = time.time() - connection_start
        print(f"[WS] Connection duration for {client_id}: {duration:.1f}s", flush=True)
        await manager.disconnect(client_id)