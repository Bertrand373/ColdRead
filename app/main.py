"""
Coachd.ai - Real-time AI Sales Assistant
Main FastAPI Application with Dual-Channel Click-to-Call
"""

import os
import json
import shutil
import asyncio
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse, Response, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from .config import settings
from .vector_db import get_vector_db
from .document_processor import DocumentProcessor
from .rag_engine import RAGEngine, CallContext
from .websocket_handler import websocket_endpoint

# Telnyx dual-channel
from .telnyx_bridge import is_telnyx_configured
from .telnyx_routes import router as telnyx_router
from .call_session import session_manager, CallStatus
from .telnyx_stream_handler import (
    get_or_create_client_handler,
    get_or_create_agent_handler,
    remove_handler,
    get_deepgram_client
)

# Database and usage tracking
from .database import (
    init_db, is_db_configured, get_usage_summary, get_usage_by_agency, 
    get_daily_usage, get_recent_logs, get_platform_config, set_platform_config
)
from .usage_tracker import (
    log_claude_usage,
    log_deepgram_usage,
    log_telnyx_usage,
    fetch_all_external_usage,
    get_platform_summary
)

# Call outcome tracking and status
from .call_outcome_routes import router as outcome_router
from .status_check import router as status_router

# Phone verification for caller ID
from .phone_verification import router as verification_router

# Power Dialer for appointment booking
from .dialer_routes import router as dialer_router

# Agent authentication
from .auth import router as auth_router, init_auth_tables


# ============ PYDANTIC MODELS ============

class AgencyValidation(BaseModel):
    code: str

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = {}
    agency: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = []

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    agency: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = {}


# ============ AGENCY DATA ============

AGENCIES = {
    "ADERHOLT": {"name": "Aderholt Agency", "active": True},
    "BROOKS": {"name": "Brooks Agency", "active": True},
}


# ============ LIFESPAN ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*50)
    print("🏈 Coachd.ai Starting Up...")
    print("="*50)
    
    db = get_vector_db()
    print(f"✓ Vector database initialized ({db.get_document_count()} chunks)")
    
    if is_db_configured():
        init_db()
        print("✓ PostgreSQL database initialized")
        # Initialize auth tables
        init_auth_tables()
    else:
        print("⚠ DATABASE_URL not set - usage tracking disabled")
    
    if settings.anthropic_api_key:
        print("✓ Anthropic API key configured")
    else:
        print("⚠ ANTHROPIC_API_KEY not set")
    
    if settings.deepgram_api_key:
        print("✓ Deepgram API key configured")
        # Pre-initialize Deepgram client at startup to avoid first-connection issues
        dg_client = get_deepgram_client()
        if dg_client:
            print("✓ Deepgram client pre-initialized")
    else:
        print("⚠ DEEPGRAM_API_KEY not set")
    
    if is_telnyx_configured():
        print("✓ Telnyx configured - DUAL-CHANNEL CLICK-TO-CALL ACTIVE")
        print(f"  Phone: {settings.telnyx_phone_number}")
        print(f"  Webhooks: {settings.base_url}/api/telnyx/")
    else:
        print("⚠ Telnyx not configured")
    
    print("\n" + "="*50)
    print("✅ Coachd.ai Ready")
    print(f"   URL: http://{settings.host}:{settings.port}")
    print("="*50 + "\n")
    
    yield
    
    print("\n👋 Coachd.ai Shutting Down...")


# ============ APP SETUP ============

app = FastAPI(
    title="Coachd.ai",
    description="Real-time AI sales assistant for life insurance agents",
    version="3.2.0",  # Added agent authentication
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).parent.parent / "static"
templates_path = Path(__file__).parent.parent / "templates"

if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

templates = Jinja2Templates(directory=str(templates_path))

app.include_router(outcome_router)
app.include_router(status_router)
app.include_router(telnyx_router)
app.include_router(verification_router)
app.include_router(dialer_router)
app.include_router(auth_router)


# Service worker must be served from root for proper scope
@app.get("/sw.js")
async def service_worker():
    return FileResponse(
        static_path / "sw.js",
        media_type="application/javascript"
    )


# ============ WEBSOCKET ROUTES ============

@app.websocket("/ws/{client_id}")
async def websocket_route(websocket: WebSocket, client_id: str):
    """WebSocket for browser mic mode"""
    await websocket_endpoint(websocket, client_id)


@app.websocket("/ws/telnyx/session/{session_id}")
async def telnyx_session_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for frontend to receive real-time updates"""
    await websocket.accept()
    
    session = await session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    await session_manager.register_websocket(session_id, websocket)
    
    try:
        await websocket.send_json({
            "type": "session_state",
            "data": session.to_dict()
        })
        
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception:
                break
                
    finally:
        await session_manager.unregister_websocket(session_id, websocket)


@app.websocket("/ws/telnyx/stream/client/{session_id}")
async def telnyx_client_stream(websocket: WebSocket, session_id: str):
    """
    CLIENT audio stream - 100% client audio.
    Sent to Deepgram for transcription and guidance triggers.
    """
    await websocket.accept()
    print(f"[ClientStream] WebSocket accepted: {session_id}", flush=True)
    
    handler = None
    message_count = 0
    
    try:
        handler = await get_or_create_client_handler(session_id)
        await session_manager.update_session(session_id, status=CallStatus.IN_PROGRESS)
        
        while True:
            try:
                message = await websocket.receive()
                message_count += 1
                
                if message_count <= 3:
                    print(f"[ClientStream] Msg #{message_count}", flush=True)
                
                if "text" in message:
                    data = json.loads(message["text"])
                    await handler.handle_telnyx_message(data)
                elif "bytes" in message:
                    if handler.connection:
                        await handler.connection.send(message["bytes"])
                elif message.get("type") == "websocket.disconnect":
                    break
                    
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        print(f"[ClientStream] Error: {e}", flush=True)
        
    finally:
        print(f"[ClientStream] Disconnected. Messages: {message_count}", flush=True)
        if handler:
            await handler.stop()


@app.websocket("/ws/telnyx/stream/agent/{session_id}")
async def telnyx_agent_stream(websocket: WebSocket, session_id: str):
    """
    AGENT audio stream - 100% agent audio.
    Sent to Deepgram for transcription (context extraction, no guidance triggers).
    """
    await websocket.accept()
    print(f"[AgentStream] WebSocket accepted: {session_id}", flush=True)
    
    handler = None
    message_count = 0
    
    try:
        handler = await get_or_create_agent_handler(session_id)
        
        while True:
            try:
                message = await websocket.receive()
                message_count += 1
                
                if message_count <= 3:
                    print(f"[AgentStream] Msg #{message_count}", flush=True)
                
                if "text" in message:
                    data = json.loads(message["text"])
                    await handler.handle_telnyx_message(data)
                elif "bytes" in message:
                    # Forward audio to Deepgram for transcription
                    if handler.connection:
                        await handler.connection.send(message["bytes"])
                elif message.get("type") == "websocket.disconnect":
                    break
                    
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        print(f"[AgentStream] Error: {e}", flush=True)
        
    finally:
        print(f"[AgentStream] Disconnected. Messages: {message_count}", flush=True)
        if handler:
            await handler.stop()


# ============ PAGE ROUTES ============

@app.get("/", response_class=HTMLResponse)
async def agent_home(request: Request):
    """Main agent entry point"""
    return templates.TemplateResponse("agent.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/call", response_class=HTMLResponse)
async def call_page(request: Request):
    """Live call interface"""
    if is_telnyx_configured():
        return templates.TemplateResponse("call_telnyx.html", {"request": request})
    return templates.TemplateResponse("call_new.html", {"request": request})


@app.get("/call/browser", response_class=HTMLResponse)
async def call_browser_page(request: Request):
    """Browser mic mode"""
    return templates.TemplateResponse("call_new.html", {"request": request})


@app.get("/call/telnyx", response_class=HTMLResponse)
async def call_telnyx_page(request: Request):
    """Telnyx click-to-call mode"""
    return templates.TemplateResponse("call_telnyx.html", {"request": request})


@app.get("/call/test", response_class=HTMLResponse)
async def call_test_page(request: Request):
    """Test page"""
    return templates.TemplateResponse("call_new.html", {"request": request})


@app.get("/dialer", response_class=HTMLResponse)
async def dialer_page(request: Request):
    """Power Dialer - rapid appointment booking"""
    return templates.TemplateResponse("power_dialer.html", {"request": request})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Agent settings and profile"""
    return templates.TemplateResponse("settings.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """Admin dashboard"""
    db = get_vector_db()
    
    agency_docs = {}
    for agency_code in AGENCIES:
        agency_docs[agency_code] = {
            "name": AGENCIES[agency_code]["name"],
            "document_count": db.get_document_count(agency=agency_code),
            "documents": db.list_documents(agency=agency_code)
        }
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "agencies": AGENCIES,
        "agency_docs": agency_docs
    })


@app.get("/platform-admin", response_class=HTMLResponse)
async def platform_admin_page(request: Request):
    """Platform admin dashboard"""
    return templates.TemplateResponse("platform_admin.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_redirect(request: Request):
    return RedirectResponse(url="/admin")


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})


@app.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})


# ============ HEALTH CHECK ============

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/status")
async def api_status():
    """Detailed status check"""
    db = get_vector_db()
    
    return {
        "status": "healthy",
        "services": {
            "vector_db": {
                "connected": True,
                "document_count": db.get_document_count()
            },
            "anthropic": {
                "configured": bool(settings.anthropic_api_key)
            },
            "deepgram": {
                "configured": bool(settings.deepgram_api_key)
            },
            "telnyx": {
                "configured": is_telnyx_configured()
            },
            "postgres": {
                "configured": is_db_configured()
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# ============ PLATFORM USAGE ============

@app.get("/api/platform/summary")
async def platform_summary():
    if not is_db_configured():
        return {
            "internal": {"by_service": {}, "by_agency": {}, "total_cost": 0},
            "external": fetch_all_external_usage(),
            "daily_trends": [],
            "totals": {"estimated_monthly": 0, "telnyx_actual": 0},
            "database_configured": False,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    return get_platform_summary()


@app.get("/api/platform/usage")
async def platform_usage(days: int = 30, agency: Optional[str] = None):
    if not is_db_configured():
        return {"error": "Database not configured", "data": {}}
    
    return {
        "by_service": get_usage_summary(agency_code=agency),
        "by_agency": get_usage_by_agency(),
        "daily": get_daily_usage(days=days, agency_code=agency)
    }


@app.get("/api/platform/logs")
async def platform_logs(limit: int = 100, agency: Optional[str] = None):
    if not is_db_configured():
        return {"error": "Database not configured", "logs": []}
    
    return {"logs": get_recent_logs(limit=limit, agency_code=agency)}


@app.get("/api/platform/external")
async def platform_external():
    return fetch_all_external_usage()


# ============ AGENCY VALIDATION ============

@app.get("/api/agencies")
async def list_agencies():
    """List all active agencies"""
    return {
        "agencies": [
            {"code": code, "name": data["name"]}
            for code, data in AGENCIES.items()
            if data["active"]
        ]
    }


@app.post("/api/agency/validate")
async def validate_agency(data: AgencyValidation):
    code = data.code.strip().upper()
    
    if code in AGENCIES and AGENCIES[code]["active"]:
        return {
            "valid": True,
            "code": code,
            "name": AGENCIES[code]["name"]
        }
    
    return {
        "valid": False,
        "code": code,
        "error": "Invalid agency code"
    }


# ============ CHAT STREAMING ============

@app.post("/api/chat/stream")
async def chat_stream(data: ChatRequest):
    """Streaming chat endpoint"""
    import anthropic
    
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="AI service not configured")
    
    agency = (data.agency or "ADERHOLT").upper()
    
    async def generate():
        try:
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            
            # Get RAG context
            db = get_vector_db()
            results = db.search(data.message, top_k=5, agency=agency)
            
            context_text = ""
            if results:
                context_text = "\n\nRelevant training materials:\n"
                for r in results[:3]:
                    context_text += f"- {r['content'][:500]}\n"
            
            system_prompt = f"""You are Coachd, an AI sales coach for Globe Life insurance agents.
You help agents with:
- Handling objections
- Product knowledge
- Sales techniques
- Appointment setting
- Closing strategies

Be conversational, supportive, and practical. Keep responses focused and actionable.
{context_text}"""
            
            # Build messages from history
            messages = []
            for h in (data.history or [])[-10:]:
                role = "user" if h.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": h.get("content", "")})
            
            messages.append({"role": "user", "content": data.message})
            
            input_tokens = 0
            output_tokens = 0
            
            async with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system_prompt,
                messages=messages
            ) as stream:
                async for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_delta':
                            if hasattr(event.delta, 'text'):
                                yield f"data: {json.dumps({'text': event.delta.text})}\n\n"
                        elif event.type == 'message_start':
                            if hasattr(event.message, 'usage'):
                                input_tokens = event.message.usage.input_tokens
                        elif event.type == 'message_delta':
                            if hasattr(event, 'usage'):
                                output_tokens = event.usage.output_tokens
            
            # Log usage
            if is_db_configured() and (input_tokens or output_tokens):
                log_claude_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model="claude-sonnet-4-20250514",
                    agency_code=agency,
                    feature="chat"
                )
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# ============ DOCUMENT MANAGEMENT ============

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form("general"),
    agency: str = Form("ADERHOLT")
):
    """Upload a document to the knowledge base"""
    agency = agency.upper()
    if agency not in AGENCIES:
        raise HTTPException(status_code=400, detail=f"Invalid agency: {agency}")
    
    # Read file with size limit
    contents = bytearray()
    file_size = 0
    chunk_size = 1024 * 64  # 64KB chunks
    
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        file_size += len(chunk)
        contents.extend(chunk)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
    
    allowed_extensions = {'.pdf', '.docx', '.txt', '.md'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    file_path = Path(settings.documents_dir) / file.filename
    
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        chunks = processor.process_document(str(file_path), category=category)
        
        db = get_vector_db()
        added = db.add_chunks(chunks, agency=agency)
        
        return {
            "success": True,
            "filename": file.filename,
            "chunks_created": added,
            "category": category,
            "agency": agency,
            "message": f"Successfully processed {file.filename} into {added} chunks for {AGENCIES[agency]['name']}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if file_path.exists():
            os.remove(file_path)


@app.get("/api/documents")
async def list_documents(agency: Optional[str] = None):
    """List documents"""
    db = get_vector_db()
    
    if agency:
        agency = agency.upper()
        return {
            "agency": agency,
            "total_chunks": db.get_document_count(agency=agency),
            "documents": db.list_documents(agency=agency)
        }
    
    all_docs = {}
    for agency_code in AGENCIES:
        all_docs[agency_code] = {
            "total_chunks": db.get_document_count(agency=agency_code),
            "documents": db.list_documents(agency=agency_code)
        }
    return all_docs


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, agency: str = "ADERHOLT"):
    """Delete a document"""
    agency = agency.upper()
    if agency not in AGENCIES:
        raise HTTPException(status_code=400, detail=f"Invalid agency: {agency}")
    
    db = get_vector_db()
    deleted = db.delete_document(document_id, agency=agency)
    
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "success": True,
        "document_id": document_id,
        "agency": agency,
        "chunks_deleted": deleted
    }


# ============ RAG SEARCH & QUERY ============

@app.post("/api/search")
async def search_knowledge(data: SearchRequest):
    """Search knowledge base"""
    db = get_vector_db()
    
    agency = getattr(data, 'agency', None)
    results = db.search(data.query, top_k=data.top_k, agency=agency)
    
    return {
        "query": data.query,
        "agency": agency,
        "results": [
            {
                "content": r["content"],
                "metadata": r["metadata"],
                "score": r.get("relevance_score", 0)
            }
            for r in results
        ]
    }


@app.post("/api/query")
async def query_ai(data: QueryRequest):
    """Query AI with RAG context"""
    
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="AI service not configured")
    
    try:
        engine = RAGEngine()
        
        context = CallContext(
            current_product=data.context.get("product", "whole_life"),
            client_age=data.context.get("age"),
            client_occupation=data.context.get("occupation"),
            client_family=data.context.get("family")
        )
        
        response = engine.generate_guidance(data.query, context)
        
        return {
            "query": data.query,
            "response": response,
            "context": data.context
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/test-objection")
async def test_objection(data: QueryRequest):
    """Test objection handling"""
    
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="AI service not configured")
    
    try:
        engine = RAGEngine()
        
        context = CallContext(
            current_product=data.context.get("product", "whole_life"),
            client_age=data.context.get("age", 45)
        )
        
        context.objections_faced.append(data.query)
        
        response = engine.generate_guidance(
            f"Client objection: {data.query}",
            context
        )
        
        return {
            "objection": data.query,
            "guidance": response,
            "product": context.current_product
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TestGuidanceRequest(BaseModel):
    text: str
    history: list = []
    context: dict = {}


@app.post("/api/test-guidance")
async def test_guidance_stream(data: TestGuidanceRequest):
    """
    Streaming test endpoint for guidance UI testing.
    Simulates what Claude would respond with during a live call.
    """
    import anthropic
    from fastapi.responses import StreamingResponse
    
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="AI service not configured")
    
    async def generate():
        try:
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            
            # Build conversation context from history
            history_text = ""
            if data.history:
                for item in data.history[-6:]:  # Last 6 exchanges
                    role = "CLIENT" if item.get("type") == "client" else "AGENT GUIDANCE"
                    history_text += f"{role}: {item.get('text', '')}\n"
            
            system_prompt = """You are an AI sales coach for life insurance agents. 
When a client raises an objection, provide a brief, conversational response the agent should say.

Rules:
- Keep responses to 2-3 sentences max
- Be conversational, not scripted
- Address the specific objection directly
- Guide toward next steps when appropriate
- If it's not really an objection (just conversation), respond with: NO_GUIDANCE_NEEDED

Examples of good responses:
- "I completely understand. Let me ask you this - if something happened to you tomorrow, would your family still need that financial protection? [wait for yes] Then let's make sure they have it."
- "That makes sense. Most of my clients felt the same way until they saw how affordable it actually is. What if I could show you a plan that fits your budget?"
"""
            
            messages = [{
                "role": "user",
                "content": f"""Previous conversation:
{history_text}

CLIENT just said: "{data.text}"

What should the agent say in response? Remember: 2-3 sentences max, conversational tone."""
            }]
            
            async with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=system_prompt,
                messages=messages
            ) as stream:
                async for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
