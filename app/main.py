"""
Coachd.ai - Real-time AI Sales Assistant
Main FastAPI Application with Telnyx Bridge Integration
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
from fastapi.responses import HTMLResponse, StreamingResponse, Response, RedirectResponse
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

# Telnyx - simplified import
from .telnyx_bridge import is_telnyx_configured
from .telnyx_routes import router as telnyx_router
from .call_session import session_manager, CallStatus
from .telnyx_stream_handler import get_or_create_handler, remove_handler

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
    
    # Initialize vector DB
    db = get_vector_db()
    print(f"✓ Vector database initialized ({db.get_document_count()} chunks)")
    
    # Initialize PostgreSQL database
    if is_db_configured():
        init_db()
        print("✓ PostgreSQL database initialized (usage tracking active)")
    else:
        print("⚠ DATABASE_URL not set - usage tracking disabled")
    
    # Check API keys
    if settings.anthropic_api_key:
        print("✓ Anthropic API key configured")
    else:
        print("⚠ ANTHROPIC_API_KEY not set - AI features disabled")
    
    if settings.deepgram_api_key:
        print("✓ Deepgram API key configured")
    else:
        print("⚠ DEEPGRAM_API_KEY not set - transcription disabled")
    
    # Check Telnyx
    if is_telnyx_configured():
        print("✓ Telnyx configured - 3-way bridge ACTIVE")
        print(f"  Phone: {settings.telnyx_phone_number}")
        print(f"  Webhooks: {settings.base_url}/api/telnyx/")
    else:
        print("⚠ Telnyx not configured - using browser mic mode")
    
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
    version="2.5.0",  # Version bump for Telnyx TeXML migration
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
static_path = Path(__file__).parent.parent / "static"
templates_path = Path(__file__).parent.parent / "templates"

if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

templates = Jinja2Templates(directory=str(templates_path))

# Include routers
app.include_router(outcome_router)
app.include_router(status_router)
app.include_router(telnyx_router)  # All /api/telnyx/* routes


# ============ WEBSOCKET ROUTES ============

@app.websocket("/ws/{client_id}")
async def websocket_route(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time transcription (browser mic mode)"""
    await websocket_endpoint(websocket, client_id)


@app.websocket("/ws/telnyx/session/{session_id}")
async def telnyx_session_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time updates from Telnyx calls"""
    await websocket.accept()
    
    session = await session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    await session_manager.register_websocket(session_id, websocket)
    
    try:
        # Send current state
        await websocket.send_json({
            "type": "session_state",
            "data": session.to_dict()
        })
        
        # Keep connection alive
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


@app.websocket("/ws/telnyx/stream/{session_id}")
async def telnyx_stream_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for receiving Telnyx media streams.
    Telnyx sends audio here, we process through Deepgram with diarization,
    and broadcast transcripts + guidance to the frontend session WebSocket.
    """
    await websocket.accept()
    print(f"[TelnyxStream] WebSocket connected for session {session_id}")
    
    handler = None
    try:
        # Get or create stream handler for this session
        try:
            handler = await get_or_create_handler(session_id)
        except Exception as e:
            print(f"[TelnyxStream] Failed to create handler: {e}")
            # Continue without handler - at least keep connection alive
            # Audio won't be transcribed but call continues
            handler = None
        
        # Update session status
        await session_manager.update_session(session_id, status=CallStatus.IN_PROGRESS)
        
        # Process incoming messages from Telnyx
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if handler:
                    await handler.handle_telnyx_message(data)
                # If no handler, just consume messages silently
                
            except json.JSONDecodeError as e:
                print(f"[TelnyxStream] Invalid JSON: {e}")
                continue
                
    except Exception as e:
        print(f"[TelnyxStream] WebSocket error: {e}")
        
    finally:
        print(f"[TelnyxStream] WebSocket disconnected for session {session_id}")
        if handler:
            await remove_handler(session_id)


# ============ PAGE ROUTES ============

@app.get("/", response_class=HTMLResponse)
async def agent_home(request: Request):
    """Main agent entry point - agency code + context form"""
    return templates.TemplateResponse("agent.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat interface for agent Q&A"""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/call", response_class=HTMLResponse)
async def call_page(request: Request):
    """Live call interface - auto-selects best mode"""
    # If Telnyx is configured, use Telnyx mode
    if is_telnyx_configured():
        return templates.TemplateResponse("call_telnyx.html", {"request": request})
    # Otherwise fall back to browser mic mode
    return templates.TemplateResponse("call_new.html", {"request": request})


@app.get("/call/browser", response_class=HTMLResponse)
async def call_browser_page(request: Request):
    """Browser mic mode (legacy/fallback)"""
    return templates.TemplateResponse("call_new.html", {"request": request})


@app.get("/call/telnyx", response_class=HTMLResponse)
async def call_telnyx_page(request: Request):
    """Telnyx 3-way bridge mode (explicit)"""
    return templates.TemplateResponse("call_telnyx.html", {"request": request})


@app.get("/call/test", response_class=HTMLResponse)
async def call_test_page(request: Request):
    """Browser mic test page with outcome tracking"""
    return templates.TemplateResponse("call_new.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """Admin dashboard for document management"""
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
    """Platform admin dashboard for usage and cost tracking"""
    return templates.TemplateResponse("platform_admin.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_redirect(request: Request):
    """Redirect old dashboard to admin"""
    return RedirectResponse(url="/admin")


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    """Privacy policy page"""
    return templates.TemplateResponse("privacy.html", {"request": request})


@app.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    """Terms of service page"""
    return templates.TemplateResponse("terms.html", {"request": request})


# ============ HEALTH CHECK ============

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": "2.5.0",
        "database": is_db_configured(),
        "telnyx": is_telnyx_configured()
    }


# ============ PLATFORM ADMIN API ============

@app.get("/api/platform/summary")
async def platform_summary():
    """Get complete platform usage summary"""
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
    """Get usage summary"""
    if not is_db_configured():
        return {"error": "Database not configured", "data": {}}
    
    return {
        "by_service": get_usage_summary(agency_code=agency),
        "by_agency": get_usage_by_agency(),
        "daily": get_daily_usage(days=days, agency_code=agency)
    }


@app.get("/api/platform/logs")
async def platform_logs(limit: int = 100, agency: Optional[str] = None):
    """Get recent usage logs"""
    if not is_db_configured():
        return {"error": "Database not configured", "logs": []}
    
    return {"logs": get_recent_logs(limit=limit, agency_code=agency)}


@app.get("/api/platform/external")
async def platform_external():
    """Fetch fresh data from external service APIs"""
    return fetch_all_external_usage()


# ============ AGENCY VALIDATION ============

@app.post("/api/agency/validate")
async def validate_agency(data: AgencyValidation):
    """Validate an agency code"""
    code = data.code.strip().upper()
    
    if code in AGENCIES and AGENCIES[code]["active"]:
        return {
            "valid": True,
            "code": code,
            "name": AGENCIES[code]["name"]
        }
    
    raise HTTPException(status_code=400, detail="Invalid agency code")


# ============ CHAT API ============

@app.post("/api/chat")
async def chat_endpoint(data: ChatRequest):
    """Process a chat message and return AI response (non-streaming)"""
    
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="AI service not configured")
    
    try:
        engine = RAGEngine()
        
        # Build context from client profile
        context = data.context or {}
        client_info = []
        if context.get("name"):
            client_info.append(f"Name: {context['name']}")
        if context.get("age"):
            client_info.append(f"Age: {context['age']}")
        if context.get("occupation"):
            client_info.append(f"Occupation: {context['occupation']}")
        if context.get("marital"):
            client_info.append(f"Marital Status: {context['marital']}")
        
        client_info = "\n".join(client_info) if client_info else "Not specified"
        product = context.get("product", "whole_life")
        issue = context.get("issue", "")
        
        agency = data.agency or "ADERHOLT"
        relevant_context = engine.get_relevant_context(
            data.message,
            category=product,
            agency=agency
        )
        
        messages = []
        for msg in (data.history or []):
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": data.message})
        
        system_prompt = f"""You are Coachd, an expert AI sales coach for life insurance agents.

CLIENT: {client_info}
Product: {product}
{f"Issue: {issue}" if issue else ""}

TRAINING MATERIALS:
{relevant_context if relevant_context else "No specific materials found - use general knowledge."}

YOUR JOB:
Answer the agent's question. Give them what they need to close the deal.
- If they need a rebuttal, give them the exact words to say
- If they need strategy, explain the approach
- If they need a quick answer, be brief
- If they need depth, go deep

Match your response to what the question actually requires. No filler, no fluff, no unnecessary padding. But don't cut corners if the situation calls for a thorough answer.

Bold **key phrases** the agent should say out loud."""

        response = engine.client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        )
        
        # Log Claude usage
        log_claude_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            agency_code=agency,
            model=settings.claude_model,
            operation='chat'
        )
        
        return {
            "response": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream_endpoint(data: ChatRequest):
    """Process a chat message and stream the AI response"""
    
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="AI service not configured")
    
    async def generate():
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            
            # Build context
            context = data.context or {}
            client_info = []
            if context.get("name"):
                client_info.append(f"Name: {context['name']}")
            if context.get("age"):
                client_info.append(f"Age: {context['age']}")
            if context.get("occupation"):
                client_info.append(f"Occupation: {context['occupation']}")
            if context.get("marital"):
                client_info.append(f"Marital Status: {context['marital']}")
            
            client_info = "\n".join(client_info) if client_info else "Not specified"
            product = context.get("product", "whole_life")
            issue = context.get("issue", "")
            
            agency = data.agency or "ADERHOLT"
            engine = RAGEngine()
            relevant_context = engine.get_relevant_context(
                data.message,
                category=product,
                agency=agency
            )
            
            messages = []
            for msg in (data.history or []):
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": data.message})
            
            system_prompt = f"""You are Coachd, an expert AI sales coach for life insurance agents.

CLIENT: {client_info}
Product: {product}
{f"Issue: {issue}" if issue else ""}

TRAINING MATERIALS:
{relevant_context if relevant_context else "No specific materials found - use general knowledge."}

YOUR JOB:
Answer the agent's question. Give them what they need to close the deal.
- If they need a rebuttal, give them the exact words to say
- If they need strategy, explain the approach
- If they need a quick answer, be brief
- If they need depth, go deep

Match your response to what the question actually requires. No filler, no fluff, no unnecessary padding. But don't cut corners if the situation calls for a thorough answer.

Bold **key phrases** the agent should say out loud."""

            # Use synchronous streaming in a thread
            usage_info = {'input': 0, 'output': 0}
            loop = asyncio.get_event_loop()
            chunk_queue = asyncio.Queue()
            
            def stream_claude():
                try:
                    with client.messages.stream(
                        model=settings.claude_model,
                        max_tokens=1024,
                        system=system_prompt,
                        messages=messages
                    ) as stream:
                        for text in stream.text_stream:
                            asyncio.run_coroutine_threadsafe(
                                chunk_queue.put(('text', text)), loop
                            )
                        
                        final_message = stream.get_final_message()
                        if final_message and final_message.usage:
                            usage_info['input'] = final_message.usage.input_tokens
                            usage_info['output'] = final_message.usage.output_tokens
                    
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(('done', None)), loop
                    )
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(('error', str(e))), loop
                    )
            
            # Start Claude streaming in background thread
            thread = threading.Thread(target=stream_claude, daemon=True)
            thread.start()
            
            # Yield chunks as they arrive from the queue
            while True:
                msg_type, content = await chunk_queue.get()
                
                if msg_type == 'text':
                    yield f"data: {json.dumps({'text': content})}\n\n"
                elif msg_type == 'done':
                    break
                elif msg_type == 'error':
                    yield f"data: {json.dumps({'error': content})}\n\n"
                    break
            
            # Log usage
            if usage_info['input'] or usage_info['output']:
                log_claude_usage(
                    input_tokens=usage_info['input'],
                    output_tokens=usage_info['output'],
                    agency_code=agency,
                    model=settings.claude_model,
                    operation='chat_stream'
                )
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============ DOCUMENT MANAGEMENT ============

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form("general"),
    agency: str = Form("ADERHOLT")
):
    """Upload and process a document for a specific agency"""
    
    agency = agency.upper()
    if agency not in AGENCIES:
        raise HTTPException(status_code=400, detail=f"Invalid agency: {agency}")
    
    file_size = 0
    chunk_size = 1024 * 1024
    contents = bytearray()
    
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
        # Ensure directory exists
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
    """List all documents in an agency's knowledge base"""
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
    """Delete a document from an agency's knowledge base"""
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
    """Search an agency's knowledge base"""
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
    """Query the AI with RAG context"""
    
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