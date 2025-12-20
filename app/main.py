"""
Coachd.ai - Real-time AI Sales Assistant
Main FastAPI Application with Twilio Bridge Integration
"""

import os
import json
import shutil
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

# Twilio imports
from .twilio_bridge import (
    is_twilio_configured,
    initiate_agent_call,
    generate_agent_conference_twiml,
    generate_client_conference_twiml,
    add_client_to_conference,
    end_conference,
    get_recording_url
)
from .call_session import session_manager, CallStatus

# Database and usage tracking
from .database import init_db, is_db_configured, get_usage_summary, get_usage_by_agency, get_daily_usage, get_recent_logs
from .usage_tracker import (
    log_claude_usage,
    log_deepgram_usage,
    log_twilio_usage,
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

# Twilio request models
class TwilioCallRequest(BaseModel):
    agent_phone: str

class TwilioDialRequest(BaseModel):
    session_id: str
    client_phone: str
    agent_caller_id: Optional[str] = None

class TwilioEndRequest(BaseModel):
    session_id: str


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
    
    # Check Twilio
    if is_twilio_configured():
        print("✓ Twilio configured - 3-way bridge ACTIVE")
    else:
        print("⚠ Twilio not configured - using browser mic mode")
    
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
    version="2.3.0",
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


# ============ WEBSOCKET ROUTES ============

@app.websocket("/ws/{client_id}")
async def websocket_route(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time transcription (browser mic mode)"""
    await websocket_endpoint(websocket, client_id)


@app.websocket("/ws/twilio/session/{session_id}")
async def twilio_session_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time updates from Twilio calls"""
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
    # If Twilio is configured, use Twilio mode
    if is_twilio_configured():
        return templates.TemplateResponse("call_twilio.html", {"request": request})
    # Otherwise fall back to browser mic mode
    return templates.TemplateResponse("call_new.html", {"request": request})


@app.get("/call/browser", response_class=HTMLResponse)
async def call_browser_page(request: Request):
    """Browser mic mode (legacy/fallback)"""
    return templates.TemplateResponse("call_new.html", {"request": request})


@app.get("/call/twilio", response_class=HTMLResponse)
async def call_twilio_page(request: Request):
    """Twilio 3-way bridge mode (explicit)"""
    return templates.TemplateResponse("call_twilio.html", {"request": request})


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
# Note: /api/status is now handled by status_router with proper service verification

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": "2.3.0",
        "database": is_db_configured()
    }


# ============ PLATFORM ADMIN API ============

@app.get("/api/platform/summary")
async def platform_summary():
    """Get complete platform usage summary"""
    if not is_db_configured():
        # Return structure even without DB for external API data
        return {
            "internal": {"by_service": {}, "by_agency": {}, "total_cost": 0},
            "external": fetch_all_external_usage(),
            "daily_trends": [],
            "totals": {"estimated_monthly": 0, "twilio_actual": 0},
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


# ============ TWILIO API ENDPOINTS ============

@app.post("/api/twilio/start-call")
async def start_twilio_call(data: TwilioCallRequest):
    """Start a new Twilio-bridged call session"""
    if not is_twilio_configured():
        raise HTTPException(status_code=503, detail="Twilio not configured")
    
    # Normalize phone number
    agent_phone = data.agent_phone
    if not agent_phone.startswith("+"):
        digits = agent_phone.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
        agent_phone = f"+1{digits}" if len(digits) == 10 else f"+{digits}"
    
    # Create session
    session = await session_manager.create_session(agent_phone)
    
    # Initiate call to agent
    result = initiate_agent_call(agent_phone, session.session_id)
    
    if not result["success"]:
        await session_manager.update_session(session.session_id, status=CallStatus.FAILED)
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to start call"))
    
    await session_manager.update_session(
        session.session_id,
        agent_call_sid=result["call_sid"],
        status=CallStatus.AGENT_RINGING
    )
    
    return {
        "success": True,
        "session_id": session.session_id,
        "message": "Calling your phone now. Answer to connect."
    }


@app.post("/api/twilio/dial-client")
async def dial_twilio_client(data: TwilioDialRequest):
    """Add client to existing Twilio conference"""
    session = await session_manager.get_session(data.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Normalize phone number
    client_phone = data.client_phone
    if not client_phone.startswith("+"):
        digits = client_phone.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
        client_phone = f"+1{digits}" if len(digits) == 10 else f"+{digits}"
    
    result = add_client_to_conference(client_phone, data.session_id, data.agent_caller_id)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to dial client"))
    
    await session_manager.update_session(
        data.session_id,
        client_phone=client_phone,
        client_call_sid=result["call_sid"],
        status=CallStatus.CLIENT_RINGING
    )
    
    return {"success": True, "message": f"Dialing {client_phone}..."}


@app.post("/api/twilio/end-call")
async def end_twilio_call(data: TwilioEndRequest):
    """End a Twilio call session"""
    session = await session_manager.get_session(data.session_id)
    
    # Log Twilio usage when call ends
    if session and session.started_at:
        duration = session.get_duration() or 0
        log_twilio_usage(
            call_duration_seconds=duration,
            agency_code=getattr(session, 'agency_code', None),
            session_id=data.session_id,
            call_sid=session.agent_call_sid
        )
    
    end_conference(data.session_id)
    await session_manager.end_session(data.session_id)
    return {"success": True, "message": "Call ended"}


@app.get("/api/twilio/session/{session_id}")
async def get_twilio_session(session_id: str):
    """Get Twilio session details"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


# ============ TWILIO WEBHOOKS (Called by Twilio) ============

@app.post("/api/twilio/agent-joined")
async def twilio_agent_joined(request: Request):
    """Webhook: Agent has answered, put them in conference"""
    session_id = request.query_params.get("session_id", "unknown")
    
    await session_manager.update_session(
        session_id,
        status=CallStatus.AGENT_CONNECTED,
        started_at=datetime.utcnow()
    )
    
    twiml = generate_agent_conference_twiml(session_id)
    return Response(content=twiml, media_type="application/xml")


@app.post("/api/twilio/client-joined")
async def twilio_client_joined(request: Request):
    """Webhook: Client has answered, put them in conference"""
    session_id = request.query_params.get("session_id", "unknown")
    
    await session_manager.update_session(session_id, status=CallStatus.IN_PROGRESS)
    
    twiml = generate_client_conference_twiml(session_id)
    return Response(content=twiml, media_type="application/xml")


@app.post("/api/twilio/call-status")
async def twilio_call_status(request: Request):
    """Webhook: Call status updates"""
    form_data = await request.form()
    call_status = form_data.get("CallStatus")
    session_id = request.query_params.get("session_id")
    party = request.query_params.get("party", "agent")
    
    if session_id and call_status == "completed":
        session = await session_manager.get_session(session_id)
        if session:
            await session_manager.end_session(session_id)
    
    return Response(content="", status_code=200)


@app.post("/api/twilio/conference-status")
async def twilio_conference_status(request: Request):
    """Webhook: Conference events"""
    form_data = await request.form()
    conference_sid = form_data.get("ConferenceSid")
    session_id = request.query_params.get("session_id")
    
    if session_id:
        await session_manager.update_session(session_id, conference_sid=conference_sid)
    
    return Response(content="", status_code=200)


@app.post("/api/twilio/recording-complete")
async def twilio_recording_complete(request: Request):
    """Webhook: Recording finished"""
    form_data = await request.form()
    recording_sid = form_data.get("RecordingSid")
    recording_status = form_data.get("RecordingStatus")
    session_id = request.query_params.get("session_id")
    
    if session_id and recording_status == "completed":
        auth_url = get_recording_url(recording_sid)
        await session_manager.update_session(
            session_id,
            recording_sid=recording_sid,
            recording_url=auth_url
        )
    
    return Response(content="", status_code=200)


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
            "model": settings.claude_model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(data: ChatRequest):
    """Stream a chat response"""
    
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="AI service not configured")
    
    async def generate():
        try:
            engine = RAGEngine()
            
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

            with engine.client.messages.stream(
                model=settings.claude_model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
                
                # Log usage after stream completes
                final_message = stream.get_final_message()
                if final_message and final_message.usage:
                    log_claude_usage(
                        input_tokens=final_message.usage.input_tokens,
                        output_tokens=final_message.usage.output_tokens,
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