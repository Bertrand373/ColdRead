"""
Coachd Usage Tracker
Pulls usage data from external APIs and calculates costs
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .database import log_usage, is_db_configured, get_db, ExternalServiceSnapshot


# ============ PRICING CONSTANTS ============
# Updated pricing as of 2024 - adjust as needed

PRICING = {
    'deepgram': {
        'nova-2': 0.0043,  # per minute
        'nova': 0.0040,
        'enhanced': 0.0145,
        'base': 0.0125,
    },
    'telnyx': {
        # Telnyx pricing - dual channel means 2 outbound legs per call
        # Rates based on Telnyx Call Control API pricing
        'call_per_minute_outbound': 0.007,  # outbound to US/Canada (per leg)
        'call_per_minute_inbound': 0.0035,  # inbound
        'phone_number_monthly': 1.00,
        'recording_per_minute': 0.0025,
        'conference_per_minute_per_participant': 0.001,  # mixing included in call control
        'media_streaming_per_minute': 0.002,  # WebSocket audio streaming
    },
    'claude': {
        'claude-sonnet-4-20250514': {
            'input_per_1k': 0.003,
            'output_per_1k': 0.015,
        },
        'claude-3-5-sonnet-20241022': {
            'input_per_1k': 0.003,
            'output_per_1k': 0.015,
        },
        'claude-3-haiku-20240307': {
            'input_per_1k': 0.00025,
            'output_per_1k': 0.00125,
        },
    },
    'render': {
        'starter_monthly': 7.00,  # PostgreSQL starter
        'web_service_free': 0.00,
        'web_service_starter': 7.00,
    }
}


# ============ COST CALCULATION HELPERS ============

def calculate_deepgram_cost(minutes: float, model: str = 'nova-2') -> float:
    """Calculate Deepgram transcription cost"""
    rate = PRICING['deepgram'].get(model, PRICING['deepgram']['nova-2'])
    return minutes * rate


def calculate_telnyx_cost(
    call_minutes: float, 
    recording_minutes: float = 0, 
    is_inbound: bool = False,
    is_dual_channel: bool = True,
    has_media_streaming: bool = True
) -> float:
    """
    Calculate Telnyx call cost for dual-channel architecture
    
    Dual-channel call costs:
    - 2 outbound call legs (agent + client)
    - Conference bridge (2 participants)
    - Media streaming for transcription (2 streams)
    - Optional recording
    """
    pricing = PRICING['telnyx']
    
    if is_inbound:
        call_rate = pricing['call_per_minute_inbound']
    else:
        call_rate = pricing['call_per_minute_outbound']
    
    if is_dual_channel:
        # Two outbound legs
        call_cost = call_minutes * call_rate * 2
        # Conference bridge cost (2 participants)
        conference_cost = call_minutes * pricing['conference_per_minute_per_participant'] * 2
        # Media streaming for both channels
        streaming_cost = call_minutes * pricing['media_streaming_per_minute'] * 2 if has_media_streaming else 0
    else:
        call_cost = call_minutes * call_rate
        conference_cost = 0
        streaming_cost = 0
    
    recording_cost = recording_minutes * pricing['recording_per_minute']
    
    return call_cost + conference_cost + streaming_cost + recording_cost


def calculate_claude_cost(input_tokens: int, output_tokens: int, model: str = 'claude-sonnet-4-20250514') -> float:
    """Calculate Claude API cost"""
    model_pricing = PRICING['claude'].get(model, PRICING['claude']['claude-sonnet-4-20250514'])
    input_cost = (input_tokens / 1000) * model_pricing['input_per_1k']
    output_cost = (output_tokens / 1000) * model_pricing['output_per_1k']
    return input_cost + output_cost


def get_dual_channel_cost_breakdown(call_minutes: float) -> Dict[str, float]:
    """
    Get detailed cost breakdown for a dual-channel call
    Useful for displaying per-call costs in admin
    
    Returns breakdown for a single coaching session
    """
    pricing = PRICING['telnyx']
    
    # Two outbound legs (agent + client)
    call_legs = call_minutes * pricing['call_per_minute_outbound'] * 2
    
    # Conference bridge (2 participants)
    conference = call_minutes * pricing['conference_per_minute_per_participant'] * 2
    
    # Media streaming (2 WebSocket streams for transcription)
    streaming = call_minutes * pricing['media_streaming_per_minute'] * 2
    
    # Total Telnyx cost
    telnyx_total = call_legs + conference + streaming
    
    # Add Deepgram cost (both channels transcribed)
    deepgram_cost = call_minutes * PRICING['deepgram']['nova-2'] * 2
    
    # Estimate Claude cost (~1 analysis per minute, ~500 input + 100 output tokens each)
    claude_analyses = call_minutes  # roughly 1 per minute
    claude_cost = claude_analyses * calculate_claude_cost(500, 100)
    
    total = telnyx_total + deepgram_cost + claude_cost
    
    return {
        'call_minutes': round(call_minutes, 2),
        'telnyx': {
            'call_legs': round(call_legs, 4),
            'conference': round(conference, 4),
            'streaming': round(streaming, 4),
            'subtotal': round(telnyx_total, 4)
        },
        'deepgram': round(deepgram_cost, 4),
        'claude': round(claude_cost, 4),
        'total': round(total, 4),
        'per_minute': round(total / call_minutes, 4) if call_minutes > 0 else 0
    }


# ============ LOGGING WRAPPERS ============
# Use these throughout the app to automatically track usage

def log_deepgram_usage(
    duration_seconds: float,
    agency_code: Optional[str] = None,
    session_id: Optional[str] = None,
    model: str = 'nova-2'
):
    """Log Deepgram transcription usage"""
    minutes = duration_seconds / 60
    cost = calculate_deepgram_cost(minutes, model)
    
    log_usage(
        service='deepgram',
        operation='transcribe',
        quantity=minutes,
        unit='minutes',
        estimated_cost=cost,
        agency_code=agency_code,
        session_id=session_id,
        metadata={'model': model, 'seconds': duration_seconds}
    )
    
    return cost


def log_telnyx_usage(
    call_duration_seconds: float,
    agency_code: Optional[str] = None,
    session_id: Optional[str] = None,
    recording_seconds: float = 0,
    call_control_id: Optional[str] = None,
    is_inbound: bool = False,
    is_dual_channel: bool = True,
    agent_duration_seconds: Optional[float] = None,
    client_duration_seconds: Optional[float] = None
):
    """
    Log Telnyx call usage for dual-channel architecture
    
    For accurate tracking, pass agent_duration_seconds and client_duration_seconds
    separately. If not provided, assumes both legs = call_duration_seconds.
    """
    call_minutes = call_duration_seconds / 60
    recording_minutes = recording_seconds / 60
    
    # Calculate accurate cost
    cost = calculate_telnyx_cost(
        call_minutes, 
        recording_minutes, 
        is_inbound,
        is_dual_channel=is_dual_channel,
        has_media_streaming=True
    )
    
    # Track both legs if provided
    agent_mins = (agent_duration_seconds / 60) if agent_duration_seconds else call_minutes
    client_mins = (client_duration_seconds / 60) if client_duration_seconds else call_minutes
    
    log_usage(
        service='telnyx',
        operation='dual_channel_call' if is_dual_channel else 'call',
        quantity=call_minutes,
        unit='minutes',
        estimated_cost=cost,
        agency_code=agency_code,
        session_id=session_id,
        metadata={
            'call_control_id': call_control_id,
            'recording_minutes': recording_minutes,
            'is_inbound': is_inbound,
            'is_dual_channel': is_dual_channel,
            'agent_minutes': round(agent_mins, 2),
            'client_minutes': round(client_mins, 2),
            'cost_breakdown': {
                'call_legs': round(call_minutes * PRICING['telnyx']['call_per_minute_outbound'] * 2, 4),
                'conference': round(call_minutes * PRICING['telnyx']['conference_per_minute_per_participant'] * 2, 4),
                'streaming': round(call_minutes * PRICING['telnyx']['media_streaming_per_minute'] * 2, 4),
                'recording': round(recording_minutes * PRICING['telnyx']['recording_per_minute'], 4)
            }
        }
    )
    
    return cost


def log_claude_usage(
    input_tokens: int,
    output_tokens: int,
    agency_code: Optional[str] = None,
    session_id: Optional[str] = None,
    model: str = 'claude-sonnet-4-20250514',
    operation: str = 'completion'
):
    """Log Claude API usage"""
    cost = calculate_claude_cost(input_tokens, output_tokens, model)
    total_tokens = input_tokens + output_tokens
    
    log_usage(
        service='claude',
        operation=operation,
        quantity=total_tokens,
        unit='tokens',
        estimated_cost=cost,
        agency_code=agency_code,
        session_id=session_id,
        metadata={
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }
    )
    
    return cost


# ============ EXTERNAL API FETCHERS ============

def fetch_anthropic_usage() -> Dict[str, Any]:
    """
    Fetch usage from Anthropic API
    Note: Anthropic doesn't have a public usage API yet,
    so we rely on our internal logging. This is a placeholder.
    """
    api_key = os.getenv('ANTHROPIC_API_KEY', '')
    
    # Anthropic doesn't expose a usage API publicly yet
    # Return empty dict - we track usage internally
    return {
        'source': 'internal_tracking',
        'note': 'Anthropic usage tracked via internal logging',
        'fetched_at': datetime.utcnow().isoformat()
    }


def fetch_deepgram_usage() -> Dict[str, Any]:
    """
    Fetch usage from Deepgram API
    https://developers.deepgram.com/reference/get-all-balances
    """
    api_key = os.getenv('DEEPGRAM_API_KEY', '')
    
    if not api_key:
        return {'error': 'DEEPGRAM_API_KEY not configured'}
    
    try:
        # Get project balances
        headers = {
            'Authorization': f'Token {api_key}',
            'Content-Type': 'application/json'
        }
        
        # First get projects
        projects_response = requests.get(
            'https://api.deepgram.com/v1/projects',
            headers=headers,
            timeout=10
        )
        
        if projects_response.status_code != 200:
            return {'error': f'Failed to fetch projects: {projects_response.status_code}'}
        
        projects_data = projects_response.json()
        projects = projects_data.get('projects', [])
        
        if not projects:
            return {'error': 'No projects found'}
        
        # Get usage for first project
        project_id = projects[0]['project_id']
        
        # Get balances
        balances_response = requests.get(
            f'https://api.deepgram.com/v1/projects/{project_id}/balances',
            headers=headers,
            timeout=10
        )
        
        balances_data = {}
        if balances_response.status_code == 200:
            balances_data = balances_response.json()
        
        # Get usage summary (last 30 days)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        usage_response = requests.get(
            f'https://api.deepgram.com/v1/projects/{project_id}/usage',
            headers=headers,
            params={
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            },
            timeout=10
        )
        
        usage_data = {}
        if usage_response.status_code == 200:
            usage_data = usage_response.json()
        
        return {
            'project_id': project_id,
            'balances': balances_data,
            'usage_30d': usage_data,
            'fetched_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {'error': str(e)}


def fetch_telnyx_usage() -> Dict[str, Any]:
    """
    Fetch usage from Telnyx API with accurate cost tracking
    Uses multiple endpoints for comprehensive data
    """
    api_key = os.getenv('TELNYX_API_KEY', '')
    
    if not api_key:
        return {'error': 'TELNYX_API_KEY not configured'}
    
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        end_date = datetime.utcnow()
        start_date = end_date.replace(day=1)  # First of month
        
        # 1. Get account balance (shows actual spend)
        balance_response = requests.get(
            'https://api.telnyx.com/v2/balance',
            headers=headers,
            timeout=10
        )
        
        balance_data = {}
        account_balance = None
        if balance_response.status_code == 200:
            balance_data = balance_response.json()
            account_balance = balance_data.get('data', {}).get('balance')
        
        # 2. Get phone numbers
        numbers_response = requests.get(
            'https://api.telnyx.com/v2/phone_numbers',
            headers=headers,
            params={'page[size]': 100},
            timeout=10
        )
        
        phone_numbers = []
        if numbers_response.status_code == 200:
            numbers_data = numbers_response.json()
            phone_numbers = numbers_data.get('data', [])
        
        # 3. Get call events for actual usage
        # Using call_events gives us more detail than /calls
        calls_response = requests.get(
            'https://api.telnyx.com/v2/call_events',
            headers=headers,
            params={
                'page[size]': 250,
                'filter[event_type]': 'call.hangup',
                'filter[occurred_at][gte]': start_date.isoformat() + 'Z',
                'filter[occurred_at][lte]': end_date.isoformat() + 'Z'
            },
            timeout=15
        )
        
        call_events = []
        total_call_seconds = 0
        call_count = 0
        
        if calls_response.status_code == 200:
            events_data = calls_response.json()
            call_events = events_data.get('data', [])
            
            for event in call_events:
                payload = event.get('payload', {})
                # Duration in seconds from hangup events
                duration = payload.get('duration_secs', 0) or payload.get('billsec', 0)
                if duration:
                    total_call_seconds += duration
                    call_count += 1
        
        total_call_minutes = total_call_seconds / 60
        
        # 4. Try to get billing/usage report (if available)
        billing_data = {}
        try:
            billing_response = requests.get(
                'https://api.telnyx.com/v2/billing/group_costs',
                headers=headers,
                params={
                    'filter[date][gte]': start_date.strftime('%Y-%m-%d'),
                    'filter[date][lte]': end_date.strftime('%Y-%m-%d')
                },
                timeout=15
            )
            if billing_response.status_code == 200:
                billing_data = billing_response.json()
        except:
            pass  # Billing API may not be available on all accounts
        
        # Calculate costs
        pricing = PRICING['telnyx']
        phone_number_monthly_cost = len(phone_numbers) * pricing['phone_number_monthly']
        
        # For dual-channel: each call session = 2 legs + conference + streaming
        # Assuming most calls are dual-channel
        sessions_count = call_count // 2 if call_count > 0 else 0
        avg_session_minutes = (total_call_minutes / 2) if call_count > 0 else 0
        
        # Detailed cost breakdown
        call_leg_cost = total_call_minutes * pricing['call_per_minute_outbound']
        conference_cost = avg_session_minutes * pricing['conference_per_minute_per_participant'] * 2 * sessions_count / max(sessions_count, 1)
        streaming_cost = total_call_minutes * pricing['media_streaming_per_minute']
        
        estimated_call_cost = call_leg_cost + conference_cost + streaming_cost
        
        # Check if we have actual billing data
        actual_cost = None
        if billing_data.get('data'):
            actual_cost = sum(
                float(item.get('total_cost', 0)) 
                for item in billing_data.get('data', [])
            )
        
        return {
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'account_balance': account_balance,
            'phone_numbers_count': len(phone_numbers),
            'phone_numbers': [n.get('phone_number') for n in phone_numbers[:5]],  # First 5
            'total_call_minutes': round(total_call_minutes, 2),
            'total_call_legs': call_count,
            'estimated_sessions': sessions_count,
            'summary': {
                'phone_numbers': round(phone_number_monthly_cost, 2),
                'call_legs': round(call_leg_cost, 4),
                'conference': round(conference_cost, 4),
                'streaming': round(streaming_cost, 4),
                'estimated_total': round(phone_number_monthly_cost + estimated_call_cost, 4),
                'actual_total': round(actual_cost, 4) if actual_cost else None,
                'total_cost': round(actual_cost if actual_cost else (phone_number_monthly_cost + estimated_call_cost), 4)
            },
            'has_actual_billing': actual_cost is not None,
            'fetched_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {'error': str(e)}


def fetch_render_usage() -> Dict[str, Any]:
    """
    Fetch usage from Render API
    https://api-docs.render.com/reference/get-bandwidth
    """
    api_key = os.getenv('RENDER_API_KEY', '')
    
    if not api_key:
        return {'error': 'RENDER_API_KEY not configured'}
    
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        }
        
        # Get services
        services_response = requests.get(
            'https://api.render.com/v1/services',
            headers=headers,
            params={'limit': 20},
            timeout=10
        )
        
        if services_response.status_code != 200:
            return {'error': f'Failed to fetch services: {services_response.status_code}'}
        
        services_data = services_response.json()
        services = []
        
        for item in services_data:
            service = item.get('service', {})
            services.append({
                'id': service.get('id'),
                'name': service.get('name'),
                'type': service.get('type'),
                'status': service.get('suspended', 'active'),
                'created_at': service.get('createdAt')
            })
        
        # Get bandwidth for each service
        for svc in services:
            if svc['id']:
                try:
                    bw_response = requests.get(
                        f"https://api.render.com/v1/services/{svc['id']}/metrics/bandwidth",
                        headers=headers,
                        params={'resolution': 'day', 'numPeriods': 30},
                        timeout=10
                    )
                    if bw_response.status_code == 200:
                        svc['bandwidth'] = bw_response.json()
                except:
                    pass
        
        return {
            'services': services,
            'fetched_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {'error': str(e)}


def fetch_all_external_usage() -> Dict[str, Any]:
    """Fetch usage from all external services"""
    return {
        'anthropic': fetch_anthropic_usage(),
        'deepgram': fetch_deepgram_usage(),
        'telnyx': fetch_telnyx_usage(),
        'render': fetch_render_usage(),
        'fetched_at': datetime.utcnow().isoformat()
    }


def save_external_snapshot(service: str, data: Dict[str, Any]):
    """Save an external service snapshot to the database"""
    if not is_db_configured():
        return
    
    try:
        with get_db() as db:
            snapshot = ExternalServiceSnapshot(
                service=service,
                data_json=json.dumps(data),
                total_cost=data.get('summary', {}).get('total_cost') if isinstance(data.get('summary'), dict) else None
            )
            db.add(snapshot)
    except Exception as e:
        print(f"Failed to save snapshot: {e}")


def get_platform_summary() -> Dict[str, Any]:
    """
    Get complete platform summary for the admin dashboard
    Combines internal tracking with external API data
    """
    from .database import get_usage_summary, get_usage_by_agency, get_daily_usage
    
    # Get internal usage data
    internal_summary = get_usage_summary()
    agency_breakdown = get_usage_by_agency()
    daily_data = get_daily_usage(days=30)
    
    # Fetch external data
    external_data = fetch_all_external_usage()
    
    # Calculate totals
    total_internal_cost = sum(
        svc.get('total_cost', 0) 
        for svc in internal_summary.values()
    )
    
    # Telnyx external cost (most accurate)
    telnyx_external_cost = 0
    if 'telnyx' in external_data and 'summary' in external_data['telnyx']:
        telnyx_external_cost = external_data['telnyx']['summary'].get('total_cost', 0)
    
    return {
        'internal': {
            'by_service': internal_summary,
            'by_agency': agency_breakdown,
            'total_cost': total_internal_cost
        },
        'external': external_data,
        'daily_trends': daily_data,
        'totals': {
            'estimated_monthly': total_internal_cost,
            'telnyx_actual': telnyx_external_cost
        },
        'generated_at': datetime.utcnow().isoformat()
    }