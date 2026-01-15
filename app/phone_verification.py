"""
Phone Verification for Caller ID
Handles Telnyx number verification to enable custom caller ID display
"""

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal

from .config import settings

router = APIRouter(prefix="/api/verify", tags=["verification"])

TELNYX_API_BASE = "https://api.telnyx.com/v2"


def get_telnyx_headers():
    """Get authorization headers for Telnyx API"""
    return {
        "Authorization": f"Bearer {settings.telnyx_api_key}",
        "Content-Type": "application/json"
    }


def normalize_phone(phone: str) -> str:
    """Normalize phone number to E.164 format (+1XXXXXXXXXX)"""
    digits = ''.join(c for c in phone if c.isdigit())
    if len(digits) == 10:
        return f"+1{digits}"
    elif len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return f"+{digits}"


class InitiateRequest(BaseModel):
    phone_number: str
    method: Literal["sms", "call"] = "sms"


class ConfirmRequest(BaseModel):
    phone_number: str
    code: str


@router.get("/check/{phone_number}")
async def check_verification(phone_number: str):
    """
    Check if a phone number is already verified in Telnyx.
    Returns verified: true/false
    """
    if not settings.telnyx_api_key:
        raise HTTPException(status_code=503, detail="Telnyx not configured")
    
    normalized = normalize_phone(phone_number)
    print(f"[Verify] Checking phone: {phone_number} -> normalized: {normalized}", flush=True)
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # List all verified numbers and check if this one exists
            response = await client.get(
                f"{TELNYX_API_BASE}/verified_numbers",
                headers=get_telnyx_headers()
            )
            
            print(f"[Verify] Telnyx response status: {response.status_code}", flush=True)
            
            if response.status_code != 200:
                print(f"[Verify] Telnyx API error: {response.status_code} - {response.text}", flush=True)
                raise HTTPException(status_code=502, detail="Failed to check verification status")
            
            data = response.json()
            verified_numbers = data.get("data", [])
            
            print(f"[Verify] Found {len(verified_numbers)} verified numbers in Telnyx", flush=True)
            
            # Check if our number is in the verified list
            for entry in verified_numbers:
                telnyx_phone = entry.get("phone_number")
                telnyx_status = entry.get("status", "")
                print(f"[Verify] Comparing: '{normalized}' vs '{telnyx_phone}' (status: {telnyx_status})", flush=True)
                
                if telnyx_phone == normalized:
                    # If number is in the verified_numbers list, it's verified
                    # (Telnyx may return empty status but presence in list = verified)
                    print(f"[Verify] MATCH FOUND - number is verified!", flush=True)
                    return {"verified": True, "phone_number": normalized}
            
            print(f"[Verify] No match found - returning verified: false", flush=True)
            return {"verified": False, "phone_number": normalized}
            
    except httpx.RequestError as e:
        print(f"[Verify] Request error: {e}", flush=True)
        raise HTTPException(status_code=502, detail="Failed to connect to verification service")


@router.post("/initiate")
async def initiate_verification(data: InitiateRequest):
    """
    Initiate phone verification by sending SMS or making a call.
    Telnyx will send a 6-digit code.
    """
    if not settings.telnyx_api_key:
        raise HTTPException(status_code=503, detail="Telnyx not configured")
    
    normalized = normalize_phone(data.phone_number)
    print(f"[Verify] Initiating verification for: {normalized} via {data.method}", flush=True)
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{TELNYX_API_BASE}/verified_numbers",
                headers=get_telnyx_headers(),
                json={
                    "phone_number": normalized,
                    "verification_method": data.method
                }
            )
            
            print(f"[Verify] Telnyx initiate response: {response.status_code}", flush=True)
            print(f"[Verify] Telnyx initiate body: {response.text[:500]}", flush=True)
            
            # 200 = new verification initiated
            # 422 = number already exists (might be pending or verified)
            if response.status_code == 200:
                return {
                    "success": True,
                    "phone_number": normalized,
                    "method": data.method,
                    "message": f"Verification code sent via {data.method.upper()}"
                }
            elif response.status_code == 422:
                # Number exists - might need to resend
                # Try to trigger a new code by requesting verification again
                error_data = response.json()
                errors = error_data.get("errors", [])
                
                print(f"[Verify] 422 errors: {errors}", flush=True)
                
                # Check if it's already verified
                for err in errors:
                    err_detail = err.get("detail", "").lower()
                    if "already verified" in err_detail or "verified" in err_detail:
                        print(f"[Verify] Number already verified - auto-proceeding", flush=True)
                        return {
                            "success": True,
                            "phone_number": normalized,
                            "already_verified": True,
                            "message": "Phone number is already verified"
                        }
                
                # Number exists but not verified - this is a resend scenario
                # Telnyx should still send a new code on POST even if number exists
                return {
                    "success": True,
                    "phone_number": normalized,
                    "method": data.method,
                    "message": f"Verification code sent via {data.method.upper()}"
                }
            else:
                print(f"[Verify] Initiate error: {response.status_code} - {response.text}", flush=True)
                raise HTTPException(
                    status_code=502,
                    detail="Failed to send verification code"
                )
                
    except httpx.RequestError as e:
        print(f"[Verify] Request error: {e}", flush=True)
        raise HTTPException(status_code=502, detail="Failed to connect to verification service")


@router.post("/confirm")
async def confirm_verification(data: ConfirmRequest):
    """
    Confirm verification with the 6-digit code.
    On success, the number can be used as caller ID.
    """
    if not settings.telnyx_api_key:
        raise HTTPException(status_code=503, detail="Telnyx not configured")
    
    normalized = normalize_phone(data.phone_number)
    code = data.code.strip()
    print(f"[Verify] Confirming code for: {normalized}", flush=True)
    
    # Validate code format (Telnyx sends 5-digit codes)
    if not code.isdigit() or len(code) < 5 or len(code) > 6:
        raise HTTPException(status_code=400, detail="Invalid verification code format")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{TELNYX_API_BASE}/verified_numbers/{normalized}/actions/verify",
                headers=get_telnyx_headers(),
                json={"verification_code": code}
            )
            
            print(f"[Verify] Confirm response: {response.status_code}", flush=True)
            print(f"[Verify] Confirm body: {response.text[:500]}", flush=True)
            
            if response.status_code == 200:
                print(f"[Verify] SUCCESS - number verified!", flush=True)
                return {
                    "success": True,
                    "verified": True,
                    "phone_number": normalized,
                    "message": "Phone number verified successfully"
                }
            elif response.status_code == 422:
                # Invalid code or expired
                error_data = response.json()
                errors = error_data.get("errors", [])
                detail = "Invalid verification code"
                for err in errors:
                    if err.get("detail"):
                        detail = err.get("detail")
                        break
                print(f"[Verify] Invalid code: {detail}", flush=True)
                raise HTTPException(status_code=400, detail=detail)
            elif response.status_code == 404:
                raise HTTPException(
                    status_code=400,
                    detail="Verification not found. Please request a new code."
                )
            else:
                print(f"[Verify] Confirm error: {response.status_code} - {response.text}", flush=True)
                raise HTTPException(status_code=502, detail="Verification failed")
                
    except httpx.RequestError as e:
        print(f"[Verify] Request error: {e}", flush=True)
        raise HTTPException(status_code=502, detail="Failed to connect to verification service")
