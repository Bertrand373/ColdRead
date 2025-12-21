"""
Coachd Configuration
Manages environment variables and application settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    deepgram_api_key: str = Field(default="", alias="DEEPGRAM_API_KEY")
    
    # Telnyx Configuration (TeXML Application)
    telnyx_api_key: str = Field(default="", alias="TELNYX_API_KEY")
    telnyx_phone_number: str = Field(default="", alias="TELNYX_PHONE_NUMBER")
    telnyx_app_id: str = Field(default="", alias="TELNYX_APP_ID")
    
    # Base URL for webhooks (your custom domain)
    base_url: str = Field(default="https://coachd.ai", alias="BASE_URL")
    
    # Application
    app_name: str = "Coachd"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database paths (uses /data on Render for persistence)
    chroma_persist_dir: str = "/data/chroma_db" if os.path.exists("/data") else "./chroma_db"
    documents_dir: str = "/data/documents" if os.path.exists("/data") else "./documents"
    
    # AI Settings
    claude_model: str = "claude-sonnet-4-20250514"
    
    # RAG Settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


# Global settings instance
settings = Settings()