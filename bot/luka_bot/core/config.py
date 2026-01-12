"""
luka_bot configuration - minimal Phase 1 settings.

Completely independent from bot_server.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Directories
DIR = Path(__file__).absolute().parent.parent.parent
BOT_DIR = Path(__file__).absolute().parent.parent
LOCALES_DIR = f"{BOT_DIR}/locales"
I18N_DOMAIN = "messages"


class EnvBaseSettings(BaseSettings):
    """Base settings with .env file loading."""
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class BotSettings(EnvBaseSettings):
    """Core Telegram bot settings."""
    BOT_TOKEN: str
    BOT_NAME: str = "luka_bot"
    RATE_LIMIT: int | float = 1
    DEBUG: bool = False
    
    # Internationalization
    DEFAULT_LOCALE: str = "en"  # Default language for bot interface ("en", "ru", etc.)
    SUPPORTED_LANGUAGES: list[str] = ["en", "ru"]  # List of supported language codes

    # Polling Lock (prevents multiple instances from polling simultaneously)
    POLLING_LOCK_ENABLED: bool = False  # Enable Redis-based polling lock (default: disabled)

    # Bot Privacy Mode (set via @BotFather -> Bot Settings -> Group Privacy)
    # False = DISABLED (bot sees all messages) - Current setting for this bot
    # True = ENABLED (bot only sees mentions/replies/commands)
    # This setting must match your @BotFather configuration
    BOT_PRIVACY_MODE_ENABLED: bool = False


class RedisSettings(EnvBaseSettings):
    """Redis cache and FSM storage settings."""
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASS: str | None = None
    REDIS_DATABASE: int = 0

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASS:
            return f"redis://{self.REDIS_PASS}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DATABASE}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DATABASE}"


class CamundaSettings(EnvBaseSettings):
    """Camunda BPMN engine settings."""
    ENGINE_URL: str = "http://localhost:8080/engine-rest"
    ENGINE_USERNAME: str = "demo"
    ENGINE_PASSWORD: str = "demo"
    ENGINE_USERS_GROUP_ID: str = "camunda-admin"
    
    # Double-write integration settings
    CAMUNDA_ENABLED: bool = False  # Start disabled for safer initial deployment
    CAMUNDA_MESSAGE_CORRELATION_ENABLED: bool = True
    CAMUNDA_TIMEOUT: int = 30


class S3Settings(EnvBaseSettings):
    """
    S3-compatible storage settings for file uploads.
    
    Standard AWS S3:
        S3_ACCESS_KEY_ID=AKIA...
        S3_SECRET_ACCESS_KEY=your_secret_key
        S3_BUCKET_NAME=my-bucket
        S3_REGION=us-east-1
        S3_ENDPOINT_URL=""  (leave empty)
        S3_CA_CERT_PATH=""  (leave empty)
        S3_PUBLIC_URL=https://my-bucket.s3.amazonaws.com  (or CloudFront URL)
    
    Cloudflare R2:
        S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
        S3_REGION=auto
        S3_CA_CERT_PATH=""  (leave empty)
    
    Selectel or custom S3:
        S3_ENDPOINT_URL=https://s3.selcdn.ru
        S3_REGION=ru-1
        S3_CA_CERT_PATH=./certs/selectel-root.crt  (if custom CA needed)
    """
    S3_ACCESS_KEY_ID: str = ""
    S3_SECRET_ACCESS_KEY: str = ""
    S3_ENDPOINT_URL: str = ""  # Leave empty for AWS S3, set for R2/Selectel/etc.
    S3_BUCKET_NAME: str = "luka-bot-uploads"
    S3_PUBLIC_URL: str = ""  # Public URL for accessing files
    S3_CA_CERT_PATH: str = ""  # Leave empty for AWS S3/R2, set for Selectel/custom CA
    S3_REGION: str = "us-east-1"  # AWS region (us-east-1, eu-west-1), 'auto' for R2, 'ru-1' for Selectel


class LLMSettings(EnvBaseSettings):
    """LLM settings for multiple providers - Phase 4 Agent Support."""
    
    # Ollama Provider Settings
    # IMPORTANT: OLLAMA_URL should be the BASE URL without /v1 suffix
    # Example: http://localhost:11434 (NOT http://localhost:11434/v1)
    # The /v1 suffix is added automatically for OpenAI-compatible API requests
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "gpt-oss"  # Default model for ollama provider
    OLLAMA_TIMEOUT: int = 15  # Reduced from 60s for faster failover
    
    # OpenAI Provider Settings (Phase 4+)
    OPENAI_API_KEY: str | None = None  # Set via env: OPENAI_API_KEY=sk-...
    OPENAI_MODEL_NAME: str = "gpt-4-turbo"  # Default model for openai provider
    OPENAI_TIMEOUT: int = 30  # Reduced from 60s, cloud is typically faster than local
    OPENAI_BASE_URL: str | None = None  # Optional: for OpenAI-compatible APIs
    
    # Default LLM Provider (used when thread.llm_provider is None)
    DEFAULT_LLM_PROVIDER: str = "ollama"  # "ollama" or "openai"
    DEFAULT_LLM_MODEL: str = "gpt-oss"  # Default model name
    
    # LLM Generation Parameters
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.9
    LLM_MAX_TOKENS: int = 2000
    LLM_FREQUENCY_PENALTY: float = 0.1
    LLM_PRESENCE_PENALTY: float = 0.1
    
    # Streaming Response Settings
    # Controls how LLM responses are sent to users via Telegram/Web
    # Streaming provides better UX by showing progressive responses
    STREAMING_ENABLED: bool = True  # Enable/disable streaming (recommended: True for better UX)
    STREAMING_UPDATE_INTERVAL: float = 2.0  # Minimum seconds between message edits (prevents API flooding)
    STREAMING_MIN_CHUNK_SIZE: int = 50  # Minimum character delta before updating (prevents tiny updates)
    
    # Available LLM Providers and Models Configuration
    # Format: provider -> list of available models
    AVAILABLE_PROVIDERS: dict = {
        "ollama": ["gpt-oss", "gpt-oss"],
        "openai": ["gpt-5", "gpt-4-turbo"],
    }
    
    def get_provider_display_name(self, provider: str) -> str:
        """Get human-readable provider name."""
        names = {
            "ollama": "Ollama (Local)",
            "openai": "OpenAI",
        }
        return names.get(provider, provider.capitalize())
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if provider is properly configured."""
        if provider == "ollama":
            return True  # Always available if running
        elif provider == "openai":
            return bool(self.OPENAI_API_KEY)
        return False


class ElasticsearchSettings(EnvBaseSettings):
    """Elasticsearch Knowledge Base settings."""
    
    # Connection Settings
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_ENABLED: bool = True  # Master switch for KB functionality
    ELASTICSEARCH_TIMEOUT: int = 30  # Request timeout in seconds
    ELASTICSEARCH_VERIFY_CERTS: bool = False  # SSL certificate verification
    
    # Index Settings
    ELASTICSEARCH_USER_KB_PREFIX: str = "tg-kb-user-"  # User KB index pattern
    ELASTICSEARCH_GROUP_KB_PREFIX: str = "tg-kb-group-"  # Group KB index pattern
    ELASTICSEARCH_TOPICS_PREFIX: str = "tg-topics-"  # Topics index pattern
    
    # Embedding Settings
    LLM_EMBEDDING_MODEL: str = "embeddinggemma:latest"  # Ollama embedding model
    EMBEDDING_DIMENSIONS: int = 768  # Vector dimensions for embeddings
    
    # Search Settings
    DEFAULT_MIN_SCORE: float = 0.1  # Minimum relevance score for search results
    DEFAULT_MAX_RESULTS: int = 5  # Default number of search results
    
    # RAG Settings
    RAG_SEARCH_METHOD: str = "hybrid"  # "text", "vector", or "hybrid"
    RAG_MAX_CONTEXT_MESSAGES: int = 20  # Max messages to pass to LLM for RAG
    
    # Indexing Settings
    BULK_INDEX_BATCH_SIZE: int = 500  # Batch size for bulk indexing
    INDEX_REFRESH_IMMEDIATE: bool = True  # Refresh index immediately after indexing


class LukaSettings(EnvBaseSettings):
    """Luka bot identity and personality settings."""
    
    # Bot Identity
    LUKA_NAME: str = "Luka"
    
    # Password Protection
    # When enabled, users must enter the password before accessing the bot
    LUKA_PASSWORD_ENABLED: bool = False
    LUKA_PASSWORD: str = ""  # Set your password here or via .env
    
    # Commands Configuration
    # Controls which commands are enabled in the bot
    # Available options: "start", "chat", "search", "tasks", "groups", "profile", "reset"
    # Example: LUKA_COMMANDS_ENABLED=["start", "groups", "profile", "reset"]
    # Note: "start" and "reset" are core commands and should generally remain enabled
    LUKA_COMMANDS_ENABLED: list[str] = [
        "start",
        # "chat",    # Thread management (Phase 3) - Currently disabled
        "search",  # Knowledge base search (Phase 4) - Currently disabled
        # "tasks",   # Task management/GTD (Phase 4) - Currently disabled
        "groups",   # Telegram group management
        "profile",  # User profile and settings
        # "reset",    # Clear all data
    ]
    
    # Default System Prompt (Bot Personality) - Phase 4 i18n
    # Note: {language} placeholder will be replaced with language name at runtime
    LUKA_DEFAULT_SYSTEM_PROMPT: str = """You are Luka, a helpful AI assistant integrated into Telegram.

You help users manage conversations through threads, execute workflows, complete tasks, and search knowledge bases.

Key capabilities:
- Thread management: Users can organize conversations into separate threads
- Workflow execution: You can launch and manage Camunda BPMN workflows
- Task management: Help users with GTD-style task organization (Inbox, Next, Waiting, Scheduled, Someday)
- Knowledge base search: Access and search indexed knowledge bases

Communication style:
- Be concise, friendly, and professional
- Use emojis sparingly and only when they add clarity
- Provide actionable answers
- Ask clarifying questions when needed
- Respect user's time and context

Remember: Each thread has its own conversation history. Always consider the thread context.

IMPORTANT: The user prefers to communicate in {language}. Please respond in {language} unless explicitly asked otherwise."""
    
    # Default Knowledge Base
    LUKA_DEFAULT_KNOWLEDGE_BASE: str = "default_kb"
    
    # Public Knowledge Base (for guest users and general access)
    # This index should contain publicly accessible content (docs, guides, FAQs)
    # When set, guest users will search this index instead of their personal KB
    # When empty (default), guests use personal KB (backward compatible)
    # Can be a single index name or comma-separated list of indices
    # Example: LUKA_PUBLIC_KNOWLEDGE_BASE=luka-public-kb
    # Example: LUKA_PUBLIC_KNOWLEDGE_BASE=luka-public-kb,another-public-kb
    LUKA_PUBLIC_KNOWLEDGE_BASE: str = ""

    # LangGraph Settings (Migration from Pydantic-AI)
    # Enable LangGraph-based agent architecture for unified web/Telegram experience
    LANGGRAPH_ENABLED: bool = Field(default=True, description="Enable LangGraph agent (replaces pydantic-ai)")
    LANGGRAPH_CHECKPOINTER: str = Field(default="redis", description="Checkpointer type: redis, postgres, memory")
    LANGGRAPH_DEBUG: bool = Field(default=False, description="Enable LangGraph debug mode and tracing")

    # Sub-Agent Settings (BMAD-Compatible Agent System)
    # Sub-agents are specialized AI personalities with their own system prompts, tools, and configurations
    # Examples: general_luka (default), crypto_analyst, trip_planner
    DEFAULT_SUB_AGENT_TELEGRAM: str = Field(
        default="general_luka",
        description="Default sub-agent for Telegram platform"
    )
    DEFAULT_SUB_AGENT_WEB: str = Field(
        default="general_luka",
        description="Default sub-agent for web platform (ag_ui_gateway)"
    )

    @property
    def public_knowledge_bases(self) -> list[str]:
        """
        Get list of public knowledge base indices for guest users.

        Parses LUKA_PUBLIC_KNOWLEDGE_BASE as comma-separated string or single value.

        Returns:
            List of KB index names, empty list if not configured
        """
        if not self.LUKA_PUBLIC_KNOWLEDGE_BASE:
            return []

        # Split by comma and strip whitespace
        indices = [
            index.strip()
            for index in self.LUKA_PUBLIC_KNOWLEDGE_BASE.split(',')
            if index.strip()
        ]
        return indices

    # Tools Configuration (Phase 4)
    # List of tools enabled by default for new threads
    # Can be set via environment variable as comma-separated string:
    # DEFAULT_ENABLED_TOOLS="knowledge_base,tripplanner"
    # Or as JSON array: DEFAULT_ENABLED_TOOLS='["knowledge_base","tripplanner"]'
    DEFAULT_ENABLED_TOOLS: Union[str, list[str]] = [
        "knowledge_base",  # KB search via Elasticsearch/Flow API
        # "support",         # Support and help resources
        # "youtube",         # YouTube transcript tool
        "workflow",        # Dialog workflows
        # Future: "camunda"
    ]
    
    # YouTube Tool Configuration (Phase 4+)
    YOUTUBE_API_KEY: str | None = None  # Set via env: YOUTUBE_API_KEY=...
    YOUTUBE_TRANSCRIPT_ENABLED: bool = True  # Enable YouTube transcript tool
    
    # Legacy KB Settings (kept for backward compatibility)
    ELASTICSEARCH_INDEX: str = "knowledge_base"
    KNOWLEDGE_BASE_GROUP_ID: str | None = None  # Default KB group for queries

    # Default Groups and Channels
    # These are automatically provisioned to all users when they first interact with the bot
    # Set LUKA_DEFAULT_GROUP_ID to enable default group provisioning
    # Set LUKA_DEFAULT_CHANNEL_ID to enable default channel provisioning
    LUKA_DEFAULT_GROUP_ID: int | None = None  # Telegram group ID (negative integer)
    LUKA_DEFAULT_GROUP_TITLE: str = ""  # Display name for the default group
    LUKA_DEFAULT_GROUP_INVITE_LINK: str = ""  # Invite link for default group (e.g., https://t.me/groupname or t.me/+invitecode)
    LUKA_DEFAULT_CHANNEL_ID: int | None = None  # Telegram channel ID (negative integer)
    LUKA_DEFAULT_CHANNEL_TITLE: str = ""  # Display name for the default channel
    LUKA_DEFAULT_CHANNEL_INVITE_LINK: str = ""  # Invite link for default channel (e.g., https://t.me/channelname or t.me/+invitecode)

    @property
    def has_default_group(self) -> bool:
        """Check if default group is configured."""
        return self.LUKA_DEFAULT_GROUP_ID is not None and self.LUKA_DEFAULT_GROUP_TITLE != ""

    @property
    def has_default_channel(self) -> bool:
        """Check if default channel is configured."""
        return self.LUKA_DEFAULT_CHANNEL_ID is not None and self.LUKA_DEFAULT_CHANNEL_TITLE != ""

    def get_default_group_kb_index(self) -> str | None:
        """
        Get KB index name for default group.

        Returns:
            KB index name in format: tg-kb-group-{group_id} or None if not configured
        """
        if not self.has_default_group:
            return None
        # Remove leading minus sign and convert to positive string
        group_id_str = str(abs(self.LUKA_DEFAULT_GROUP_ID))
        return f"{self.ELASTICSEARCH_GROUP_KB_PREFIX}{group_id_str}"

    def get_default_channel_kb_index(self) -> str | None:
        """
        Get KB index name for default channel.

        Returns:
            KB index name in format: tg-kb-group-{channel_id} or None if not configured
        """
        if not self.has_default_channel:
            return None
        # Remove leading minus sign and convert to positive string
        channel_id_str = str(abs(self.LUKA_DEFAULT_CHANNEL_ID))
        return f"{self.ELASTICSEARCH_GROUP_KB_PREFIX}{channel_id_str}"

    def get_default_group_invite_link(self) -> str | None:
        """
        Get invite link for default group.

        Returns:
            Invite link URL or None if not configured
        """
        if not self.has_default_group:
            return None
        return self.LUKA_DEFAULT_GROUP_INVITE_LINK if self.LUKA_DEFAULT_GROUP_INVITE_LINK else None

    def get_default_channel_invite_link(self) -> str | None:
        """
        Get invite link for default channel.

        Returns:
            Invite link URL or None if not configured
        """
        if not self.has_default_channel:
            return None
        return self.LUKA_DEFAULT_CHANNEL_INVITE_LINK if self.LUKA_DEFAULT_CHANNEL_INVITE_LINK else None


class FlowAPISettings(EnvBaseSettings):
    """Flow API settings for user authentication."""
    FLOW_API_URL: str = "http://localhost:8000"
    FLOW_API_SYS_KEY: str = "secret"
    FLOW_API_JWT_SECRET: str = "your-secret-key-here"


class WebhookSettings(EnvBaseSettings):
    """Webhook settings (Phase 1: polling only)."""
    USE_WEBHOOK: bool = False
    WEBHOOK_BASE_URL: str = "https://example.ngrok-free.app"
    WEBHOOK_PATH: str = "/webhook"
    WEBHOOK_SECRET: str = "change-me"
    # WEBHOOK_HOST should be a hostname or IP (e.g., "localhost", "0.0.0.0", "127.0.0.1")
    # NOT a full URL - this is used for binding the TCP socket
    WEBHOOK_HOST: str = "0.0.0.0"  # Bind to all interfaces by default
    WEBHOOK_PORT: int = 8080

    @property
    def webhook_url(self) -> str:
        if self.USE_WEBHOOK:
            return f"{self.WEBHOOK_BASE_URL}{self.WEBHOOK_PATH}"
        return f"http://{self.WEBHOOK_HOST}:{self.WEBHOOK_PORT}{self.WEBHOOK_PATH}"


class MetricsSettings(EnvBaseSettings):
    """Prometheus metrics settings."""
    # Enable metrics endpoint (enabled by default for monitoring)
    METRICS_ENABLED: bool = True
    # Port for metrics server in polling mode (separate from webhook)
    METRICS_PORT: int = 9090
    # Host to bind metrics server (polling mode only)
    METRICS_HOST: str = "0.0.0.0"


class WarehouseSettings(EnvBaseSettings):
    """Warehouse API settings for task event subscriptions via WebSocket."""
    WAREHOUSE_API_URL: str = "http://localhost:8001"
    WAREHOUSE_WS_URL: str = "ws://localhost:8001"
    WAREHOUSE_ENABLED: bool = True
    
    # WebSocket connection tuning
    WAREHOUSE_WS_ENABLED: bool = True  # Enable/disable WebSocket connections entirely
    WAREHOUSE_WS_HEARTBEAT_INTERVAL: int = 30  # Ping interval in seconds
    WAREHOUSE_WS_RECONNECT_DELAY: int = 5  # Initial reconnect delay in seconds
    WAREHOUSE_WS_MAX_RECONNECT_ATTEMPTS: int = -1  # -1 = infinite reconnection attempts


class AGUISettings(EnvBaseSettings):
    """
    AG-UI Gateway settings for REST API and WebSocket endpoints.
    
    When enabled, adds AG-UI compatible endpoints to the luka_bot FastAPI app:
    - REST API: /api/auth, /api/catalog, /api/profile, /api/files, /api/agent/luka
    - WebSocket: /ws/chat
    - Health: /health/ping, /health/status
    
    This allows using luka_bot with AG-UI Dojo frontend or as a standalone service.
    """
    # Enable AG-UI endpoints (default: True for integrated mode)
    AG_UI_ENABLED: bool = True
    
    # CORS settings for AG-UI (when DEBUG=True)
    AG_UI_ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",  # Dojo frontend
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Alt frontend
    ]
    
    # Dojo Next.js app proxy settings
    AG_UI_DOJO_ENABLED: bool = False
    AG_UI_DOJO_URL: str = "http://webchatbot-app:3000"  # Next.js server URL
    
    # PostgreSQL (optional, reserved for future AG-UI features)
    AG_UI_POSTGRES_ENABLED: bool = False
    AG_UI_POSTGRES_HOST: str = "postgres"
    AG_UI_POSTGRES_PORT: int = 5432
    AG_UI_POSTGRES_USER: str = "luka"
    AG_UI_POSTGRES_PASSWORD: str = ""
    AG_UI_POSTGRES_DB: str = "luka_bot"
    
    @property
    def ag_ui_postgres_url(self) -> str:
        """PostgreSQL connection URL."""
        if not self.AG_UI_POSTGRES_ENABLED:
            return ""
        return (
            f"postgresql+asyncpg://{self.AG_UI_POSTGRES_USER}:{self.AG_UI_POSTGRES_PASSWORD}"
            f"@{self.AG_UI_POSTGRES_HOST}:{self.AG_UI_POSTGRES_PORT}/{self.AG_UI_POSTGRES_DB}"
        )


class Settings(
    BotSettings,
    RedisSettings,
    CamundaSettings,
    S3Settings,
    LLMSettings,
    ElasticsearchSettings,
    LukaSettings,
    FlowAPISettings,
    WebhookSettings,
    MetricsSettings,
    WarehouseSettings,
    AGUISettings,
):
    """Combined settings for luka_bot."""
    pass


# Global settings instance
settings = Settings()
