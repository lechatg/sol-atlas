"""Unified LangGraph state schema for web and Telegram platforms.

This module defines the AgentState TypedDict that serves as the state schema
for both the web interface (ag_ui_gateway) and Telegram bot (luka_bot).

The state is automatically persisted by LangGraph's checkpointer, eliminating
the need for manual Redis history management that was required with pydantic-ai.

Key Features:
- Automatic message history via add_messages reducer
- Platform-agnostic (web vs telegram identified via 'platform' field)
- Workflow state tracking (active_workflow, workflow_step, suggestions)
- UI context for dynamic keyboard/button updates
- Tool execution results
- Routing logic for conditional edges
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Unified state for conversational agent (web + Telegram).

    This state is automatically persisted by LangGraph's checkpointer,
    eliminating the need for manual Redis history management.

    The same state schema is used for both platforms with platform-specific
    adaptations in rendering (CopilotKit protocol for web, Telegram messages for bot).

    Attributes:
        messages: Conversation history with automatic reducer
        user_id: Telegram user ID or web user ID
        thread_id: Conversation thread identifier
        language: User's preferred language (en, ru, etc.)
        platform: Platform identifier (web or telegram)
        is_guest: Whether user is guest (web only)
        knowledge_bases: List of KB indices user can access
        enabled_tools: List of tool names available to agent
        active_workflow: Currently active workflow domain (if any)
        workflow_step: Current step in active workflow
        workflow_progress: Progress percentage (0.0 to 1.0)
        workflow_suggestions: Dynamic suggestions from current workflow step
        tool_results: Results from tool executions
        next_action: Routing decision (agent, tools, end)
        ui_context: Platform-agnostic UI state (keyboard updates, etc.)
        metadata: Additional metadata for platform-specific features
    """

    # Message history (automatic reducer via add_messages)
    messages: Annotated[list[BaseMessage], add_messages]

    # User context
    user_id: int
    thread_id: str
    language: str
    platform: Literal["web", "telegram"]  # Platform identifier
    is_guest: bool  # Only for web platform

    # Knowledge base configuration
    knowledge_bases: list[str]
    enabled_tools: list[str]

    # Workflow state (UNIFIED for both platforms)
    active_workflow: str | None  # e.g., "sol_atlas_onboarding"
    workflow_step: str | None  # e.g., "hook", "situation"
    workflow_progress: float  # 0.0 to 1.0
    workflow_suggestions: list[str]  # Dynamic suggestions from current step
    conversation_suggestions: list[str]  # Dynamic suggestions for normal conversation (no workflow)

    # Tool execution results
    tool_results: dict[str, Any]

    # Routing logic
    next_action: Literal["agent", "tools", "end"] | None

    # UI state (platform-agnostic)
    # - Web: Sent to frontend via STATE_SNAPSHOT
    # - Telegram: Signals reply keyboard update needed
    ui_context: dict[str, Any]

    # Metadata
    metadata: dict[str, Any]

