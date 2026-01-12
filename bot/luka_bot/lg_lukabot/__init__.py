"""LangGraph integration for Luka Bot - Unified architecture for web and Telegram.

This module provides a unified LangGraph-based agent architecture that works
across both the web interface (ag_ui_gateway) and Telegram bot (luka_bot).

Key Components:
- state.py: Unified AgentState schema with automatic checkpointing
- tools.py: LangChain tool wrappers for existing bot services
- nodes.py: Graph nodes (agent, tools, workflow updates)
- graph.py: Graph builder with routing logic
- checkpointer.py: Redis-based state persistence

Usage:
    from luka_bot.lg_lukabot import AgentState, create_tools_for_user

    # Initialize state
    state = AgentState(
        messages=[],
        user_id=123,
        thread_id="user_123",
        language="en",
        platform="telegram",
        is_guest=False,
        knowledge_bases=["tg-kb-user-123"],
        enabled_tools=["knowledge_base", "workflow", "support"],
        active_workflow=None,
        workflow_step=None,
        workflow_progress=0.0,
        workflow_suggestions=[],
        tool_results={},
        next_action=None,
        ui_context={},
        metadata={}
    )

    # Create tools for user
    tools = create_tools_for_user(
        user_id=123,
        thread_id="user_123",
        knowledge_bases=["tg-kb-user-123"],
        enabled_tools=["knowledge_base", "workflow"]
    )

Migration Status:
- [x] Phase 1: Infrastructure setup (module structure, config)
- [x] Phase 2: Unified state schema (AgentState)
- [ ] Phase 3: Tool conversion (LangChain wrappers)
- [ ] Phase 4: Graph nodes (agent, workflow_update)
- [ ] Phase 5: Graph builder with checkpointing
- [ ] Phase 6: Reply keyboard updates
- [ ] Phase 7: Feature flag enablement
"""

# Lazy imports to avoid circular dependencies
# Import these directly from their modules when needed:
# - from luka_bot.lg_lukabot.checkpointer import create_redis_checkpointer, get_checkpointer
# - from luka_bot.lg_lukabot.graph import create_telegram_agent_graph, get_telegram_agent_graph
# - from luka_bot.lg_lukabot.integration import stream_langgraph_agent, invoke_langgraph_agent
# - from luka_bot.lg_lukabot.state import AgentState
# - from luka_bot.lg_lukabot.tools import create_tools_for_user, map_config_tools_to_langgraph_tools

__all__ = [
    "AgentState",
    "create_tools_for_user",
    "map_config_tools_to_langgraph_tools",
    "create_redis_checkpointer",
    "get_checkpointer",
    "create_telegram_agent_graph",
    "get_telegram_agent_graph",
    "create_state_from_telegram_message",
    "extract_response_from_state",
    "invoke_langgraph_agent",
    "stream_langgraph_agent",
]

