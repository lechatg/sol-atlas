"""LangGraph graph builder for Telegram bot.

Defines the agent execution flow:
START → agent → [tools] → workflow_update → END
             ↘ workflow_update → END
"""

from langgraph.graph import END, StateGraph
from loguru import logger

from luka_bot.lg_lukabot.checkpointer import create_redis_checkpointer
from luka_bot.lg_lukabot.nodes import (
    call_agent_telegram,
    execute_tools_telegram,
    update_workflow_suggestions_telegram,
)
from luka_bot.lg_lukabot.state import AgentState


async def create_telegram_agent_graph():
    """
    Create LangGraph agent for Telegram bot.

    Graph structure:
    ```
    START
      ↓
    agent (call LLM with tools)
      ├─→ tools (if LLM calls tools)
      │    ↓
      │  agent (back to LLM with tool results)
      │
      └─→ workflow_update (if no tool calls)
           ↓
          END
    ```

    The graph uses conditional routing based on the `next_action` field
    in the state, which is set by the agent node.

    Returns:
        Compiled LangGraph with Redis checkpointing
    """
    logger.info("🔨 Building Telegram LangGraph agent...")

    # Create state graph
    graph = StateGraph(AgentState)

    # Add nodes
    logger.debug("Adding agent node")
    graph.add_node("agent", call_agent_telegram)

    logger.debug("Adding tool node")
    # Use custom tool execution node that recreates tools dynamically per user
    graph.add_node("tools", execute_tools_telegram)

    logger.debug("Adding workflow_update node")
    graph.add_node("workflow_update", update_workflow_suggestions_telegram)
    
    logger.debug("Adding ensure_conversation_suggestions node")
    from luka_bot.lg_lukabot.nodes import ensure_conversation_suggestions_telegram
    graph.add_node("ensure_suggestions", ensure_conversation_suggestions_telegram)

    # Set entry point
    graph.set_entry_point("agent")

    # Add conditional edges from agent
    def route_after_agent(state: AgentState) -> str:
        """
        Route to next node based on agent's decision.

        Returns:
            "tools" if LLM wants to call tools
            "workflow_update" if LLM provided direct response
        """
        next_action = state.get("next_action")
        logger.debug(f"Routing after agent: next_action={next_action}")

        if next_action == "tools":
            return "tools"
        else:
            return "workflow_update"

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "workflow_update": "workflow_update"},
    )

    # After tools, go back to agent to process results
    graph.add_edge("tools", "agent")

    # After workflow_update, ensure conversation suggestions are present
    graph.add_edge("workflow_update", "ensure_suggestions")
    
    # After ensuring suggestions, end
    graph.add_edge("ensure_suggestions", END)

    # Compile graph with Redis checkpointer
    logger.info("✅ Compiling graph with Redis checkpointing...")
    
    # Create and use Redis checkpointer for automatic state persistence
    checkpointer = await create_redis_checkpointer()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    logger.info("✅ Telegram LangGraph agent created successfully!")
    return compiled_graph


# Singleton instance
_telegram_agent_graph = None


async def get_telegram_agent_graph():
    """
    Get or create the Telegram agent graph singleton.

    Returns:
        Compiled LangGraph instance
    """
    global _telegram_agent_graph

    if _telegram_agent_graph is None:
        _telegram_agent_graph = await create_telegram_agent_graph()

    return _telegram_agent_graph

