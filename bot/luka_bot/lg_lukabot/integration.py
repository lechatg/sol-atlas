"""Integration layer between Telegram handlers and LangGraph agent.

This module provides utilities to bridge the Telegram bot handlers (aiogram)
with the LangGraph agent execution.
"""

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from luka_bot.lg_lukabot.graph import get_telegram_agent_graph
from luka_bot.lg_lukabot.state import AgentState
from luka_bot.services.thread_service import get_thread_service
from luka_bot.services.user_profile_service import get_user_profile_service


async def create_state_from_telegram_message(
    message_text: str,
    user_id: int,
    thread_id: str,
    language: str = "en",
    enabled_tools: list[str] | None = None,
    active_workflow: str | None = None,
    platform: str = "telegram",
    group_bot_prompt: str | None = None,
) -> AgentState:
    """
    Create AgentState from Telegram message.

    This function explicitly loads checkpoint state and merges messages to ensure
    conversation history is preserved across messages.

    Args:
        message_text: User's message text
        user_id: Telegram user ID
        thread_id: Thread ID for conversation context
        language: User's language
        enabled_tools: List of tool names to enable
        active_workflow: Active workflow domain (if any)
        platform: Platform identifier (telegram, telegram_group)
        group_bot_prompt: Custom bot personality for this group (if any)

    Returns:
        AgentState ready for graph invocation with merged checkpoint history
    """
    # Get knowledge bases from thread (if exists) or user profile
    # Priority: Thread KBs > User profile KB > Default KB
    thread_service = get_thread_service()
    thread = await thread_service.get_thread(thread_id)
    
    if thread and thread.knowledge_bases:
        # Thread has specific KBs configured (e.g., group KB + user KB)
        knowledge_bases = thread.knowledge_bases
        logger.debug(f"📚 Using thread KBs: {knowledge_bases}")
    else:
        # Fallback to user's personal KB
        profile_service = get_user_profile_service()
        user_profile = await profile_service.get_profile(user_id)
        
        # Handle case where profile doesn't exist (e.g., in tests)
        if user_profile and hasattr(user_profile, 'kb_index'):
            knowledge_bases = [user_profile.kb_index]
        else:
            # Default KB index for testing/new users
            knowledge_bases = [f"tg-kb-user-{user_id}"]
        logger.debug(f"📚 Using user KB: {knowledge_bases}")

    # Load previous checkpoint state if it exists
    checkpoint_messages = []
    checkpoint_state = None
    
    try:
        from luka_bot.lg_lukabot.checkpointer import get_checkpointer
        checkpointer = get_checkpointer()
        
        if checkpointer:
            config = {"configurable": {"thread_id": thread_id}}
            
            # Try to get checkpoint (RedisSaver may support async or sync)
            # Use aget/get which returns a dict with channel_values, consistent with tools.py
            if hasattr(checkpointer, 'aget'):
                checkpoint = await checkpointer.aget(config)
            elif hasattr(checkpointer, 'get'):
                checkpoint = checkpointer.get(config)
            else:
                checkpoint = None
            
            if checkpoint and checkpoint.get("channel_values"):
                checkpoint_state = checkpoint["channel_values"]
                checkpoint_messages = checkpoint_state.get("messages", [])
                logger.debug(f"📚 Loaded {len(checkpoint_messages)} messages from checkpoint for thread {thread_id}")
            else:
                logger.debug(f"📝 No checkpoint found for thread {thread_id}, starting new conversation")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load checkpoint: {e}, starting with new conversation")
    
    # Merge messages: checkpoint messages + new user message
    # Use add_messages reducer logic: append new messages to existing ones
    from langgraph.graph.message import add_messages
    all_messages = checkpoint_messages if checkpoint_messages else []
    new_message = HumanMessage(content=message_text)
    
    # Apply add_messages reducer to merge
    merged_messages = add_messages(all_messages, [new_message])
    logger.debug(f"📝 Merged messages: {len(checkpoint_messages)} existing + 1 new = {len(merged_messages)} total")

    # Check for active workflow if not explicitly provided
    workflow_step = None
    workflow_suggestions = []
    
    # If we loaded checkpoint state, use its workflow info if not explicitly provided
    if checkpoint_state and active_workflow is None:
        active_workflow = checkpoint_state.get("active_workflow")
        workflow_step = checkpoint_state.get("workflow_step")
        workflow_suggestions = checkpoint_state.get("workflow_suggestions", [])
    
    if active_workflow is None:
        from luka_bot.services import get_workflow_service, get_workflow_discovery_service
        
        try:
            workflow_service = get_workflow_service()
            discovery_service = get_workflow_discovery_service()
            await discovery_service.initialize()
            
            # Check all domains for active workflows
            available_workflows = await discovery_service.get_available_workflows()
            for domain in available_workflows.keys():
                wf_status = await workflow_service.get_active_workflow_for_user(user_id, domain)
                if wf_status:
                    active_workflow = wf_status.domain
                    workflow_step = wf_status.current_step
                    logger.info(f"📋 Loaded active workflow from service: {active_workflow} (step: {workflow_step})")
                    break
        except Exception as e:
            logger.debug(f"No active workflow found: {e}")
    
    # Workflow suggestions will be dynamically generated by LLM in update_workflow_suggestions_telegram node
    # We start with empty suggestions - they will be populated based on conversation context and workflow progress
    # This allows suggestions to adapt to the actual conversation flow rather than using static BPMN definitions
    if active_workflow:
        logger.debug(f"📋 Workflow suggestions will be dynamically generated for step '{workflow_step}'")

    # Build metadata with group_bot_prompt if provided
    metadata = checkpoint_state.get("metadata", {}) if checkpoint_state else {}
    if group_bot_prompt:
        metadata["group_bot_prompt"] = group_bot_prompt
    
    # Use checkpoint state values as defaults, but allow overrides
    if checkpoint_state:
        # Merge checkpoint state with new values (new values take precedence)
        # IMPORTANT: Clear conversation_suggestions when a new user message arrives
        # because suggestions should be regenerated based on the new response, not the old one
        state: AgentState = {
            "messages": merged_messages,
            "user_id": user_id,
            "thread_id": thread_id,
            "language": checkpoint_state.get("language", language),
            "platform": platform,
            "is_guest": False,
            "knowledge_bases": checkpoint_state.get("knowledge_bases", knowledge_bases),
            "enabled_tools": enabled_tools or checkpoint_state.get("enabled_tools", ["knowledge_base", "support", "workflow"]),
            "active_workflow": active_workflow or checkpoint_state.get("active_workflow"),
            "workflow_step": workflow_step or checkpoint_state.get("workflow_step"),
            "workflow_progress": checkpoint_state.get("workflow_progress", 0.0),
            "workflow_suggestions": workflow_suggestions or checkpoint_state.get("workflow_suggestions", []),
            "conversation_suggestions": [],  # Clear old suggestions - will be regenerated for new response
            "tool_results": checkpoint_state.get("tool_results", {}),
            "next_action": None,
            "ui_context": checkpoint_state.get("ui_context", {}),
            "metadata": metadata,
        }
    else:
        # No checkpoint - create new state
        state: AgentState = {
            "messages": merged_messages,
            "user_id": user_id,
            "thread_id": thread_id,
            "language": language,
            "platform": platform,
            "is_guest": False,
            "knowledge_bases": knowledge_bases,
            "enabled_tools": enabled_tools or ["knowledge_base", "support", "workflow"],
            "active_workflow": active_workflow,
            "workflow_step": workflow_step,
            "workflow_progress": 0.0,
            "workflow_suggestions": workflow_suggestions,
            "conversation_suggestions": [],
            "tool_results": {},
            "next_action": None,
            "ui_context": {},
            "metadata": metadata,
        }

    return state


async def extract_response_from_state(state: AgentState) -> dict[str, Any]:
    """
    Extract response data from final AgentState.

    Args:
        state: Final state after graph execution

    Returns:
        Dict with:
        - text: Bot's response text
        - suggestions: Workflow suggestions (if any)
        - keyboard_update_needed: Whether reply keyboard should be updated
    """
    # Get last AI message
    messages = state.get("messages", [])
    last_message = None

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_message = msg
            break

    # Extract content and convert to string (content might be int, list, etc.)
    raw_content = last_message.content if last_message else "Sorry, I couldn't generate a response."
    response_text = str(raw_content) if raw_content else "Sorry, I couldn't generate a response."
    
    # Clean up: Remove any "SUGGESTIONS:[...]" that the LLM might have included in its response
    # This can happen if the LLM sees the tool result and incorporates it into its answer
    import re
    if response_text and isinstance(response_text, str) and "SUGGESTIONS:" in response_text:
        # Remove the SUGGESTIONS:[...] part using regex
        response_text = re.sub(r'SUGGESTIONS:\[.*?\]', '', response_text, flags=re.DOTALL).strip()
        logger.debug(f"Cleaned SUGGESTIONS marker from response text")

    # Extract suggestions (prefer workflow suggestions, fallback to conversation suggestions)
    workflow_suggestions = state.get("workflow_suggestions", [])
    conversation_suggestions = state.get("conversation_suggestions", [])
    
    # Use workflow suggestions if available, otherwise use conversation suggestions
    suggestions = workflow_suggestions if workflow_suggestions else conversation_suggestions
    
    # Check if keyboard update is needed from ui_context OR if we have conversation suggestions
    keyboard_update_needed = (
        state.get("ui_context", {}).get("keyboard_update_needed", False) 
        or bool(conversation_suggestions)
    )

    return {
        "text": response_text,
        "suggestions": suggestions,
        "keyboard_update_needed": keyboard_update_needed,
        "workflow_step": state.get("workflow_step"),
        "workflow_progress": state.get("workflow_progress", 0.0),
    }


async def invoke_langgraph_agent(
    message_text: str,
    user_id: int,
    thread_id: str,
    language: str = "en",
    enabled_tools: list[str] | None = None,
    active_workflow: str | None = None,
    platform: str = "telegram",
) -> dict[str, Any]:
    """
    High-level API to invoke LangGraph agent.

    This is the main entry point for Telegram handlers.

    Args:
        message_text: User's message
        user_id: Telegram user ID
        thread_id: Thread ID for context
        language: User's language
        enabled_tools: Tools to enable
        active_workflow: Active workflow (if any)

    Returns:
        Response dict with text, suggestions, and metadata
    """
    logger.info(f"🤖 Invoking LangGraph agent for user {user_id}, thread {thread_id}")

    try:
        # Create initial state
        state = await create_state_from_telegram_message(
            message_text=message_text,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            enabled_tools=enabled_tools,
            active_workflow=active_workflow,
            platform=platform,
        )

        # Get graph
        graph = await get_telegram_agent_graph()

        # Configure checkpointing
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            }
        }

        # Invoke graph
        final_state = await graph.ainvoke(state, config=config)

        # Extract response
        response = await extract_response_from_state(final_state)

        # Note: LangGraph checkpointer automatically saves state after each node execution
        # No need for manual history saving

        logger.info(f"✅ LangGraph agent completed for user {user_id}")
        return response

    except Exception as e:
        logger.error(f"❌ Error invoking LangGraph agent: {e}", exc_info=True)
        return {
            "text": "Sorry, I encountered an error processing your message.",
            "suggestions": [],
            "keyboard_update_needed": False,
        }


async def stream_langgraph_agent(
    message_text: str,
    user_id: int,
    thread_id: str,
    language: str = "en",
    enabled_tools: list[str] | None = None,
    active_workflow: str | None = None,
    platform: str = "telegram",
    group_bot_prompt: str | None = None,
) -> AsyncIterator[str | dict[str, Any]]:
    """
    Stream LangGraph agent execution with token-by-token responses.

    Args:
        platform: Platform identifier (telegram for private chats, telegram_group for groups)
                  Groups skip suggestion generation since reply keyboards don't work there.
        group_bot_prompt: Custom bot personality for this group (if any).
                         Injected into system prompt with safety wrapper.

    Yields:
        - str: Text tokens from LLM
        - dict: Tool notifications {"type": "tool_notification", "text": "🔧", "tool_name": "..."}
        - dict: Final state {"type": "final_state", "state": {...}} (yielded last)
    """
    logger.info(f"🤖 Streaming LangGraph agent for user {user_id}, thread {thread_id}")

    try:
        # Create initial state
        state = await create_state_from_telegram_message(
            message_text=message_text,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            enabled_tools=enabled_tools,
            active_workflow=active_workflow,
            platform=platform,
            group_bot_prompt=group_bot_prompt,
        )

        # Get graph
        graph = await get_telegram_agent_graph()

        # Configure checkpointing with thread_id for state isolation
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": str(user_id),  # Optional metadata
            }
        }

        # Stream state updates from graph
        # LangGraph checkpointer will automatically load previous state from checkpoint
        # and save state after each node execution
        full_response = ""
        last_message_count = len(state.get("messages", []))
        final_state = None  # Track final state for yielding at end
        
        logger.debug("About to call graph.astream() with checkpointing config...")
        try:
            async for state_update in graph.astream(state, config=config, stream_mode="values"):
                logger.debug(f"Got state_update with keys: {list(state_update.keys())}")
                final_state = state_update  # Update final state on each iteration
                messages = state_update.get("messages", [])
                
                # Check if there are new messages from the agent
                if len(messages) > last_message_count:
                    # Get the latest message
                    latest_message = messages[-1]
                    
                    # Only process AI messages (skip HumanMessage, ToolMessage, etc.)
                    from langchain_core.messages import AIMessage
                    if not isinstance(latest_message, AIMessage):
                        last_message_count = len(messages)
                        continue
                    
                    # If it's an AI message, extract its content
                    if hasattr(latest_message, "content") and latest_message.content:
                        # Convert content to string (it might be int, list, etc.)
                        new_content = str(latest_message.content) if latest_message.content else ""
                        
                        # Clean up: Remove any "SUGGESTIONS:[...]" that the LLM included in its response
                        # This must happen BEFORE yielding to prevent it from reaching the user
                        import re
                        if new_content and "SUGGESTIONS:" in new_content:
                            new_content = re.sub(r'SUGGESTIONS:\[.*?\]', '', new_content, flags=re.DOTALL).strip()
                            logger.debug(f"🧹 Cleaned SUGGESTIONS marker from streaming content")
                        
                        # If this is new content we haven't yielded yet
                        if new_content and not full_response:
                            full_response = new_content
                            logger.info(f"💬 LLM response START: {new_content[:200]}...")
                            yield new_content
                        elif new_content and new_content != full_response:
                            # Streaming update - yield delta
                            delta = new_content[len(full_response):]
                            full_response = new_content
                            if delta:
                                yield delta
                    
                    # Check for tool calls in the latest message
                    if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                        # Ensure tool_calls is iterable (it might be an int or other type)
                        try:
                            if isinstance(latest_message.tool_calls, (list, tuple)):
                                for tool_call in latest_message.tool_calls:
                                    # Ensure tool_call is a dict before calling .get()
                                    if isinstance(tool_call, dict):
                                        tool_name = tool_call.get("name", "unknown")
                                    else:
                                        tool_name = str(tool_call) if tool_call else "unknown"
                                    tool_emoji = _get_tool_emoji(tool_name)
                                    yield {
                                        "type": "tool_notification",
                                        "text": tool_emoji,
                                        "tool_name": tool_name,
                                    }
                        except (TypeError, AttributeError) as e:
                            logger.debug(f"⚠️ Could not iterate tool_calls: {e}, type: {type(latest_message.tool_calls)}")
                    
                    last_message_count = len(messages)
        except NotImplementedError as nie:
            logger.error(f"NotImplementedError in graph.astream(): {nie}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        
        # Clean up full_response before saving (in case it wasn't cleaned during streaming)
        import re
        if full_response and isinstance(full_response, str) and "SUGGESTIONS:" in full_response:
            full_response = re.sub(r'SUGGESTIONS:\[.*?\]', '', full_response, flags=re.DOTALL).strip()
            logger.debug(f"🧹 Cleaned SUGGESTIONS marker from final response before saving")
        
        # Log the complete final response for debugging
        if full_response:
            logger.info(f"✅ LLM final response ({len(full_response)} chars): {full_response[:300]}...")
        
        # Yield final state for handler to capture (for keyboard building, etc.)
        if final_state:
            yield {
                "type": "final_state",
                "state": final_state,
            }
            logger.debug(f"📤 Yielded final state with conversation_suggestions: {final_state.get('conversation_suggestions', [])}")
        
        # Note: LangGraph checkpointer automatically saves state after each node execution
        # No need for manual history saving

        logger.info(f"✅ LangGraph streaming completed for user {user_id}")

    except Exception as e:
        logger.error(f"❌ Error streaming LangGraph agent: {e}", exc_info=True)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error repr: {repr(e)}")
        logger.error(f"Error str: {str(e)}")
        logger.error(f"Error args: {e.args}")
        
        # Print full traceback for debugging
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        yield f"Sorry, I encountered an error: {type(e).__name__}"


def _get_tool_emoji(tool_name: str) -> str:
    """Get emoji for tool notification."""
    tool_emojis = {
        "search_knowledge_base": "🔍",
        "execute_workflow": "📋",
        "get_support_info": "ℹ️",
        "connect_to_support": "👤",
        "get_youtube_transcript": "📺",
    }
    return tool_emojis.get(tool_name, "🔧")


async def update_thread_activity(thread_id: str) -> None:
    """
    Update thread activity timestamp and message count.
    
    This is a lightweight helper to update thread metadata.
    LangGraph checkpointer handles state persistence, but thread metadata
    (like message_count) is still useful for statistics and thread management.
    
    Args:
        thread_id: Thread ID to update
    """
    try:
        thread_service = get_thread_service()
        thread = await thread_service.get_thread(thread_id)
        if thread:
            thread.update_activity()  # Increments message_count and updates updated_at
            await thread_service.update_thread(thread)
            logger.debug(f"📊 Updated thread activity: {thread_id} (message_count: {thread.message_count})")
    except Exception as e:
        logger.debug(f"⚠️  Failed to update thread activity: {e}")


# Note: save_response_to_thread() removed - LangGraph checkpointer now handles state persistence automatically
# Use update_thread_activity() if thread metadata updates are needed
