"""
⚠️ DEPRECATED: This LLM Service is being replaced by LangGraph ⚠️

**Migration Status**: ✅ COMPLETE - All handlers migrated to LangGraph
**Replacement**: Use `luka_bot.lg_lukabot.integration.stream_langgraph_agent` instead

**What Changed:**
- ✅ Direct messages (DM) → Migrated to LangGraph
- ✅ Group mentions/replies → Migrated to LangGraph
- ✅ Group-aware DM (via /groups) → Migrated to LangGraph
- ✅ Voice messages → Migrated to LangGraph
- ✅ Forwarded messages → Migrated to LangGraph

**Why LangGraph:**
- Unified architecture across web and Telegram
- Better tool management and execution
- Built-in conversation memory (Redis checkpoints)
- Workflow integration with conversation suggestions
- More flexible and maintainable

**For New Development:**
Use `stream_langgraph_agent()` from `luka_bot.lg_lukabot.integration` instead of this service.

**Legacy Use:**
This service is retained ONLY for internal Redis history methods (_get_history, _clear_history).

---

OLD DOCUMENTATION:
LLM Service - Agent-based with pydantic-ai support.
Phase 4: Integrated with pydantic-ai agents for tool support.
Phase 4: i18n support - language injection into system prompts.
Phase 4: Support tools enabled (KB, YouTube in Phase 4+).
Replaces direct Ollama calls with agent factory pattern.
"""
import asyncio
from typing import AsyncIterator, List, Dict, Optional, TYPE_CHECKING
from loguru import logger
import re

from luka_bot.core.config import settings
from luka_bot.core.loader import redis_client
from luka_bot.services.thread_service import get_thread_service
from luka_bot.agents import (
    ConversationContext,
    create_static_agent_with_basic_tools,
)
from luka_bot.agents import youtube_tools  # heuristic fallback for YouTube

if TYPE_CHECKING:
    from luka_bot.models.thread import Thread


def _get_tool_notification(tool_name: str) -> dict:
    """
    Get emoji notification for tool execution.
    
    Returns a dict to enable message editing in streaming handler.
    Emoji-only (no text) for universal i18n.
    
    Based on old bot_server pattern from streaming_service.py:232-242
    """
    # Map tool names to emoji-only notifications
    emoji_map = {
        'search_knowledge_base': '🔍',
        'get_youtube_transcript': '📺',
        'execute_task': '📋',
        'get_support_info': '🎧',
        'connect_to_support': '🎧',
    }
    
    # Try exact match first
    emoji = emoji_map.get(tool_name)
    
    # Try partial match
    if not emoji:
        for key, value in emoji_map.items():
            if key in tool_name or tool_name in key:
                emoji = value
                break
    
    # Fallback to generic tool emoji
    if not emoji:
        emoji = '🔧'
    
    return {
        "type": "tool_notification",
        "text": emoji,
        "tool_name": tool_name
    }


class LLMService:
    """
    Service for LLM interactions via pydantic-ai agents.
    
    Phase 4 Features:
    - Agent-based streaming with tool support
    - Conversation history (Redis via ConversationHistory)
    - Support tools (get_support_info, connect_to_support)
    - Per-thread configuration (provider, model, system prompt)
    
    Phase 5+:
    - Camunda tools integration
    - Knowledge base tools
    - YouTube tools
    """
    
    def __init__(self):
        # Keep for fallback compatibility
        self.base_url = settings.OLLAMA_URL.rstrip('/')
        self.model = settings.OLLAMA_MODEL_NAME
        self.timeout = settings.OLLAMA_TIMEOUT
        
    async def stream_response(
        self,
        user_message: str,
        user_id: int,
        thread_id: Optional[str] = None,
        thread: Optional["Thread"] = None,
        system_prompt: Optional[str] = None,
        save_history: bool = True,
        output_format: str = "telegram"  # "telegram" or "markdown" (for web)
    ) -> AsyncIterator[str]:
        """
        Stream LLM response using pydantic-ai agent.
        
        Phase 4: Uses agent with support tools.
        Agent automatically handles tool selection and execution during streaming.
        
        Args:
            user_message: User's input text
            user_id: Telegram user ID
            thread_id: Optional thread ID for history
            thread: Optional Thread object for per-thread settings
            system_prompt: Optional system prompt override (not used with agents)
            save_history: Whether to save to history (default True)
            
        Yields:
            Chunks of response text as they arrive from agent
        """
        logger.info(f"🚀🚀🚀 stream_response() ENTERED: user_id={user_id}, thread_id={thread_id}, message_len={len(user_message)}")
        logger.debug(f"   thread={thread}, save_history={save_history}")
        
        try:
            logger.debug("📦 Inside try block, about to get user profile service...")
            # Get user language and KB index for context
            from luka_bot.services.user_profile_service import get_user_profile_service
            profile_service = get_user_profile_service()
            user_lang = await profile_service.get_language(user_id)
            kb_index = await profile_service.get_kb_index(user_id)
            
            # Phase 5: Index USER message to KB immediately (before LLM call)
            if kb_index and settings.ELASTICSEARCH_ENABLED and save_history:
                try:
                    from luka_bot.services.elasticsearch_service import get_elasticsearch_service
                    from luka_bot.utils.message_parser import extract_mentions, extract_hashtags, extract_urls
                    from datetime import datetime
                    
                    es_service = await get_elasticsearch_service()
                    
                    # Build message document
                    message_doc = {
                        "message_id": f"{user_id}_{thread_id}_{int(datetime.utcnow().timestamp() * 1000)}",
                        "user_id": str(user_id),
                        "thread_id": thread_id or "",
                        "role": "user",
                        "message_text": user_message,
                        "message_date": datetime.utcnow().isoformat(),
                        "sender_name": f"User {user_id}",  # Will be enriched later
                        "mentions": extract_mentions(user_message),
                        "hashtags": extract_hashtags(user_message),
                        "urls": extract_urls(user_message),
                    }
                    
                    # Index immediately
                    await es_service.index_message_immediate(
                        index_name=kb_index,
                        document_id=message_doc["message_id"],
                        message_data=message_doc
                    )
                    
                except Exception as e:
                    # Don't break chat flow on KB errors
                    logger.warning(f"⚠️  KB indexing failed for USER message: {e}")
            
            # Create conversation context for tools
            # Phase 4: Use getattr for forward compatibility with Thread model updates
            # Important: If enabled_tools is empty list, use DEFAULT_ENABLED_TOOLS (empty = all enabled)
            thread_enabled_tools = getattr(thread, 'enabled_tools', []) if thread else []
            effective_enabled_tools = thread_enabled_tools if thread_enabled_tools else settings.DEFAULT_ENABLED_TOOLS
            
            ctx = ConversationContext.from_thread(
                user_id=user_id,
                thread_id=thread_id or "",
                thread_knowledge_bases=getattr(thread, 'knowledge_bases', []) if thread else [],
                enabled_tools=effective_enabled_tools,
                llm_provider=getattr(thread, 'llm_provider', None) if thread else None,
                model_name=getattr(thread, 'model_name', None) if thread else None,
                system_prompt_override=getattr(thread, 'system_prompt', None) if thread else None,
                conversation_summary=getattr(thread, 'conversation_summary', None) if thread else None
            )
            
            # Log if conversation summary is injected
            if thread and thread.conversation_summary:
                logger.debug(f"📝 Injected conversation summary into context: {len(thread.conversation_summary)} chars")

            # Store user language, output format, and message in context metadata for tools
            ctx.metadata['language'] = user_lang
            ctx.metadata['output_format'] = output_format  # "telegram" or "markdown"
            ctx.metadata['user_message'] = user_message  # For digest detection in KB tools
            
            # Add source detection: markdown = web, telegram = telegram bot
            if output_format == "markdown":
                ctx.metadata['source'] = 'web'
            else:
                ctx.metadata['source'] = 'telegram'
            
            # Add bot name and URLs for workflow tools
            # For web users (markdown output), add web app URL
            # For telegram users, we'd need bot username (would be added elsewhere)
            ctx.metadata['bot_name'] = settings.LUKA_NAME
            
            # Add web app URL for web users (can be used by workflows to invite users)
            if output_format == "markdown":
                from luka_bot.core.config import settings as luka_config
                # Try to get web app URL from AG_UI settings if available
                try:
                    from ag_ui_gateway.config.settings import settings as ag_settings
                    ctx.metadata['web_app_url'] = getattr(ag_settings, 'AG_UI_DOJO_URL', 'http://localhost:3000')
                except ImportError:
                    ctx.metadata['web_app_url'] = 'http://localhost:3000'  # Default fallback
                
                # Default bot username fallback (can be overridden by actual bot instance)
                ctx.metadata['bot_username'] = getattr(luka_config, 'BOT_USERNAME', 'SOLAtlasBOT')
            
            # Check for active workflow and inject workflow context into metadata
            try:
                from luka_bot.services.workflow_service import get_workflow_service
                workflow_service = get_workflow_service()
                
                # Check for any active workflow for this user (check multiple domains)
                active_workflow = await workflow_service.get_active_workflow_for_user(user_id, "sol_atlas_onboarding")
                if not active_workflow:
                    # Check trip planner workflow
                    active_workflow = await workflow_service.get_active_workflow_for_user(user_id, "trip_planner_onboarding")
                
                if active_workflow:
                    # Inject workflow context into metadata so tools/LLM can access it
                    ctx.metadata['active_workflow'] = {
                        'workflow_id': active_workflow.workflow_id,
                        'domain': active_workflow.domain,
                        'current_step': active_workflow.current_step,
                        'progress': active_workflow.progress,
                        'state': active_workflow.state
                    }
                    logger.debug(f"🔗 Active workflow found for user {user_id}: {active_workflow.workflow_id}, step: {active_workflow.current_step}")

                    # Check if user wants to resume a paused workflow
                    if active_workflow.state == "paused":
                        user_msg_lower = user_message.lower().strip()
                        resume_indicators = ["yes", "yeah", "yep", "continue", "resume", "да", "продолжи", "продолжить", "давай"]

                        if any(indicator in user_msg_lower for indicator in resume_indicators):
                            logger.info(f"▶️ User chose to resume workflow {active_workflow.workflow_id}")
                            await workflow_service.resume_workflow(active_workflow.workflow_id)
                            # Refresh workflow state
                            active_workflow = await workflow_service.get_active_workflow_for_user(user_id, active_workflow.domain)
                            if active_workflow:
                                ctx.metadata['active_workflow']['state'] = 'running'
                                logger.info(f"✅ Workflow resumed successfully")

                    # Check if user is interrupting workflow with off-topic question
                    if active_workflow.state == "running":
                        is_off_topic = await self._detect_workflow_interruption(
                            user_message=user_message,
                            workflow_domain=active_workflow.domain,
                            current_step=active_workflow.current_step,
                            user_id=user_id
                        )

                        if is_off_topic:
                            logger.info(f"⏸️ Detected off-topic question during workflow - pausing {active_workflow.workflow_id}")
                            await workflow_service.pause_workflow(active_workflow.workflow_id)

                            # Update metadata to reflect paused state
                            ctx.metadata['workflow_paused'] = True
                            ctx.metadata['paused_workflow_id'] = active_workflow.workflow_id
                            ctx.metadata['paused_workflow_domain'] = active_workflow.domain
                            ctx.metadata['paused_step'] = active_workflow.current_step
                            ctx.metadata['active_workflow']['state'] = 'paused'
            except Exception as e:
                # Don't break chat if workflow check fails
                logger.debug(f"⚠️  Workflow context check failed: {e}")
            
            # Heuristic fallback: If message contains a YouTube URL, call transcript tool directly
            try:
                youtube_url_match = re.search(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+)", user_message, re.IGNORECASE)
                if youtube_url_match:
                    youtube_url = youtube_url_match.group(1)
                    logger.info(f"🎬 Heuristic: YouTube URL detected: {youtube_url}")
                    transcript = await youtube_tools.get_youtube_transcript(ctx, video_url=youtube_url, language=user_lang)
                    if transcript:
                        # Persist full transcript (if available) as a system message in history
                        yt_full = None
                        yt_title = None
                        try:
                            yt_full = ctx.metadata.get('youtube_full_transcript') if hasattr(ctx, 'metadata') and isinstance(ctx.metadata, dict) else None
                            yt_title = ctx.metadata.get('youtube_video_title', 'YouTube Video') if hasattr(ctx, 'metadata') and isinstance(ctx.metadata, dict) else None
                        except Exception:
                            yt_full = None
                            yt_title = None

                        yield transcript
                        if save_history and thread_id:
                            await self._save_to_history(
                                user_id,
                                user_message,
                                transcript,
                                thread_id,
                                youtube_transcript=yt_full,
                                youtube_video_title=yt_title
                            )
                        logger.info(f"✅ Agent response complete: {len(transcript)} chars")
                        return
            except Exception as e:
                logger.warning(f"⚠️  Heuristic YouTube transcript fallback failed: {e}")
            
            logger.info(f"🤖 Creating agent for user {user_id}, thread {thread_id}")
            
            # FIX: Load conversation history BEFORE creating agent (so we can check if user already answered)
            try:
                logger.debug("📦 Step 1: Loading conversation history...")
                history = await self._get_history(user_id, thread_id, max_messages=10)
                logger.debug(f"✅ History loaded: {len(history)} messages")
            except Exception as history_error:
                logger.error(f"❌ FATAL: History loading failed: {history_error}", exc_info=True)
                raise

            # Note: Removed heuristic-based workflow advancement.
            # Workflows are now advanced naturally by the LLM after processing the user's response.
            # The LLM context includes workflow step instructions and will handle responses appropriately.

            # Create agent (Phase 4: static agent with support tools)
            # Phase 5: Will use create_agent_with_user_tasks for Camunda integration
            try:
                logger.debug("📦 Step 2: Creating agent via create_static_agent_with_basic_tools()")
                # Pass enabled_tools to filter tools at registration time
                agent = await create_static_agent_with_basic_tools(
                    user_id=user_id,
                    enabled_tools=effective_enabled_tools
                )
                logger.debug(f"✅ Agent created successfully: type={type(agent).__name__}")
            except Exception as agent_error:
                logger.error(f"❌ FATAL: Agent creation failed: {agent_error}", exc_info=True)
                raise
            
            # Convert to pydantic-ai message format
            try:
                logger.debug("📦 Step 3: Converting history to pydantic-ai format...")
                from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
                from typing import List, Union
                
                model_messages: List[Union[ModelRequest, ModelResponse]] = []
                for msg in history:
                    if msg["role"] == "user":
                        model_messages.append(
                            ModelRequest(
                                kind="request",
                                parts=[UserPromptPart(content=msg["content"])]
                            )
                        )
                    elif msg["role"] == "assistant":
                        model_messages.append(
                            ModelResponse(
                                kind="response",
                                parts=[TextPart(content=msg["content"])]
                            )
                        )
                    elif msg["role"] == "system":
                        # Add as contextual user content so the model can use it directly
                        context_text = msg.get("content") or ""
                        if context_text:
                            model_messages.append(
                                ModelRequest(
                                    kind="request",
                                    parts=[UserPromptPart(content=f"[Context]\n{context_text}")]
                                )
                            )
                
                # Track previous assistant message to avoid echoing duplicates
                prev_assistant_response = ""
                for msg in reversed(history):
                    if msg.get("role") == "assistant":
                        prev_assistant_response = (msg.get("content") or "").strip()
                        break
                
                logger.debug(f"✅ Converted {len(model_messages)} history messages to pydantic-ai format")
            except Exception as conversion_error:
                logger.error(f"❌ FATAL: Message conversion failed: {conversion_error}", exc_info=True)
                raise
            
            logger.info(f"🧠 LLM request: user={user_id}, thread={thread_id}")
            logger.info(f"📚 Context KB indices: {ctx.thread_knowledge_bases}")
            logger.info(f"🔧 Context enabled tools: {ctx.enabled_tools}")
            
            # Log agent's actual tools for debugging
            try:
                # Check various possible tool storage locations in pydantic-ai
                tool_count = 0
                tool_names = []
                
                if hasattr(agent, '_function_toolset'):
                    toolset = agent._function_toolset
                    logger.info(f"🔍 Agent._function_toolset exists: {type(toolset)}")
                    
                    # Check toolset for tools
                    for attr in ['_tools', 'tools', '_tool_defs']:
                        if hasattr(toolset, attr):
                            tools_dict = getattr(toolset, attr)
                            if isinstance(tools_dict, dict):
                                tool_count = len(tools_dict)
                                tool_names = list(tools_dict.keys())
                                logger.info(f"🛠️  Agent has {tool_count} tools via toolset.{attr}: {tool_names}")
                                break
                
                if tool_count == 0:
                    # Try direct agent attributes
                    if hasattr(agent, 'tools') and agent.tools:
                        tool_count = len(agent.tools)
                        tool_names = [str(t) for t in agent.tools]
                        logger.info(f"🛠️  Agent has {tool_count} tools via agent.tools: {tool_names}")
                    else:
                        logger.warning(f"⚠️  Could not detect tools on agent (but they may still work)")
            except Exception as e:
                logger.warning(f"⚠️  Error checking agent tools: {e}")
            
            full_response = ""
            
            # FIX 21: Use stream() instead of stream_text() to expose tool calls
            # Based on working old bot pattern (bot_server/services/streaming_service.py:152-299)
            logger.debug(f"🚀 Starting agent.run_stream for user_id={user_id}, thread_id={thread_id}")
            logger.debug(f"   user_message: {user_message[:100]}...")
            logger.debug(f"   message_history: {len(model_messages)} messages")
            
            async with agent.run_stream(
                user_prompt=user_message,
                deps=ctx,
                message_history=model_messages  # Add history to agent context
            ) as stream:
                logger.debug(f"✅ run_stream context entered, starting execution...")

                # Initialize variables for both streaming and non-streaming modes
                tool_notification_shown = False
                is_cumulative_streaming = True  # Default to cumulative mode
                
                if settings.STREAMING_ENABLED:
                    # STREAMING ENABLED: Try streaming first
                    logger.info("🔄 Streaming mode enabled - attempting to stream response")
                    
                    try:
                        # Use stream() to see ALL message parts (text + tool calls)
                        # NOT stream_text() which only shows text and hides tool execution
                        chunk_count = 0
                        
                        # Determine streaming mode based on ACTUAL provider being used
                        # (not ctx.llm_provider which may be outdated after fallback)
                        # OpenAI: cumulative (each chunk contains full text so far)
                        # Ollama: delta (each chunk contains only new text)
                        try:
                            from luka_bot.services.llm_provider_fallback import get_llm_provider_fallback
                            fallback = get_llm_provider_fallback()
                            cached_provider = await fallback.redis.get(fallback.PREFERRED_PROVIDER_KEY)
                            actual_provider = cached_provider.decode() if cached_provider else (ctx.llm_provider or settings.DEFAULT_LLM_PROVIDER)
                        except Exception:
                            actual_provider = ctx.llm_provider or settings.DEFAULT_LLM_PROVIDER
                        
                        # IMPORTANT: Both OpenAI and Ollama use cumulative streaming
                        # (each chunk contains the full response so far, not just new text)
                        is_cumulative_streaming = True  # Always use delta extraction
                        logger.info(f"🔄 Streaming mode: cumulative (both providers) [actual_provider={actual_provider}, ctx_provider={ctx.llm_provider}]")
                        
                        logger.debug("🔍 About to iterate stream.stream()...")
                        async for chunk in stream.stream():  # FIX 21: Changed from stream_text()
                            chunk_count += 1
                            chunk_type = chunk.__class__.__name__
                            
                            # Log tool calls and first few chunks
                            if chunk_count <= 3 or chunk_type == 'ToolCallPart':
                                logger.info(f"📦 Chunk {chunk_count}: {chunk_type}")
                            
                            if chunk_count == 1:
                                logger.debug(f"🎉 First chunk received! Type: {chunk_type}, Content: {str(chunk)[:200]}")
                            
                            # FIX 22/24: Handle both message parts AND plain strings
                            # stream() can return either ModelResponse parts or strings depending on the model
                            
                            # Case 1: Plain string (OpenAI/Ollama streaming)
                            if isinstance(chunk, str):
                                if chunk:
                                    if is_cumulative_streaming:
                                        # Cumulative streaming: extract only the delta (new text)
                                        if chunk.startswith(full_response):
                                            delta = chunk[len(full_response):]
                                            if delta:
                                                full_response = chunk
                                                yield delta
                                        else:
                                            # Fallback: if logic fails, treat as delta
                                            logger.warning(f"⚠️ Cumulative chunk doesn't start with previous response, treating as delta")
                                            logger.warning(f"   Previous: {full_response[:50]}...")
                                            logger.warning(f"   Current: {chunk[:50]}...")
                                            full_response += chunk
                                            yield chunk
                                    else:
                                        # Delta mode: accumulate directly (not currently used)
                                        full_response += chunk
                                        yield chunk
                            
                            # Case 2: ToolCallPart - show notification
                            elif chunk_type == 'ToolCallPart':
                                tool_name = chunk.tool_name if hasattr(chunk, 'tool_name') else 'unknown'
                                logger.info(f"🔧 ✅ TOOL CALLED: {tool_name}")
                                
                                if not tool_notification_shown:
                                    notification_dict = _get_tool_notification(tool_name)
                                    logger.info(f"🔧 Showing tool notification: {tool_name}")
                                    yield notification_dict  # Yield dict to enable message editing
                                    tool_notification_shown = True
                            
                            # Case 3: TextPart - extract text content
                            elif chunk_type == 'TextPart':
                                if hasattr(chunk, 'content') and chunk.content:
                                    text_content = str(chunk.content)
                                    if text_content:
                                        full_response += text_content
                                        yield text_content  # Yield text as it arrives
                            
                            # Case 4: response_so_far attribute (old bot pattern)
                            # Some models return chunks with response_so_far instead of content
                            elif hasattr(chunk, 'response_so_far'):
                                response_text = chunk.response_so_far or ""
                                if response_text and response_text != full_response:
                                    # Yield only the new part
                                    if full_response and response_text.startswith(full_response):
                                        new_text = response_text[len(full_response):]
                                        if new_text:
                                            full_response = response_text
                                            yield new_text
                                    else:
                                        full_response = response_text
                                        yield response_text
                            
                            # Case 5: content attribute (fallback)
                            elif hasattr(chunk, 'content'):
                                content_text = str(chunk.content) if chunk.content else ""
                                if content_text and content_text != full_response:
                                    full_response = content_text
                                    yield content_text
                            
                            # Case 6: Other parts (RetryPromptPart, etc.) - skip
                            elif chunk_type in ('RetryPromptPart', 'UserPromptPart'):
                                pass  # Skip silently
                        
                        logger.info(f"✅ Streaming complete: {chunk_count} chunks, {len(full_response)} chars")
                        
                    except Exception as stream_error:
                        # Log but don't crash - partial response is OK
                        error_name = stream_error.__class__.__name__
                        logger.warning(f"⚠️ Streaming failed ({error_name}), will try get_output() fallback")
                        full_response = ""  # Reset to trigger fallback
                    
                    # Check if streaming produced content
                    if full_response and len(full_response.strip()) > 0:
                        logger.debug("✅ Streaming produced content, skipping get_output() to avoid duplicate tool execution")
                        # Don't call get_output() - we have a complete response from streaming
                    else:
                        # Streaming produced no content - check if tools were called first
                        # If tools were called, we'll extract results manually to avoid re-execution
                        logger.info("⚠️ Streaming produced no content, checking for tool calls...")
                        
                        # Quick check if any tools were called (before calling get_output which might re-execute)
                        all_msgs_preview = stream.all_messages()
                        tool_was_called_preview = False
                        for msg in all_msgs_preview:
                            if msg.kind == 'response':
                                for part in msg.parts:
                                    if part.__class__.__name__ == 'ToolCallPart':
                                        tool_was_called_preview = True
                                        break
                                if tool_was_called_preview:
                                    break
                        
                        if tool_was_called_preview:
                            # Tools were called - don't use get_output() as it might re-execute
                            # Instead, we'll extract tool results in the continuation logic below
                            logger.info("🔧 Tools were called - skipping get_output() to avoid re-execution, will extract results manually")
                        else:
                            # No tools called - safe to use get_output()
                            logger.info("⚠️ No tools called, falling back to get_output()")
                            try:
                                final_output = await stream.get_output()
                                if final_output:
                                    full_response = str(final_output)
                                    logger.info(f"✅ get_output() fallback successful: {len(full_response)} chars")
                                    yield full_response
                                else:
                                    logger.warning("⚠️ get_output() also returned empty")
                            except Exception as e:
                                logger.error(f"❌ get_output() fallback failed: {e}")
                
                else:
                    # STREAMING DISABLED: Still need to iterate stream to trigger execution
                    logger.info("🔄 Streaming disabled - executing agent without streaming")
                    
                    try:
                        # CRITICAL: Must iterate through stream to trigger agent execution
                        # Even though we don't want to yield chunks, the agent needs to run
                        async for chunk in stream.stream():
                            # Accumulate response without yielding (non-streaming mode)
                            if isinstance(chunk, str) and chunk:
                                if chunk.startswith(full_response):
                                    full_response = chunk
                                else:
                                    full_response += chunk
                            elif hasattr(chunk, '__class__'):
                                chunk_type = chunk.__class__.__name__
                                if chunk_type == 'TextPart' and hasattr(chunk, 'content'):
                                    full_response += str(chunk.content)
                        
                        logger.info(f"✅ Agent execution complete: {len(full_response)} chars")
                        
                        # Only call get_output() if we didn't accumulate text from stream
                        if not full_response:
                            final_output = await stream.get_output()
                            if final_output:
                                full_response = str(final_output)
                                logger.info(f"✅ Got response from get_output(): {len(full_response)} chars")
                        
                        # Yield the complete response once
                        if full_response:
                            yield full_response
                        else:
                            logger.warning("⚠️ Agent execution produced no output")
                    except Exception as e:
                        logger.error(f"❌ Agent execution failed: {e}")
                
                # FIX 21: Removed 125+ lines of complex manual message parsing (FIX 9/17/18/19/20)
                # The old approach was trying to manually extract tool results from message parts
                # pydantic-ai handles this automatically and reliably via get_output()
                
                # Check if any tools were called
                logger.info(f"🔍 Checking if KB tool was called (tool_notification_shown={tool_notification_shown})")
                logger.info(f"🔍 Total response length after streaming: {len(full_response)} chars")
                if full_response:
                    logger.info(f"🔍 Response preview: {full_response[:200]}...")
                else:
                    logger.warning(f"⚠️  LLM produced NO text output!")
                
                # Extract KB tool results and append them if missing from LLM response
                # Check if formatted snippets are already in the response
                has_formatted_snippets = '━━━━━━━━━━━━━━━━━━━━' in full_response
                
                if not has_formatted_snippets:
                    # LLM consumed the tool results without showing the formatted snippets
                    # Extract them from tool returns and append
                    all_msgs = stream.all_messages()
                    kb_tool_results = []
                    
                    logger.info(f"🔍 KB snippets not in response, scanning {len(all_msgs)} messages for tool results...")
                    
                    # Also check if tool was actually called
                    tool_was_called = False
                    called_tool_name = None  # Track which tool was called
                    for msg in all_msgs:
                        if msg.kind == 'response':
                            for part in msg.parts:
                                if part.__class__.__name__ == 'ToolCallPart':
                                    tool_name = getattr(part, 'tool_name', '')
                                    tool_was_called = True
                                    called_tool_name = tool_name  # Store the tool name
                                    logger.info(f"  ✅ Found ToolCallPart: {tool_name}")
                    
                    if not tool_was_called:
                        logger.warning(f"⚠️  LLM did NOT call any tools (answered from memory/history)")
                    
                    # Track unique results to avoid duplicates
                    seen_results = set()
                    kb_empty_result_found = False
                    last_non_kb_tool_result = None
                    last_non_kb_tool_name = None  # Track which tool produced the result
                    
                    for msg in all_msgs:
                        if msg.kind == 'request':
                            for part in msg.parts:
                                if part.__class__.__name__ == 'ToolReturnPart':
                                    # Extract the tool result
                                    result = None
                                    if hasattr(part, 'content') and part.content:
                                        result = str(part.content)
                                    elif hasattr(part, 'tool_return') and part.tool_return:
                                        result = str(part.tool_return)
                                    
                                    # Check for empty KB result instruction
                                    tool_name = getattr(part, 'tool_name', '')

                                    if result and '[No messages found in knowledge base' in result:
                                        kb_empty_result_found = True
                                        logger.info(f"📚 KB returned empty result instruction")
                                    elif result and '━━━━━━━━━━━━━━━━━━━━' in result:
                                        # Found formatted KB snippets - deduplicate by content
                                        result_hash = hash(result)
                                        if result_hash not in seen_results:
                                            kb_tool_results.append(result)
                                            seen_results.add(result_hash)
                                            logger.info(f"📚 Found KB snippets in tool result: {len(result)} chars")
                                        else:
                                            logger.info(f"⚠️  Skipping duplicate KB result: {len(result)} chars")
                                    elif result:
                                        # Track last non-KB tool output (e.g., workflow execution) for fallback
                                        # Don't overwrite if we already have a result and the new one is a workflow instruction marker
                                        if last_non_kb_tool_result and "[WORKFLOW_INSTRUCTION]" in result:
                                            logger.debug(
                                                f"⏭️  Skipping workflow instruction update - keeping first result ({len(last_non_kb_tool_result)} chars)"
                                            )
                                        else:
                                            last_non_kb_tool_result = result
                                            last_non_kb_tool_name = tool_name  # Track which tool produced this result
                                            logger.info(
                                                "🧩 Captured tool result for fallback: "
                                                f"{tool_name or 'unknown'} ({len(result)} chars)"
                                            )
                    
                    # Handle empty KB result - provide helpful default response
                    if kb_empty_result_found and not kb_tool_results and not full_response:
                        logger.info("💡 KB was empty, providing helpful default response")
                        # Get user language
                        user_lang = "en"
                        if thread and hasattr(thread, 'language'):
                            user_lang = thread.language
                        
                        if user_lang == "ru":
                            empty_kb_response = "В базе знаний пока нет сообщений по этому запросу. Возможно, это недавняя тема или используйте другие ключевые слова для поиска."
                        else:
                            empty_kb_response = "I don't have any messages in the knowledge base matching that query yet. This might be a new topic, or try using different keywords."
                        
                        yield empty_kb_response
                        full_response = empty_kb_response
                        logger.info(f"✅ Provided empty KB default response: {len(empty_kb_response)} chars")
                    
                    # Append KB snippets to response if found
                    elif kb_tool_results:
                        logger.info(f"✅ Appending {len(kb_tool_results)} KB snippet section(s) that LLM didn't include")
                        
                        # Check if LLM generated a proper summary
                        has_summary = full_response and len(full_response.strip()) >= 20
                        
                        # If LLM didn't generate meaningful text, generate one with LLM
                        if not has_summary:
                            logger.info("🤖 Generating LLM summary for KB results (main LLM produced no text)...")
                            try:
                                # Create a simple agent for summarization
                                from pydantic_ai import Agent
                                from luka_bot.services.llm_model_factory import create_llm_model_with_fallback
                                
                                summary_model = await create_llm_model_with_fallback(context=f"kb_summary_{user_id}")
                                
                                # Get first KB result for content analysis
                                kb_content = kb_tool_results[0] if kb_tool_results else ""
                                
                                summary_prompt = f"""Based on the search results below, provide a 2-3 sentence summary that:
1. DIRECTLY ANSWERS the user's question
2. SUMMARIZES the key points found in the messages
3. Is conversational and helpful

User's question: {user_message}

Search results:
{kb_content}

Generate ONLY the summary text (no formatting, no extra text)."""
                                
                                summary_agent: Agent[None, str] = Agent(
                                    summary_model,
                                    system_prompt="You are a helpful assistant that summarizes search results.",
                                    retries=0
                                )
                                
                                # Get summary (non-streaming for simplicity)
                                summary_result = await summary_agent.run(summary_prompt)
                                llm_summary = summary_result.output.strip()
                                
                                if llm_summary and len(llm_summary) >= 20:
                                    # Yield LLM summary
                                    yield llm_summary
                                    full_response = llm_summary
                                    logger.info(f"✅ Generated LLM summary: {len(llm_summary)} chars")
                                else:
                                    # Summary too short, use default intro
                                    intro_lang = "en"
                                    if thread and hasattr(thread, 'language'):
                                        intro_lang = thread.language
                                    
                                    if intro_lang == "ru":
                                        default_intro = "Вот последние сообщения, которые соответствуют вашему запросу. Если нужно что-то конкретное, дайте знать!"
                                    else:
                                        default_intro = "I've found relevant messages in your history. Let me know if you need something more specific!"
                                    
                                    yield default_intro
                                    full_response = default_intro
                                    logger.warning(f"⚠️ Summary too short, using default intro")
                                    
                            except Exception as summary_err:
                                logger.warning(f"⚠️ Summary generation failed: {summary_err}, using default intro")
                                # Fall back to default intro
                                intro_lang = "en"
                                if thread and hasattr(thread, 'language'):
                                    intro_lang = thread.language
                                
                                if intro_lang == "ru":
                                    default_intro = "Вот последние сообщения, которые соответствуют вашему запросу. Если нужно что-то конкретное, дайте знать!"
                                else:
                                    default_intro = "I've found relevant messages in your history. Let me know if you need something more specific!"
                                
                                yield default_intro
                                full_response = default_intro
                        else:
                            logger.info(f"✅ LLM generated proper summary: {len(full_response)} chars")
                        
                        # Append KB snippets after summary
                        for kb_result in kb_tool_results:
                            # Add separator if needed
                            if full_response and not full_response.endswith('\n\n'):
                                yield '\n\n'
                                full_response += '\n\n'
                            yield kb_result
                            full_response += kb_result
                    else:
                        logger.info("📚 No KB snippets found in tool results")

                    # If LLM produced no text but a tool returned rich output, force continuation
                    # This handles cases where Ollama calls tools but doesn't generate follow-up text
                    if (not full_response or not full_response.strip()) and last_non_kb_tool_result and tool_was_called:
                        # Filter out empty or instruction-only messages (e.g., "Workflow is already active...")
                        tool_result_trimmed = last_non_kb_tool_result.strip()
                        
                        # For workflow tool results that are questions (new workflow start), extract the actual question
                        # Format: "What's your name?" -> extract to What's your name?
                        is_workflow_question = False
                        workflow_question_content = None
                        if '"' in tool_result_trimmed and tool_result_trimmed.count('"') >= 2:
                            # Check if it's a simple quoted question (new workflow start)
                            quoted_match = re.match(r'^"([^"]+)"$', tool_result_trimmed)
                            if quoted_match:
                                workflow_question_content = quoted_match.group(1)
                                is_workflow_question = True
                                logger.debug(f"📝 Extracted workflow question from quotes: {workflow_question_content}")
                        
                        should_skip = (
                            not tool_result_trimmed or
                            "Do not display this message to the user" in tool_result_trimmed or
                            "Workflow is already active" in tool_result_trimmed or
                            (  # Filter workflow step instructions that have the marker
                                "[WORKFLOW_INSTRUCTION]" in tool_result_trimmed and 
                                "Ask the user:" in tool_result_trimmed  # Only skip if it's explicitly an instruction template
                            )
                        )
                        
                        if not should_skip:
                            # If it's a workflow question, use the extracted content (without quotes)
                            if is_workflow_question and workflow_question_content:
                                logger.info(f"🧩 Using workflow question as response: {workflow_question_content}")
                                yield workflow_question_content
                                full_response = workflow_question_content
                            else:
                                # FORCE CONTINUATION: LLM called tool but didn't generate text
                                # Create a continuation prompt that includes the tool result
                                logger.info(f"🔄 Forcing LLM continuation after tool call: {last_non_kb_tool_name or 'unknown'}")
                                
                                try:
                                    # Get user language
                                    user_lang = "en"
                                    if thread and hasattr(thread, 'language'):
                                        user_lang = thread.language
                                    
                                    # Create continuation prompt based on tool type
                                    tool_name = last_non_kb_tool_name or called_tool_name or 'tool'
                                    
                                    # Truncate tool result if too long (keep first 2000 chars for context)
                                    tool_result_preview = tool_result_trimmed[:2000]
                                    if len(tool_result_trimmed) > 2000:
                                        tool_result_preview += f"\n\n[... {len(tool_result_trimmed) - 2000} more characters ...]"
                                    
                                    if tool_name == 'plan_trip':
                                        continuation_prompt = f"""The user asked about planning a trip. The trip planning tool returned the following results:

                                            {tool_result_preview}
                                            
                                            Now provide a natural, conversational response to the user that:
                                            1. Acknowledges their trip planning request
                                            2. Highlights the key information from the trip plan (route, stops, duration)
                                            3. Is enthusiastic and helpful
                                            4. Presents the trip plan in a friendly, engaging way
                                            
                                            Write your response now:"""
                                    elif tool_name in ('search_locations', 'find_nearby_locations', 'get_location_details'):
                                        continuation_prompt = f"""The user asked about locations. The location search tool returned:

                                            {tool_result_preview}
                                            
                                            Provide a helpful response that summarizes the key locations found and presents them naturally to the user."""
                                    elif tool_name == 'suggest_route_stops':
                                        continuation_prompt = f"""The user asked for route suggestions. The route planning tool returned:

                                            {tool_result_preview}
                                            
                                            Provide a helpful response that presents these route suggestions in an engaging way."""
                                    else:
                                        # Generic continuation for other tools
                                        continuation_prompt = f"""A tool was called and returned the following results:

                                            {tool_result_preview}
                                            
                                            The user's original question was: "{user_message}"
                                            
                                            Now provide a natural, helpful response that incorporates this tool result and answers the user's question. Be conversational and engaging."""
                                    
                                    # Create a simple continuation agent (no tools, just text generation)
                                    from pydantic_ai import Agent
                                    from luka_bot.services.llm_model_factory import create_llm_model_with_fallback
                                    
                                    continuation_model = await create_llm_model_with_fallback(
                                        context=f"continuation_{user_id}",
                                        model_settings=None  # Use default settings
                                    )
                                    
                                    continuation_agent = Agent(
                                        model=continuation_model,
                                        system_prompt=f"You are a helpful assistant. Process tool results and provide natural, conversational responses to users. Always respond in {user_lang}.",
                                        retries=0
                                    )
                                    
                                    # Generate continuation response
                                    logger.info(f"🤖 Generating continuation response for tool: {tool_name}")
                                    continuation_result = await continuation_agent.run(continuation_prompt)
                                    continuation_text = continuation_result.output.strip() if hasattr(continuation_result, 'output') else str(continuation_result).strip()
                                    
                                    if continuation_text and len(continuation_text) >= 20:
                                        # Successfully generated continuation
                                        logger.info(f"✅ Generated continuation response: {len(continuation_text)} chars")
                                        yield continuation_text
                                        full_response = continuation_text
                                        
                                        # Append the full tool result after the continuation text
                                        if not continuation_text.endswith('\n\n'):
                                            yield '\n\n'
                                            full_response += '\n\n'
                                        yield last_non_kb_tool_result
                                        full_response += last_non_kb_tool_result
                                    else:
                                        # Continuation failed or too short, fall back to direct tool result
                                        logger.warning(f"⚠️ Continuation response too short ({len(continuation_text) if continuation_text else 0} chars), using tool result directly")
                                        yield last_non_kb_tool_result
                                        full_response = last_non_kb_tool_result
                                        
                                except Exception as continuation_err:
                                    logger.error(f"❌ Continuation generation failed: {continuation_err}", exc_info=True)
                                    # Fall back to direct tool result
                                    logger.info("🧩 Falling back to tool result as primary response")
                                    yield last_non_kb_tool_result
                                    full_response = last_non_kb_tool_result
                        else:
                            logger.debug("⚠️ Skipping tool result - appears to be instruction-only message")
                            # If we skipped a workflow instruction, log what it contained for debugging
                            if "[WORKFLOW_INSTRUCTION]" in tool_result_trimmed:
                                logger.debug(f"   Skipped workflow instruction: {tool_result_trimmed[:100]}")
                else:
                    logger.info("✅ KB snippets already present in LLM response")
            
            # Save to history after completion
            if save_history and thread_id:
                # Check if YouTube transcript was processed and store it in history
                youtube_transcript = ctx.metadata.get('youtube_full_transcript')
                youtube_video_title = ctx.metadata.get('youtube_video_title', 'YouTube Video')
                
                await self._save_to_history(
                    user_id, 
                    user_message, 
                    full_response, 
                    thread_id,
                    youtube_transcript=youtube_transcript,
                    youtube_video_title=youtube_video_title
                )
                
                # Update conversation summary in background (non-blocking)
                if save_history and thread:
                    asyncio.create_task(self._update_summary_async(user_id, thread))
                
                # Phase 5: Index ASSISTANT message to KB after LLM response
                if kb_index and settings.ELASTICSEARCH_ENABLED:
                    try:
                        from luka_bot.services.elasticsearch_service import get_elasticsearch_service
                        from luka_bot.utils.message_parser import extract_mentions, extract_hashtags, extract_urls
                        from datetime import datetime
                        
                        es_service = await get_elasticsearch_service()
                        
                        # FIX 39: Skip indexing KB search results to prevent recursive pollution
                        is_kb_search_result = "📝 Summary:" in full_response and "📋 References" in full_response
                        
                        if not is_kb_search_result:
                            # Import utilities for double-write
                            from luka_bot.utils.document_id_generator import DocumentIDGenerator
                            from luka_bot.services.camunda_service import get_camunda_service
                            
                            # Generate assistant document ID
                            kb_doc_id = DocumentIDGenerator.generate_dm_assistant_id(
                                thread_id=thread_id or "default",
                                telegram_message_id=int(datetime.utcnow().timestamp() * 1000)  # Use timestamp as message ID
                            )
                            
                            # Build assistant message document
                            assistant_doc = {
                                "message_id": kb_doc_id,  # Use generated document ID
                                "user_id": str(user_id),
                                "thread_id": thread_id or "",
                                "role": "assistant",
                                "message_text": full_response,
                                "message_date": datetime.utcnow().isoformat(),
                                "sender_name": settings.LUKA_NAME or "Luka",
                                "mentions": extract_mentions(full_response),
                                "hashtags": extract_hashtags(full_response),
                                "urls": extract_urls(full_response),
                                "media_type": "text",
                            }
                            
                            # Enhance message data with thread context
                            camunda_service = get_camunda_service()
                            enhanced_assistant_doc = await camunda_service._build_enhanced_message_data(assistant_doc, thread)
                            
                            # Double-write: Call both services asynchronously
                            es_task = None
                            camunda_task = None
                            
                            try:
                                # Start both operations asynchronously
                                if settings.ELASTICSEARCH_ENABLED:
                                    es_task = asyncio.create_task(
                                        es_service.index_message_immediate(
                                            index_name=kb_index,
                                            message_data=enhanced_assistant_doc,
                                            document_id=kb_doc_id
                                        )
                                    )
                                
                                if settings.CAMUNDA_ENABLED and settings.CAMUNDA_MESSAGE_CORRELATION_ENABLED:
                                    camunda_task = asyncio.create_task(
                                        camunda_service.correlate_message(
                                            user_id=str(user_id),
                                            message_data=enhanced_assistant_doc,
                                            message_type="ASSISTANT_MESSAGE",
                                            kb_doc_id=kb_doc_id
                                        )
                                    )
                                
                                # Wait for both to complete
                                es_success = False
                                camunda_success = False
                                
                                if es_task:
                                    es_success = await es_task
                                if camunda_task:
                                    camunda_success = await camunda_task
                                
                                # Log results
                                if es_success and camunda_success:
                                    logger.info(f"✅ Assistant double-write successful: {kb_doc_id}")
                                elif es_success:
                                    logger.warning(f"⚠️ Assistant ES success, Camunda failed: {kb_doc_id}")
                                elif camunda_success:
                                    logger.warning(f"⚠️ Assistant Camunda success, ES failed: {kb_doc_id}")
                                else:
                                    logger.error(f"❌ Assistant double-write failed: {kb_doc_id}")
                                    
                            except Exception as e:
                                logger.error(f"❌ Error in assistant double-write: {e}")
                                # Cancel any pending tasks
                                if es_task and not es_task.done():
                                    es_task.cancel()
                                if camunda_task and not camunda_task.done():
                                    camunda_task.cancel()
                        else:
                            logger.debug(f"⏭️  Skipped indexing KB search result (to prevent recursion)")
                        
                        # Phase 5: Also index YouTube transcript as system message if present
                        if youtube_transcript and youtube_video_title:
                            system_doc = {
                                "message_id": f"{user_id}_{thread_id}_system_yt_{int(datetime.utcnow().timestamp() * 1000)}",
                                "user_id": str(user_id),
                                "thread_id": thread_id or "",
                                "role": "system",
                                "message_text": f"[YouTube: {youtube_video_title}]\n{youtube_transcript}",
                                "message_date": datetime.utcnow().isoformat(),
                                "sender_name": "YouTube Transcript",
                                "mentions": [],
                                "hashtags": [],
                                "urls": [],
                            }
                            
                            await es_service.index_message_immediate(
                                index_name=kb_index,
                                message_id=system_doc["message_id"],
                                message_data=system_doc
                            )
                            logger.debug(f"📚 Indexed SYSTEM (YouTube) message to KB: {kb_index}")
                        
                    except Exception as e:
                        # Don't break chat flow on KB errors
                        logger.warning(f"⚠️  KB indexing failed for ASSISTANT message: {e}")
            
            # RETRY: If LLM produced empty output, retry once with modified prompt
            if not full_response or not full_response.strip():
                logger.warning("⚠️  LLM produced NO text output! Retrying once with emphatic prompt...")
                
                # Detect if this is a trip planning request that requires tool usage
                user_msg_lower = user_message.lower()
                trip_keywords = ['trip', 'plan', 'itinerary', 'route', 'travel', 'journey', 'from', 'to']
                is_trip_request = any(keyword in user_msg_lower for keyword in trip_keywords) and ('from' in user_msg_lower or 'to' in user_msg_lower)
                
                if is_trip_request:
                    # Force tool usage for trip planning
                    retry_message = f"{user_message}\n\n[CRITICAL INSTRUCTION: The user is asking about trip planning. You MUST use the plan_trip tool immediately. Do not try to answer from memory. Call plan_trip with the locations mentioned in the user's message. After calling the tool, present the results to the user.]"
                    logger.info("🔧 Detected trip planning request - forcing tool usage in retry")
                else:
                    # Generic retry for other cases
                    retry_message = f"{user_message}\n\n[IMPORTANT: Please provide a direct, helpful response to the user's message above. If you need to use tools, use them now. Do not remain silent.]"
                
                try:
                    # Re-run the agent with modified prompt
                    async with agent.run_stream(
                        user_prompt=retry_message,
                        deps=ctx,
                        message_history=model_messages
                    ) as retry_stream:
                        retry_response = ""
                        async for chunk in retry_stream.stream():
                            if isinstance(chunk, str) and chunk:
                                if is_cumulative_streaming:
                                    # Extract delta from cumulative chunk
                                    if chunk.startswith(retry_response):
                                        delta = chunk[len(retry_response):]
                                        if delta:
                                            retry_response = chunk
                                            yield delta
                                    else:
                                        # Fallback: treat as delta
                                        retry_response += chunk
                                        yield chunk
                                else:
                                    # Delta mode
                                    retry_response += chunk
                                    yield chunk
                        
                        full_response = retry_response
                        logger.info(f"🔄 Retry result: {len(full_response)} chars")
                        
                except Exception as retry_error:
                    logger.error(f"❌ Retry failed: {retry_error}")
            
            # SAFETY NET: Ensure we always have a response (after retry)
            if not full_response or not full_response.strip():
                logger.warning("⚠️  No response generated after retry! Providing fallback message.")

                # Check if we're in a workflow step and can provide a step-specific fallback
                if active_workflow and active_workflow.current_step == "complete":
                    # For complete step, extract user name from context or history
                    user_name = "friend"  # default
                    if hasattr(active_workflow, 'context') and active_workflow.context.get("user_name"):
                        user_name = active_workflow.context.get("user_name")
                    elif history and len(history) > 0:
                        # Try to extract from last user message
                        last_user_msg = next((msg for msg in reversed(history) if msg["role"] == "user"), None)
                        if last_user_msg and last_user_msg.get("content"):
                            content = last_user_msg.get("content", "").strip()
                            # Simple heuristic: if it's short and doesn't look like a question, it's probably their name
                            if len(content) < 50 and "?" not in content:
                                user_name = content

                    fallback_message = f"Great to meet you, {user_name}! 🎉 Onboarding finished! Now you can explore the options with research."
                    logger.info(f"✅ Generated workflow-specific fallback for complete step with name: {user_name}")
                else:
                    # Generic fallback for non-workflow or other steps
                    fallback_message = "I'm having trouble processing your request right now. Please try again or rephrase your question."

                yield fallback_message
                full_response = fallback_message
            
            logger.info(f"✅ Response complete: {len(full_response)} chars")

            # Add resume prompt if workflow was paused
            if ctx.metadata.get('workflow_paused'):
                workflow_domain = ctx.metadata.get('paused_workflow_domain', 'workflow')
                workflow_name = workflow_domain.replace("_", " ").title()

                resume_prompt = f"\n\n_Would you like to continue the **{workflow_name}** where we left off? (Yes/No)_"
                full_response += resume_prompt
                yield resume_prompt
                logger.info(f"💬 Added resume prompt for paused workflow: {workflow_domain}")

            # Clear KB search cache after conversation turn completes
            try:
                from luka_bot.agents.tools.knowledge_base_tools import clear_kb_search_cache
                clear_kb_search_cache()
            except Exception as cache_error:
                logger.debug(f"Failed to clear KB search cache: {cache_error}")
            
            # Log which provider successfully handled the request
            if hasattr(agent, 'model') and hasattr(agent.model, 'provider'):
                provider_class = agent.model.provider.__class__.__name__
                logger.info(f"✅ [SUCCESS] Provider {provider_class} handled request successfully")
            
        except asyncio.TimeoutError:
            # Get provider info for logging
            provider_info = "unknown"
            if hasattr(agent, 'model') and hasattr(agent.model, 'provider'):
                provider_info = agent.model.provider.__class__.__name__
            
            logger.error(f"⏱️  [TIMEOUT] Provider {provider_info} timed out after {self.timeout}s")
            
            # Clear cache to trigger fallback on next request
            try:
                from luka_bot.services.llm_provider_fallback import get_llm_provider_fallback
                fallback = get_llm_provider_fallback()
                await fallback.redis.delete(fallback.PREFERRED_PROVIDER_KEY)
                logger.warning(f"🗑️  Cleared provider cache - next request will try fallback")
            except Exception as cache_error:
                logger.warning(f"Failed to clear provider cache: {cache_error}")
            
            raise Exception(f"Agent request timed out (provider: {provider_info})")
            
        except Exception as e:
            # Get provider info for logging
            provider_info = "unknown"
            if hasattr(agent, 'model') and hasattr(agent.model, 'provider'):
                provider_info = agent.model.provider.__class__.__name__
            
            logger.error(f"❌ [ERROR] Provider {provider_info} failed: {e}")
            logger.exception("Full agent error traceback:")
            
            # Clear cache on timeout-like errors
            from openai import APITimeoutError
            if isinstance(e, (APITimeoutError, asyncio.TimeoutError)):
                logger.warning(f"⏱️  LLM timeout detected on {provider_info}, invalidating cache...")
                try:
                    from luka_bot.services.llm_provider_fallback import get_llm_provider_fallback
                    fallback = get_llm_provider_fallback()
                    await fallback.redis.delete(fallback.PREFERRED_PROVIDER_KEY)
                    logger.info("🗑️  Cleared preferred provider cache")
                except Exception as cache_error:
                    logger.warning(f"Failed to clear provider cache: {cache_error}")
            
            raise
    
    async def _get_history(
        self,
        user_id: int,
        thread_id: Optional[str] = None,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get conversation history from Redis.
        
        Args:
            user_id: Telegram user ID
            thread_id: Optional thread ID (Phase 3+)
            max_messages: Maximum messages to retrieve
            
        Returns:
            List of message dicts with role and content
        """
        # Use thread-scoped history if available, otherwise user-scoped
        if thread_id:
            key = f"thread_history:{thread_id}"
        else:
            key = f"llm_history:{user_id}"
        
        try:
            # Get last N messages from Redis list
            history_raw = await redis_client.lrange(key, -max_messages * 2, -1)
            
            history = []
            for item in history_raw:
                try:
                    import json
                    msg = json.loads(item)
                    history.append(msg)
                except:
                    continue
            
            logger.info(f"📚 Loaded {len(history)} messages from history")
            return history
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load history: {e}")
            return []

    async def _detect_workflow_interruption(
        self,
        user_message: str,
        workflow_domain: str,
        current_step: str,
        user_id: int
    ) -> bool:
        """
        Detect if user's message is off-topic from current workflow step.

        Uses LLM classification to determine if the user is:
        A) Answering the workflow question
        B) Asking something completely different (off-topic)

        Args:
            user_message: User's current message
            workflow_domain: Domain of the active workflow
            current_step: Current workflow step ID
            user_id: User ID for logging

        Returns:
            True if off-topic (should pause workflow), False otherwise
        """
        try:
            # Get step instruction for context
            from luka_bot.services.workflow_context_service import get_workflow_context_service
            context_service = get_workflow_context_service()
            step_guidance = await context_service.get_workflow_step_guidance(workflow_domain, current_step)

            if not step_guidance:
                # No guidance available, assume on-topic to be safe
                logger.debug(f"No step guidance for {workflow_domain}:{current_step}, assuming on-topic")
                return False

            # Use fast classification with lightweight model
            classification_prompt = f"""Analyze if the user's response is relevant to the workflow question.

Workflow asks:
{step_guidance}

User responded: "{user_message}"

Is the response:
A - Directly answering the workflow question
B - Asking something completely different or off-topic

Reply with just the letter A or B."""

            # Use fast model for classification (timeout 5s)
            from luka_bot.services.llm_model_factory import create_llm_model_with_fallback
            model = await create_llm_model_with_fallback(f"classification_user_{user_id}", timeout=5)

            # Get classification
            from pydantic_ai import Agent
            classifier = Agent(model=model)

            result_obj = await classifier.run(classification_prompt)
            result = str(result_obj.data).strip().upper()

            is_off_topic = "B" in result

            logger.info(f"🤔 Interruption detection: user_message='{user_message[:50]}...', result={result}, off_topic={is_off_topic}")

            return is_off_topic

        except Exception as e:
            # If classification fails, assume on-topic to avoid falsely pausing workflows
            logger.warning(f"⚠️  Workflow interruption detection failed: {e}, assuming on-topic")
            return False

    async def _save_to_history(
        self,
        user_id: int,
        user_message: str,
        assistant_message: str,
        thread_id: Optional[str] = None,
        max_history_size: int = 20,
        youtube_transcript: Optional[str] = None,
        youtube_video_title: Optional[str] = None
    ) -> None:
        """
        Save conversation turn to Redis history.
        
        If YouTube transcript is provided, it's saved as a system message
        for LLM context without being shown to the user.
        
        Args:
            user_id: Telegram user ID
            user_message: User's message
            assistant_message: Assistant's response
            thread_id: Optional thread ID (Phase 3+)
            max_history_size: Maximum messages to keep (pairs × 2)
            youtube_transcript: Optional full YouTube transcript to store
            youtube_video_title: Optional video title for context
        """
        # Use thread-scoped history if available, otherwise user-scoped
        if thread_id:
            key = f"thread_history:{thread_id}"
        else:
            key = f"llm_history:{user_id}"
        
        try:
            import json
            
            # Save user message
            user_msg = json.dumps({
                "role": "user",
                "content": user_message
            })
            await redis_client.rpush(key, user_msg)
            
            # If YouTube transcript available, save as system message for LLM context
            if youtube_transcript:
                system_msg = json.dumps({
                    "role": "system",
                    "content": f"[YouTube Transcript: {youtube_video_title}]\n\n{youtube_transcript}",
                    "metadata": {
                        "message_type": "youtube_transcript",
                        "video_title": youtube_video_title
                    }
                })
                await redis_client.rpush(key, system_msg)
                logger.info(f"💾 Saved YouTube transcript to history: {len(youtube_transcript)} chars")
            
            # Save assistant message
            assistant_msg = json.dumps({
                "role": "assistant",
                "content": assistant_message
            })
            await redis_client.rpush(key, assistant_msg)
            
            # Trim to max size
            await redis_client.ltrim(key, -max_history_size, -1)
            
            # Set expiry (7 days)
            await redis_client.expire(key, 7 * 24 * 60 * 60)

            # Update thread activity and message count
            if thread_id:
                try:
                    thread_service = get_thread_service()
                    thread = await thread_service.get_thread(thread_id)
                    if thread:
                        thread.update_activity()  # Increments message_count
                        await thread_service.update_thread(thread)
                        logger.debug(f"📊 Updated thread message count: {thread.message_count}")
                except Exception as thread_error:
                    logger.warning(f"⚠️  Failed to update thread message count: {thread_error}")

            logger.info(f"💾 Saved conversation turn to history")
            
            # Check if we should advance workflow to next step
            try:
                from luka_bot.services.workflow_service import get_workflow_service
                workflow_service = get_workflow_service()
                
                # Check for active workflow directly using user_id (don't rely on ctx which isn't in scope)
                active_workflow = await workflow_service.get_active_workflow_for_user(
                    user_id=user_id,
                    domain="sol_atlas_onboarding"  # Currently only this workflow is enabled
                )
                
                if active_workflow:
                    workflow_id = active_workflow.workflow_id
                    
                    # Don't advance workflow if message is workflow initiation command
                    # (e.g., "Please execute the sol_atlas_onboarding workflow")
                    is_workflow_trigger = (
                        "execute" in user_message.lower() and "workflow" in user_message.lower()
                    ) or "sol_atlas_onboarding" in user_message.lower()
                    
                    # Skip advancement if we're already on the last step (complete)
                    # (This prevents double advancement if we advanced before agent creation)
                    if active_workflow.current_step == "complete":
                        logger.debug(f"⏸️ Skipping workflow advancement - already on complete step")
                    # Check if current step received a substantive user response
                    # (indicates step completion) AND it's not a workflow trigger command
                    elif user_message and len(user_message.strip()) > 3 and not is_workflow_trigger:
                        # Prepare context with user_id, thread_id, language for background tasks
                        advancement_context = {
                            "user_id": user_id,
                            "response": user_message
                        }
                        if thread_id:
                            advancement_context["thread_id"] = thread_id
                        if thread and hasattr(thread, 'language'):
                            advancement_context["language"] = thread.language

                        # Advance to next step
                        next_step = await workflow_service.advance_workflow_to_next_step(
                            workflow_id=workflow_id,
                            context=advancement_context
                        )
                        
                        if next_step:
                            logger.info(f"✅ Advanced workflow {workflow_id} to step: {next_step}")
                        else:
                            logger.info(f"🎉 Workflow {workflow_id} completed - no more steps")
                    elif is_workflow_trigger:
                        logger.debug(f"⏸️ Skipping workflow advancement - message is workflow trigger command")
            except Exception as e:
                logger.debug(f"⚠️ Workflow advancement check failed: {e}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to save history: {e}")
    
    async def clear_history(self, user_id: int) -> None:
        """
        Clear conversation history for user.
        
        Args:
            user_id: Telegram user ID
        """
        key = f"llm_history:{user_id}"
        try:
            await redis_client.delete(key)
            logger.info(f"🗑️  Cleared history for user {user_id}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to clear history: {e}")
    
    async def _update_summary_async(self, user_id: int, thread: "Thread"):
        """
        Background task to update conversation summary.
        
        This is called asynchronously after saving history to update
        the conversation summary if needed (every N messages).
        
        Args:
            user_id: User ID
            thread: Thread object to update
        """
        try:
            from luka_bot.services.conversation_summary_service import get_conversation_summary_service
            summary_service = get_conversation_summary_service()
            
            # Update summary if needed (will check interval internally)
            new_summary = await summary_service.update_summary_if_needed(user_id, thread)
            
            if new_summary:
                logger.info(f"📝 Background task updated summary for thread {thread.thread_id}")
        except Exception as e:
            logger.warning(f"Failed to update conversation summary in background: {e}")
    
    def _get_default_system_prompt(self, language: str = "en") -> str:
        """
        Get default system prompt from config with language injection.
        
        Args:
            language: User's language preference ("en" or "ru")
            
        Returns:
            System prompt with language placeholder replaced
        """
        # Map language code to full name for the prompt
        language_names = {
            "en": "English",
            "ru": "Russian"
        }
        language_name = language_names.get(language, "English")
        
        # Inject language into the system prompt
        # Use safe formatting to handle prompts with unexpected format specifiers
        try:
            prompt = settings.LUKA_DEFAULT_SYSTEM_PROMPT.format(language=language_name)
        except (KeyError, IndexError, ValueError) as e:
            # If format fails (e.g., prompt has {0} or other specifiers), use replace
            logger.warning(f"⚠️ System prompt format failed: {e}, using string replacement")
            prompt = settings.LUKA_DEFAULT_SYSTEM_PROMPT.replace("{language}", language_name)
        return prompt


# Global service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
