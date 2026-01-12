"""
Streaming DM handler - Phase 3 with lazy thread creation.

Phase 2: Basic streaming
Phase 3: Thread-scoped conversations + lazy thread creation
Phase 4: Groups navigation support
Phase 5: Configurable streaming with throttling
"""
import asyncio
import time
from datetime import datetime

from aiogram import F, Router
from aiogram.enums import ChatAction, ChatType
from aiogram.filters import BaseFilter, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from loguru import logger

from luka_bot.core.config import settings
from luka_bot.core.loader import redis_client
from luka_bot.handlers.keyboard_actions import ThreadCreationStates
from luka_bot.handlers.states import NavigationStates
# Import FSM states from group_admin that have their own handlers
from luka_bot.handlers.group_admin import ModerationPromptEditForm, StoplistEditForm, BotPersonalityEditForm
from luka_bot.keyboards.threads_menu import get_threads_keyboard
from luka_bot.lg_lukabot.tools import map_config_tools_to_langgraph_tools
from luka_bot.services.divider_service import send_thread_divider
from luka_bot.services.group_service import get_group_service
from luka_bot.services.group_thread_service import get_group_thread_service
from luka_bot.services.llm_service import get_llm_service
from luka_bot.services.message_state_service import get_message_state_service
from luka_bot.services.messaging_service import edit_and_send_parts
from luka_bot.services.thread_name_generator import generate_thread_name
from luka_bot.services.thread_service import get_thread_service
from luka_bot.services.user_profile_service import get_user_profile_service
from luka_bot.utils.formatting import escape_html
from luka_bot.services.messaging_service import split_long_message
from luka_bot.utils.i18n_helper import _


# Form filtering is handled by handler registration order and backup guard
# processes_router (form handlers) is registered before streaming_router
# If a form handler matches, it consumes the update before streaming handlers run


router = Router()


# Note: Custom PrivateChatFilter removed in favor of aiogram's built-in F.chat.type filter
# The custom filter was unreliable and caused group messages to leak through to DM handlers
# See: /docs/issues/group-message-filter-bug.md


class NotCommandFilter(BaseFilter):
    """Filter to exclude messages that start with / (commands)"""

    def __init__(self):
        logger.info("🔧 NotCommandFilter instance created")
        super().__init__()

    async def __call__(self, message: Message) -> bool:
        # Only pass non-command text messages
        logger.debug(f"🔍 NotCommandFilter __call__ invoked for message: {message.text[:30] if message.text else 'NO TEXT'}...")

        if not message.text:
            logger.debug("🔍 NotCommandFilter: Allowing non-text message")
            return True  # Allow non-text messages

        # Exclude any message starting with "/"
        is_command = message.text.startswith("/")
        result = not is_command

        logger.debug(f"🔍 NotCommandFilter: is_command={is_command}, result={result}")

        return result


async def handle_group_selection_in_streaming(message: Message, state: FSMContext, group_id: int) -> None:
    """
    Handle group selection when detected in streaming handler.
    
    This is a fallback when GroupSelectionFilter doesn't catch the message.
    """
    user_id = message.from_user.id if message.from_user else None
    if not user_id:
        return

    logger.info(f"🔀 Handling group selection in streaming: user {user_id}, group {group_id}")

    try:
        # Get user language
        from luka_bot.utils.i18n_helper import get_user_language
        lang = await get_user_language(user_id)

        # Create or get user's thread for this group
        group_thread_service = await get_group_thread_service()
        user_group_thread = await group_thread_service.get_or_create_user_group_thread(
            user_id=user_id,
            group_id=group_id
        )

        # Set as active thread
        thread_service = get_thread_service()
        await thread_service.set_active_thread(user_id, user_group_thread.thread_id)

        # IMPORTANT: Ensure we stay in groups_mode
        await state.set_state(NavigationStates.groups_mode)

        # Update state with current group
        await state.update_data(current_group_id=group_id)

        # Rebuild keyboard with new selection
        group_service = await get_group_service()
        groups = await group_service.list_user_groups(user_id, active_only=True)

        from luka_bot.keyboards.groups_menu import get_groups_keyboard
        keyboard = await get_groups_keyboard(
            groups=groups,
            current_group_id=group_id,
            language=lang
        )

        from luka_bot.utils.i18n_helper import _
        intro_text = _('groups.intro', lang, count=len(groups))
        await message.answer(intro_text, parse_mode="HTML", reply_markup=keyboard)

        # Send inline actions keyboard
        from luka_bot.keyboards.groups_actions_inline import build_groups_actions_inline_keyboard
        actions_inline = await build_groups_actions_inline_keyboard(language=lang)
        await message.answer("🔧 Group Actions", reply_markup=actions_inline)

        # Send GROUP divider with updated inline controls
        from luka_bot.services.group_divider_service import send_group_divider
        await send_group_divider(
            user_id=user_id,
            group_id=group_id,
            divider_type="switch",
            bot=message.bot
        )

        logger.info(f"✅ Switched to group {group_id} for user {user_id} (via streaming handler)")

    except Exception as e:
        logger.error(f"❌ Error switching group in streaming handler: {e}", exc_info=True)
        await message.answer("❌ Error switching group. Please try again.")


async def handle_group_aware_message(message: Message, state: FSMContext) -> None:
    """
    Handle messages when user is in groups mode.

    Converses with group-aware AI agent that has:
    - Group knowledge base access
    - Group context (info, stats, recent messages)
    - User's question history in this group context
    """
    user_id = message.from_user.id if message.from_user else None
    text = message.text or ""

    if not user_id or not text:
        return

    # SAFETY CHECK: This handler should ONLY be called for private chats
    # Group-aware mode means user is chatting about a group FROM their DM
    if message.chat.type in ("group", "supergroup"):
        logger.error(f"🚨 CRITICAL: handle_group_aware_message called with GROUP message! chat_id={message.chat.id}")
        logger.error("🚨 This should NEVER happen - responses would go to the group instead of DM!")
        return

    logger.info(f"💬🏘 Group-aware message from user {user_id}: {text[:50]}...")
    logger.debug(f"🔍 Group-aware message source: chat_id={message.chat.id}, chat_type={message.chat.type}")

    # Skip all keyboard button filtering if user is filling a form
    # Check for TRUTHY values, not just key existence (keys may exist with None values after form completion)
    data = await state.get_data()
    if data.get("form_context") or data.get("start_form") or data.get("task_dialog"):
        logger.info(f"⏭️ GROUP_AWARE: Skipping button filters - user {user_id} is filling a form (text='{text[:50]}')")
        return  # Let the form handler process it

    # Filter out keyboard button texts - don't send to LLM
    if text:
        # Check if message is a keyboard button selection
        from luka_bot.keyboards.groups_menu import is_control_button
        from luka_bot.keyboards.threads_menu import is_thread_button

        # Get user's groups and threads for comparison
        group_service = await get_group_service()
        groups = await group_service.list_user_groups(user_id, active_only=True)

        thread_service = get_thread_service()
        threads = await thread_service.list_threads(user_id)

        # Check if text matches any keyboard button
        if is_control_button(text):
            logger.info(f"⚠️ Filtered control button from LLM: {text}")
            return

        # Check if text is a group selection button
        from luka_bot.keyboards.groups_menu import is_group_button
        group_id = await is_group_button(text, groups)
        if group_id:
            logger.info(f"🔀 Group selection detected: {text} -> group {group_id}")
            # Handle group selection here instead of filtering it out
            await handle_group_selection_in_streaming(message, state, group_id)
            return

        # Check if text is a thread selection button
        if is_thread_button(text, threads):
            logger.info(f"⚠️ Filtered thread selection from LLM: {text}")
            return

        # Check for common keyboard button patterns
        button_patterns = [
            "💬", "📋", "👤", "🏠", "❌",  # Navigation emojis
            "⚙️", "🌐", "🎯",  # Scope controls
            "➕ New", "➕ Начать",  # New item buttons
            "🏘 Groups", "🏘 Группы",  # Section headers
        ]
        if any(pattern in text for pattern in button_patterns):
            logger.info(f"⚠️ Filtered keyboard button from LLM: {text}")
            return

    try:
        # Get current group from state
        state_data = await state.get_data()
        current_group_id = state_data.get("current_group_id")

        if not current_group_id:
            logger.warning(f"⚠️ No current group ID in state for user {user_id}")
            # Exit groups mode - let user chat normally in DM
            logger.info(f"🔙 Exiting groups mode for user {user_id} - switching to normal DM chat")
            await state.clear()
            # Re-route to normal DM handler by calling it directly
            from luka_bot.handlers.streaming_dm import handle_streaming_message
            await handle_streaming_message(message, state)
            return

        # Get services
        from luka_bot.lg_lukabot.integration import stream_langgraph_agent
        from luka_bot.core.config import settings
        thread_service = get_thread_service()
        group_thread_service = await get_group_thread_service()

        # Get user-group thread
        user_group_thread = await group_thread_service.get_or_create_user_group_thread(
            user_id=user_id,
            group_id=current_group_id
        )

        thread_id = user_group_thread.thread_id

        # Show typing status (not editing message like in regular DM)
        try:
            await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        except Exception as e:
            logger.debug(f"Could not send typing action: {e}")

        # Stream response with group context using LangGraph
        response_chunks = []
        final_state = None  # Will be populated by LangGraph stream

        # Throttling variables for streaming mode
        last_update_time = time.time()
        last_update_length = 0
        last_sent_text = ""
        edit_count = 0
        bot_message = None

        # Map config-style tools to individual LangGraph tool names
        config_tools = user_group_thread.enabled_tools or settings.DEFAULT_ENABLED_TOOLS
        enabled_tools = map_config_tools_to_langgraph_tools(config_tools)
        logger.debug(f"🔧 Mapped config tools {config_tools} → LangGraph tools: {enabled_tools}")

        async for event in stream_langgraph_agent(
            message_text=text,
            user_id=user_id,
            thread_id=thread_id,
            language=user_group_thread.language,
            enabled_tools=enabled_tools,
        ):
            # Extract event type and content
            event_type = event.get("type") if isinstance(event, dict) else "content"
            
            # Handle final state (contains conversation_suggestions)
            if event_type == "final_state":
                final_state = event.get("state")
                logger.debug(f"📦 Captured final_state with keys: {list(final_state.keys()) if final_state else 'None'}")
                continue
            
            # Handle tool notifications
            if event_type == "tool_call":
                tool_name = event.get("tool_name", "tool")
                tool_emoji = "🔧"
                if tool_name == "search_knowledge_base":
                    tool_emoji = "🔍"
                elif tool_name == "plan_trip":
                    tool_emoji = "🗺️"
                
                # Send or edit to show tool emoji
                if bot_message:
                    try:
                        await bot_message.edit_text(tool_emoji)
                    except Exception as e:
                        logger.debug(f"Skipped tool notification edit: {e}")
                else:
                    bot_message = await message.answer(tool_emoji)
                continue
            
            # Handle content chunks
            if event_type == "content":
                chunk = event.get("content", "") if isinstance(event, dict) else str(event)
            else:
                continue

            # Only collect string chunks - don't create intermediate messages
            # (We'll send the final response with keyboard attached)
            if chunk:
                response_chunks.append(chunk)
                # Note: No intermediate updates in group-aware mode to ensure keyboard
                # is always attached to the final message

        # Final response (only string chunks)
        final_response = "".join(response_chunks)
        # Don't escape HTML if response contains KB snippets (they have proper HTML formatting)
        has_kb_snippets = '━━━━━━━━━━━━━━━━━━━━' in final_response
        if has_kb_snippets:
            # KB snippets already have proper HTML formatting with <a href> tags
            formatted_response = final_response
        else:
            # Regular response - escape HTML for safety
            formatted_response = escape_html(final_response)

        # Build reply keyboard with suggestions (workflow or conversation)
        suggestions_keyboard = None
        if final_state:
            # Check for workflow suggestions first (if in a workflow)
            workflow_suggestions = final_state.get("workflow_suggestions", [])
            conversation_suggestions = final_state.get("conversation_suggestions", [])
            
            # Use workflow suggestions if available, otherwise conversation suggestions
            suggestions = workflow_suggestions if workflow_suggestions else conversation_suggestions
            keyboard_type = "workflow" if workflow_suggestions else "conversation"
            
            if suggestions:
                try:
                    from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
                    
                    # Build keyboard with suggestions
                    buttons = [[KeyboardButton(text=suggestion)] for suggestion in suggestions[:3]]
                    suggestions_keyboard = ReplyKeyboardMarkup(
                        keyboard=buttons,
                        resize_keyboard=True,
                        one_time_keyboard=False,
                    )
                    logger.debug(f"📋 Prepared {keyboard_type} keyboard with {len(suggestions)} suggestions")
                except Exception as kb_error:
                    logger.debug(f"Failed to build suggestions keyboard: {kb_error}")

        # Send final message with keyboard attached (no intermediate streaming messages)
        try:
            bot_message = await message.answer(
                formatted_response, 
                parse_mode="HTML",
                reply_markup=suggestions_keyboard
            )
            if suggestions_keyboard:
                kb_type = "workflow" if final_state and final_state.get("workflow_suggestions") else "conversation"
                logger.info(f"✅ Sent response with {kb_type} keyboard ({len(suggestions_keyboard.keyboard)} suggestions)")

            # Log summary
            if settings.STREAMING_ENABLED:
                logger.info(f"✅🏘 Group-aware streaming complete: {len(final_response)} chars, {edit_count} edits")
            else:
                logger.info(f"✅🏘 Group-aware response complete: {len(final_response)} chars, non-streaming mode")
        except Exception:
            pass  # Best effort

        # Update thread metadata (timestamp)
        thread = await thread_service.get_thread(thread_id)
        if thread:
            from datetime import datetime
            thread.updated_at = datetime.utcnow()
            thread.message_count += 1
            await thread_service.update_thread(thread)

        logger.info(f"✅🏘 Group-aware conversation complete for user {user_id} in group {current_group_id}")

    except Exception as e:
        logger.error(f"❌ Error in group-aware message handling: {e}", exc_info=True)
        try:
            await message.answer("❌ Sorry, I encountered an error. Please try again.")
        except:
            pass


@router.message(
    F.chat.type == "private",  # Only private chats
    ~StateFilter(ModerationPromptEditForm.waiting_for_prompt),  # Let group_admin handler process
    ~StateFilter(StoplistEditForm.waiting_for_words),  # Let group_admin handler process
)
async def handle_streaming_message(message: Message, state: FSMContext) -> None:
    """
    Handle text messages in DM with streaming LLM responses.

    Phase 3: Thread-scoped conversations + lazy thread creation
    - Creates thread ONLY on first message (lazy creation)
    - Generates thread name from first message
    - Uses active thread for context
    - Updates thread activity

    Phase 4: Groups navigation support
    - Detects groups_mode and routes to group-aware handler

    Note: Uses F.chat.type == "private" to match private chats only.
    Command exclusion is done via early return check (not filter) because
    aiogram's F.text.startswith() filter doesn't work reliably for command exclusion.

    Phase 5: Form input exclusion
    - Uses NotFillingFormFilter() to prevent matching when user is filling a form
    - This allows form handlers to process the message instead

    Phase 6: Command exclusion
    - Uses early return to exclude all commands (messages starting with "/")
    - Checks for bot_command entity (most reliable method)
    - Fallback check for text starting with "/"
    - This ensures command handlers (like /verify, /start, /reset, etc.) process commands first
    - Only non-command text messages are sent to the LLM
    
    Phase 7: FSM state exclusion
    - Uses ~StateFilter() to exclude messages when user is in specific FSM states
    - ModerationPromptEditForm.waiting_for_prompt - editing moderation rules
    - StoplistEditForm.waiting_for_words - editing stoplist words
    - These states have dedicated handlers in group_admin router
    """
    user_id = message.from_user.id if message.from_user else None
    text = message.text or ""

    if not user_id or not text:
        return

    # CRITICAL: Skip if this is a command - let command handlers process it
    # Check for bot_command entity (more reliable than startswith)
    if message.entities:
        for entity in message.entities:
            if entity.type == "bot_command" and entity.offset == 0:
                logger.debug(f"⏭️ STREAMING_DM: Skipping command message: {text[:50]}")
                return

    # Fallback: also check if text starts with /
    if text.startswith("/"):
        logger.debug(f"⏭️ STREAMING_DM: Skipping message starting with /: {text[:50]}")
        return

    # CHECK: If form is active, log and skip LLM processing
    # Form handlers in processes_router should match first (registered before streaming_router)
    # However, if this handler matched first, we log it and skip processing
    # Check for TRUTHY values, not just key existence (keys may exist with None values after form completion)
    data = await state.get_data()
    if data.get("form_context") or data.get("start_form") or data.get("task_dialog"):
        logger.warning(
            f"⚠️ STREAMING_DM: Handler matched while form active for user {user_id} "
            f"(text length={len(text)}). This shouldn't happen - form handler should match first. "
            f"Update has been consumed by this handler."
        )
        return  # Skip processing but update is already consumed

    # EXPLICIT FSM STATE ROUTING FOR GROUP ADMIN SETTINGS
    # 
    # Problem: When admins edit group settings (moderation rules, stoplist, bot personality)
    # via the admin menu in private chat, they enter an FSM state and send text input.
    # However, aiogram's ~StateFilter() doesn't work reliably across routers, so this
    # streaming handler catches the message instead of the group_admin FSM handlers.
    #
    # Solution: Manually check FSM state and route to the correct handler.
    # See: https://github.com/aiogram/aiogram/discussions/1387
    current_fsm_state = await state.get_state()
    if current_fsm_state:
        # Route to specific FSM handlers in group_admin.py
        if current_fsm_state == "ModerationPromptEditForm:waiting_for_prompt":
            logger.debug(f"🔀 Routing to moderation prompt handler (FSM state active)")
            from luka_bot.handlers.group_admin import handle_moderation_prompt_input
            await handle_moderation_prompt_input(message, state)
            return
        
        if current_fsm_state == "StoplistEditForm:waiting_for_words":
            logger.debug(f"🔀 Routing to stoplist handler (FSM state active)")
            from luka_bot.handlers.group_admin import handle_stoplist_words_input
            await handle_stoplist_words_input(message, state)
            return
        
        if current_fsm_state == "BotPersonalityEditForm:waiting_for_prompt":
            logger.debug(f"🔀 Routing to bot personality handler (FSM state active)")
            from luka_bot.handlers.group_admin import handle_bot_personality_input
            await handle_bot_personality_input(message, state)
            return

    # WORKAROUND: If this is a group message, route it to the group handler
    if message.chat.type in ("group", "supergroup"):
        logger.warning(f"⚠️ STREAMING_DM received group message - routing to group handler")
        logger.warning(f"   chat_id={message.chat.id}, type={message.chat.type}, text='{text[:50]}'")

        # Import and call the group handler directly
        from luka_bot.handlers.group_messages import handle_group_message
        await handle_group_message(message)
        return
    
    # Debug: Log successful private chat message acceptance
    logger.debug(f"✅ STREAMING_DM: Accepted private chat message from user {user_id}: {text[:50]}...")

    logger.info(f"💬 Streaming message from user {user_id}: {text[:50]}...")
    logger.debug(f"🔍 Message source: chat_id={message.chat.id}, chat_type={message.chat.type}")

    # Show typing indicator (with rate limit protection)
    try:
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    except Exception as e:
        logger.debug(f"Skipped typing action: {e}")

    try:
        # Get services
        llm_service = get_llm_service()
        thread_service = get_thread_service()
        
        # Get thread_id first to check for active workflows/suggestions
        thread_id = await thread_service.get_active_thread(user_id)

        # Check FSM state - are we waiting for first message?
        current_state = await state.get_state()

        # NEW: Check if user is in groups mode - route to group-aware handler
        if current_state == NavigationStates.groups_mode:
            await handle_group_aware_message(message, state)
            return

        if current_state == ThreadCreationStates.waiting_for_first_message:
            # LAZY THREAD CREATION - User's first message!
            logger.info(f"✨ First message detected for user {user_id} - checking for existing thread with default agent")

            # Get default sub-agent for platform
            from luka_agent.core.config import settings as luka_settings
            default_sub_agent = getattr(luka_settings, "DEFAULT_SUB_AGENT_TELEGRAM", "general_luka")
            
            # Check if user already has a private thread with this agent
            existing_thread = await thread_service.has_private_thread_with_agent(
                user_id=user_id,
                platform="telegram",
                sub_agent_id=default_sub_agent
            )
            
            if existing_thread:
                # User has existing thread with this agent - route them there
                logger.info(
                    f"✅ Found existing thread {existing_thread.thread_id} with agent {default_sub_agent} "
                    f"for user {user_id} - routing to existing thread"
                )
                
                # Set as active thread
                await thread_service.set_active_thread(user_id, "telegram", existing_thread.thread_id)
                
                # Clear FSM state to allow normal streaming
                await state.clear()
                
                # Update thread_id for streaming
                thread_id = existing_thread.thread_id
                
                logger.info(f"✅ Routed user {user_id} to existing thread {thread_id}")
            else:
                # No existing thread - create new one with default agent
                logger.info(f"📝 No existing thread with agent {default_sub_agent} - creating new thread")

            # Use Redis lock to prevent race conditions
            lock_key = f"thread_creation_lock:{user_id}"

            # Try to acquire lock
            lock_acquired = await redis_client.set(lock_key, "locked", ex=5, nx=True)

            if lock_acquired:
                try:
                    # FIX 4: Create thread with temporary name FIRST (before streaming)
                    # This prevents interference with LLM streaming
                    temp_thread_name = "New Chat"
                    thread = await thread_service.create_thread(user_id, temp_thread_name)
                    thread_id = thread.thread_id

                    # Clear FSM state
                    await state.clear()

                    # Get user language for Step 2 confirmation
                    profile_service = get_user_profile_service()
                    lang = await profile_service.get_language(user_id)

                    # Send Step 2 confirmation (onboarding completion)
                    step2_text = _('onboarding.step2_confirmation', lang)
                    await message.answer(step2_text, parse_mode="HTML")

                    # Prepare keyboard with new thread
                    threads = await thread_service.list_threads(user_id)
                    keyboard = await get_threads_keyboard(threads, thread_id, lang)

                    # Send divider for new thread with updated keyboard
                    await send_thread_divider(
                        user_id,
                        thread_id,
                        divider_type="new",
                        bot=message.bot,
                        reply_markup=keyboard
                    )

                    logger.info(f"✅ Created thread {thread_id} with agent {default_sub_agent}, sent Step 2 confirmation, will generate real name after streaming")

                finally:
                    # Release lock
                    await redis_client.delete(lock_key)
            else:
                # Lock not acquired - another request is creating thread
                # Wait a moment and get the thread
                logger.info(f"⏳ Waiting for thread creation lock for user {user_id}")
                await asyncio.sleep(0.5)
                thread_id = await thread_service.get_active_thread(user_id)

                if not thread_id:
                    # Fallback: create with generic name
                    thread = await thread_service.create_thread(
                        user_id=user_id,
                        platform="telegram",
                        title="Quick Chat",
                        sub_agent_id=default_sub_agent
                    )
                    thread_id = thread.thread_id
                    logger.warning(f"⚠️  Fallback thread created for user {user_id} with agent {default_sub_agent}")

        else:
            # Normal flow - get active thread
            thread_id = await thread_service.get_active_thread(user_id)

            if not thread_id:
                # Check for existing thread with default agent before creating new one
                from luka_agent.core.config import settings as luka_settings
                default_sub_agent = getattr(luka_settings, "DEFAULT_SUB_AGENT_TELEGRAM", "general_luka")
                
                existing_thread = await thread_service.has_private_thread_with_agent(
                    user_id=user_id,
                    platform="telegram",
                    sub_agent_id=default_sub_agent
                )
                
                if existing_thread:
                    # Route to existing thread
                    await thread_service.set_active_thread(user_id, "telegram", existing_thread.thread_id)
                    thread_id = existing_thread.thread_id
                    logger.info(f"✅ Routed to existing thread {thread_id} with agent {default_sub_agent}")
                else:
                    # Shouldn't happen, but fallback just in case
                    thread = await thread_service.create_thread(
                        user_id=user_id,
                        platform="telegram",
                        title="New Chat",
                        sub_agent_id=default_sub_agent
                    )
                    thread_id = thread.thread_id
                    logger.warning(f"⚠️  No active thread found, created fallback for user {user_id} with agent {default_sub_agent}")

        # Get thread object for settings (Phase 4)
        thread = await thread_service.get_thread(thread_id) if thread_id else None

        # Remove reply keyboard when user sends a message (they clicked a button or typed)
        # We'll add it back with the response if suggestions are available
        # Note: We can't send empty messages, so we'll remove keyboard when sending the response instead
        # (Telegram will automatically replace the keyboard when we send a new message with a keyboard)

        # Keep typing status active (no initial message - we'll send final response as new message)
        # This allows us to attach keyboard directly to the response message

        # Stream response with thread context and settings (Phase 4)
        # Phase 5: Configurable streaming with throttling
        # Phase 6: LangGraph integration with feature flag
        full_response = ""  # Initialize accumulator
        last_tool_emoji = None
        final_state = None  # Will be populated by LangGraph stream (if using LangGraph)

        # Throttling variables for streaming mode (for typing status updates)
        last_update_time = time.time()
        last_update_length = 0
        last_sent_text = ""
        edit_count = 0

        # FEATURE FLAG: Use LangGraph agent if enabled
        if settings.LANGGRAPH_ENABLED:
            logger.info(f"🚀 Using LangGraph agent for user {user_id}")
            from luka_bot.lg_lukabot.integration import stream_langgraph_agent
            from luka_bot.services.user_profile_service import get_user_profile_service
            
            # Get user language
            profile_service = get_user_profile_service()
            lang = await profile_service.get_language(user_id)
            
            # Get enabled tools from config and map to LangGraph tool names
            config_tools = settings.DEFAULT_ENABLED_TOOLS
            enabled_tools = map_config_tools_to_langgraph_tools(config_tools)
            logger.debug(f"🔧 Mapped config tools {config_tools} → LangGraph tools: {enabled_tools}")

            # Stream from LangGraph agent
            async for chunk in stream_langgraph_agent(
                message_text=text,
                user_id=user_id,
                thread_id=thread_id,
                language=lang,
                enabled_tools=enabled_tools,
            ):
                # Handle final state (yielded last by stream_langgraph_agent)
                if isinstance(chunk, dict) and chunk.get("type") == "final_state":
                    final_state = chunk.get("state")
                    logger.debug(f"📥 Captured final state from LangGraph stream")
                    continue  # Don't process this as a tool notification or text
                
                # Handle tool notifications - keep typing status active
                if isinstance(chunk, dict) and chunk.get("type") == "tool_notification":
                    tool_emoji = chunk.get("text", "🔧")
                    tool_name = chunk.get("tool_name", "tool")
                    last_tool_emoji = tool_emoji

                    # Keep typing status active while tool executes
                    try:
                        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
                    except Exception:
                        pass
                    
                    logger.info(f"🔧 Tool executing: {tool_name} ({tool_emoji})")
                    continue

                # Regular text chunk - just accumulate, no edits
                if isinstance(chunk, str):
                    full_response += chunk

                    # Keep typing status active during streaming
                    if settings.STREAMING_ENABLED:
                        current_time = time.time()
                        time_elapsed = current_time - last_update_time
                        length_delta = len(full_response) - last_update_length

                        # Update typing status periodically during streaming
                        if (time_elapsed >= settings.STREAMING_UPDATE_INTERVAL and
                            length_delta >= settings.STREAMING_MIN_CHUNK_SIZE):
                            try:
                                await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
                                last_update_time = current_time
                                last_update_length = len(full_response)
                                logger.debug(f"🔄 LangGraph streaming: {len(full_response)} chars accumulated")
                            except Exception:
                                pass
        else:
            # Use luka_agent unified graph
            logger.info(f"🤖 Using luka_agent unified graph for user {user_id}")

            from luka_agent.integration.telegram import stream_telegram_response

            async for event in stream_telegram_response(
                user_message=text,
                user_id=user_id,
                thread_id=thread_id,
                knowledge_bases=thread.knowledge_bases if thread else [f"tg-kb-user-{user_id}"],
                language=thread.language if thread else "en",
                enabled_tools=thread.enabled_tools if thread else ["knowledge_base", "sub_agent", "youtube"]
            ):
                # Convert luka_agent events to chunks
                if event.get("type") == "text_chunk":
                    chunk = event.get("content", "")
                elif event.get("type") == "tool_notification":
                    # Keep typing status active while tool executes
                    tool_emoji = event.get("content", "🔧")
                    last_tool_emoji = tool_emoji

                    try:
                        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
                    except Exception:
                        pass

                    logger.info(f"🔧 Tool executing: {tool_emoji}")
                    continue
                elif event.get("type") == "suggestions":
                    # Suggestions handled after stream completes
                    continue
                else:
                    continue

                # Regular text chunk (string) - ACCUMULATE, don't replace!
                if chunk:
                    full_response += chunk

                    # Keep typing status active during streaming
                    if settings.STREAMING_ENABLED:
                        current_time = time.time()
                        time_elapsed = current_time - last_update_time
                        length_delta = len(full_response) - last_update_length

                        # Update typing status periodically during streaming
                        if (time_elapsed >= settings.STREAMING_UPDATE_INTERVAL and
                                length_delta >= settings.STREAMING_MIN_CHUNK_SIZE):
                            try:
                                await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
                                last_update_time = current_time
                                last_update_length = len(full_response)
                                logger.debug(f"🔄 Streaming: {len(full_response)} chars accumulated")
                            except Exception:
                                pass

        # Final response: Send as NEW message with keyboard attached (if available)
        # This allows us to attach keyboard directly to the response message
        workflow_keyboard = None
        keyboard_type = None  # Track whether it's "workflow" or "conversation"
        
        if settings.LANGGRAPH_ENABLED:
            try:
                from luka_bot.services import get_workflow_service, get_workflow_discovery_service
                from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
                
                workflow_service = get_workflow_service()
                discovery_service = get_workflow_discovery_service()
                await discovery_service.initialize()
                
                # Get all available workflow domains
                available_workflows = await discovery_service.get_available_workflows()
                
                # Check each domain for active workflows for this user
                active_workflow = None
                for domain in available_workflows.keys():
                    wf_status = await workflow_service.get_active_workflow_for_user(user_id, domain)
                    if wf_status:
                        active_workflow = wf_status
                        break
                
                if active_workflow:
                    # Use dynamically generated workflow suggestions from LangGraph state (not hardcoded BPMN)
                    if final_state:
                        suggestions = final_state.get("workflow_suggestions", [])
                        
                        if suggestions:
                            # Build keyboard with dynamic suggestions
                            buttons = [[KeyboardButton(text=suggestion)] for suggestion in suggestions[:3]]
                            workflow_keyboard = ReplyKeyboardMarkup(
                                keyboard=buttons,
                                resize_keyboard=True,
                                one_time_keyboard=False,
                            )
                            keyboard_type = "workflow"
                            logger.info(f"✅ Will attach workflow keyboard with {len(suggestions)} dynamic suggestions to response")
                        else:
                            logger.debug(f"⚠️ No dynamic workflow suggestions in state for step '{active_workflow.current_step}'")
                else:
                    # No active workflow - check for conversation suggestions
                    if final_state:
                        conversation_suggestions = final_state.get("conversation_suggestions", [])
                        if conversation_suggestions:
                            # Build keyboard with conversation suggestions
                            buttons = [[KeyboardButton(text=suggestion)] for suggestion in conversation_suggestions[:3]]
                            workflow_keyboard = ReplyKeyboardMarkup(
                                keyboard=buttons,
                                resize_keyboard=True,
                                one_time_keyboard=False,
                            )
                            keyboard_type = "conversation"
                            logger.info(f"✅ Will attach conversation keyboard with {len(conversation_suggestions)} suggestions to response")
            except Exception as wf_error:
                logger.debug(f"No keyboard to attach: {wf_error}")
        
        try:
            # Don't escape HTML if response contains KB snippets (they have proper HTML formatting)
            has_kb_snippets = '━━━━━━━━━━━━━━━━━━━━' in full_response
            if has_kb_snippets:
                # KB snippets already have proper HTML formatting with <a href> tags
                formatted_response = full_response
            else:
                # Regular response - escape HTML for safety
                formatted_response = escape_html(full_response)

            # Send final response as NEW message (not edited) with keyboard attached
            # Use chunking if response exceeds Telegram's 4096 char limit
            if formatted_response:
                # Split long messages into chunks (respects Telegram's 4096 char limit)
                chunks = split_long_message(formatted_response, max_length=4096)
                
                # Send all chunks, attaching keyboard only to the LAST one
                for idx, chunk in enumerate(chunks):
                    is_last_chunk = (idx == len(chunks) - 1)
                    
                    await message.answer(
                        chunk,
                        parse_mode="HTML",
                        reply_markup=workflow_keyboard if is_last_chunk else None,
                    )
                
                # Log results
                if len(chunks) > 1:
                    logger.info(f"✅ Sent response in {len(chunks)} chunks (total: {len(formatted_response)} chars)")
                
                if workflow_keyboard:
                    kb_label = keyboard_type or "unknown"
                    logger.info(f"✅ Sent response with {kb_label} keyboard ({len(workflow_keyboard.keyboard)} suggestions)")
                else:
                    logger.info(f"✅ Sent response without keyboard")

            # Log summary
            if settings.STREAMING_ENABLED:
                logger.info(f"✅ Streaming complete: {len(full_response)} chars")
            else:
                logger.info(f"✅ Response complete: {len(full_response)} chars, non-streaming mode")
        except Exception as e:
            logger.warning(f"⚠️  Failed to send final response: {e}")

        # FIX 4: Generate thread name AFTER streaming completes (for first messages)
        # This avoids interference with the LLM streaming process
        # Note: We check the thread name instead of FSM state since state was cleared earlier
        try:
            thread = await thread_service.get_thread(thread_id)
            if thread and thread.name == "New Chat":
                # Generate proper thread name from first message
                thread_name = await generate_thread_name(text, language="en")
                logger.info(f"📝 Generated thread name AFTER streaming: '{thread_name}' from '{text[:30]}...'")

                # Update thread with real name
                thread.name = thread_name
                await thread_service.update_thread(thread)
                logger.info(f"✅ Updated thread {thread_id} with generated name")
        except Exception as e:
            logger.warning(f"⚠️  Failed to generate/update thread name: {e}")

        # Update thread activity
        try:
            thread = await thread_service.get_thread(thread_id)
            if thread:
                thread.update_activity()
                await thread_service.update_thread(thread)
        except Exception as e:
            logger.warning(f"⚠️  Failed to update thread activity: {e}")
        
        # Double-write: Index message to ES and send to Camunda
        try:
            # Import utilities for double-write
            from luka_bot.utils.document_id_generator import DocumentIDGenerator
            from luka_bot.services.camunda_service import get_camunda_service
            from luka_bot.services.elasticsearch_service import get_elasticsearch_service
            from luka_bot.services.user_profile_service import get_user_profile_service
            
            # Check if ES is enabled
            if not settings.ELASTICSEARCH_ENABLED:
                logger.debug("⏭️ ES disabled, skipping double-write")
                return
            
            # Get user KB index
            profile_service = get_user_profile_service()
            user_profile = await profile_service.get_profile(user_id)
            if not user_profile:
                logger.warning(f"⚠️ No user profile found for user {user_id}, skipping double-write")
                return
            kb_index = user_profile.kb_index or f"tg-kb-user-{user_id}"
            
            # Generate document ID upfront
            kb_doc_id = DocumentIDGenerator.generate_dm_message_id(
                user_id=user_id,
                thread_id=thread_id,
                telegram_message_id=message.message_id
            )
            
            # Prepare message data
            # Extract parent message details for replies
            parent_message_text = None
            parent_message_id = None
            parent_message_user_id = None
            if message.reply_to_message:
                parent_message_text = message.reply_to_message.text or message.reply_to_message.caption
                parent_message_id = str(message.reply_to_message.message_id)
                parent_message_user_id = str(message.reply_to_message.from_user.id) if message.reply_to_message.from_user else None

            message_data = {
                "message_id": kb_doc_id,  # Use generated document ID
                "user_id": str(user_id),
                "thread_id": thread_id,
                "telegram_topic_id": None,  # DMs don't have topics
                "role": "user",  # DM messages are always from users
                "message_text": text,
                "message_date": message.date.isoformat() if message.date else datetime.utcnow().isoformat(),
                "sender_name": message.from_user.full_name if message.from_user else "Unknown",
                "reply_to_message_id": str(message.reply_to_message.message_id) if message.reply_to_message else "",
                "parent_message_text": parent_message_text,
                "parent_message_id": parent_message_id,
                "parent_message_user_id": parent_message_user_id,
                "mentions": [],  # DM messages don't have mentions
                "hashtags": [],  # DM messages don't have hashtags
                "urls": [],  # DM messages don't have URLs
                "media_type": message.content_type or "text",
            }
            
            # Enhance message data with thread context
            camunda_service = get_camunda_service()
            enhanced_message_data = await camunda_service._build_enhanced_message_data(message_data, thread)
            
            # Double-write: Call both services asynchronously
            es_task = None
            camunda_task = None
            
            try:
                # Start both operations asynchronously
                if settings.ELASTICSEARCH_ENABLED:
                    es_service = await get_elasticsearch_service()
                    es_task = asyncio.create_task(
                        es_service.index_message_immediate(
                            index_name=kb_index,
                            message_data=enhanced_message_data,
                            document_id=kb_doc_id
                        )
                    )
                
                if settings.CAMUNDA_ENABLED and settings.CAMUNDA_MESSAGE_CORRELATION_ENABLED:
                    camunda_task = asyncio.create_task(
                        camunda_service.correlate_message(
                            user_id=str(user_id),
                            message_data=enhanced_message_data,
                            message_type="DM_MESSAGE",
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
                    logger.info(f"✅ Double-write successful: {kb_doc_id}")
                elif es_success:
                    logger.warning(f"⚠️ ES success, Camunda failed: {kb_doc_id}")
                elif camunda_success:
                    logger.warning(f"⚠️ Camunda success, ES failed: {kb_doc_id}")
                else:
                    logger.error(f"❌ Double-write failed: {kb_doc_id}")
                    
            except Exception as e:
                logger.error(f"❌ Error in double-write: {e}")
                # Cancel any pending tasks
                if es_task and not es_task.done():
                    es_task.cancel()
                if camunda_task and not camunda_task.done():
                    camunda_task.cancel()
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to perform double-write: {e}")

    except Exception as e:
        logger.error(f"❌ Streaming error: {e}")

        # Send error message
        error_msg = """❌ Sorry, I encountered an error processing your message.

Please try again or use /start to restart."""

        try:
            # Send error as new message (no bot_message to edit anymore)
                await message.answer(error_msg)
        except:
            pass

