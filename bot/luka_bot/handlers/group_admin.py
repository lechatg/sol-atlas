"""
Group admin control handlers.
"""
from aiogram import Router, F
from aiogram.types import CallbackQuery, Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from loguru import logger
from datetime import datetime
import html

from luka_bot.keyboards.group_admin import create_group_admin_menu, create_group_settings_menu
from luka_bot.services.group_service import get_group_service
from luka_bot.utils.permissions import is_user_admin_in_group
from luka_bot.utils.i18n_helper import _
from luka_bot.core.config import settings

router = Router()


# FSM States for stoplist editing
class StoplistEditForm(StatesGroup):
    """FSM states for editing stoplist."""
    waiting_for_words = State()
    group_id = State()


# FSM States for moderation prompt editing
class ModerationPromptEditForm(StatesGroup):
    """FSM states for editing moderation prompt."""
    waiting_for_prompt = State()
    settings_id = State()


# FSM States for bot personality editing
class BotPersonalityEditForm(StatesGroup):
    """FSM states for editing bot personality (group_bot_prompt)."""
    waiting_for_prompt = State()
    settings_id = State()


# ============================================================================
# Helper Functions
# ============================================================================

async def refresh_admin_menu_simple(callback: CallbackQuery, id: int):
    """
    Helper to refresh admin menu for both user defaults and groups.
    Simplified version that just refreshes the keyboard.
    """
    if id > 0:
        # User defaults
        from luka_bot.handlers.groups_enhanced import handle_user_group_defaults
        await handle_user_group_defaults(callback)
    else:
        # Group - call the main handler which has all the logic
        # We need to simulate the callback with correct data
        # Create a mock callback-like object
        class MockCallback:
            def __init__(self, original):
                self.message = original.message
                self.from_user = original.from_user
                self.bot = original.bot
                self.data = f"group_admin_menu:{id}"
                
        mock = MockCallback(callback)
        # Copy attributes we need
        for attr in dir(callback):
            if not attr.startswith('_') and attr not in ['data', 'message', 'from_user', 'bot']:
                try:
                    setattr(mock, attr, getattr(callback, attr))
                except:
                    pass
        
        await handle_back_to_admin_menu(mock)


@router.callback_query(F.data.startswith("close_menu:"))
async def handle_close_menu(callback: CallbackQuery):
    """Handle close menu button - restores group divider with Settings/Delete buttons."""
    try:
        raw_id = callback.data.split(":")[1]
        target_id = int(raw_id)
        user_id = callback.from_user.id if callback.from_user else None

        if not user_id:
            await callback.answer("❌ Error: User ID not found")
            return

        await callback.message.delete()

        if target_id < 0:
            # Group settings - restore divider
            group_id = target_id
            from luka_bot.services.group_divider_service import send_group_divider
            await send_group_divider(
                user_id=user_id,
                group_id=group_id,
                divider_type="settings_return",
                bot=callback.bot
            )
        else:
            # User defaults - provide friendly notice
            from luka_bot.utils.i18n_helper import get_user_language, _
            lang = await get_user_language(target_id)

            if lang == "ru":
                text = "✅ Меню настроек по умолчанию закрыто. Используйте /groups, чтобы открыть снова."
            else:
                text = "✅ Default settings menu closed. Use /groups to open it again."

            await callback.message.answer(text)

        await callback.answer()
    except Exception as e:
        logger.error(f"Failed to close menu and restore divider: {e}")
        await callback.answer("❌ Failed to close menu")


@router.callback_query(F.data.startswith("group_admin_menu:"))
async def handle_back_to_admin_menu(callback: CallbackQuery):
    """Handle back to admin menu - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Route based on ID sign
        if id > 0:
            # User defaults - call handler directly (can't modify frozen callback.data)
            from luka_bot.handlers.groups_enhanced import handle_user_group_defaults
            return await handle_user_group_defaults(callback)
        
        # Continue with group handling (id < 0)
        group_id = id
        
        # Get services
        from luka_bot.services.moderation_service import get_moderation_service
        from luka_bot.services.thread_service import get_thread_service
        from luka_bot.keyboards.group_admin import create_group_admin_menu
        from luka_bot.services.elasticsearch_service import get_elasticsearch_service
        
        group_service = await get_group_service()
        moderation_service = await get_moderation_service()
        thread_service = get_thread_service()
        
        # Get thread info
        thread = await thread_service.get_group_thread(group_id)
        
        if not thread:
            await callback.answer("❌ Group info not found", show_alert=True)
            return
        
        # 🌟 NEW: Get or refresh metadata
        metadata = await group_service.refresh_group_metadata(group_id, callback.bot, force=False)
        
        if not metadata:
            # Fallback: collect fresh metadata if not cached
            logger.warning(f"⚠️  No metadata for group {group_id}, collecting now")
            metadata = await group_service.collect_group_metadata(
                group_id=group_id,
                bot=callback.bot,
                added_by_user_id=user_id,
                added_by_username=callback.from_user.username,
                added_by_full_name=callback.from_user.full_name or "Unknown"
            )
            metadata.bot_name = thread.agent_name or settings.LUKA_NAME
            metadata.kb_index = thread.knowledge_bases[0] if thread.knowledge_bases else ""
            metadata.thread_id = thread.thread_id
            await group_service.cache_group_metadata(metadata)
        
        # Get moderation settings
        group_settings = await moderation_service.get_group_settings(group_id)
        moderation_enabled = group_settings.moderation_enabled if group_settings else False
        stoplist_count = len(group_settings.stoplist_words) if group_settings else 0
        silent_mode = group_settings.silent_mode if group_settings else False
        ai_assistant_enabled = group_settings.ai_assistant_enabled if group_settings else True
        kb_indexation_enabled = group_settings.kb_indexation_enabled if group_settings else True
        moderate_admins_enabled = group_settings.moderate_admins_enabled if group_settings else False
        
        # Build comprehensive info card using metadata
        language = thread.language
        kb_index = thread.knowledge_bases[0] if thread.knowledge_bases else "Not set"
        
        # Get group type display
        if metadata.group_type == "supergroup":
            group_type = "Supergroup"
            group_type_icon = "👥📌"
            if metadata.has_topics:
                group_type += " (Forum Topics)"
        else:
            group_type = "Group"
            group_type_icon = "👥"
        
        # Language flag
        lang_flag = "🇬🇧 English" if language == "en" else "🇷🇺 Русский"
        
        # Get ES stats
        try:
            es_service = await get_elasticsearch_service()
            index_stats = await es_service.get_index_stats(kb_index)
            message_count = index_stats.get("message_count", 0)
            size_mb = index_stats.get("size_mb", 0.0)
        except:
            message_count = 0
            size_mb = 0.0
        
        # Build settings menu text
        info_text = f"""⚙️ <b>{_('group.settings_menu.title', language)}</b>

{_('group.settings_menu.description', language)}

<b>📊 {metadata.group_title}</b>"""
        
        if metadata.group_username:
            info_text += f" (@{metadata.group_username})"
        
        # Create admin menu keyboard
        admin_keyboard = create_group_admin_menu(
            group_id, 
            metadata.group_title,
            moderation_enabled,
            stoplist_count,
            language,
            silent_mode,
            ai_assistant_enabled,
            kb_indexation_enabled,
            moderate_admins_enabled
        )
        
        try:
            await callback.message.edit_text(
                info_text,
                reply_markup=admin_keyboard,
                parse_mode="HTML"
            )
        except Exception as edit_error:
            # If message is not modified, that's OK
            if "message is not modified" in str(edit_error):
                pass
            else:
                raise
        
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show admin menu: {e}", exc_info=True)
        await callback.answer("❌ Error loading menu")


@router.callback_query(F.data.startswith("group_settings:"))
async def handle_group_settings(callback: CallbackQuery):
    """Handle group settings button - same as handle_back_to_admin_menu."""
    # Just redirect to the universal back handler
    # Replace the callback data to match the pattern
    callback.data = f"group_admin_menu:{callback.data.split(':')[1]}"
    await handle_back_to_admin_menu(callback)


@router.callback_query(F.data.startswith("group_import:"))
async def handle_group_import(callback: CallbackQuery, state: FSMContext):
    """
    Handle history import button - Start BPMN process.
    Initiates the community_audit workflow.
    """
    from aiogram.exceptions import TelegramBadRequest
    
    try:
        group_id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin permissions
        is_admin = await is_user_admin_in_group(callback.bot, group_id, user_id)
        if not is_admin:
            try:
                await callback.answer("⚠️ You must be an admin to import history", show_alert=True)
            except TelegramBadRequest as e:
                if "query is too old" in str(e):
                    logger.warning(f"⏰ Callback query expired (admin check)")
            return
        
        # Store group context in state
        await state.update_data({"current_group_id": group_id})
        
        # Trigger process start (it has its own callback answer handling)
        from luka_bot.handlers.processes.start_process import handle_start_history_import
        await handle_start_history_import(callback, state)
        
        logger.info(f"User {user_id} started history import for group {group_id}")
    
    except TelegramBadRequest as e:
        if "query is too old" in str(e):
            logger.warning(f"⏰ Callback query expired during process start")
        else:
            logger.error(f"Telegram error: {e}")
    except Exception as e:
        logger.error(f"Failed to start import process: {e}")
        try:
            await callback.answer("❌ Error starting process", show_alert=True)
        except TelegramBadRequest as e:
            if "query is too old" in str(e):
                logger.warning(f"⏰ Callback query expired (error handler)")
                try:
                    await callback.message.answer("❌ Error starting process")
                except Exception:
                    pass


@router.callback_query(F.data.startswith("group_stats:"))
async def handle_group_stats(callback: CallbackQuery):
    """Show comprehensive group statistics with weekly activity."""
    try:
        group_id = int(callback.data.split(":")[1])
        
        # Get group info
        group_service = await get_group_service()
        kb_index = await group_service.get_group_kb_index(group_id)
        
        if not kb_index:
            await callback.answer("⚠️ Group not fully initialized", show_alert=True)
            return
        
        # Get language for i18n
        from luka_bot.utils.i18n_helper import _
        current_language = await group_service.get_group_language(group_id)
        
        # Fetch stats from multiple sources
        from luka_bot.services.elasticsearch_service import get_elasticsearch_service
        es_service = await get_elasticsearch_service()
        
        # 1. Get basic index statistics
        index_stats = await es_service.get_index_stats(kb_index)
        total_messages = index_stats.get("message_count", 0)
        size_mb = index_stats.get("size_mb", 0.0)
        
        # 2. Get weekly statistics from Elasticsearch
        weekly_stats = await es_service.get_group_weekly_stats(kb_index)
        unique_users_week = weekly_stats.get("unique_users_week", 0)
        messages_week = weekly_stats.get("total_messages_week", 0)
        top_users_week = weekly_stats.get("top_users_week", [])
        
        # 3. Get member count from Telegram API
        member_count = 0
        try:
            member_count = await callback.bot.get_chat_member_count(group_id)
        except Exception as e:
            logger.warning(f"Failed to get member count for {group_id}: {e}")
        
        # Build stats text with i18n
        title = _('group.stats.title', current_language)
        group_id_label = _('group.stats.group_id', current_language)
        kb_index_label = _('group.stats.kb_index', current_language)
        
        # Use existing translations or add new ones
        if current_language == "en":
            members_label = "👥 Total Members"
            total_messages_label = "📝 Total Messages"
            kb_size_label = "💾 KB Size"
            week_header = "\n📊 <b>Last 7 Days:</b>"
            active_users_label = "👤 Active Users"
            messages_week_label = "💬 Messages Sent"
            top_users_label = "\n🏆 <b>Most Active Members:</b>"
            no_activity_msg = "No activity in the last 7 days"
            no_data_msg = _('group.stats.no_data', current_language)
        else:  # Russian
            members_label = "👥 Всего участников"
            total_messages_label = "📝 Всего сообщений"
            kb_size_label = "💾 Размер БЗ"
            week_header = "\n📊 <b>За последние 7 дней:</b>"
            active_users_label = "👤 Активных пользователей"
            messages_week_label = "💬 Отправлено сообщений"
            top_users_label = "\n🏆 <b>Самые активные:</b>"
            no_activity_msg = "Нет активности за последние 7 дней"
            no_data_msg = _('group.stats.no_data', current_language)
        
        # Build main stats section
        stats_text = f"""{title}

{group_id_label} <code>{group_id}</code>
{kb_index_label} <code>{kb_index}</code>

{members_label}: <b>{member_count:,}</b>
{total_messages_label}: <b>{total_messages:,}</b>
{kb_size_label}: <b>{size_mb:.2f} MB</b>"""
        
        # Add weekly stats if there's any data
        if total_messages > 0:
            if messages_week > 0:
                stats_text += f"""{week_header}
{active_users_label}: <b>{unique_users_week}</b>
{messages_week_label}: <b>{messages_week:,}</b>"""
                
                # Add top users if available
                if top_users_week:
                    stats_text += f"{top_users_label}"
                    for i, user_data in enumerate(top_users_week[:5], 1):  # Show top 5
                        name = user_data['sender_name']
                        count = user_data['message_count']
                        # Medal emojis for top 3
                        if i == 1:
                            medal = "🥇"
                        elif i == 2:
                            medal = "🥈"
                        elif i == 3:
                            medal = "🥉"
                        else:
                            medal = f"{i}."
                        stats_text += f"\n{medal} {name}: {count:,}"
            else:
                # Has historical data but no recent activity
                stats_text += f"\n\n{no_activity_msg}"
        else:
            # No data at all
            stats_text += f"\n\n{no_data_msg}"
        
        # Add privacy mode warning if applicable
        from luka_bot.core.config import settings
        if settings.BOT_PRIVACY_MODE_ENABLED:
            metadata = await group_service.get_cached_group_metadata(group_id)
            
            if metadata and not metadata.bot_is_admin:
                if current_language == "en":
                    privacy_note = (
                        "\n\n⚠️ <b>Note:</b> Privacy Mode is ON. "
                        "Stats only reflect @mention messages. "
                        "Make bot admin to track all messages."
                    )
                else:
                    privacy_note = (
                        "\n\n⚠️ <b>Примечание:</b> Режим приватности включён. "
                        "Статистика учитывает только @упоминания. "
                        "Сделайте бота админом для учёта всех сообщений."
                    )
                
                stats_text += privacy_note
        
        # Add Back button
        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
        back_text = _('common.back', current_language)
        back_keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text=back_text,
                callback_data=f"close_menu:{group_id}"
            )]
        ])
        
        await callback.message.edit_text(
            stats_text,
            reply_markup=back_keyboard,
            parse_mode="HTML"
        )
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show stats: {e}", exc_info=True)
        await callback.answer("❌ Error loading stats", show_alert=True)


@router.callback_query(F.data.startswith("group_threads:"))
async def handle_group_threads(callback: CallbackQuery):
    """Manage group-linked threads."""
    try:
        group_id = int(callback.data.split(":")[1])
        
        # Check if user is admin
        is_admin = await is_user_admin_in_group(callback.bot, group_id, callback.from_user.id)
        if not is_admin:
            await callback.answer("⚠️ You must be an admin to manage threads", show_alert=True)
            return
        
        await callback.answer(
            "🔗 Thread management coming soon!\n\n"
            "This will let you link specific threads to group topics.",
            show_alert=True
        )
    except Exception as e:
        logger.error(f"Failed to handle threads: {e}")
        await callback.answer("❌ Error")


@router.callback_query(F.data.startswith("group_search:"))
async def handle_group_search(callback: CallbackQuery):
    """Quick search group KB."""
    try:
        group_id = int(callback.data.split(":")[1])
        
        await callback.answer(
            "🔍 Use /search command to search this group's knowledge base!",
            show_alert=True
        )
    except Exception as e:
        logger.error(f"Failed to handle search: {e}")
        await callback.answer("❌ Error")


# Placeholder handlers for settings sub-menus
@router.callback_query(F.data.startswith("group_notif:"))
async def handle_notifications(callback: CallbackQuery):
    """Handle notification settings."""
    await callback.answer("🔔 Notification settings coming soon!", show_alert=True)


@router.callback_query(F.data.startswith("group_kb_rules:"))
async def handle_kb_rules(callback: CallbackQuery):
    """Handle KB indexing rules."""
    await callback.answer("📝 KB indexing rules coming soon!", show_alert=True)


@router.callback_query(F.data.startswith("group_access:"))
async def handle_member_access(callback: CallbackQuery):
    """Handle member access settings."""
    await callback.answer("👥 Member access settings coming soon!", show_alert=True)


# New moderation configuration handlers
@router.callback_query(F.data.startswith("group_stoplist_config:"))
async def handle_stoplist_config(callback: CallbackQuery, state: FSMContext = None):
    """Handle stoplist configuration - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        
        # Check if user is admin (only for groups, not user defaults)
        if id < 0:  # Group
            is_admin = await is_user_admin_in_group(callback.bot, id, callback.from_user.id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        group_id = id  # Keep for compatibility
        
        # Get current stoplist (works for both groups and user defaults!)
        from luka_bot.services.moderation_service import get_moderation_service
        from luka_bot.services.group_service import get_group_service
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        
        moderation_service = await get_moderation_service()
        group_service = await get_group_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Get language for i18n
        if id > 0:  # User defaults
            from luka_bot.utils.i18n_helper import get_user_language
            language = await get_user_language(callback.from_user.id)
        else:  # Group
            language = await group_service.get_group_language(group_id)
        
        # Show current stoplist
        stoplist_preview = ", ".join(settings.stoplist_words[:10]) if settings.stoplist_words else _('group.stoplist.empty', language)
        if len(settings.stoplist_words) > 10:
            stoplist_preview += _('group.stoplist.more_words', language, count=len(settings.stoplist_words) - 10)
        
        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
        
        # Build keyboard with i18n
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text=_('group.stoplist.btn.edit', language),
                callback_data=f"group_stoplist_edit:{group_id}"
            )],
            [InlineKeyboardButton(
                text=_('group.stoplist.btn.clear', language),
                callback_data=f"group_stoplist_clear:{group_id}"
            )],
            [InlineKeyboardButton(
                text=_('common.back', language),
                callback_data=f"moderation_menu:{group_id}"
            )]
        ])
        
        message_text = (
            f"{_('group.stoplist.config.title', language)}\n\n"
            f"{_('group.stoplist.config.current', language, count=len(settings.stoplist_words))}\n"
            f"<code>{stoplist_preview}</code>\n\n"
            f"{_('group.stoplist.config.description', language)}\n\n"
            f"{_('group.stoplist.config.tip', language)}"
        )
        
        await callback.message.edit_text(
            message_text,
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show stoplist config: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("sys_msg_menu:"))
async def handle_sys_msg_menu(callback: CallbackQuery):
    """Show system messages filter menu - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        
        # Check admin (only for groups, not user defaults)
        if id < 0:  # Group
            is_admin = await is_user_admin_in_group(callback.bot, id, callback.from_user.id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        group_id = id  # Keep for compatibility
        
        # Get current settings (works for both groups and user defaults!)
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        from luka_bot.keyboards.group_admin import create_system_messages_menu
        from luka_bot.services.group_service import get_group_service
        
        group_service = await get_group_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Get language for i18n
        if id > 0:  # User defaults
            from luka_bot.utils.i18n_helper import get_user_language
            language = await get_user_language(callback.from_user.id)
        else:  # Group
            language = await group_service.get_group_language(group_id)
        
        # Create menu with current state
        keyboard = create_system_messages_menu(group_id, settings.service_message_types, language)
        
        # Build detailed status text (matching content types format)
        type_groups = {
            "joined": {
                "i18n_key": "group.sys_msg.user_joined_left",
                "types": ["new_chat_members", "left_chat_member"]
            },
            "title": {
                "i18n_key": "group.sys_msg.name_title_changes",
                "types": ["new_chat_title"]
            },
            "pinned": {
                "i18n_key": "group.sys_msg.pinned_messages",
                "types": ["pinned_message"]
            },
            "voice": {
                "i18n_key": "group.sys_msg.voice_chat_events",
                "types": ["voice_chat_started", "voice_chat_ended", "voice_chat_scheduled"]
            },
            "photo": {
                "i18n_key": "group.sys_msg.group_photo_changed",
                "types": ["new_chat_photo", "delete_chat_photo"]
            }
        }
        
        # Build header text with status details
        legend = _('group.sys_msg.legend', language)

        header_text = (
            f"{_('group.sys_msg.config.title', language)}\n\n"
            f"{_('group.sys_msg.config.description', language)}\n\n"
            f"{legend}\n\n"
            f"{_('group.sys_msg.config.current_status', language)}"
        )
        
        # Add status for each type
        for key, config in type_groups.items():
            is_filtered = any(t in settings.service_message_types for t in config["types"])
            if is_filtered:
                status = "❌ " + ("скрываем" if language == "ru" else "hidden")
            else:
                status = "✅ " + ("показываем" if language == "ru" else "visible")
            label = _(config['i18n_key'], language)
            header_text += f"\n• {label}: {status}"
        
        await callback.message.edit_text(
            header_text,
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show system messages menu: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("sys_msg_toggle:"))
async def handle_sys_msg_toggle(callback: CallbackQuery):
    """Toggle a system message type filter - works for BOTH groups AND user defaults!"""
    try:
        parts = callback.data.split(":")
        msg_type = parts[1]  # joined, title, pinned, voice, photo
        id = int(parts[2])
        
        # Check admin (only for groups, not user defaults)
        if id < 0:  # Group
            is_admin = await is_user_admin_in_group(callback.bot, id, callback.from_user.id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        group_id = id  # Keep for compatibility
        
        # Type mapping (same as in keyboard)
        type_groups = {
            "joined": ["new_chat_members", "left_chat_member"],
            "title": ["new_chat_title"],
            "pinned": ["pinned_message"],
            "voice": ["voice_chat_started", "voice_chat_ended", "voice_chat_scheduled"],
            "photo": ["new_chat_photo", "delete_chat_photo"]
        }
        
        # Get current settings (works for both groups and user defaults!)
        from luka_bot.services.moderation_service import get_moderation_service
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        from luka_bot.keyboards.group_admin import create_system_messages_menu
        
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Toggle the types
        types_to_toggle = type_groups.get(msg_type, [])
        current_types = set(settings.service_message_types)
        
        # Check if any are currently enabled
        is_enabled = any(t in current_types for t in types_to_toggle)
        
        if is_enabled:
            # Disable: remove all types in this group
            for t in types_to_toggle:
                current_types.discard(t)
            status = "disabled"
        else:
            # Enable: add all types in this group
            current_types.update(types_to_toggle)
            status = "enabled"
        
        # Update settings
        settings.service_message_types = list(current_types)
        settings.delete_service_messages = len(current_types) > 0
        await moderation_service.save_group_settings(settings)
        
        # Get language for proper i18n
        from luka_bot.services.group_service import get_group_service
        group_service = await get_group_service()
        language = await group_service.get_group_language(group_id)
        
        keyboard = create_system_messages_menu(group_id, settings.service_message_types, language)

        # Rebuild descriptive text with legend
        legend = _('group.sys_msg.legend', language)
        header_text = (
            f"{_('group.sys_msg.config.title', language)}\n\n"
            f"{_('group.sys_msg.config.description', language)}\n\n"
            f"{legend}\n\n"
            f"{_('group.sys_msg.config.current_status', language)}"
        )

        type_groups = {
            "joined": {
                "i18n_key": "group.sys_msg.user_joined_left",
                "types": ["new_chat_members", "left_chat_member"]
            },
            "title": {
                "i18n_key": "group.sys_msg.name_title_changes",
                "types": ["new_chat_title"]
            },
            "pinned": {
                "i18n_key": "group.sys_msg.pinned_messages",
                "types": ["pinned_message"]
            },
            "voice": {
                "i18n_key": "group.sys_msg.voice_chat_events",
                "types": ["voice_chat_started", "voice_chat_ended", "voice_chat_scheduled"]
            },
            "photo": {
                "i18n_key": "group.sys_msg.group_photo_changed",
                "types": ["new_chat_photo", "delete_chat_photo"]
            }
        }

        for key, config in type_groups.items():
            is_filtered = any(t in settings.service_message_types for t in config["types"])
            if is_filtered:
                status_text = "❌ " + ("скрываем" if language == "ru" else "hidden")
            else:
                status_text = "✅ " + ("показываем" if language == "ru" else "visible")
            label = _(config['i18n_key'], language)
            header_text += f"• {label}: {status_text}"

        await callback.message.edit_text(header_text, reply_markup=keyboard, parse_mode="HTML")

        if status == "enabled":
            message = "❌ Hidden" if language != "ru" else "❌ Скрываем"
        else:
            message = "✅ Visible" if language != "ru" else "✅ Показываем"
        await callback.answer(message)

    except Exception as e:
        logger.error(f"Failed to toggle system message filter: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("scheduled_content:"))
async def handle_scheduled_content(callback: CallbackQuery):
    """Show scheduled content placeholder."""
    try:
        group_id = int(callback.data.split(":")[1])
        
        # Check admin
        is_admin = await is_user_admin_in_group(callback.bot, group_id, callback.from_user.id)
        if not is_admin:
            await callback.answer("🔒 Admin only", show_alert=True)
            return
        
        # Get language for i18n
        from luka_bot.services.group_service import get_group_service
        from luka_bot.utils.i18n_helper import _
        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
        
        group_service = await get_group_service()
        language = await group_service.get_group_language(group_id)
        
        # Build message with proper i18n
        message_text = f"""{_('group.scheduled_content.title', language)}

{_('group.scheduled_content.coming_soon', language)}

{_('group.scheduled_content.features', language)}

{_('group.scheduled_content.stay_tuned', language)}"""
        
        back_button_text = _('common.back', language)
        
        await callback.message.edit_text(
            message_text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text=back_button_text, callback_data=f"close_menu:{group_id}")]
            ]),
            parse_mode="HTML"
        )
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show scheduled content: {e}")
        await callback.answer("❌ Error", show_alert=True)


# DEPRECATED: Old "coming soon" handler - replaced by Camunda BPMN process
# The actual handler is handle_group_import() above (line ~279)
# @router.callback_query(F.data.startswith("group_import_kb:"))
# async def handle_group_import_kb(callback: CallbackQuery):
#     """Show import history placeholder."""
#     # This handler is no longer used - Import History now triggers Camunda BPMN process


# Stoplist edit handlers
@router.callback_query(F.data.startswith("group_stoplist_edit:"))
async def handle_stoplist_edit(callback: CallbackQuery, state: FSMContext):
    """Prompt user to edit stoplist - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        group_id = id  # Keep for compatibility
        
        # Get current settings to show existing stoplist
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Get language for i18n
        from luka_bot.services.group_service import get_group_service
        group_service = await get_group_service()
        
        if id > 0:  # User defaults
            from luka_bot.utils.i18n_helper import get_user_language
            language = await get_user_language(callback.from_user.id)
        else:  # Group
            language = await group_service.get_group_language(group_id)
        
        # Store id in FSM (can be user_id or group_id)
        await state.update_data(group_id=id)
        await state.set_state(StoplistEditForm.waiting_for_words)
        
        # Build prompt with current stoplist or example
        if settings.stoplist_words:
            current_words = ", ".join(settings.stoplist_words)
            prompt_text = (
                f"{_('group.stoplist.edit.title', language)}\n\n"
                f"{_('group.stoplist.edit.instruction', language)}\n\n"
                f"<b>{_('group.stoplist.edit.current', language)}</b>\n"
                f"<code>{current_words}</code>\n\n"
                f"{_('group.stoplist.edit.format', language)}\n\n"
                f"{_('group.stoplist.edit.note', language)}"
            )
        else:
            prompt_text = (
                f"{_('group.stoplist.edit.title', language)}\n\n"
                f"{_('group.stoplist.edit.instruction', language)}\n\n"
                f"<i>{_('group.stoplist.empty', language)}</i>\n\n"
                f"{_('group.stoplist.edit.format', language)}\n\n"
                f"{_('group.stoplist.edit.example', language)}\n\n"
                f"{_('group.stoplist.edit.note', language)}"
            )
        
        await callback.message.edit_text(prompt_text, parse_mode="HTML")
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to start stoplist edit: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.message(StoplistEditForm.waiting_for_words, F.text)
async def handle_stoplist_words_input(message: Message, state: FSMContext):
    """Handle stoplist words input."""
    try:
        # Get id from FSM first (needed for both cancel and submit)
        data = await state.get_data()
        id = data.get("group_id")  # Keep name for compatibility
        
        if not id:
            await message.reply("❌ Session expired. Please try again.")
            await state.clear()
            return
        
        group_id = id  # Keep for compatibility
        
        # Get language for i18n (works for both groups and user defaults)
        from luka_bot.utils.i18n_helper import _, get_user_language
        from luka_bot.services.group_service import get_group_service
        
        group_service = await get_group_service()
        
        if id > 0:  # User defaults
            language = await get_user_language(message.from_user.id)
        else:  # Group
            language = await group_service.get_group_language(group_id)
        
        # Check for cancel
        if message.text.lower().strip() == "/cancel":
            await state.clear()
            
            # Get settings for stoplist menu
            from luka_bot.handlers.groups_enhanced import get_settings_for_id
            settings = await get_settings_for_id(id)
            
            if not settings:
                await message.reply("⚠️ Settings not found")
                return
            
            # Show stoplist config menu
            stoplist_preview = ", ".join(settings.stoplist_words[:10]) if settings.stoplist_words else _('group.stoplist.empty', language)
            if len(settings.stoplist_words) > 10:
                stoplist_preview += _('group.stoplist.more_words', language, count=len(settings.stoplist_words) - 10)
            
            from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(
                    text=_('group.stoplist.btn.edit', language),
                    callback_data=f"group_stoplist_edit:{group_id}"
                )],
                [InlineKeyboardButton(
                    text=_('group.stoplist.btn.clear', language),
                    callback_data=f"group_stoplist_clear:{group_id}"
                )],
                [InlineKeyboardButton(
                    text=_('common.back', language),
                    callback_data=f"group_admin_menu:{group_id}"
                )]
            ])
            
            message_text = (
                f"{_('group.stoplist.config.title', language)}\n\n"
                f"{_('group.stoplist.config.current', language, count=len(settings.stoplist_words))}\n"
                f"<code>{stoplist_preview}</code>\n\n"
                f"{_('group.stoplist.config.description', language)}\n\n"
                f"{_('group.stoplist.config.tip', language)}"
            )
            
            cancel_msg = "❌ " + (_('common.cancelled', language) if language == "en" else "Отменено")
            await message.reply(cancel_msg)
            await message.answer(message_text, reply_markup=keyboard, parse_mode="HTML")
            return
        
        # Verify admin status (only for groups, not user defaults)
        if id < 0:  # Group
            is_admin = await is_user_admin_in_group(message.bot, id, message.from_user.id)
            if not is_admin:
                await message.reply("🔒 You must be an admin in that group")
                await state.clear()
                return
        
        # Parse words
        words = [w.strip().lower() for w in message.text.split(",") if w.strip()]
        
        if not words:
            await message.reply("❌ No valid words provided. Try again or send /cancel")
            return
        
        # Update stoplist (works for both groups and user defaults!)
        from luka_bot.services.moderation_service import get_moderation_service
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await message.reply("⚠️ Settings not found")
            await state.clear()
            return
        
        settings.stoplist_words = words
        settings.stoplist_enabled = True  # Enable stoplist when words are set
        await moderation_service.save_group_settings(settings)
        
        # Clear FSM
        await state.clear()
        
        # Show confirmation with i18n
        preview = ", ".join(words[:10])
        if len(words) > 10:
            preview += _('group.stoplist.more_words', language, count=len(words) - 10)
        
        confirmation_text = (
            f"{_('group.stoplist.updated', language)}\n\n"
            f"{_('group.stoplist.total_words', language, count=len(words))}\n"
            f"{_('group.stoplist.preview', language, preview=preview)}\n\n"
            f"{_('group.stoplist.auto_delete_notice', language)}"
        )
        
        await message.reply(confirmation_text, parse_mode="HTML")
        
        # Show stoplist config menu
        stoplist_preview = ", ".join(settings.stoplist_words[:10]) if settings.stoplist_words else _('group.stoplist.empty', language)
        if len(settings.stoplist_words) > 10:
            stoplist_preview += _('group.stoplist.more_words', language, count=len(settings.stoplist_words) - 10)
        
        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text=_('group.stoplist.btn.edit', language),
                callback_data=f"group_stoplist_edit:{group_id}"
            )],
            [InlineKeyboardButton(
                text=_('group.stoplist.btn.clear', language),
                callback_data=f"group_stoplist_clear:{group_id}"
            )],
            [InlineKeyboardButton(
                text=_('common.back', language),
                callback_data=f"group_admin_menu:{group_id}"
            )]
        ])
        
        message_text = (
            f"{_('group.stoplist.config.title', language)}\n\n"
            f"{_('group.stoplist.config.current', language, count=len(settings.stoplist_words))}\n"
            f"<code>{stoplist_preview}</code>\n\n"
            f"{_('group.stoplist.config.description', language)}\n\n"
            f"{_('group.stoplist.config.tip', language)}"
        )
        
        await message.answer(message_text, reply_markup=keyboard, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Failed to update stoplist: {e}")
        await message.reply(f"❌ Error: {e}")
        await state.clear()


@router.callback_query(F.data.startswith("group_stoplist_clear:"))
async def handle_stoplist_clear(callback: CallbackQuery):
    """Clear the stoplist - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        
        # Check if user is admin (only for groups, not user defaults)
        if id < 0:  # Group
            is_admin = await is_user_admin_in_group(callback.bot, id, callback.from_user.id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        group_id = id  # Keep for compatibility
        
        # Clear stoplist (works for both groups and user defaults!)
        from luka_bot.services.moderation_service import get_moderation_service
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if settings:
            settings.stoplist_words = []
            settings.stoplist_enabled = False  # Disable stoplist when cleared
            await moderation_service.save_group_settings(settings)
            await callback.answer("✅ Stoplist cleared!", show_alert=True)
            
            # Refresh the view
            await handle_stoplist_config(callback)
        else:
            await callback.answer("⚠️ Group not initialized", show_alert=True)
            
    except Exception as e:
        logger.error(f"Failed to clear stoplist: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("group_refresh_metadata:"))
async def handle_refresh_metadata(callback: CallbackQuery):
    """Handle metadata refresh button - force refresh group metadata from Telegram API."""
    try:
        group_id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check if user is admin
        is_admin = await is_user_admin_in_group(callback.bot, group_id, user_id)
        if not is_admin:
            await callback.answer("⚠️ You must be an admin to refresh metadata", show_alert=True)
            return
        
        # Show loading indicator
        await callback.answer("🔄 Refreshing group info...", show_alert=False)
        
        # Force refresh metadata
        group_service = await get_group_service()
        metadata = await group_service.refresh_group_metadata(group_id, callback.bot, force=True)
        
        if metadata:
            logger.info(f"✅ Metadata refreshed for group {group_id} by user {user_id}")
            
            # Redirect to admin menu to show updated info
            callback.data = f"group_admin_menu:{group_id}"
            await handle_back_to_admin_menu(callback)
        else:
            await callback.answer("❌ Failed to refresh metadata", show_alert=True)
            
    except Exception as e:
        logger.error(f"Failed to refresh metadata: {e}", exc_info=True)
        await callback.answer("❌ Error refreshing metadata", show_alert=True)



@router.callback_query(F.data.startswith("group_toggle_silent:"))
async def handle_toggle_silent_mode(callback: CallbackQuery):
    """Toggle silent mode - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin (only for groups)
        if id < 0:  # Group
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        # Get settings (works for both!)
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        from luka_bot.services.moderation_service import get_moderation_service
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Toggle silent mode
        settings.silent_mode = not settings.silent_mode
        settings.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(settings)
        
        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(user_id)
        else:
            from luka_bot.services.group_service import get_group_service
            group_service = await get_group_service()
            lang = await group_service.get_group_language(id)

        if settings.silent_mode:
            message = "🔕 Silent mode enabled" if lang != "ru" else "🔕 Режим тишины включён"
        else:
            message = "🔔 Notifications enabled" if lang != "ru" else "🔔 Уведомления включены"
        await callback.answer(message)

        # Refresh menu
        await refresh_admin_menu_simple(callback, id)

    except Exception as e:
        logger.error(f"Failed to toggle silent mode: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("group_toggle_ai:"))
async def handle_toggle_ai_assistant(callback: CallbackQuery):
    """Toggle AI assistant - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin (only for groups)
        if id < 0:  # Group
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        # Get settings (works for both!)
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        from luka_bot.services.moderation_service import get_moderation_service
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Toggle
        settings.ai_assistant_enabled = not settings.ai_assistant_enabled
        settings.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(settings)
        
        # Determine language for responses
        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(user_id)
        else:
            from luka_bot.services.group_service import get_group_service
            group_service = await get_group_service()
            lang = await group_service.get_group_language(id)

        # Re-render submenu with updated state
        await _render_ai_assistant_menu(callback, id, user_id)

        if settings.ai_assistant_enabled:
            message = "🤖 AI assistant enabled" if lang != "ru" else "🤖 Ассистент включён"
        else:
            message = "🤖 AI assistant disabled" if lang != "ru" else "🤖 Ассистент отключён"
        await callback.answer(message)
        
    except Exception as e:
        logger.error(f"Failed to toggle AI assistant: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("group_toggle_respond_all:"))
async def handle_toggle_respond_to_all(callback: CallbackQuery):
    """Toggle respond to all messages - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        if id < 0:
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        from luka_bot.services.moderation_service import get_moderation_service
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Only allow toggle if AI assistant is enabled
        if not settings.ai_assistant_enabled:
            await callback.answer("⚠️ Enable AI Assistant first", show_alert=True)
            return
        
        settings.respond_to_all_messages = not settings.respond_to_all_messages
        settings.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(settings)

        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(user_id)
        else:
            from luka_bot.services.group_service import get_group_service
            group_service = await get_group_service()
            lang = await group_service.get_group_language(id)

        await _render_ai_assistant_menu(callback, id, user_id)

        if settings.respond_to_all_messages:
            message = "💬 Responding to all messages" if lang != "ru" else "💬 Ответ на все сообщения"
        else:
            message = "📝 Only mentions/replies" if lang != "ru" else "📝 Только упоминания/ответы"

        await callback.answer(message)

    except Exception as e:
        logger.error(f"Failed to toggle respond to all: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("group_toggle_kb:"))
async def handle_toggle_kb_indexation(callback: CallbackQuery):
    """Toggle KB indexation - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        if id < 0:
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        from luka_bot.services.moderation_service import get_moderation_service
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        settings.kb_indexation_enabled = not settings.kb_indexation_enabled
        settings.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(settings)

        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(user_id)
        else:
            from luka_bot.services.group_service import get_group_service
            group_service = await get_group_service()
            lang = await group_service.get_group_language(id)

        await _render_ai_assistant_menu(callback, id, user_id)

        if settings.kb_indexation_enabled:
            message = "📚 Indexation enabled" if lang != "ru" else "📚 Индексация включена"
        else:
            message = "📚 Indexation disabled" if lang != "ru" else "📚 Индексация отключена"

        from luka_bot.core.config import settings as bot_settings
        if id < 0 and bot_settings.BOT_PRIVACY_MODE_ENABLED and settings.kb_indexation_enabled:
            warning = "⚠️ Privacy Mode: only @mentions will be indexed" if lang != "ru" else "⚠️ Режим приватности: индексируются только упоминания"
            message += f"\n\n{warning}"

        await callback.answer(message, show_alert="\n" in message)

    except Exception as e:
        logger.error(f"Failed to toggle KB indexation: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("group_toggle_manual_kb:"))
async def handle_toggle_manual_kb_gathering(callback: CallbackQuery):
    """Toggle Manual KB Gathering - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        if id < 0:
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        from luka_bot.services.moderation_service import get_moderation_service
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        settings.manual_kb_gathering_enabled = not settings.manual_kb_gathering_enabled
        settings.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(settings)

        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(user_id)
        else:
            from luka_bot.services.group_service import get_group_service
            group_service = await get_group_service()
            lang = await group_service.get_group_language(id)

        await _render_ai_assistant_menu(callback, id, user_id)

        if settings.manual_kb_gathering_enabled:
            message = "➕ Manual adding to KB enabled" if lang != "ru" else "➕ Ручное добавление в БЗ включено"
        else:
            message = "➕ Manual adding to KB disabled" if lang != "ru" else "➕ Ручное добавление в БЗ отключено"

        await callback.answer(message)

    except Exception as e:
        logger.error(f"Failed to toggle manual KB gathering: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("ai_assistant_menu:"))
async def handle_ai_assistant_menu(callback: CallbackQuery):
    """Show AI Assistant submenu with ON/OFF toggles."""
    try:
        settings_id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id

        if settings_id < 0:
            is_admin = await is_user_admin_in_group(callback.bot, settings_id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return

        if not await _render_ai_assistant_menu(callback, settings_id, user_id):
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return

        await callback.answer()

    except Exception as e:
        logger.error(f"Failed to show AI Assistant menu: {e}")
        await callback.answer("❌ Error", show_alert=True)


# ============================================================================
# AI Assistant Submenu
# ============================================================================

async def _render_ai_assistant_menu(callback: CallbackQuery, settings_id: int, user_id: int) -> bool:
    from luka_bot.handlers.groups_enhanced import get_settings_for_id
    from luka_bot.services.group_service import get_group_service
    from luka_bot.keyboards.group_admin import create_ai_assistant_menu

    settings = await get_settings_for_id(settings_id)
    if not settings:
        return False

    group_service = await get_group_service()
    if settings_id > 0:
        from luka_bot.utils.i18n_helper import get_user_language
        lang = await get_user_language(user_id)
    else:
        lang = await group_service.get_group_language(settings_id)

    keyboard = create_ai_assistant_menu(
        settings_id,
        settings.ai_assistant_enabled,
        settings.respond_to_all_messages,
        settings.kb_indexation_enabled,
        settings.manual_kb_gathering_enabled,
        lang
    )

    text = f"""🤖 <b>{_('group.ai_assistant_menu.title', lang)}</b>

{_('group.ai_assistant_menu.description', lang)}"""

    await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    return True


# ============================================================================
# Moderation Submenu
# ============================================================================

async def _render_moderation_menu(callback: CallbackQuery, settings_id: int, user_id: int) -> bool:
    from luka_bot.handlers.groups_enhanced import get_settings_for_id
    from luka_bot.services.group_service import get_group_service
    from luka_bot.keyboards.group_admin import create_moderation_menu

    settings = await get_settings_for_id(settings_id)
    if not settings:
        return False

    group_service = await get_group_service()
    if settings_id > 0:
        from luka_bot.utils.i18n_helper import get_user_language
        lang = await get_user_language(user_id)
    else:
        lang = await group_service.get_group_language(settings_id)

    keyboard = create_moderation_menu(
        settings_id,
        settings.moderation_enabled,
        settings.moderate_admins_enabled,
        lang
    )

    text = f"""🛡️ <b>{_('group.moderation_menu.title', lang)}</b>

{_('group.moderation_menu.description', lang)}"""

    await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    return True


@router.callback_query(F.data.startswith("moderation_menu:"))
async def handle_moderation_menu(callback: CallbackQuery):
    """Show Moderation submenu with ON/OFF toggles and links to other submenus."""
    try:
        settings_id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id

        if settings_id < 0:
            is_admin = await is_user_admin_in_group(callback.bot, settings_id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return

        if not await _render_moderation_menu(callback, settings_id, user_id):
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return

        await callback.answer()

    except Exception as e:
        logger.error(f"Failed to show Moderation menu: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("group_toggle_mod_admins:"))
async def handle_toggle_moderate_admins(callback: CallbackQuery):
    """Toggle moderate admins - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin (only for groups)
        if id < 0:  # Group
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        # Get settings (works for both!)
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        from luka_bot.services.moderation_service import get_moderation_service
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Toggle
        settings.moderate_admins_enabled = not settings.moderate_admins_enabled
        settings.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(settings)
        
        # Determine language for messaging
        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(user_id)
        else:
            from luka_bot.services.group_service import get_group_service
            group_service = await get_group_service()
            lang = await group_service.get_group_language(id)

        await _render_moderation_menu(callback, id, user_id)

        if settings.moderate_admins_enabled:
            message = "🛡️ Admin moderation enabled" if lang != "ru" else "🛡️ Модерация админов включена"
        else:
            message = "🛡️ Admin moderation disabled" if lang != "ru" else "🛡️ Модерация админов отключена"
        await callback.answer(message)
        
    except Exception as e:
        logger.error(f"Failed to toggle moderate admins: {e}")
        await callback.answer("❌ Error", show_alert=True)


# ============================================================================
# Content Types Filter (works for BOTH groups and user defaults)
# ============================================================================

@router.callback_query(F.data.startswith("content_types_menu:"))
async def handle_content_types_menu(callback: CallbackQuery):
    """Show content types filter menu - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin for groups
        if id < 0:  # Group
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        # Get settings (works for both!)
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Get language
        from luka_bot.services.group_service import get_group_service
        group_service = await get_group_service()
        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(user_id)
        else:
            lang = await group_service.get_group_language(id)
        
        # Create menu using the keyboard function
        from luka_bot.keyboards.group_admin import create_content_types_menu
        keyboard = create_content_types_menu(id, settings, lang)
        
        # Build text
        legend = _('user_group_defaults.content_types.legend', lang)

        text = f"""🗂️ <b>{_('user_group_defaults.content_types', lang)}</b>

{_('user_group_defaults.content_types_desc', lang)}

{legend}

<b>{_('user_group_defaults.content_type_status', lang)}:</b>"""
        
        # Add status for each type
        content_types = [
            ("delete_links", "user_group_defaults.content_links"),
            ("delete_images", "user_group_defaults.content_images"),
            ("delete_videos", "user_group_defaults.content_videos"),
            ("delete_stickers", "user_group_defaults.content_stickers"),
            ("delete_forwarded", "user_group_defaults.content_forwarded")
        ]
        
        for attr, i18n_key in content_types:
            is_filtered = getattr(settings, attr, False)
            if is_filtered:
                status = "❌ " + ("скрываем" if lang == "ru" else "hidden")
            else:
                status = "✅ " + ("показываем" if lang == "ru" else "visible")
            label = _(i18n_key, lang)
            text += f"\n• {label}: {status}"
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show content types menu: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("content_type_toggle:"))
async def handle_content_type_toggle(callback: CallbackQuery):
    """Toggle a content type filter - works for BOTH groups AND user defaults!"""
    try:
        parts = callback.data.split(":")
        content_type = parts[1]  # e.g., "delete_links"
        id = int(parts[2])
        user_id = callback.from_user.id

        # Check admin for groups
        if id < 0:  # Group
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return

        from luka_bot.services.moderation_service import get_moderation_service
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)

        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return

        current_value = getattr(settings, content_type, False)
        setattr(settings, content_type, not current_value)
        settings.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(settings)

        from luka_bot.services.group_service import get_group_service
        group_service = await get_group_service()
        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(user_id)
        else:
            lang = await group_service.get_group_language(id)

        from luka_bot.keyboards.group_admin import create_content_types_menu
        keyboard = create_content_types_menu(id, settings, lang)

        legend = _('user_group_defaults.content_types.legend', lang)
        text_body = f"""🗂️ <b>{_('user_group_defaults.content_types', lang)}</b>

{_('user_group_defaults.content_types_desc', lang)}

{legend}

<b>{_('user_group_defaults.content_type_status', lang)}:</b>"""

        content_types = [
            ("delete_links", "user_group_defaults.content_links"),
            ("delete_images", "user_group_defaults.content_images"),
            ("delete_videos", "user_group_defaults.content_videos"),
            ("delete_stickers", "user_group_defaults.content_stickers"),
            ("delete_forwarded", "user_group_defaults.content_forwarded")
        ]

        for attr, i18n_key in content_types:
            is_filtered = getattr(settings, attr, False)
            if is_filtered:
                status_text = "❌ " + ("скрываем" if lang == "ru" else "hidden")
            else:
                status_text = "✅ " + ("показываем" if lang == "ru" else "visible")
            label = _(i18n_key, lang)
            text_body += f"\n• {label}: {status_text}"

        await callback.message.edit_text(text_body, reply_markup=keyboard, parse_mode="HTML")

        if getattr(settings, content_type, False):
            message = "❌ Hidden" if lang != "ru" else "❌ Скрываем"
        else:
            message = "✅ Visible" if lang != "ru" else "✅ Показываем"
        await callback.answer(message)

    except Exception as e:
        logger.error(f"Failed to toggle content type: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("moderation_prompt_menu:"))
async def handle_moderation_prompt_menu(callback: CallbackQuery):
    """Show moderation rules/prompt menu - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin for groups
        if id < 0:  # Group
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        # Get settings
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Get language
        from luka_bot.services.group_service import get_group_service
        group_service = await get_group_service()
        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(user_id)
        else:
            lang = await group_service.get_group_language(id)
        
        # Build text
        text = f"""📝 <b>{_('user_group_defaults.moderation_rules', lang)}</b>

{_('user_group_defaults.moderation_prompt_desc', lang)}

<b>{_('user_group_defaults.current_prompt', lang)}:</b>"""
        
        if settings.moderation_prompt:
            text += f"\n\n<code>{settings.moderation_prompt[:500]}</code>"
            if len(settings.moderation_prompt) > 500:
                text += "..."
        else:
            text += f"\n\n<i>{_('user_group_defaults.using_default_prompt', lang)}</i>"
        
        # Create keyboard
        buttons = []
        
        # View full prompt
        if settings.moderation_prompt:
            buttons.append([
                InlineKeyboardButton(
                    text=f"👁️ {_('user_group_defaults.view_full_prompt', lang)}",
                    callback_data=f"mod_prompt_view:{id}"
                )
            ])
        
        # Edit prompt
        buttons.append([
            InlineKeyboardButton(
                text=f"✏️ {_('user_group_defaults.edit_prompt', lang)}",
                callback_data=f"mod_prompt_edit:{id}"
            )
        ])
        
        # Reset to default (if custom prompt exists)
        if settings.moderation_prompt:
            buttons.append([
                InlineKeyboardButton(
                    text=f"🔄 {_('user_group_defaults.reset_to_default', lang)}",
                    callback_data=f"mod_prompt_reset:{id}"
                )
            ])
        
        # Back button - return to AI Assistant menu
        buttons.append([
            InlineKeyboardButton(
                text=_('common.back', lang),
                callback_data=f"ai_assistant_menu:{id}"
            )
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show moderation prompt menu: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("mod_prompt_view:"))
async def handle_mod_prompt_view(callback: CallbackQuery):
    """Send the full moderation prompt to the user."""
    try:
        raw_id = callback.data.split(":", 1)[1]
        settings_id = int(raw_id)
        user_id = callback.from_user.id

        # Determine language and permissions
        if settings_id < 0:
            group_id = settings_id
            is_admin = await is_user_admin_in_group(callback.bot, group_id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return

            from luka_bot.services.group_service import get_group_service
            group_service = await get_group_service()
            language = await group_service.get_group_language(group_id)
        else:
            from luka_bot.utils.i18n_helper import get_user_language
            language = await get_user_language(settings_id)

        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        settings = await get_settings_for_id(settings_id)

        if not settings or not settings.moderation_prompt:
            await callback.answer("⚠️ No custom prompt found", show_alert=True)
            return

        prompt = settings.moderation_prompt
        escaped_prompt = html.escape(prompt)

        header = "📝 <b>Full Moderation Prompt:</b>\n\n"
        chunk_size = 3900  # Leave room for HTML wrappers

        parts = []
        for start in range(0, len(escaped_prompt), chunk_size):
            chunk = escaped_prompt[start : start + chunk_size]
            parts.append(f"<code>{chunk}</code>")

        if parts:
            # Send header with first chunk
            first_message = header + parts[0]
            await callback.message.answer(first_message, parse_mode="HTML")
            for part in parts[1:]:
                await callback.message.answer(part, parse_mode="HTML")

        await callback.answer("📄 Prompt sent below")

    except Exception as e:
        logger.error(f"Failed to show full prompt: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("mod_prompt_edit:"))
async def handle_moderation_prompt_edit(callback: CallbackQuery, state: FSMContext):
    """Start editing moderation prompt - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin for groups
        if id < 0:
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        # Get current settings
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Get language
        if id > 0:  # User defaults
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(user_id)
        else:  # Group
            from luka_bot.services.group_service import get_group_service
            group_service = await get_group_service()
            lang = await group_service.get_group_language(id)
        
        # Store ID in FSM
        await state.update_data(settings_id=id)
        await state.set_state(ModerationPromptEditForm.waiting_for_prompt)
        
        # Log FSM state for debugging (can be removed once stable)
        logger.debug(f"FSM state set: ModerationPromptEditForm, settings_id={id}")
        
        # Get current prompt or show default message
        current_prompt = settings.moderation_prompt
        
        # Send prompt instructions with current value
        if lang == "en":
            if current_prompt:
                text = f"""✏️ <b>Edit Moderation Rules</b>

Send me the new moderation prompt/rules you want to use.

<b>Current prompt:</b>
<code>{current_prompt[:500]}</code>{"..." if len(current_prompt) > 500 else ""}

Send /cancel to abort."""
            else:
                text = """✏️ <b>Edit Moderation Rules</b>

Send me the custom moderation prompt/rules you want to use.

<i>Currently using default prompt.</i>

<b>Example:</b>
<code>Be strict about spam and advertising. 
Allow friendly discussions and questions.
Remove offensive language.</code>

Send /cancel to abort."""
        else:
            if current_prompt:
                text = f"""✏️ <b>Редактирование правил модерации</b>

Отправьте мне новый промпт/правила модерации.

<b>Текущий промпт:</b>
<code>{current_prompt[:500]}</code>{"..." if len(current_prompt) > 500 else ""}

Отправьте /cancel для отмены."""
            else:
                text = """✏️ <b>Редактирование правил модерации</b>

Отправьте мне пользовательский промпт/правила модерации.

<i>Сейчас используется промпт по умолчанию.</i>

<b>Пример:</b>
<code>Строго относитесь к спаму и рекламе.
Разрешайте дружеские обсуждения и вопросы.
Удаляйте оскорбительные выражения.</code>

Отправьте /cancel для отмены."""
        
        await callback.message.edit_text(text, parse_mode="HTML")
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to start prompt edit: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.message(ModerationPromptEditForm.waiting_for_prompt, F.text)
async def handle_moderation_prompt_input(message: Message, state: FSMContext):
    """Handle moderation prompt input."""
    try:
        # Get ID from FSM first (needed for both cancel and submit)
        data = await state.get_data()
        id = data.get("settings_id")
        
        if not id:
            await message.reply("❌ Session expired. Please try again.")
            await state.clear()
            return
        
        # Get language
        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(message.from_user.id)
        else:
            from luka_bot.services.group_service import get_group_service
            group_service = await get_group_service()
            lang = await group_service.get_group_language(id)
        
        # Check for cancel
        if message.text.lower().strip() == "/cancel":
            await state.clear()
            
            # Get settings and show moderation prompt menu
            from luka_bot.handlers.groups_enhanced import get_settings_for_id
            settings = await get_settings_for_id(id)
            
            if not settings:
                await message.reply("⚠️ Settings not found")
                return
            
            # Build moderation prompt menu
            text = f"""📝 <b>{_('user_group_defaults.moderation_rules', lang)}</b>

{_('user_group_defaults.moderation_prompt_desc', lang)}

<b>{_('user_group_defaults.current_prompt', lang)}:</b>"""
            
            if settings.moderation_prompt:
                text += f"\n\n<code>{settings.moderation_prompt[:500]}</code>"
                if len(settings.moderation_prompt) > 500:
                    text += "..."
            else:
                text += f"\n\n<i>{_('user_group_defaults.using_default_prompt', lang)}</i>"
            
            # Create keyboard
            buttons = []
            
            # View full prompt
            if settings.moderation_prompt:
                buttons.append([
                    InlineKeyboardButton(
                        text=f"👁️ {_('user_group_defaults.view_full_prompt', lang)}",
                        callback_data=f"mod_prompt_view:{id}"
                    )
                ])
            
            # Edit prompt
            buttons.append([
                InlineKeyboardButton(
                    text=f"✏️ {_('user_group_defaults.edit_prompt', lang)}",
                    callback_data=f"mod_prompt_edit:{id}"
                )
            ])
            
            # Reset to default (if custom prompt exists)
            if settings.moderation_prompt:
                buttons.append([
                    InlineKeyboardButton(
                        text=f"🔄 {_('user_group_defaults.reset_to_default', lang)}",
                        callback_data=f"mod_prompt_reset:{id}"
                    )
                ])
            
            # Back button
            buttons.append([
                InlineKeyboardButton(
                    text=_('common.back', lang),
                    callback_data=f"group_admin_menu:{id}"
                )
            ])
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
            
            cancel_msg = "❌ " + (_('common.cancelled', lang) if lang == "en" else "Отменено")
            await message.reply(cancel_msg)
            await message.answer(text, reply_markup=keyboard, parse_mode="HTML")
            return
        
        # Verify admin for groups
        if id < 0:
            is_admin = await is_user_admin_in_group(message.bot, id, message.from_user.id)
            if not is_admin:
                await message.reply("🔒 You must be an admin")
                await state.clear()
                return
        
        # Update prompt
        from luka_bot.services.moderation_service import get_moderation_service
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await message.reply("⚠️ Settings not found")
            await state.clear()
            return
        
        settings.moderation_prompt = message.text.strip()
        settings.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(settings)
        
        # Clear FSM
        await state.clear()
        
        # Send confirmation
        if lang == "en":
            confirmation = f"""✅ <b>Moderation Rules Updated!</b>

Your custom rules have been saved.

<b>Preview:</b>
<code>{message.text[:200]}</code>{"..." if len(message.text) > 200 else ""}"""
        else:
            confirmation = f"""✅ <b>Правила модерации обновлены!</b>

Ваши пользовательские правила сохранены.

<b>Предпросмотр:</b>
<code>{message.text[:200]}</code>{"..." if len(message.text) > 200 else ""}"""
        
        await message.reply(confirmation, parse_mode="HTML")
        
        # Show moderation prompt menu
        text = f"""📝 <b>{_('user_group_defaults.moderation_rules', lang)}</b>

{_('user_group_defaults.moderation_prompt_desc', lang)}

<b>{_('user_group_defaults.current_prompt', lang)}:</b>"""
        
        if settings.moderation_prompt:
            text += f"\n\n<code>{settings.moderation_prompt[:500]}</code>"
            if len(settings.moderation_prompt) > 500:
                text += "..."
        else:
            text += f"\n\n<i>{_('user_group_defaults.using_default_prompt', lang)}</i>"
        
        # Create keyboard
        buttons = []
        
        # View full prompt
        if settings.moderation_prompt:
            buttons.append([
                InlineKeyboardButton(
                    text=f"👁️ {_('user_group_defaults.view_full_prompt', lang)}",
                    callback_data=f"mod_prompt_view:{id}"
                )
            ])
        
        # Edit prompt
        buttons.append([
            InlineKeyboardButton(
                text=f"✏️ {_('user_group_defaults.edit_prompt', lang)}",
                callback_data=f"mod_prompt_edit:{id}"
            )
        ])
        
        # Reset to default (if custom prompt exists)
        if settings.moderation_prompt:
            buttons.append([
                InlineKeyboardButton(
                    text=f"🔄 {_('user_group_defaults.reset_to_default', lang)}",
                    callback_data=f"mod_prompt_reset:{id}"
                )
            ])
        
        # Back button
        buttons.append([
            InlineKeyboardButton(
                text=_('common.back', lang),
                callback_data=f"group_admin_menu:{id}"
            )
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
        await message.answer(text, reply_markup=keyboard, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Failed to update moderation prompt: {e}")
        await message.reply(f"❌ Error: {e}")
        await state.clear()


@router.callback_query(F.data.startswith("mod_prompt_reset:"))
async def handle_moderation_prompt_reset(callback: CallbackQuery):
    """Reset moderation prompt to default - works for BOTH groups AND user defaults!"""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin for groups
        if id < 0:
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        # Reset prompt
        from luka_bot.services.moderation_service import get_moderation_service
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        settings.moderation_prompt = None  # Reset to None (will use default)
        settings.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(settings)
        
        await callback.answer("✅ Reset to default prompt")
        
        # Refresh the prompt menu
        callback.data = f"moderation_prompt_menu:{id}"
        await handle_moderation_prompt_menu(callback)
        
    except Exception as e:
        logger.error(f"Failed to reset prompt: {e}")
        await callback.answer("❌ Error", show_alert=True)


# ============================================================================
# Bot Personality (group_bot_prompt) Handlers
# ============================================================================

@router.callback_query(F.data.startswith("bot_personality_menu:"))
async def handle_bot_personality_menu(callback: CallbackQuery):
    """Show bot personality menu - customize how bot responds in this group."""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin for groups
        if id < 0:  # Group
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        # Get settings
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Get language
        from luka_bot.services.group_service import get_group_service
        group_service = await get_group_service()
        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(user_id)
        else:
            lang = await group_service.get_group_language(id)
        
        # Build text
        text = f"""<b>{_('user_group_defaults.bot_personality_title', lang)}</b>

{_('user_group_defaults.bot_personality_desc', lang)}

<b>{_('user_group_defaults.current_personality', lang)}:</b>"""
        
        if settings.group_bot_prompt:
            text += f"\n\n<code>{settings.group_bot_prompt[:500]}</code>"
            if len(settings.group_bot_prompt) > 500:
                text += "..."
        else:
            text += f"\n\n<i>{_('user_group_defaults.using_default_personality', lang)}</i>"
        
        # Create keyboard
        buttons = []
        
        # View full prompt
        if settings.group_bot_prompt:
            buttons.append([
                InlineKeyboardButton(
                    text=f"👁️ {_('user_group_defaults.view_full_prompt', lang)}",
                    callback_data=f"bot_personality_view:{id}"
                )
            ])
        
        # Edit prompt
        buttons.append([
            InlineKeyboardButton(
                text=f"✏️ {_('user_group_defaults.edit_prompt', lang)}",
                callback_data=f"bot_personality_edit:{id}"
            )
        ])
        
        # Reset to default (if custom prompt exists)
        if settings.group_bot_prompt:
            buttons.append([
                InlineKeyboardButton(
                    text=f"🔄 {_('user_group_defaults.reset_to_default', lang)}",
                    callback_data=f"bot_personality_reset:{id}"
                )
            ])
        
        # Back button - return to AI Assistant menu
        buttons.append([
            InlineKeyboardButton(
                text=_('common.back', lang),
                callback_data=f"ai_assistant_menu:{id}"
            )
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show bot personality menu: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("bot_personality_view:"))
async def handle_bot_personality_view(callback: CallbackQuery):
    """Send the full bot personality prompt to the user."""
    try:
        settings_id = int(callback.data.split(":", 1)[1])
        user_id = callback.from_user.id

        # Check permissions
        if settings_id < 0:
            is_admin = await is_user_admin_in_group(callback.bot, settings_id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return

        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        settings = await get_settings_for_id(settings_id)

        if not settings or not settings.group_bot_prompt:
            await callback.answer("⚠️ No custom personality set", show_alert=True)
            return

        prompt = settings.group_bot_prompt
        escaped_prompt = html.escape(prompt)

        header = "🎭 <b>Full Bot Personality:</b>\n\n"
        await callback.message.answer(f"{header}<code>{escaped_prompt}</code>", parse_mode="HTML")
        await callback.answer("📄 Personality sent below")

    except Exception as e:
        logger.error(f"Failed to show full bot personality: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("bot_personality_edit:"))
async def handle_bot_personality_edit(callback: CallbackQuery, state: FSMContext):
    """Start editing bot personality."""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin for groups
        if id < 0:
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        # Get current settings
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        # Get language
        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(user_id)
        else:
            from luka_bot.services.group_service import get_group_service
            group_service = await get_group_service()
            lang = await group_service.get_group_language(id)
        
        # Store ID in FSM
        await state.update_data(settings_id=id)
        await state.set_state(BotPersonalityEditForm.waiting_for_prompt)
        
        # Get current prompt
        current_prompt = settings.group_bot_prompt
        
        # Build edit message using localization
        title = _('user_group_defaults.edit_personality_title', lang)
        send_new = _('user_group_defaults.edit_personality_send_new', lang)
        examples = _('user_group_defaults.personality_examples', lang)
        cancel = _('user_group_defaults.edit_personality_cancel', lang)
        
        if current_prompt:
            current_label = _('user_group_defaults.edit_personality_current', lang)
            truncated = current_prompt[:500] + ("..." if len(current_prompt) > 500 else "")
            text = f"""{title}

{send_new}

{current_label}
<code>{truncated}</code>

{examples}

{cancel}"""
        else:
            using_default = _('user_group_defaults.edit_personality_using_default', lang)
            text = f"""{title}

{send_new}

{using_default}

{examples}

{cancel}"""
        
        await callback.message.edit_text(text, parse_mode="HTML")
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to start bot personality edit: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.message(BotPersonalityEditForm.waiting_for_prompt, F.text)
async def handle_bot_personality_input(message: Message, state: FSMContext):
    """Handle bot personality input."""
    try:
        # Get ID from FSM
        data = await state.get_data()
        id = data.get("settings_id")
        
        if not id:
            await message.reply("❌ Session expired. Please try again.")
            await state.clear()
            return
        
        # Get language
        if id > 0:
            from luka_bot.utils.i18n_helper import get_user_language
            lang = await get_user_language(message.from_user.id)
        else:
            from luka_bot.services.group_service import get_group_service
            group_service = await get_group_service()
            lang = await group_service.get_group_language(id)
        
        # Check for cancel
        if message.text.lower().strip() == "/cancel":
            await state.clear()
            cancel_msg = "❌ Cancelled" if lang == "en" else "❌ Отменено"
            await message.reply(cancel_msg)
            return
        
        # Verify admin for groups
        if id < 0:
            is_admin = await is_user_admin_in_group(message.bot, id, message.from_user.id)
            if not is_admin:
                await message.reply("🔒 You must be an admin")
                await state.clear()
                return
        
        # Validate length (max 1000 chars recommended)
        if len(message.text) > 2000:
            await message.reply("⚠️ Personality is too long (max 2000 characters). Please shorten it.")
            return
        
        # Update prompt
        from luka_bot.services.moderation_service import get_moderation_service
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await message.reply("⚠️ Settings not found")
            await state.clear()
            return
        
        settings.group_bot_prompt = message.text.strip()
        settings.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(settings)
        
        # Clear FSM
        await state.clear()
        
        # Send confirmation
        if lang == "en":
            confirmation = f"""✅ <b>Bot Personality Updated!</b>

Your custom personality has been saved.

<b>Preview:</b>
<code>{message.text[:200]}</code>{"..." if len(message.text) > 200 else ""}

The bot will now use this style when responding in this group."""
        else:
            confirmation = f"""✅ <b>Личность бота обновлена!</b>

Ваша пользовательская личность сохранена.

<b>Предпросмотр:</b>
<code>{message.text[:200]}</code>{"..." if len(message.text) > 200 else ""}

Бот теперь будет использовать этот стиль при ответах в этой группе."""
        
        await message.reply(confirmation, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Failed to update bot personality: {e}")
        await message.reply(f"❌ Error: {e}")
        await state.clear()


@router.callback_query(F.data.startswith("bot_personality_reset:"))
async def handle_bot_personality_reset(callback: CallbackQuery):
    """Reset bot personality to default."""
    try:
        id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin for groups
        if id < 0:
            is_admin = await is_user_admin_in_group(callback.bot, id, user_id)
            if not is_admin:
                await callback.answer("🔒 Admin only", show_alert=True)
                return
        
        # Reset personality
        from luka_bot.services.moderation_service import get_moderation_service
        from luka_bot.handlers.groups_enhanced import get_settings_for_id
        
        moderation_service = await get_moderation_service()
        settings = await get_settings_for_id(id)
        
        if not settings:
            await callback.answer("⚠️ Settings not found", show_alert=True)
            return
        
        settings.group_bot_prompt = None  # Reset to None (will use default)
        settings.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(settings)
        
        await callback.answer("✅ Reset to default personality")
        
        # Refresh the personality menu
        callback.data = f"bot_personality_menu:{id}"
        await handle_bot_personality_menu(callback)
        
    except Exception as e:
        logger.error(f"Failed to reset bot personality: {e}")
        await callback.answer("❌ Error", show_alert=True)


# ============================================================================
# User Defaults: Reset to Factory Defaults
# ============================================================================

@router.callback_query(F.data.startswith("user_defaults_reset_confirm:"))
async def handle_user_defaults_reset_confirm(callback: CallbackQuery):
    """Show confirmation dialog for resetting user defaults to factory values."""
    try:
        user_id = int(callback.data.split(":")[1])
        
        # Verify this is the user's own defaults
        if user_id != callback.from_user.id:
            await callback.answer("🔒 Access denied", show_alert=True)
            return
        
        # Get current language
        from luka_bot.services.user_profile_service import get_user_profile_service
        profile_service = get_user_profile_service()
        profile = await profile_service.get_or_create_profile(user_id)
        lang = profile.language or "en"
        
        # Build confirmation dialog
        title = _('user_group_defaults.reset_confirm_title', lang)
        text = _('user_group_defaults.reset_confirm_text', lang)
        
        full_text = f"{title}\n\n{text}"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=_('user_group_defaults.reset_confirm_yes', lang),
                    callback_data=f"user_defaults_reset_do:{user_id}"
                ),
                InlineKeyboardButton(
                    text=_('user_group_defaults.reset_confirm_no', lang),
                    callback_data=f"group_admin_menu:{user_id}"
                )
            ]
        ])
        
        await callback.message.edit_text(
            full_text,
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show reset confirmation: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("user_defaults_reset_do:"))
async def handle_user_defaults_reset_do(callback: CallbackQuery):
    """Actually reset user defaults to factory values."""
    try:
        user_id = int(callback.data.split(":")[1])
        
        # Verify this is the user's own defaults
        if user_id != callback.from_user.id:
            await callback.answer("🔒 Access denied", show_alert=True)
            return
        
        # Get current language before reset
        from luka_bot.services.user_profile_service import get_user_profile_service
        profile_service = get_user_profile_service()
        profile = await profile_service.get_or_create_profile(user_id)
        lang = profile.language or "en"
        
        # Perform reset
        from luka_bot.services.moderation_service import get_moderation_service
        moderation_service = await get_moderation_service()
        await moderation_service.reset_user_default_settings(user_id)
        
        logger.info(f"✨ User {user_id} reset their default settings to factory values")
        
        # Show success message
        success_text = _('user_group_defaults.reset_success', lang)
        
        # Create a keyboard to go back to defaults menu
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=_('common.back_to_settings', lang),
                    callback_data=f"group_admin_menu:{user_id}"
                )
            ]
        ])
        
        await callback.message.edit_text(
            success_text,
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        await callback.answer("✅ Reset complete!", show_alert=False)
        
    except Exception as e:
        logger.error(f"Failed to reset user defaults: {e}")
        await callback.answer("❌ Error", show_alert=True)


# ============================================================================
# Group: Reset to User Defaults
# ============================================================================

@router.callback_query(F.data.startswith("group_reset_to_user_defaults:"))
async def handle_group_reset_to_user_defaults(callback: CallbackQuery):
    """Reset group settings to user's default settings."""
    try:
        group_id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin
        is_admin = await is_user_admin_in_group(callback.bot, group_id, user_id)
        if not is_admin:
            await callback.answer("🔒 Admin only", show_alert=True)
            return
        
        # Get user's default settings
        from luka_bot.services.moderation_service import get_moderation_service
        from luka_bot.services.group_service import get_group_service
        moderation_service = await get_moderation_service()
        group_service = await get_group_service()
        
        # Get user's default settings (positive user_id)
        user_defaults = await moderation_service.get_group_settings(user_id)
        
        if not user_defaults:
            await callback.answer("⚠️ User defaults not found", show_alert=True)
            return
        
        # Get current group settings
        group_settings = await moderation_service.get_group_settings(group_id)
        
        if not group_settings:
            await callback.answer("⚠️ Group settings not found", show_alert=True)
            return
        
        # Copy user default values to group settings (preserve identity fields)
        group_settings.moderation_enabled = user_defaults.moderation_enabled
        group_settings.silent_mode = user_defaults.silent_mode
        group_settings.ai_assistant_enabled = user_defaults.ai_assistant_enabled
        group_settings.respond_to_all_messages = user_defaults.respond_to_all_messages
        group_settings.kb_indexation_enabled = user_defaults.kb_indexation_enabled
        group_settings.moderate_admins_enabled = user_defaults.moderate_admins_enabled
        group_settings.moderation_prompt = user_defaults.moderation_prompt
        group_settings.delete_service_messages = user_defaults.delete_service_messages
        group_settings.service_message_types = user_defaults.service_message_types.copy() if user_defaults.service_message_types else []
        group_settings.stoplist_enabled = user_defaults.stoplist_enabled
        group_settings.stoplist_words = user_defaults.stoplist_words.copy() if user_defaults.stoplist_words else []
        group_settings.stoplist_case_sensitive = user_defaults.stoplist_case_sensitive
        group_settings.stoplist_auto_delete = user_defaults.stoplist_auto_delete
        group_settings.delete_links = user_defaults.delete_links
        group_settings.delete_images = user_defaults.delete_images
        group_settings.delete_videos = user_defaults.delete_videos
        group_settings.delete_stickers = user_defaults.delete_stickers
        group_settings.delete_forwarded = user_defaults.delete_forwarded
        group_settings.auto_delete_threshold = user_defaults.auto_delete_threshold
        group_settings.auto_warn_threshold = user_defaults.auto_warn_threshold
        group_settings.quality_threshold = user_defaults.quality_threshold
        group_settings.updated_at = datetime.utcnow()
        
        # Save updated group settings
        await moderation_service.save_group_settings(group_settings)
        
        # Get language
        lang = await group_service.get_group_language(group_id)
        
        # Answer and refresh
        await callback.answer("✅ Reset to your default settings", show_alert=True)
        
        # Refresh admin menu
        await refresh_admin_menu_simple(callback, group_id)
        
    except Exception as e:
        logger.error(f"Failed to reset group to user defaults: {e}")
        await callback.answer("❌ Error", show_alert=True)


# ============================================================================
# Delete Group (DANGER ZONE)
# ============================================================================

@router.callback_query(F.data.startswith("group_delete_confirm:"))
async def handle_group_delete_confirm(callback: CallbackQuery):
    """Show confirmation dialog for deleting a group's data."""
    try:
        group_id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin
        is_admin = await is_user_admin_in_group(callback.bot, group_id, user_id)
        if not is_admin:
            await callback.answer("🔒 Admin only", show_alert=True)
            return
        
        # Get group info
        from luka_bot.services.group_service import get_group_service
        group_service = await get_group_service()
        language = await group_service.get_group_language(group_id)
        
        # Get group title
        try:
            chat = await callback.bot.get_chat(group_id)
            group_title = chat.title or f"Group {group_id}"
        except:
            group_title = f"Group {group_id}"
        
        # Build confirmation dialog
        title = _('group_settings.delete_group_confirm_title', language)
        text = _('group_settings.delete_group_confirm_text', language, group_title=group_title)
        
        full_text = f"{title}\n\n{text}"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=_('group_settings.delete_group_confirm_yes', language),
                    callback_data=f"group_delete_do:{group_id}"
                ),
                InlineKeyboardButton(
                    text=_('group_settings.delete_group_confirm_no', language),
                    callback_data=f"group_admin_menu:{group_id}"
                )
            ]
        ])
        
        await callback.message.edit_text(
            full_text,
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show delete confirmation: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("group_delete_do:"))
async def handle_group_delete_do(callback: CallbackQuery):
    """Actually delete the group's data (reuses /reset logic)."""
    try:
        group_id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Check admin
        is_admin = await is_user_admin_in_group(callback.bot, group_id, user_id)
        if not is_admin:
            await callback.answer("🔒 Admin only", show_alert=True)
            return
        
        # Get services
        from luka_bot.services.group_service import get_group_service
        from luka_bot.services.moderation_service import get_moderation_service
        
        group_service = await get_group_service()
        moderation_service = await get_moderation_service()
        language = await group_service.get_group_language(group_id)
        
        logger.info(f"🗑️ Deleting all data for group {group_id} by user {user_id}")
        
        # Delete group data (same as /reset command)
        await group_service.delete_group_link(user_id, group_id)
        
        # Delete group settings using the proper method
        settings_deleted = await moderation_service.delete_group_settings(group_id)
        if settings_deleted:
            logger.info(f"🗑️ Deleted group settings for {group_id}")
        
        # Show success message
        success_text = _('group_settings.delete_group_success', language)
        
        # Create a keyboard to go back to groups list
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=_('common.btn.back_to_list', language),
                    callback_data="groups_back"
                )
            ]
        ])
        
        await callback.message.edit_text(
            success_text,
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        await callback.answer("✅ Group deleted!", show_alert=False)
        
    except Exception as e:
        logger.error(f"Failed to delete group: {e}")
        await callback.answer("❌ Error", show_alert=True)
