"""
Enhanced /groups command handler.

Shows actual group information with KB, agent details, and management options.
Allows switching between group contexts for conversations.

NEW: Reply keyboard navigation (similar to /chats)
"""
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from loguru import logger

from luka_bot.utils.i18n_helper import _, get_user_language
from luka_bot.services.group_service import get_group_service
from luka_bot.services.thread_service import get_thread_service
from luka_bot.services.group_thread_service import get_group_thread_service
from luka_bot.services.group_divider_service import send_group_divider
from luka_bot.services.default_groups_service import get_default_groups_service
from luka_bot.utils.permissions import is_user_admin_in_group
from luka_bot.handlers.states import NavigationStates
from luka_bot.core.config import settings
from luka_bot.keyboards.groups_menu import get_groups_keyboard, get_empty_groups_keyboard
from luka_bot.keyboards.group_settings_inline import (
    get_welcome_message_with_settings,
    create_group_settings_inline
)

router = Router()


# ============================================================================
# Helper Functions
# ============================================================================

async def get_settings_for_id(id: int):
    """
    Get settings for ID - works for BOTH user defaults and group settings.
    
    Args:
        id: positive = user_id (user defaults), negative = group_id (group settings)
    
    Returns:
        GroupSettings object
    """
    from luka_bot.services.moderation_service import get_moderation_service
    moderation_service = await get_moderation_service()
    
    if id > 0:
        # User defaults
        return await moderation_service.get_or_create_user_default_settings(id)
    else:
        # Group settings
        return await moderation_service.get_group_settings(id)


@router.message(Command("groups"))
async def handle_groups_enhanced(message: Message, state: FSMContext) -> None:
    """
    Enhanced /groups command - show user's groups with reply keyboard navigation.
    
    NEW APPROACH (similar to /chats):
    - Reply keyboard with groups list
    - Auto-select first group
    - Show group divider with inline Settings/Delete buttons
    - User can chat with group-aware AI agent
    """
    user_id = message.from_user.id if message.from_user else None
    
    if not user_id:
        return
    
    logger.info(f"👥 /groups (new navigation) from user {user_id}")
    
    # Set navigation state
    await state.set_state(NavigationStates.groups_mode)
    
    # Get user language
    lang = await get_user_language(user_id)
    
    # Get services
    group_service = await get_group_service()
    default_groups_service = get_default_groups_service()
    
    # Auto-provision default group and channel if not already present
    await default_groups_service.provision_default_group_for_user(user_id)
    await default_groups_service.provision_default_channel_for_user(user_id)
    
    # Get user's groups
    user_groups = await group_service.list_user_groups(user_id, active_only=True)
    
    # Sort groups: default group first, then default channel, then rest
    user_groups = default_groups_service.sort_groups_by_priority(user_groups)
    
    if not user_groups:
        # Empty state - no groups yet
        logger.info(f"✨ No groups for user {user_id} - showing empty state")
        
        # Show empty state keyboard
        keyboard = await get_empty_groups_keyboard(lang)
        
        intro_text = _('groups.intro', lang, count=0)
        await message.answer(intro_text, parse_mode="HTML")
        
        no_groups_text = _('groups.no_groups', lang)
        await message.answer(
            no_groups_text,
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        
        # Send inline actions keyboard for empty state
        from luka_bot.keyboards.groups_actions_inline import build_groups_actions_inline_keyboard
        actions_inline = await build_groups_actions_inline_keyboard(language=lang)
        await message.answer("🔧 Group Actions", reply_markup=actions_inline)
        return
    
    # User has groups - show reply keyboard and auto-select default group (or first group)
    logger.info(f"📚 User {user_id} has {len(user_groups)} groups")
    
    # Since groups are sorted, default group is first (if it exists)
    # Otherwise, use the first group in the list
    first_group_id = user_groups[0].group_id if user_groups else None
    
    if first_group_id is None:
        # This shouldn't happen if user_groups is not empty, but handle it
        logger.warning(f"⚠️ No groups found for user {user_id} after sorting")
        return
    
    # Create reply keyboard with all groups
    keyboard = await get_groups_keyboard(
        groups=user_groups,
        current_group_id=first_group_id,
        language=lang
    )
    
    # Create or get user's thread for this group
    group_thread_service = await get_group_thread_service()
    user_group_thread = await group_thread_service.get_or_create_user_group_thread(
        user_id=user_id,
        group_id=first_group_id
    )
    
    # Set as active thread
    thread_service = get_thread_service()
    await thread_service.set_active_thread(user_id, user_group_thread.thread_id)
    
    # Store current group in state for message handling
    await state.update_data(current_group_id=first_group_id)
    
    # Send intro message
    intro_text = _('groups.intro', lang, count=len(user_groups))
    await message.answer(intro_text, parse_mode="HTML", reply_markup=keyboard)
    
    # Send inline actions keyboard
    from luka_bot.keyboards.groups_actions_inline import build_groups_actions_inline_keyboard
    actions_inline = await build_groups_actions_inline_keyboard(language=lang)
    await message.answer("🔧 Group Actions", reply_markup=actions_inline)
    
    # Send group divider with inline buttons (Settings/Delete) and reply keyboard
    await send_group_divider(
        user_id=user_id,
        group_id=first_group_id,
        divider_type="initial",
        bot=message.bot
    )
    
    logger.info(f"✅ Showed {len(user_groups)} groups to user {user_id}, auto-selected group {first_group_id}")


@router.callback_query(F.data.startswith("group_view:"))
async def handle_group_view(callback: CallbackQuery):
    """View detailed information about a specific group with concise info card and admin controls."""
    try:
        group_id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Get services
        group_service = await get_group_service()
        thread_service = get_thread_service()
        
        # Check if user is member
        user_groups = await group_service.list_user_groups(user_id, active_only=True)
        is_member = any(g.group_id == group_id for g in user_groups)
        
        if not is_member:
            await callback.answer("⚠️ You are not a member of this group", show_alert=True)
            return
        
        # Get thread info
        thread = await thread_service.get_group_thread(group_id)
        
        if not thread:
            await callback.answer("❌ Group info not found", show_alert=True)
            return
        
        # Get moderation settings
        from luka_bot.services.moderation_service import get_moderation_service
        from luka_bot.keyboards.group_admin import create_group_admin_menu
        
        moderation_service = await get_moderation_service()
        group_settings = await moderation_service.get_group_settings(group_id)
        moderation_enabled = group_settings.moderation_enabled if group_settings else False
        stoplist_count = len(group_settings.stoplist_words) if group_settings else 0
        
        # Build concise info card
        group_title = thread.name or f"Group {group_id}"
        kb_index = thread.knowledge_bases[0] if thread.knowledge_bases else "Not set"
        language = thread.language
        
        # Get user who added the bot (from group link)
        added_by_name = "Unknown"
        try:
            group_link = await group_service.get_group_link(user_id, group_id)
            if group_link and group_link.added_by_name:
                added_by_name = group_link.added_by_name
        except:
            pass
        
        # Get group type and check for topics
        group_type = "Group"
        group_type_icon = "👥"
        is_supergroup = False
        try:
            chat_info = await callback.bot.get_chat(group_id)
            if chat_info.type == "supergroup":
                is_supergroup = True
                group_type = "Supergroup"
                group_type_icon = "👥📌"
        except Exception as e:
            logger.warning(f"Failed to get chat info: {e}")
        
        # If supergroup, get all topic threads
        topic_threads = []
        if is_supergroup:
            try:
                # Get all threads for this group
                all_threads = await thread_service.list_threads(user_id)
                # Filter for topic threads in this group
                for t in all_threads:
                    if (t.thread_type == "topic" and 
                        t.group_id == group_id and 
                        t.topic_id is not None):
                        topic_threads.append(t)
                logger.info(f"Found {len(topic_threads)} topic threads for supergroup {group_id}")
            except Exception as e:
                logger.warning(f"Failed to list topic threads: {e}")
        
        # Language flag
        lang_flag = "🇬🇧 English" if language == "en" else "🇷🇺 Русский"
        
        # Status (always active if we can see it)
        status_text = "Active and indexing" if language == "en" else "Активен и индексирует"
        
        # Build topics list if any
        topics_section = ""
        if topic_threads:
            if language == "en":
                topics_section = "\n\n<b>📌 Connected Topics:</b>\n"
                for topic in topic_threads:
                    topic_name = topic.name or f"Topic {topic.topic_id}"
                    topic_kb = topic.knowledge_bases[0] if topic.knowledge_bases else "No KB"
                    topics_section += f"  • {topic_name} (KB: <code>{topic_kb}</code>)\n"
            else:  # Russian
                topics_section = "\n\n<b>📌 Подключенные темы:</b>\n"
                for topic in topic_threads:
                    topic_name = topic.name or f"Тема {topic.topic_id}"
                    topic_kb = topic.knowledge_bases[0] if topic.knowledge_bases else "Нет KB"
                    topics_section += f"  • {topic_name} (KB: <code>{topic_kb}</code>)\n"
        
        # Build info text
        if language == "en":
            info_text = f"""📊 <b>{group_title}</b>

<b>Setup Complete:</b>
• {group_type_icon} Type: {group_type}
• 🆔 Group ID: <code>{group_id}</code>
• 📚 KB Index: <code>{kb_index}</code>
• 👤 Added by: {added_by_name}
• 🌍 Language: {lang_flag}
• ✅ Status: {status_text}{topics_section}

<i>Use buttons below to manage group settings:</i>"""
        else:  # Russian
            info_text = f"""📊 <b>{group_title}</b>

<b>Настройка завершена:</b>
• {group_type_icon} Тип: {group_type}
• 🆔 ID группы: <code>{group_id}</code>
• 📚 KB Index: <code>{kb_index}</code>
• 👤 Добавил: {added_by_name}
• 🌍 Язык: {lang_flag}
• ✅ Статус: {status_text}{topics_section}

<i>Используйте кнопки ниже для управления настройками:</i>"""
        
        # Create admin menu keyboard directly (no separate Settings button)
        admin_keyboard = create_group_admin_menu(
            group_id=group_id,
            group_title=group_title,
            moderation_enabled=moderation_enabled,
            stoplist_count=stoplist_count,
            current_language=language,
            silent_mode=group_settings.silent_mode if group_settings else False,
            ai_assistant_enabled=group_settings.ai_assistant_enabled if group_settings else True,
            kb_indexation_enabled=group_settings.kb_indexation_enabled if group_settings else True,
            moderate_admins_enabled=group_settings.moderate_admins_enabled if group_settings else False
        )
        
        # Add back button at the end
        admin_keyboard.inline_keyboard.append([
            InlineKeyboardButton(
                text=_('common.btn.back_to_list', language),
                callback_data="groups_back"
            )
        ])
        
        await callback.message.edit_text(
            info_text,
            reply_markup=admin_keyboard,
            parse_mode="HTML"
        )
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to view group: {e}", exc_info=True)
        await callback.answer("❌ Error loading group info", show_alert=True)


@router.callback_query(F.data == "groups_back")
async def handle_groups_back(callback: CallbackQuery, state: FSMContext):
    """Go back to groups list."""
    try:
        user_id = callback.from_user.id
        
        # Get user language
        lang = await get_user_language(user_id)
        
        # Get services
        group_service = await get_group_service()
        thread_service = get_thread_service()
        default_groups_service = get_default_groups_service()
        
        # Get user's groups
        user_groups = await group_service.list_user_groups(user_id, active_only=True)
        
        # Sort groups: default group first, then default channel, then rest
        user_groups = default_groups_service.sort_groups_by_priority(user_groups)
        
        if not user_groups:
            # No groups yet - show message with Default Settings and Refresh buttons
            bot_info = await callback.bot.get_me()
            no_groups_text = _('groups.back.no_groups', lang, bot_username=bot_info.username)
            
            # Build keyboard with Default Settings and Refresh buttons (same as main /groups)
            keyboard_buttons = [
                [
                    InlineKeyboardButton(
                        text=_('groups.btn.default_settings', lang),
                        callback_data="user_group_defaults"
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=_('common.btn.refresh', lang),
                        callback_data="groups_refresh"
                    ),
                    InlineKeyboardButton(
                        text=_('common.btn.close', lang),
                        callback_data="groups_close"
                    )
                ]
            ]
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
            await callback.message.edit_text(
                no_groups_text, 
                reply_markup=keyboard,
                parse_mode="HTML"
            )
            await callback.answer()
            return
        
        # Build groups list
        header = _('groups.back.header', lang, count=len(user_groups))
        footer = _('groups.back.footer', lang)
        
        # Build inline keyboard with groups
        keyboard_buttons = []
        
        for group_link in user_groups[:20]:  # Limit to 20 groups for UI
            group_id = group_link.group_id
            
            # Get thread to get group title and agent info
            thread = await thread_service.get_group_thread(group_id)
            
            if not thread:
                continue
            
            group_title = thread.name or f"Group {group_id}"
            
            # Get admin status
            try:
                is_admin = await is_user_admin_in_group(callback.bot, group_id, user_id)
                admin_badge = " 👑" if is_admin else ""
            except:
                admin_badge = ""
            
            # Create button
            keyboard_buttons.append([
                InlineKeyboardButton(
                    text=f"{group_title}{admin_badge}",
                    callback_data=f"group_view:{group_id}"
                )
            ])
        
        # Add "Default Settings" button
        keyboard_buttons.append([
            InlineKeyboardButton(
                text=_('groups.btn.default_settings', lang),
                callback_data="user_group_defaults"
            )
        ])
        
        # Add action buttons
        keyboard_buttons.append([
            InlineKeyboardButton(
                text=_('common.btn.refresh', lang),
                callback_data="groups_refresh"
            ),
            InlineKeyboardButton(
                text=_('common.btn.close', lang),
                callback_data="groups_close"
            )
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        
        message_text = header + footer
        
        try:
            await callback.message.edit_text(message_text, reply_markup=keyboard, parse_mode="HTML")
        except Exception as edit_error:
            # If message is not modified (same content), that's OK - just ignore
            if "message is not modified" in str(edit_error):
                pass
            else:
                raise
        
        await callback.answer()
        
        logger.info(f"✅ Showed {len(user_groups)} groups to user {user_id}")
        
    except Exception as e:
        logger.error(f"Failed to go back to groups: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data == "groups_refresh")
async def handle_groups_refresh(callback: CallbackQuery, state: FSMContext):
    """Refresh groups list."""
    try:
        await handle_groups_back(callback, state)
        # Don't answer here - handle_groups_back already answers
    except Exception as e:
        logger.error(f"Failed to refresh groups: {e}")
        await callback.answer("❌ Error refreshing", show_alert=True)


@router.callback_query(F.data == "groups_list")
async def handle_groups_list(callback: CallbackQuery, state: FSMContext):
    """Show groups list from inline button (e.g., from group addition notification)."""
    try:
        # Reuse the groups_back handler which shows the full groups list
        await handle_groups_back(callback, state)
        # Don't answer here - handle_groups_back already answers
    except Exception as e:
        logger.error(f"Failed to show groups list: {e}")
        await callback.answer("❌ Error showing groups", show_alert=True)


@router.callback_query(F.data == "groups_close")
async def handle_groups_close(callback: CallbackQuery):
    """Close groups menu."""
    await callback.message.delete()
    lang = await get_user_language(callback.from_user.id)
    await callback.answer(_('groups.menu_closed', lang))


@router.callback_query(F.data.startswith("group_switch:"))
async def handle_group_switch(callback: CallbackQuery, state: FSMContext):
    """
    Switch conversation context to group agent.
    
    Future: This will allow talking to the group's agent with group KB context.
    """
    try:
        group_id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        
        # Get language
        lang = await get_user_language(user_id)
        
        # For now, show coming soon
        await callback.answer(
            _('groups.context_switch.coming_soon', lang),
            show_alert=True
        )
        
        logger.info(f"🚧 User {user_id} tried to switch to group {group_id} context (coming soon)")
        
    except Exception as e:
        logger.error(f"Failed to switch group context: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data.startswith("group_digest:"))
async def handle_group_digest(callback: CallbackQuery):
    """Show group digest (coming soon)."""
    lang = await get_user_language(callback.from_user.id)
    
    await callback.answer(
        _('groups.digest.coming_soon', lang),
        show_alert=True
    )


@router.callback_query(F.data.startswith("group_admin_settings:"))
async def handle_group_admin_settings(callback: CallbackQuery):
    """Open group admin settings (redirect to /moderation)."""
    try:
        group_id = int(callback.data.split(":")[1])
        user_id = callback.from_user.id
        lang = await get_user_language(user_id)
        
        await callback.answer(
            _('groups.moderation_tip', lang),
            show_alert=True
        )
        
    except Exception as e:
        logger.error(f"Failed to open admin settings: {e}")
        await callback.answer("❌ Error", show_alert=True)



@router.callback_query(F.data == "user_group_defaults")
async def handle_user_group_defaults(callback: CallbackQuery):
    """Show user's default group settings - REUSES group admin menu!"""
    try:
        user_id = callback.from_user.id
        
        # Get user defaults using helper
        user_defaults = await get_settings_for_id(user_id)
        
        # Use language from user defaults if available, otherwise from user profile
        # This allows user to set a different default language for new groups
        if hasattr(user_defaults, 'language') and user_defaults.language:
            lang = user_defaults.language
        else:
            lang = await get_user_language(user_id)
        
        # Header text
        text = f"""<b>{_('user_group_defaults.title', lang)}</b>

<i>{_('user_group_defaults.intro', lang)}</i>"""
        
        # REUSE existing group admin menu - just pass is_user_defaults=True!
        from luka_bot.keyboards.group_admin import create_group_admin_menu
        
        keyboard = create_group_admin_menu(
            group_id=user_id,  # Use user_id as "group_id"
            group_title=None,
            moderation_enabled=user_defaults.moderation_enabled,
            kb_indexation_enabled=user_defaults.kb_indexation_enabled,
            moderate_admins_enabled=user_defaults.moderate_admins_enabled,
            stoplist_count=len(user_defaults.stoplist_words),
            current_language=lang,
            silent_mode=user_defaults.silent_mode,
            ai_assistant_enabled=user_defaults.ai_assistant_enabled,
            is_user_defaults=True  # This hides group-specific buttons!
        )
        
        # Replace close button with back to groups
        new_buttons = []
        for row in keyboard.inline_keyboard:
            new_row = []
            for button in row:
                if button.callback_data and button.callback_data.startswith("close_menu"):
                    # Replace close with back to groups
                    new_row.append(InlineKeyboardButton(
                        text=_('user_group_defaults.btn_back', lang),
                        callback_data="groups_back"
                    ))
                else:
                    new_row.append(button)
            if new_row:
                new_buttons.append(new_row)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=new_buttons)
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show user group defaults: {e}")
        await callback.answer("❌ Error loading settings", show_alert=True)


@router.callback_query(F.data.startswith("toggle_user_default:"))
async def handle_toggle_user_default(callback: CallbackQuery):
    """Toggle a user default setting."""
    try:
        setting_name = callback.data.split(":")[1]
        user_id = callback.from_user.id
        lang = await get_user_language(user_id)
        
        # Get service
        from luka_bot.services.moderation_service import get_moderation_service
        from datetime import datetime
        
        moderation_service = await get_moderation_service()
        
        # Get user defaults
        user_defaults = await moderation_service.get_or_create_user_default_settings(user_id)
        
        # Toggle the requested setting
        if setting_name == "silent_mode":
            user_defaults.silent_mode = not user_defaults.silent_mode
        elif setting_name == "ai_assistant":
            user_defaults.ai_assistant_enabled = not user_defaults.ai_assistant_enabled
        elif setting_name == "moderation_enabled":
            user_defaults.moderation_enabled = not user_defaults.moderation_enabled
        elif setting_name == "reputation_enabled":
            user_defaults.reputation_enabled = not user_defaults.reputation_enabled
        elif setting_name == "stoplist_enabled":
            user_defaults.stoplist_enabled = not user_defaults.stoplist_enabled
        elif setting_name == "delete_links":
            user_defaults.delete_links = not user_defaults.delete_links
        elif setting_name == "delete_images":
            user_defaults.delete_images = not user_defaults.delete_images
        elif setting_name == "delete_videos":
            user_defaults.delete_videos = not user_defaults.delete_videos
        elif setting_name == "delete_stickers":
            user_defaults.delete_stickers = not user_defaults.delete_stickers
        elif setting_name == "delete_forwarded":
            user_defaults.delete_forwarded = not user_defaults.delete_forwarded
        elif setting_name == "delete_service_messages":
            user_defaults.delete_service_messages = not user_defaults.delete_service_messages
        else:
            await callback.answer(f"⚠️ Unknown setting: {setting_name}", show_alert=True)
            return
        
        # Save
        user_defaults.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(user_defaults)
        
        logger.info(f"User {user_id} toggled default setting {setting_name}")
        
        # Refresh the appropriate view based on which setting was toggled
        if setting_name == "stoplist_enabled":
            # Refresh stoplist view
            callback.data = "user_default_stoplist"
            await handle_user_default_stoplist(callback)
        elif setting_name == "delete_service_messages":
            # Refresh system messages view
            callback.data = "user_default_system_messages"
            await handle_user_default_system_messages(callback)
        elif setting_name in ["delete_links", "delete_images", "delete_videos", 
                              "delete_stickers", "delete_forwarded"]:
            # Refresh content types view
            callback.data = "user_default_content_types"
            await handle_user_default_content_types(callback)
        else:
            # All other settings: refresh main defaults view
            await handle_user_group_defaults(callback)
        
    except Exception as e:
        logger.error(f"Failed to toggle user default: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data == "user_default_system_messages")
async def handle_user_default_system_messages(callback: CallbackQuery):
    """Show system messages configuration with types (matching group admin)."""
    try:
        user_id = callback.from_user.id
        lang = await get_user_language(user_id)
        
        from luka_bot.services.moderation_service import get_moderation_service
        moderation_service = await get_moderation_service()
        user_defaults = await moderation_service.get_or_create_user_default_settings(user_id)
        
        # Overall status
        overall_status = _('common.on', lang) if user_defaults.delete_service_messages else _('common.off', lang)
        
        # System message types (matching group admin menu structure)
        type_groups = {
            "joined": {
                "label": _('group.sys_msg.user_joined_left', lang),
                "desc": _('group.sys_msg.user_joined_left_desc', lang)
            },
            "title": {
                "label": _('group.sys_msg.name_title_changes', lang),
                "desc": _('group.sys_msg.name_title_changes_desc', lang)
            },
            "pinned": {
                "label": _('group.sys_msg.pinned_messages', lang),
                "desc": _('group.sys_msg.pinned_messages_desc', lang)
            },
            "voice": {
                "label": _('group.sys_msg.voice_chat_events', lang),
                "desc": _('group.sys_msg.voice_chat_events_desc', lang)
            },
            "photo": {
                "label": _('group.sys_msg.group_photo_changed', lang),
                "desc": _('group.sys_msg.group_photo_changed_desc', lang)
            }
        }
        
        # Build description text
        types_text = "\n".join([f"• {info['label']}: {info['desc']}" for info in type_groups.values()])
        
        text = f"""🔧 <b>{_('group.btn.system_messages', lang)}</b>

<b>{_('common.status', lang)}:</b> {overall_status}

<b>{_('user_group_defaults.system_message_types', lang)}:</b>
{types_text}

<i>{_('user_group_defaults.system_messages_hint', lang)}</i>"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"{_('common.toggle', lang)}: {overall_status}",
                    callback_data="toggle_user_default:delete_service_messages"
                )
            ],
            [
                InlineKeyboardButton(
                    text=_('user_group_defaults.btn_back', lang),
                    callback_data="user_group_defaults"
                )
            ]
        ])
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show system messages config: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data == "user_default_content_types")
async def handle_user_default_content_types(callback: CallbackQuery):
    """Show content type filters for user defaults."""
    try:
        user_id = callback.from_user.id
        lang = await get_user_language(user_id)
        
        from luka_bot.services.moderation_service import get_moderation_service
        moderation_service = await get_moderation_service()
        user_defaults = await moderation_service.get_or_create_user_default_settings(user_id)
        
        # Helper for status
        def get_status(enabled: bool) -> str:
            return _('common.on', lang) if enabled else _('common.off', lang)
        
        text = f"""📎 <b>{_('user_group_defaults.content_types', lang)}</b>

<b>{_('user_group_defaults.allowed_content', lang)}:</b>
• 🔗 {_('user_group_defaults.links', lang)}: {get_status(not user_defaults.delete_links)}
• 🖼️ {_('user_group_defaults.images', lang)}: {get_status(not user_defaults.delete_images)}
• 🎬 {_('user_group_defaults.videos', lang)}: {get_status(not user_defaults.delete_videos)}
• 🎭 {_('user_group_defaults.stickers', lang)}: {get_status(not user_defaults.delete_stickers)}
• 📤 {_('user_group_defaults.forwarded', lang)}: {get_status(not user_defaults.delete_forwarded)}

<i>{_('user_group_defaults.content_types_hint', lang)}</i>"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            # Row 1: Links & Images
            [
                InlineKeyboardButton(
                    text=f"🔗 {get_status(not user_defaults.delete_links)}",
                    callback_data="toggle_user_default:delete_links"
                ),
                InlineKeyboardButton(
                    text=f"🖼️ {get_status(not user_defaults.delete_images)}",
                    callback_data="toggle_user_default:delete_images"
                )
            ],
            # Row 2: Videos & Stickers
            [
                InlineKeyboardButton(
                    text=f"🎬 {get_status(not user_defaults.delete_videos)}",
                    callback_data="toggle_user_default:delete_videos"
                ),
                InlineKeyboardButton(
                    text=f"🎭 {get_status(not user_defaults.delete_stickers)}",
                    callback_data="toggle_user_default:delete_stickers"
                )
            ],
            # Row 3: Forwarded
            [
                InlineKeyboardButton(
                    text=f"📤 {get_status(not user_defaults.delete_forwarded)}",
                    callback_data="toggle_user_default:delete_forwarded"
                )
            ],
            # Row 4: Back
            [
                InlineKeyboardButton(
                    text=_('user_group_defaults.btn_back', lang),
                    callback_data="user_group_defaults"
                )
            ]
        ])
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show content types config: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data == "user_default_stoplist")
async def handle_user_default_stoplist(callback: CallbackQuery):
    """Show stoplist configuration with description and regex examples."""
    try:
        user_id = callback.from_user.id
        lang = await get_user_language(user_id)
        
        from luka_bot.services.moderation_service import get_moderation_service
        moderation_service = await get_moderation_service()
        user_defaults = await moderation_service.get_or_create_user_default_settings(user_id)
        
        # Status
        status = _('common.on', lang) if user_defaults.stoplist_enabled else _('common.off', lang)
        
        # Preview stoplist (up to 5 words)
        stoplist_preview = "\n".join([f"<code>{word}</code>" for word in user_defaults.stoplist_words[:5]]) if user_defaults.stoplist_words else f"<i>{_('common.empty', lang)}</i>"
        if len(user_defaults.stoplist_words) > 5:
            stoplist_preview += f"\n<i>...{_('common.and_more', lang).format(count=len(user_defaults.stoplist_words) - 5)}</i>"
        
        text = f"""🚫 <b>{_('group.btn.stoplist', lang)}</b>

<b>{_('common.status', lang)}:</b> {status}
<b>{_('common.words', lang)}:</b> {len(user_defaults.stoplist_words)}

<b>{_('user_group_defaults.stoplist_description', lang)}:</b>
{_('user_group_defaults.stoplist_desc_text', lang)}

<b>{_('user_group_defaults.current_list', lang)}:</b>
{stoplist_preview}

<b>{_('user_group_defaults.regex_examples', lang)}:</b>
• <code>spam</code> - {_('user_group_defaults.regex_ex_exact', lang)}
• <code>.*scam.*</code> - {_('user_group_defaults.regex_ex_contains', lang)}
• <code>^buy now.*</code> - {_('user_group_defaults.regex_ex_starts', lang)}
• <code>.*(viagra|cialis).*</code> - {_('user_group_defaults.regex_ex_multiple', lang)}

<i>{_('user_group_defaults.stoplist_hint', lang)}</i>"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"{_('user_group_defaults.stoplist_toggle', lang)}: {status}",
                    callback_data="toggle_user_default:stoplist_enabled"
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"✏️ {_('user_group_defaults.edit_stoplist', lang)}",
                    callback_data="user_default_stoplist_edit"
                )
            ],
            [
                InlineKeyboardButton(
                    text=_('user_group_defaults.btn_back', lang),
                    callback_data="user_group_defaults"
                )
            ]
        ])
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show stoplist config: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data == "user_default_advanced")
async def handle_user_default_advanced(callback: CallbackQuery):
    """Show advanced settings for user defaults."""
    try:
        user_id = callback.from_user.id
        lang = await get_user_language(user_id)
        
        from luka_bot.services.moderation_service import get_moderation_service
        moderation_service = await get_moderation_service()
        user_defaults = await moderation_service.get_or_create_user_default_settings(user_id)
        
        text = f"""⚙️ <b>Advanced Default Settings</b>

<b>Content Filters:</b>
• Delete Links: {'ON' if user_defaults.delete_links else 'OFF'}
• Delete Images: {'ON' if user_defaults.delete_images else 'OFF'}
• Delete Videos: {'ON' if user_defaults.delete_videos else 'OFF'}
• Delete Stickers: {'ON' if user_defaults.delete_stickers else 'OFF'}
• Delete Forwarded: {'ON' if user_defaults.delete_forwarded else 'OFF'}

<b>Service Messages:</b>
• Delete Service Messages: {'ON' if user_defaults.delete_service_messages else 'OFF'}

<b>Thresholds:</b>
• Auto-delete: {user_defaults.auto_delete_threshold}/10
• Auto-warn: {user_defaults.auto_warn_threshold}/10
• Quality reward: {user_defaults.quality_threshold}/10

<i>Note: These settings will be applied to all new groups you add the bot to.</i>"""
        
        if lang == "ru":
            text = f"""⚙️ <b>Расширенные настройки по умолчанию</b>

<b>Фильтры контента:</b>
• Удалять ссылки: {'ВКЛ' if user_defaults.delete_links else 'ВЫКЛ'}
• Удалять изображения: {'ВКЛ' if user_defaults.delete_images else 'ВЫКЛ'}
• Удалять видео: {'ВКЛ' if user_defaults.delete_videos else 'ВЫКЛ'}
• Удалять стикеры: {'ВКЛ' if user_defaults.delete_stickers else 'ВЫКЛ'}
• Удалять пересланное: {'ВКЛ' if user_defaults.delete_forwarded else 'ВЫКЛ'}

<b>Служебные сообщения:</b>
• Удалять служебные сообщения: {'ВКЛ' if user_defaults.delete_service_messages else 'ВЫКЛ'}

<b>Пороги:</b>
• Авто-удаление: {user_defaults.auto_delete_threshold}/10
• Авто-предупреждение: {user_defaults.auto_warn_threshold}/10
• Награда за качество: {user_defaults.quality_threshold}/10

<i>Примечание: Эти настройки будут применены ко всем новым группам, куда вы добавите бота.</i>"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"🔗 Links: {'ON' if user_defaults.delete_links else 'OFF'}",
                    callback_data="toggle_user_default:delete_links"
                ),
                InlineKeyboardButton(
                    text=f"🖼️ Images: {'ON' if user_defaults.delete_images else 'OFF'}",
                    callback_data="toggle_user_default:delete_images"
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"🎬 Videos: {'ON' if user_defaults.delete_videos else 'OFF'}",
                    callback_data="toggle_user_default:delete_videos"
                ),
                InlineKeyboardButton(
                    text=f"🎭 Stickers: {'ON' if user_defaults.delete_stickers else 'OFF'}",
                    callback_data="toggle_user_default:delete_stickers"
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"📤 Forwarded: {'ON' if user_defaults.delete_forwarded else 'OFF'}",
                    callback_data="toggle_user_default:delete_forwarded"
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"🔧 Service Msgs: {'ON' if user_defaults.delete_service_messages else 'OFF'}",
                    callback_data="toggle_user_default:delete_service_messages"
                )
            ],
            [
                InlineKeyboardButton(
                    text="⬅️ Back",
                    callback_data="user_group_defaults"
                )
            ]
        ])
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show advanced settings: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data == "user_default_moderation_prompt")
async def handle_user_default_moderation_prompt(callback: CallbackQuery):
    """Show moderation prompt configuration for user defaults."""
    try:
        user_id = callback.from_user.id
        lang = await get_user_language(user_id)
        
        from luka_bot.services.moderation_service import get_moderation_service
        moderation_service = await get_moderation_service()
        user_defaults = await moderation_service.get_or_create_user_default_settings(user_id)
        
        prompt_preview = user_defaults.moderation_prompt[:200] + "..." if user_defaults.moderation_prompt and len(user_defaults.moderation_prompt) > 200 else (user_defaults.moderation_prompt or _('user_group_defaults.using_default_prompt', lang))
        
        text = f"""📝 <b>{_('user_group_defaults.moderation_prompt_title', lang)}</b>

<b>{_('user_group_defaults.current_prompt', lang)}:</b>
<code>{prompt_preview}</code>

{_('user_group_defaults.moderation_prompt_hint', lang)}"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=_('user_group_defaults.use_default_prompt', lang),
                    callback_data="user_default_prompt_reset"
                )
            ],
            [
                InlineKeyboardButton(
                    text=_('user_group_defaults.btn_back', lang),
                    callback_data="user_group_defaults"
                )
            ]
        ])
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
        await callback.answer()
        
    except Exception as e:
        logger.error(f"Failed to show moderation prompt config: {e}")
        await callback.answer("❌ Error", show_alert=True)


@router.callback_query(F.data == "user_default_prompt_reset")
async def handle_user_default_prompt_reset(callback: CallbackQuery):
    """Reset moderation prompt to default."""
    try:
        user_id = callback.from_user.id
        lang = await get_user_language(user_id)
        
        from luka_bot.services.moderation_service import get_moderation_service
        from datetime import datetime
        
        moderation_service = await get_moderation_service()
        user_defaults = await moderation_service.get_or_create_user_default_settings(user_id)
        
        user_defaults.moderation_prompt = None  # Reset to default
        user_defaults.updated_at = datetime.utcnow()
        await moderation_service.save_group_settings(user_defaults)
        
        await callback.answer(_('user_group_defaults.prompt_reset_success', lang))
        
        # Refresh view
        callback.data = "user_default_moderation_prompt"
        await handle_user_default_moderation_prompt(callback)
        
    except Exception as e:
        logger.error(f"Failed to reset moderation prompt: {e}")
        await callback.answer("❌ Error", show_alert=True)
