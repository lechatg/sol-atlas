"""
Inline keyboard for group actions (New Group, Default Settings).

Similar to camunda_tasks_inline.py, provides action buttons as inline keyboard
to avoid filtering issues in streaming_dm handler.
"""
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from loguru import logger

from luka_bot.utils.i18n_helper import _


async def build_groups_actions_inline_keyboard(language: str = "en") -> InlineKeyboardMarkup:
    """
    Build inline keyboard with group action buttons.
    
    Layout:
    - "➕ New Group" button
    - "⚙️ Default Settings" button
    - Both in one row
    
    Args:
        language: User language (for i18n)
        
    Returns:
        Inline keyboard markup
    """
    buttons = []
    
    # Get translated button texts
    new_group_text = _('groups.keyboard.btn_new_group', language)
    default_settings_text = _('groups.keyboard.btn_default_settings', language)
    
    # Single row with both buttons
    buttons.append([
        InlineKeyboardButton(
            text=new_group_text,
            callback_data="groups:new_group"
        ),
        InlineKeyboardButton(
            text=default_settings_text,
            callback_data="groups:default_settings"
        )
    ])
    
    logger.debug(f"📋 Created groups actions inline keyboard with 2 buttons")
    return InlineKeyboardMarkup(inline_keyboard=buttons)

