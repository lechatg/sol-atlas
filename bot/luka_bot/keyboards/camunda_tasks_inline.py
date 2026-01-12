"""
Camunda tasks inline keyboard - Shows tasks from chatbot_start process.

Replaces the groups inline menu with task management interface.
"""
from typing import List
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from loguru import logger

from luka_bot.utils.i18n_helper import _


def _get_task_emoji(task_id: str) -> str:
    """
    Generate a varied emoji for each task based on task ID.

    Uses task ID hash to pick from a diverse set of emojis.
    """
    emojis = [
        "📋", "✅", "🎯", "💡", "🚀", "⭐", "🔥", "💼",
        "📝", "🎨", "🔧", "⚡", "🎪", "🎭", "🎬", "🎮",
        "🎲", "🎯", "🎪", "🎨", "🔔", "🔑", "🔨", "🔬",
        "📌", "📍", "📎", "📊", "📈", "📉", "📱", "💎"
    ]
    # Use hash of task_id to consistently pick emoji for same task
    emoji_index = hash(task_id) % len(emojis)
    return emojis[emoji_index]


async def build_camunda_tasks_inline_keyboard(
    tasks: List,  # List of TaskSchema from camunda_client
    language: str = "en"
) -> InlineKeyboardMarkup:
    """
    Build inline keyboard with Camunda tasks from chatbot_start process.

    Layout:
    - Each task = one button per row
    - callback_data = "task:{task_id}"
    - No limit on number of tasks

    Args:
        tasks: List of TaskSchema from chatbot_start process
        language: User language (not used, kept for compatibility)

    Returns:
        Inline keyboard markup
    """
    buttons = []

    # Show all tasks, one per row
    for task in tasks:
        # Task name from BPMN definition
        task_name = task.name or f"Task {task.id[:8]}"
        # Get varied emoji for this task
        emoji = _get_task_emoji(task.id)

        buttons.append([
            InlineKeyboardButton(
                text=f"{emoji} {task_name}",
                callback_data=f"task:{task.id}"
            )
        ])

    logger.info(f"📋 Created Camunda tasks inline keyboard with {len(tasks)} tasks (one per row)")
    return InlineKeyboardMarkup(inline_keyboard=buttons)

