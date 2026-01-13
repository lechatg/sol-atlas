"""Utility helpers to ensure the chatbot_start BPMN stays alive."""

from __future__ import annotations

import logging
from typing import Optional

from luka_bot.services.camunda_service import CamundaService

logger = logging.getLogger(__name__)

CHATBOT_START_KEY = "chatbot_start"


async def ensure_chatbot_start_running(
    *,
    user_id: str,
    telegram_user_id: int,
    chat_id: int,
) -> tuple[Optional[str], bool]:
    """
    Ensure the chatbot_start BPMN process is active for the user.

    Args:
        user_id: Flow API UUID (string) - primary identifier
        telegram_user_id: Telegram user ID (int) - for logging/reference
        chat_id: Telegram chat ID

    Returns:
        Tuple of (process_instance_id, was_just_created)
        - process_instance_id: The process instance ID if found/started, otherwise None
        - was_just_created: True if a new process was started, False if existing found

    Note:
        Checks process definition cache first to avoid errors if process not deployed.
    """
    # Check if process definition exists before attempting to start
    from luka_bot.services.process_definition_cache import get_process_definition_cache

    process_cache = get_process_definition_cache()
    if not process_cache.has_process(CHATBOT_START_KEY):
        logger.debug(
            "chatbot_start process not deployed in Camunda - skipping for user %s",
            user_id
        )
        return None, False

    business_key = f"{user_id}-chatbot-start"

    try:
        camunda_service = CamundaService.get_instance()

        # Check if process instance already exists
        existing = await camunda_service.get_process_instance_by_business_key(
            user_id=user_id,
            business_key=business_key,
        )
        if existing:
            logger.debug("chatbot_start already running for user %s (instance=%s)", user_id, existing.id)
            return str(existing.id), False  # Existing process, not just created

        # Start new process instance
        variables = {
            "config_chatID": chat_id,
            "form_replyMessageType": "DM",
            "config_chatHumanName": "DM",
        }

        process_instance = await camunda_service.start_process(
            user_id=user_id,
            process_key=CHATBOT_START_KEY,
            variables=variables,
            business_key=business_key,
        )
        logger.info("Started chatbot_start for user %s (instance=%s)", user_id, process_instance.id)
        return str(process_instance.id), True  # New process, just created
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to ensure chatbot_start for user %s: %s", user_id, exc)
        return None, False
