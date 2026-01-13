"""
Unified form handlers for both start forms and tasks.

Handles button interactions:
- form_close: X button (close form)
- form_begin: Start button (begin form completion)

These handlers delegate to UnifiedFormService for actual processing.
"""
from aiogram import Router, F
from aiogram.types import CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.filters import CommandObject
from loguru import logger

from luka_bot.models.form_models import FormType
from luka_bot.services.unified_form_service import get_unified_form_service
from luka_bot.handlers.start import handle_start


router = Router(name="unified_form")


@router.callback_query(F.data.startswith("form_close:"))
async def handle_form_close(callback: CallbackQuery, state: FSMContext):
    """
    Handle back button click - close form and return to /start menu.
    
    Format: form_close:{form_type}
    Where form_type is "start_form" or "task"
    """
    try:
        form_type_str = callback.data.split(":")[1]
        form_type = FormType(form_type_str)
        
        # Clean up form messages
        service = get_unified_form_service()
        await service.handle_form_close(callback, form_type, state)
        
        # Return to main menu by calling /start
        pseudo_message = callback.message.model_copy(update={
            "text": "/start",
            "from_user": callback.from_user,
        })
        await handle_start(pseudo_message, state, command=CommandObject(command="/start", args=None))
        
    except Exception as e:
        logger.error(f"❌ Error in handle_form_close: {e}")
        await callback.answer("❌ Ошибка", show_alert=True)


@router.callback_query(F.data.startswith("form_begin:"))
async def handle_form_begin(callback: CallbackQuery, state: FSMContext):
    """
    Handle Start button click - begin form completion.
    
    Format: form_begin:{form_type}
    Where form_type is "start_form" or "task"
    
    Determines next step based on form variables:
    1. If has action buttons → Show action buttons
    2. If no editable vars → Complete immediately
    3. If has editable vars → Start dialog collection
    """
    try:
        form_type_str = callback.data.split(":")[1]
        form_type = FormType(form_type_str)
        
        service = get_unified_form_service()
        await service.handle_form_begin(callback, form_type, state)
        
    except Exception as e:
        logger.error(f"❌ Error in handle_form_begin: {e}")
        await callback.answer("❌ Ошибка", show_alert=True)

