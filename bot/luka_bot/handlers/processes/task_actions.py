"""
Task action handler for button callbacks.
"""
from aiogram import Router
from aiogram.types import CallbackQuery
from aiogram.fsm.context import FSMContext
from loguru import logger

from luka_bot.keyboards.inline.task_keyboards import TaskActionCallback
from luka_bot.services.task_service import get_task_service
from luka_bot.services.message_cleanup_service import get_message_cleanup_service
from luka_bot.handlers.processes.start_process import poll_and_render_next_task

router = Router(name="task_actions")


@router.callback_query(TaskActionCallback.filter())
async def handle_task_action(
    callback: CallbackQuery,
    callback_data: TaskActionCallback,
    state: FSMContext
):
    """
    Handle task action button clicks.
    
    Args:
        callback: Callback query from button press
        callback_data: Parsed callback data
        state: FSM context
    """
    await callback.answer()
    
    user_id = callback.from_user.id
    task_service = get_task_service()
    
    # Get full task ID from state
    data = await state.get_data()
    current_task_id = data.get("current_task_id")
    
    if not current_task_id:
        logger.warning(f"Task action called but no current_task_id in state for user {user_id}")
        await callback.answer()
        return
    
    # Handle different actions
    if callback_data.action == "cancel" or callback_data.action == "cancel_upload":
        await handle_task_cancel(callback, current_task_id, state)
    else:
        # Complete task with action
        logger.info(f"Completing task {current_task_id} with action {callback_data.action}")
        
        # Check if there's an active dialog and collect partial form data
        form_context = data.get("form_context", {})
        collected_values = form_context.get("collected_values", {})

        if collected_values:
            logger.info(f"📦 Found {len(collected_values)} collected form values during dialog, including with action")

        # Complete task with action + any collected form values
        success = await task_service.complete_task_with_action(
            task_id=current_task_id,
            action_name=callback_data.action,
            user_id=user_id,
            state=state,
            partial_form_values=collected_values
        )
        
        if success:
            # Delete the intro message (action button message)
            try:
                await callback.message.delete()
            except Exception as e:
                logger.debug(f"Could not delete intro message: {e}")
            
            # Clean up dialog messages if dialog was active
            cleanup_service = get_message_cleanup_service()

            # Delete dialog prompt messages
            dialog_message_ids = form_context.get("dialog_message_ids", [])
            for msg_id in dialog_message_ids:
                try:
                    await callback.bot.delete_message(chat_id=callback.message.chat.id, message_id=msg_id)
                except Exception as e:
                    logger.debug(f"Could not delete dialog message {msg_id}: {e}")

            # Log action completion
            action_display = callback_data.action.replace('action_', '').replace('_', ' ').title()
            logger.info(f"✅ Action completed: {action_display} (task: {current_task_id}, fields: {len(collected_values)})")


            # Clean up other messages
            await cleanup_service.delete_task_messages(current_task_id, callback.bot, state)
            
            # Clear form context and dialog state
            await state.update_data({
                "form_context": None,
                "current_task_id": None
            })

            # Poll for next task
            process_id = data.get("active_process")
            if process_id:
                # Create a message object for poll function
                from aiogram.types import Message, User, Chat
                placeholder_msg = await callback.bot.send_message(chat_id=callback.message.chat.id, text="⏳")
                await poll_and_render_next_task(placeholder_msg, user_id, process_id, state)
            else:
                logger.info(f"✅ Task {current_task_id} completed, no active process")
        else:
            await callback.answer("❌ Failed to complete task", show_alert=True)


async def handle_task_cancel(callback: CallbackQuery, task_id: str, state: FSMContext):
    """
    Handle task/process cancellation.
    
    Args:
        callback: Callback query
        task_id: Task ID to cancel
        state: FSM context
    """
    # Delete the intro message (with action buttons)
    try:
        await callback.message.delete()
    except Exception as e:
        logger.debug(f"Could not delete intro message: {e}")

    # Delete other task messages
    cleanup_service = get_message_cleanup_service()
    await cleanup_service.delete_task_messages(task_id, callback.bot, state)
    
    # Clear process state
    await state.set_state(None)
    await state.update_data({
        "active_process": None,
        "current_task_id": None,
        "current_s3_variable": None,
        "expected_file_extension": None
    })
    
    await callback.bot.send_message(
        chat_id=callback.message.chat.id,
        text="❌ Process cancelled"
    )
    logger.info(f"User {callback.from_user.id} cancelled task {task_id}")

