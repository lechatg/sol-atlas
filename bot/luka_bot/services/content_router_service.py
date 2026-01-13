"""
Content Router Service for Knowledge Base Gathering System.

This service routes detected content to appropriate analysis tools and
manages the workflow for adding content to the knowledge base.

Phase 1: Simple routing to Twitter tool with inline confirmation
Phase 2+: Full Camunda workflow integration with moderation

Related to: docs/feature/knowledge-base-gathering-system.md
"""
from typing import Optional, Dict, Any
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from loguru import logger

from luka_bot.services.content_detection_service import DetectedContent, get_content_detection_service
from luka_bot.agents.context import ConversationContext
from luka_bot.agents.tools.twitter_tools import get_twitter_account_info
from luka_bot.utils.i18n_helper import _


class ContentRouterService:
	"""Service for routing content to appropriate analysis tools."""

	def __init__(self):
		"""Initialize the content router service."""
		logger.info("🚦 ContentRouterService initialized")

	async def route_content(
		self,
		content: DetectedContent,
		message: Message,
		user_id: int,
		group_id: int,
		language: str = "en"
	) -> Optional[str]:
		"""
		Route content to appropriate analysis tool and return analysis result.

		Args:
			content: Detected content item
			message: Original Telegram message
			user_id: User who submitted the content
			group_id: Group where content was submitted
			language: User's language preference

		Returns:
			Analysis result text or None if routing failed
		"""
		logger.info(f"🚦 Routing content: type={content.content_type}, platform={content.source_platform}")

		try:
			# Route based on content type
			if content.content_type == "twitter_profile":
				return await self._route_twitter_profile(content, message, user_id, group_id, language)

			elif content.content_type == "twitter_tweet":
				# Phase 2: Analyze tweet content
				logger.info(f"📝 Twitter tweet detected (Phase 2): {content.content_url}")
				return None

			elif content.content_type == "telegram_channel":
				# Phase 2: Analyze Telegram channel
				logger.info(f"📝 Telegram channel detected (Phase 2): {content.content_url}")
				return None

			elif content.content_type == "website":
				# Phase 2: Analyze website content
				logger.info(f"📝 Website detected (Phase 2): {content.content_url}")
				return None

			elif content.content_type == "attachment":
				# Phase 2: Analyze attachment
				logger.info(f"📝 Attachment detected (Phase 2): {content.metadata}")
				return None

			elif content.content_type == "forwarded":
				# Index forwarded content to knowledge base
				return await self._route_forwarded_content(content, message, user_id, group_id, language)

			else:
				logger.warning(f"⚠️ Unknown content type: {content.content_type}")
				return None

		except Exception as e:
			logger.error(f"❌ Error routing content: {e}", exc_info=True)
			return None

	async def _route_forwarded_content(
		self,
		content: DetectedContent,
		message: Message,
		user_id: int,
		group_id: int,
		language: str
	) -> Optional[str]:
		"""
		Route forwarded Telegram message content to knowledge base indexing.
		
		Extracts the forwarded message text and indexes it with proper metadata.
		
		Args:
			content: Detected forwarded content
			message: Original Telegram message
			user_id: User who forwarded the message
			group_id: Group where message was forwarded
			language: User's language preference
			
		Returns:
			Analysis result text or None if indexing failed
		"""
		try:
			# Extract forwarded message text
			forwarded_text = content.raw_text or message.text or message.caption or ""
			
			if not forwarded_text:
				logger.warning(f"⚠️ Forwarded message has no text content to index")
				return None
			
			# Get metadata about the forwarded source
			metadata = content.metadata or {}
			forward_source = ""
			
			if metadata.get("from_chat_username"):
				forward_source = f"Channel: @{metadata['from_chat_username']}"
				if metadata.get("from_chat_title"):
					forward_source = f"{metadata['from_chat_title']} (@{metadata['from_chat_username']})"
			elif metadata.get("from_chat_title"):
				forward_source = f"Channel: {metadata['from_chat_title']}"
			elif metadata.get("from_user_name"):
				forward_source = f"User: {metadata['from_user_name']}"
				if metadata.get("from_user_username"):
					forward_source = f"{metadata['from_user_name']} (@{metadata['from_user_username']})"
			
			# Build attribution text
			attribution = f"[Forwarded from {forward_source}]"
			if content.content_url:
				attribution += f" - {content.content_url}"
			
			# Index to knowledge base if KB indexation is enabled
			from luka_bot.services.moderation_service import get_moderation_service
			from luka_bot.services.group_service import get_group_service
			from luka_bot.services.elasticsearch_service import get_elasticsearch_service
			from luka_bot.utils.document_id_generator import DocumentIDGenerator
			from datetime import datetime
			
			moderation_service = await get_moderation_service()
			group_settings = await moderation_service.get_group_settings(group_id)
			
			# Check if KB indexation is enabled
			if not group_settings or not group_settings.kb_indexation_enabled:
				logger.debug(f"⏭️ KB indexation disabled for group {group_id}, skipping forwarded content indexing")
				return f"📋 Forwarded content detected: {attribution}\n\n{forwarded_text[:200]}..."
			
			# Get KB index
			group_service = await get_group_service()
			kb_index = await group_service.get_group_kb_index(group_id)
			
			if not kb_index:
				logger.debug(f"⏭️ No KB index for group {group_id}, skipping forwarded content indexing")
				return f"📋 Forwarded content detected: {attribution}\n\n{forwarded_text[:200]}..."
			
			# Generate document ID for forwarded content
			forwarded_doc_id = DocumentIDGenerator.generate_group_message_id(
				user_id=user_id,
				group_id=group_id,
				telegram_message_id=message.message_id,
				thread_id=None
			)
			# Add suffix to distinguish from regular message
			forwarded_doc_id = f"{forwarded_doc_id}_forwarded"
			
			# Prepare message data with forwarded content
			sender_name = message.from_user.full_name if message.from_user else "Unknown"
			group_name = message.chat.title or f"Group {group_id}"
			
			# Combine attribution with forwarded text
			indexed_text = f"{attribution}\n\n{forwarded_text}"
			
			forwarded_message_data = {
				"message_id": forwarded_doc_id,
				"group_id": str(group_id),
				"user_id": str(user_id),
				"group_name": group_name,
				"role": "user",
				"thread_id": "",
				"telegram_topic_id": "",
				"message_text": indexed_text,  # Include attribution in the text
				"message_date": message.date.isoformat() if message.date else datetime.utcnow().isoformat(),
				"sender_name": sender_name,
				"reply_to_message_id": "",
				"parent_message_text": None,
				"parent_message_id": None,
				"parent_message_user_id": None,
				"mentions": [],
				"hashtags": [],
				"urls": [content.content_url] if content.content_url else [],
				"media_type": "forwarded",
				# Additional metadata for forwarded content
				"forwarded_from": forward_source,
				"forwarded_url": content.content_url or "",
				"forward_date": metadata.get("forward_date"),
				"content_type": "forwarded",
			}
			
			# Index to Elasticsearch
			try:
				es_service = await get_elasticsearch_service()
				index_success = await es_service.index_message_immediate(
					index_name=kb_index,
					message_data=forwarded_message_data,
					document_id=forwarded_doc_id
				)
				
				if index_success:
					logger.info(f"✅ Indexed forwarded content to {kb_index}: {forwarded_doc_id}")
					return f"📋 Forwarded content indexed to knowledge base:\n\n{attribution}\n\n{forwarded_text[:300]}..."
				else:
					logger.warning(f"⚠️ Failed to index forwarded content: {forwarded_doc_id}")
					return f"📋 Forwarded content detected (indexing failed): {attribution}\n\n{forwarded_text[:200]}..."
					
			except Exception as e:
				logger.error(f"❌ Error indexing forwarded content: {e}", exc_info=True)
				return f"📋 Forwarded content detected (error during indexing): {attribution}\n\n{forwarded_text[:200]}..."
				
		except Exception as e:
			logger.error(f"❌ Error routing forwarded content: {e}", exc_info=True)
			return None

	async def _route_twitter_profile(
		self,
		content: DetectedContent,
		message: Message,
		user_id: int,
		group_id: int,
		language: str
	) -> Optional[str]:
		"""Route Twitter profile to twitter_tools for analysis."""
		try:
			# Create conversation context
			ctx = ConversationContext(
				user_id=user_id,
				metadata={"language": language}
			)

			# Get Twitter account info using the tool
			username = content.metadata.get("username") if content.metadata else None
			if not username and content.content_url:
				# Extract from URL as fallback
				username = content.content_url

			logger.info(f"🐦 Analyzing Twitter account: {username}")

			# Call the Twitter tool directly
			analysis_result = await get_twitter_account_info(ctx, username)

			# Store analysis result in metadata for downstream processing
			usefulness_score = ctx.metadata.get("twitter_usefulness_score", 0)

			logger.info(f"✅ Twitter analysis complete: score={usefulness_score}/10")

			return analysis_result

		except Exception as e:
			logger.error(f"❌ Error analyzing Twitter profile: {e}", exc_info=True)
			return None

	async def should_prompt_for_kb_addition(self, content: DetectedContent, usefulness_score: int) -> bool:
		"""
		Determine if we should prompt user to add content to KB.

		Phase 1: Simple threshold check
		Phase 2+: ML-based decision with user preferences

		Args:
			content: Detected content
			usefulness_score: Calculated usefulness score (-10 to +10)

		Returns:
			True if we should prompt user
		"""
		# Threshold from PRD: score >= 7
		MIN_SCORE_THRESHOLD = 7

		if usefulness_score >= MIN_SCORE_THRESHOLD:
			logger.info(f"✅ Content meets threshold (score={usefulness_score} >= {MIN_SCORE_THRESHOLD})")
			return True
		else:
			logger.info(f"⏭️ Content below threshold (score={usefulness_score} < {MIN_SCORE_THRESHOLD})")
			return False

	def create_kb_addition_keyboard(
		self,
		content: DetectedContent,
		usefulness_score: int,
		language: str = "en"
	) -> InlineKeyboardMarkup:
		"""
		Create inline keyboard for KB addition confirmation.

		Phase 1: Simple Yes/No buttons
		Phase 2+: Advanced options (edit, skip, settings)

		Args:
			content: Detected content
			usefulness_score: Calculated usefulness score
			language: User's language

		Returns:
			Inline keyboard markup
		"""
		# Create callback data with content info
		callback_data_prefix = f"kb_add:{content.content_type}:"

		# Truncate URL for callback data (max 64 bytes)
		url_hash = hash(content.content_url or "") % 10000
		callback_yes = f"{callback_data_prefix}yes:{url_hash}"
		callback_no = f"{callback_data_prefix}no:{url_hash}"

		buttons = [
			[
				InlineKeyboardButton(
					text=_("kb_gathering.button_yes", language) if language == "ru" else "✅ Yes, Add to KB",
					callback_data=callback_yes
				),
				InlineKeyboardButton(
					text=_("kb_gathering.button_no", language) if language == "ru" else "❌ No, Skip",
					callback_data=callback_no
				)
			]
		]

		# Phase 2: Add "Edit Score" button for admins
		# buttons.append([
		#     InlineKeyboardButton(text="✏️ Edit Score", callback_data=f"{callback_data_prefix}edit:{url_hash}")
		# ])

		return InlineKeyboardMarkup(inline_keyboard=buttons)

	async def format_kb_prompt_message(
		self,
		content: DetectedContent,
		analysis_result: str,
		usefulness_score: int,
		language: str = "en"
	) -> str:
		"""
		Format the prompt message asking user to add content to KB.

		Args:
			content: Detected content
			analysis_result: Tool analysis result
			usefulness_score: Calculated score
			language: User's language

		Returns:
			Formatted message text
		"""
		# Build prompt message
		prompt_lines = [
			"",
			"━━━━━━━━━━━━━━━━━━━━",
			"",
			_("kb_gathering.prompt_header", language) if language == "ru" else "📚 **Add to Knowledge Base?**",
			"",
			_("kb_gathering.prompt_score", language, score=usefulness_score) if language == "ru"
				else f"**Usefulness Score:** {usefulness_score}/10",
			_("kb_gathering.prompt_potential_reward", language, reward=usefulness_score * 10) if language == "ru"
				else f"**Potential Reward:** {usefulness_score * 10} Atlas tokens",
			"",
			_("kb_gathering.prompt_question", language) if language == "ru"
				else "Would you like to submit this for moderation and add it to the knowledge base?",
		]

		return "\n".join(prompt_lines)


# Singleton instance
_content_router_service: Optional[ContentRouterService] = None


async def get_content_router_service() -> ContentRouterService:
	"""Get or create the content router service instance."""
	global _content_router_service
	if _content_router_service is None:
		_content_router_service = ContentRouterService()
	return _content_router_service
