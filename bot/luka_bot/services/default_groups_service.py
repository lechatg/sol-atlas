"""
Default Groups/Channels Service

Manages automatic provisioning of default groups and channels to all users.
"""
from typing import Optional, List
from loguru import logger
from aiogram import Bot

from luka_bot.core.config import settings
from luka_bot.models.group_link import GroupLink
from luka_bot.services.group_service import get_group_service


class DefaultGroupsService:
    """Service for managing default groups and channels."""

    def __init__(self):
        self.redis = None  # Initialized on first use

    async def verify_bot_membership(self, bot: Bot) -> dict:
        """
        Verify bot is member/admin of configured default group/channel.

        Args:
            bot: Bot instance for API calls

        Returns:
            Dict with verification results
        """
        results = {
            "default_group_ok": False,
            "default_channel_ok": False,
            "errors": []
        }

        # Verify default group
        if settings.has_default_group:
            try:
                bot_member = await bot.get_chat_member(
                    settings.LUKA_DEFAULT_GROUP_ID,
                    bot.id
                )

                if bot_member.status in ["member", "administrator", "creator"]:
                    results["default_group_ok"] = True
                    logger.info(
                        f"✅ Bot is {bot_member.status} in default group "
                        f"{settings.LUKA_DEFAULT_GROUP_TITLE} ({settings.LUKA_DEFAULT_GROUP_ID})"
                    )
                else:
                    error = f"Bot status '{bot_member.status}' not valid for default group"
                    results["errors"].append(error)
                    logger.error(f"❌ {error}")

            except Exception as e:
                error = f"Cannot access default group {settings.LUKA_DEFAULT_GROUP_ID}: {e}"
                results["errors"].append(error)
                logger.error(f"❌ {error}")

        # Verify default channel
        if settings.has_default_channel:
            try:
                bot_member = await bot.get_chat_member(
                    settings.LUKA_DEFAULT_CHANNEL_ID,
                    bot.id
                )

                if bot_member.status in ["administrator", "creator"]:
                    results["default_channel_ok"] = True
                    logger.info(
                        f"✅ Bot is {bot_member.status} in default channel "
                        f"{settings.LUKA_DEFAULT_CHANNEL_TITLE} ({settings.LUKA_DEFAULT_CHANNEL_ID})"
                    )
                else:
                    error = f"Bot must be admin in default channel (current: {bot_member.status})"
                    results["errors"].append(error)
                    logger.error(f"❌ {error}")

            except Exception as e:
                error = f"Cannot access default channel {settings.LUKA_DEFAULT_CHANNEL_ID}: {e}"
                results["errors"].append(error)
                logger.error(f"❌ {error}")

        return results

    async def provision_default_group_for_user(self, user_id: int) -> Optional[GroupLink]:
        """
        Provision default group link for a user.

        Creates GroupLink and ensures group thread exists with KB.
        This happens automatically when user first interacts with bot.

        Args:
            user_id: User ID to provision for

        Returns:
            GroupLink if provisioned, None if no default group configured
        """
        if not settings.has_default_group:
            return None

        try:
            group_service = await get_group_service()

            # Check if link already exists
            existing = await group_service.get_group_link(
                user_id,
                settings.LUKA_DEFAULT_GROUP_ID
            )

            if existing:
                logger.debug(f"✅ User {user_id} already has default group link")
                return existing

            # Create group link
            link = await group_service.create_group_link(
                user_id=user_id,
                group_id=settings.LUKA_DEFAULT_GROUP_ID,
                group_title=settings.LUKA_DEFAULT_GROUP_TITLE,
                language="en",  # Default, will inherit from group thread
                user_role="member"  # Default role
            )

            logger.info(
                f"✨ Provisioned default group for user {user_id}: "
                f"{settings.LUKA_DEFAULT_GROUP_TITLE}"
            )

            return link

        except Exception as e:
            logger.error(f"❌ Error provisioning default group for user {user_id}: {e}")
            return None

    async def provision_default_channel_for_user(self, user_id: int) -> Optional[GroupLink]:
        """
        Provision default channel link for a user.

        Creates GroupLink and ensures channel thread exists with KB.
        This happens automatically when user first interacts with bot.

        Args:
            user_id: User ID to provision for

        Returns:
            GroupLink if provisioned, None if no default channel configured
        """
        if not settings.has_default_channel:
            return None

        try:
            group_service = await get_group_service()

            # Check if link already exists
            existing = await group_service.get_group_link(
                user_id,
                settings.LUKA_DEFAULT_CHANNEL_ID
            )

            if existing:
                logger.debug(f"✅ User {user_id} already has default channel link")
                return existing

            # Create group link (channels are also stored as GroupLink)
            link = await group_service.create_group_link(
                user_id=user_id,
                group_id=settings.LUKA_DEFAULT_CHANNEL_ID,
                group_title=settings.LUKA_DEFAULT_CHANNEL_TITLE,
                language="en",  # Default, will inherit from channel thread
                user_role="member"  # Default role
            )

            logger.info(
                f"✨ Provisioned default channel for user {user_id}: "
                f"{settings.LUKA_DEFAULT_CHANNEL_TITLE}"
            )

            return link

        except Exception as e:
            logger.error(f"❌ Error provisioning default channel for user {user_id}: {e}")
            return None

    def sort_groups_by_priority(self, groups: List[GroupLink]) -> List[GroupLink]:
        """
        Sort groups list prioritizing default group, then default channel, then rest.

        Args:
            groups: List of GroupLink instances

        Returns:
            Sorted list with default group first, default channel second, then rest
        """
        default_group_id = settings.LUKA_DEFAULT_GROUP_ID if settings.has_default_group else None
        default_channel_id = settings.LUKA_DEFAULT_CHANNEL_ID if settings.has_default_channel else None

        # Separate groups into priority categories
        default_group = None
        default_channel = None
        other_groups = []

        for group in groups:
            if default_group_id and group.group_id == default_group_id:
                default_group = group
            elif default_channel_id and group.group_id == default_channel_id:
                default_channel = group
            else:
                other_groups.append(group)

        # Build sorted list: default group, default channel, then rest
        sorted_groups = []
        if default_group:
            sorted_groups.append(default_group)
        if default_channel:
            sorted_groups.append(default_channel)
        sorted_groups.extend(other_groups)

        return sorted_groups

    async def get_default_group_kbs(self, user_id: int) -> List[str]:
        """
        Get KB indices for default groups/channels available to user.

        Args:
            user_id: User ID

        Returns:
            List of KB index names
        """
        kb_indices = []

        # Add default group KB
        if settings.has_default_group:
            group_kb = settings.get_default_group_kb_index()
            if group_kb:
                kb_indices.append(group_kb)

        # Add default channel KB (prepared for /channels feature)
        if settings.has_default_channel:
            channel_kb = settings.get_default_channel_kb_index()
            if channel_kb:
                kb_indices.append(channel_kb)

        return kb_indices


# Singleton instance
_default_groups_service: Optional[DefaultGroupsService] = None


def get_default_groups_service() -> DefaultGroupsService:
    """Get or create DefaultGroupsService singleton."""
    global _default_groups_service
    if _default_groups_service is None:
        _default_groups_service = DefaultGroupsService()
        logger.info("✅ DefaultGroupsService singleton created")
    return _default_groups_service
