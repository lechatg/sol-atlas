"""LangChain tool wrappers for luka_bot services.

This module wraps existing pydantic-ai tools from luka_bot/agents/tools/
into LangChain StructuredTool format for use with LangGraph.

The wrappers CALL the existing implementations to avoid code duplication,
enabling gradual migration with rollback capability.

Priority Tools (Phase 3):
- search_knowledge_base: Search user's message history
- execute_workflow: Start/manage workflows
- get_support_info: Support resources
- connect_to_support: Escalate to human support
- get_youtube_transcript: Fetch video transcripts

Additional Tools (Future):
- Twitter analysis, trip planning, menu actions
"""

from typing import Optional

from langchain_core.tools import BaseTool, StructuredTool
from loguru import logger
from pydantic import BaseModel, Field

# ============================================================================
# TOOL INPUT SCHEMAS (Pydantic Models)
# ============================================================================


class SearchKnowledgeBaseInput(BaseModel):
    """Input schema for knowledge base search tool."""

    query: str = Field(
        ...,
        description=(
            "Search query. Use '*' for ALL messages in time period (digests), "
            "or keywords for specific topics like 'postgres issues'"
        ),
    )
    from_user: str | None = Field(
        None, description="Filter by sender name (optional)"
    )
    date_from: str | None = Field(
        None,
        description=(
            "Start date: '7d' (days), '1w' (weeks), '1m' (months), '1y' (years), "
            "or 'YYYY-MM-DD'. Leave empty to search all history."
        ),
    )
    date_to: str | None = Field(
        None, description="End date in 'YYYY-MM-DD' format. Leave empty for current time."
    )
    max_results: int = Field(
        5, description="Maximum results (1-20, use 20+ for digests)", ge=1, le=100
    )


class ListRecentMessagesInput(BaseModel):
    """Input schema for list recent messages tool."""

    max_results: int = Field(
        10, description="Number of recent messages (5-50, default 10)", ge=5, le=50
    )


class GetKnowledgeBaseStatsInput(BaseModel):
    """Input schema for knowledge base statistics tool."""

    date_from: str = Field(
        default="7d",
        description=(
            "Start date: '7d' (days), '1w' (weeks), '1m' (months), '1y' (years), "
            "or 'YYYY-MM-DD'. Default: 7d (last 7 days)"
        ),
    )
    date_to: str = Field(
        default="", description="End date in 'YYYY-MM-DD' format. Leave empty for current time."
    )
    include_timeline: bool = Field(
        default=False, description="Include daily message timeline (histogram)"
    )
    include_hourly_activity: bool = Field(
        default=False, description="Include hourly activity pattern (peak hours)"
    )
    include_hashtags: bool = Field(
        default=False, description="Include most used hashtags"
    )
    include_media_breakdown: bool = Field(
        default=False, description="Include media types breakdown"
    )
    include_user_engagement: bool = Field(
        default=False, description="Include user engagement distribution"
    )
    include_unanswered_questions: bool = Field(
        default=True, description="Include unanswered questions analysis"
    )
    top_n: int = Field(
        default=5, description="Number of top items to show (users, hashtags, etc). Default: 5", ge=1, le=20
    )


class ExecuteWorkflowInput(BaseModel):
    """Input schema for workflow execution tool."""

    domain: str = Field(
        ...,
        description="Workflow domain identifier (e.g., 'sol_atlas_onboarding')",
    )


class GetSupportInfoInput(BaseModel):
    """Input schema for support information tool."""

    question: str = Field(..., description="User's support question or issue")


class ConnectToSupportInput(BaseModel):
    """Input schema for support escalation tool."""

    reason: str = Field(..., description="Reason for contacting human support")


class GetYouTubeTranscriptInput(BaseModel):
    """Input schema for YouTube transcript tool."""

    video_url: str = Field(..., description="YouTube video URL")
    language: str = Field(
        "en", description="Preferred transcript language (e.g., 'en', 'ru')"
    )


class GetConversationSuggestionsInput(BaseModel):
    """Input schema for conversation suggestions tool."""
    
    # No input needed - tool analyzes conversation history automatically
    pass


class AccessConversationHistoryInput(BaseModel):
    """Input schema for accessing conversation history from checkpoint."""
    
    query: str = Field(
        ...,
        description=(
            "What to search for in conversation history. Examples: "
            "'What number did the user ask me to remember?', "
            "'What did we discuss about workflows?', "
            "'Find when the user mentioned X'"
        ),
    )
    max_messages: int = Field(
        20,
        description="Maximum number of messages to search through (default: 20, max: 100)",
        ge=1,
        le=100,
    )


class SearchCryptoTweetsInput(BaseModel):
    """Input schema for crypto tweets search tool."""
    
    query: str = Field(
        ...,
        description=(
            "Search query for crypto/blockchain topics. Examples: 'Solana', 'Bitcoin', 'DeFi', 'NFT'"
        ),
    )
    from_user: str | None = Field(
        None, description="Filter by Twitter author username (optional)"
    )
    date_from: str | None = Field(
        None,
        description=(
            "Start date: '7d' (days), '1w' (weeks), '1m' (months), '1y' (years), "
            "or 'YYYY-MM-DD'. Leave empty to search all history."
        ),
    )
    date_to: str | None = Field(
        None, description="End date in 'YYYY-MM-DD' format. Leave empty for current time."
    )
    max_results: int = Field(
        20, description="Maximum results (1-50)", ge=1, le=50
    )


class SearchMessageHistoryInput(BaseModel):
    """Input schema for Telegram message history search tool."""
    
    query: str = Field(
        ...,
        description=(
            "Search query. Use '*' for ALL messages in time period (digests), "
            "or keywords for specific topics like 'postgres issues'"
        ),
    )
    from_user: str | None = Field(
        None, description="Filter by sender name (optional)"
    )
    date_from: str | None = Field(
        None,
        description=(
            "Start date: '7d' (days), '1w' (weeks), '1m' (months), '1y' (years), "
            "or 'YYYY-MM-DD'. Leave empty to search all history."
        ),
    )
    date_to: str | None = Field(
        None, description="End date in 'YYYY-MM-DD' format. Leave empty for current time."
    )
    max_results: int = Field(
        5, description="Maximum results (1-20, use 20+ for digests)", ge=1, le=100
    )


# ============================================================================
# TOOL IMPLEMENTATIONS (Wrappers calling existing code)
# ============================================================================


async def search_knowledge_base_tool(
    query: str,
    from_user: str | None,
    date_from: str | None,
    date_to: str | None,
    max_results: int,
    user_id: int,
    thread_id: str,
    language: str,
    knowledge_bases: list[str],
) -> str:
    """Search user's knowledge base for relevant information.

    This wrapper calls the existing pydantic-ai implementation with full features:
    - Crypto Twitter search with smart routing
    - LLM synthesis of results
    - Date and user filters
    - Batched digest processing
    - Formatted message cards with deeplinks

    Args:
        query: Search query
        from_user: Filter by sender name
        date_from: Start date filter
        date_to: End date filter
        max_results: Maximum results
        user_id: User ID for context
        thread_id: Thread ID for context
        language: User's language
        knowledge_bases: List of KB indices to search

    Returns:
        Formatted search results or error message
    """
    try:
        from pydantic_ai import RunContext
        from luka_bot.agents.context import ConversationContext
        from luka_bot.agents.tools.knowledge_base_tools import search_knowledge_base
        
        logger.info(f"🔍 KB Search (LangGraph): query='{query}', user={user_id}, thread={thread_id}")
        logger.info(f"  └─ Filters: from_user={from_user}, date_from={date_from}, date_to={date_to}, max={max_results}")
        logger.info(f"  └─ KBs: {knowledge_bases}")
        
        # Create ConversationContext with full metadata
        conv_ctx = ConversationContext(
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            thread_knowledge_bases=knowledge_bases,
            platform="telegram",
            enabled_tools=["knowledge_base"],
            metadata={
                "output_format": "telegram",  # Can be "telegram" or "markdown"
                "user_message": query,  # Pass query for crypto routing
            }
        )
        
        # Create RunContext wrapper for pydantic-ai compatibility
        # In pydantic-ai 1.0+, RunContext requires model and usage parameters
        from pydantic_ai.usage import Usage
        run_ctx = RunContext(
            deps=conv_ctx, 
            retry=0,
            model="langgraph",  # Placeholder model name
            usage=Usage()  # Empty usage tracker
        )
        
        # Call the REAL implementation with all features:
        # - Crypto Twitter search (_search_crypto_tweets)
        # - Smart routing (crypto_keywords, group_keywords)
        # - LLM synthesis (_synthesize_answer_from_data)
        # - Date/user filters with proper parsing
        # - Batched digest processing (_process_large_result_set_batched)
        # - Formatted cards with deeplinks
        result = await search_knowledge_base(
            ctx=run_ctx,
            query=query,
            max_results=max_results,
            from_user=from_user or "",
            date_from=date_from or "",
            date_to=date_to or "",
            min_score=0.1,
        )
        
        logger.info(f"✅ KB Search (LangGraph) completed: {len(result)} chars returned")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in search_knowledge_base_tool: {e}", exc_info=True)
        return f"Error searching knowledge base: {str(e)}"


async def search_crypto_tweets_tool(
    query: str,
    from_user: str | None,
    date_from: str | None,
    date_to: str | None,
    max_results: int,
    user_id: int,
    thread_id: str,
    language: str,
) -> str:
    """Search crypto Twitter for recent tweets from crypto KOLs.
    
    This tool searches the crypto-tweets index and synthesizes answers using LLM.
    Use this when users ask about crypto projects, tokens, blockchain topics, or market updates.
    
    Args:
        query: Search query (e.g., "Solana", "Bitcoin", "DeFi")
        from_user: Filter by Twitter author username
        date_from: Start date filter
        date_to: End date filter
        max_results: Maximum results
        user_id: User ID for context
        thread_id: Thread ID for context
        language: User's language
        
    Returns:
        Synthesized answer from crypto tweets
    """
    try:
        from pydantic_ai import RunContext
        from luka_bot.agents.context import ConversationContext
        from luka_bot.agents.tools.knowledge_base_tools import (
            _search_crypto_tweets,
            _synthesize_answer_from_data,
        )
        from luka_bot.services.elasticsearch_service import get_elasticsearch_service
        from datetime import datetime, timedelta
        
        logger.info(f"🐦 Crypto Tweets Search: query='{query}', user={user_id}, thread={thread_id}")
        logger.info(f"  └─ Filters: from_user={from_user}, date_from={date_from}, date_to={date_to}, max={max_results}")
        
        # Create ConversationContext
        conv_ctx = ConversationContext(
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            platform="telegram",
            enabled_tools=["knowledge_base"],
            metadata={
                "output_format": "telegram",
                "user_message": query,
            }
        )
        
        # Create RunContext wrapper
        from pydantic_ai.usage import Usage
        run_ctx = RunContext(
            deps=conv_ctx,
            retry=0,
            model="langgraph",
            usage=Usage()
        )
        
        # Get Elasticsearch service
        es_service = await get_elasticsearch_service()
        
        # Parse date filters
        date_from_dt = None
        date_to_dt = None
        
        if date_from:
            try:
                if date_from.endswith('d') and date_from[:-1].isdigit():
                    days = int(date_from[:-1])
                    date_from_dt = datetime.utcnow() - timedelta(days=days)
                elif date_from.endswith('w') and date_from[:-1].isdigit():
                    weeks = int(date_from[:-1])
                    date_from_dt = datetime.utcnow() - timedelta(weeks=weeks)
                elif date_from.endswith('m') and date_from[:-1].isdigit():
                    months = int(date_from[:-1])
                    date_from_dt = datetime.utcnow() - timedelta(days=months * 30)
                else:
                    date_from_dt = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            except Exception as e:
                logger.warning(f"⚠️  Could not parse date_from '{date_from}': {e}")
        
        if date_to:
            try:
                date_to_dt = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
                date_to_dt = date_to_dt + timedelta(days=1)  # Inclusive
            except Exception as e:
                logger.warning(f"⚠️  Could not parse date_to '{date_to}': {e}")
        
        # Search crypto tweets
        crypto_results = await _search_crypto_tweets(
            es_service=es_service,
            query=query,
            date_from_dt=date_from_dt,
            date_to_dt=date_to_dt,
            from_user=from_user,
            max_results=max_results,
            min_score=0.1
        )
        
        if not crypto_results:
            return f"No crypto tweets found for '{query}'. Try refining your search or checking different time periods."
        
        # Synthesize answer from tweets
        logger.info(f"🤖 Synthesizing answer from {len(crypto_results)} tweets...")
        result = await _synthesize_answer_from_data(
            user_question=query,
            data=crypto_results,
            data_type="tweets",
            user_lang=language,
            conv_ctx=conv_ctx
        )
        
        logger.info(f"✅ Crypto Tweets Search completed: {len(result)} chars returned")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in search_crypto_tweets_tool: {e}", exc_info=True)
        return f"Error searching crypto tweets: {str(e)}"


async def search_message_history_tool(
    query: str,
    from_user: str | None,
    date_from: str | None,
    date_to: str | None,
    max_results: int,
    user_id: int,
    thread_id: str,
    language: str,
    knowledge_bases: list[str],
) -> str:
    """Search user's Telegram message history.
    
    This tool searches the user's indexed Telegram messages (personal KB or group KBs).
    Use this when users ask about past conversations, messages, or stored information.
    
    Args:
        query: Search query (use '*' for all messages in time period)
        from_user: Filter by sender name
        date_from: Start date filter
        date_to: End date filter
        max_results: Maximum results
        user_id: User ID for context
        thread_id: Thread ID for context
        language: User's language
        knowledge_bases: List of KB indices to search
        
    Returns:
        Formatted search results from message history
    """
    try:
        from pydantic_ai import RunContext
        from luka_bot.agents.context import ConversationContext
        from luka_bot.agents.tools.knowledge_base_tools import search_knowledge_base
        
        logger.info(f"💬 Message History Search: query='{query}', user={user_id}, thread={thread_id}")
        logger.info(f"  └─ Filters: from_user={from_user}, date_from={date_from}, date_to={date_to}, max={max_results}")
        logger.info(f"  └─ KBs: {knowledge_bases}")
        
        # Create ConversationContext - force Telegram message search only
        conv_ctx = ConversationContext(
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            thread_knowledge_bases=knowledge_bases,
            platform="telegram",
            enabled_tools=["knowledge_base"],
            metadata={
                "output_format": "telegram",
                "user_message": query,
                "force_message_search": True,  # Flag to skip crypto tweets
            }
        )
        
        # Create RunContext wrapper
        from pydantic_ai.usage import Usage
        run_ctx = RunContext(
            deps=conv_ctx,
            retry=0,
            model="langgraph",
            usage=Usage()
        )
        
        # Call search_knowledge_base but it will skip crypto tweets due to force_message_search flag
        result = await search_knowledge_base(
            ctx=run_ctx,
            query=query,
            max_results=max_results,
            from_user=from_user or "",
            date_from=date_from or "",
            date_to=date_to or "",
            min_score=0.1,
        )
        
        logger.info(f"✅ Message History Search completed: {len(result)} chars returned")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in search_message_history_tool: {e}", exc_info=True)
        return f"Error searching message history: {str(e)}"


async def list_recent_messages_tool(
    max_results: int,
    user_id: int,
    thread_id: str,
    language: str,
    knowledge_bases: list[str],
) -> str:
    """List the most recent messages from the knowledge base without search query.

    This wrapper calls the existing pydantic-ai implementation.

    Args:
        max_results: Number of recent messages (5-50)
        user_id: User ID for context
        thread_id: Thread ID for context
        language: User's language
        knowledge_bases: List of KB indices to search

    Returns:
        Formatted list of recent messages
    """
    try:
        from pydantic_ai import RunContext
        from luka_bot.agents.context import ConversationContext
        from luka_bot.agents.tools.knowledge_base_tools import list_recent_messages
        
        logger.info(f"📋 List Recent Messages (LangGraph): user={user_id}, max={max_results}")
        
        # Create ConversationContext
        conv_ctx = ConversationContext(
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            thread_knowledge_bases=knowledge_bases,
            platform="telegram",
            enabled_tools=["knowledge_base"],
            metadata={
                "output_format": "telegram",
            }
        )
        
        # Create RunContext wrapper
        from pydantic_ai.usage import Usage
        run_ctx = RunContext(
            deps=conv_ctx, 
            retry=0,
            model="langgraph",
            usage=Usage()
        )
        
        # Call the original implementation
        result = await list_recent_messages(
            ctx=run_ctx,
            max_results=max_results,
        )
        
        logger.info(f"✅ List Recent Messages (LangGraph) completed: {len(result)} chars")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in list_recent_messages_tool: {e}", exc_info=True)
        return f"Error listing recent messages: {str(e)}"


async def get_knowledge_base_stats_tool(
    date_from: str,
    date_to: str,
    include_timeline: bool,
    include_hourly_activity: bool,
    include_hashtags: bool,
    include_media_breakdown: bool,
    include_user_engagement: bool,
    include_unanswered_questions: bool,
    top_n: int,
    user_id: int,
    thread_id: str,
    language: str,
    knowledge_bases: list[str],
) -> str:
    """Get comprehensive KB statistics using advanced Elasticsearch aggregations.

    This wrapper calls the existing pydantic-ai implementation.

    Args:
        date_from: Start date filter
        date_to: End date filter
        include_timeline: Include daily message timeline
        include_hourly_activity: Include hourly activity pattern
        include_hashtags: Include most used hashtags
        include_media_breakdown: Include media types breakdown
        include_user_engagement: Include user engagement distribution
        include_unanswered_questions: Include unanswered questions analysis
        top_n: Number of top items to show
        user_id: User ID for context
        thread_id: Thread ID for context
        language: User's language
        knowledge_bases: List of KB indices to search

    Returns:
        Formatted statistics report
    """
    try:
        from pydantic_ai import RunContext
        from luka_bot.agents.context import ConversationContext
        from luka_bot.agents.tools.knowledge_base_tools import get_knowledge_base_stats
        
        logger.info(f"📊 KB Stats (LangGraph): user={user_id}, date_from={date_from}, date_to={date_to}")
        logger.info(f"  └─ Flags: timeline={include_timeline}, hourly={include_hourly_activity}, hashtags={include_hashtags}")
        logger.info(f"  └─ Flags: media={include_media_breakdown}, engagement={include_user_engagement}, unanswered={include_unanswered_questions}")
        
        # Create ConversationContext
        conv_ctx = ConversationContext(
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            thread_knowledge_bases=knowledge_bases,
            platform="telegram",
            enabled_tools=["knowledge_base"],
            metadata={
                "output_format": "telegram",
            }
        )
        
        # Create RunContext wrapper
        from pydantic_ai.usage import Usage
        run_ctx = RunContext(
            deps=conv_ctx, 
            retry=0,
            model="langgraph",
            usage=Usage()
        )
        
        # Call the original implementation with all advanced stats flags
        result = await get_knowledge_base_stats(
            ctx=run_ctx,
            date_from=date_from,
            date_to=date_to,
            include_timeline=include_timeline,
            include_hourly_activity=include_hourly_activity,
            include_hashtags=include_hashtags,
            include_media_breakdown=include_media_breakdown,
            include_user_engagement=include_user_engagement,
            include_unanswered_questions=include_unanswered_questions,
            top_n=top_n,
        )
        
        logger.info(f"✅ KB Stats (LangGraph) completed: {len(result)} chars")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in get_knowledge_base_stats_tool: {e}", exc_info=True)
        return f"Error getting KB stats: {str(e)}"


async def execute_workflow_tool(
    domain: str,
    user_id: int,
    thread_id: str,
    language: str,
) -> str:
    """Execute or manage workflow instances.

    Calls workflow services directly, bypassing pydantic-ai RunContext.

    Args:
        domain: Workflow domain identifier
        user_id: User ID
        thread_id: Thread ID for workflow context
        language: User's language

    Returns:
        Workflow execution status and next steps
    """
    try:
        from luka_bot.services import (
            get_workflow_discovery_service,
            get_workflow_service,
        )
        from luka_bot.utils.i18n_helper import _

        logger.info(f"Executing workflow '{domain}' for user {user_id}")

        workflow_service = get_workflow_service()
        discovery_service = get_workflow_discovery_service()
        
        # Initialize services
        await discovery_service.initialize()
        
        if not workflow_service._initialized:  # pylint: disable=protected-access
            await workflow_service.initialize()

        workflow_def = discovery_service.get_workflow(domain)

        if not workflow_def:
            return _(
                f"Workflow with domain '{domain}' not found. "
                "Available workflows: Use search_knowledge_base to find onboarding guides.",
                language=language,
            )

        # Check if user already has an active workflow for this domain
        existing_workflow = await workflow_service.get_active_workflow_for_user(
            user_id=user_id, domain=domain
        )

        if existing_workflow:
            # User already has an active workflow - return current step instruction
            logger.info(
                f"User {user_id} already has active workflow for {domain}, "
                "returning step instruction"
            )

            # Get current step instruction
            current_step_id = existing_workflow.current_step
            if current_step_id:
                steps = workflow_def.tool_chain.get("steps", [])
                current_step = next(
                    (step for step in steps if step.get("id") == current_step_id), None
                )

                if current_step:
                    instruction = current_step.get("instruction", "").strip()
                    if instruction:
                        return _(
                            f"You're already in the {workflow_def.name} workflow. "
                            f"Current step: {instruction}",
                            language=language,
                        )

            return _(
                f"You have an active {workflow_def.name} workflow. "
                "Continue where you left off!",
                language=language,
            )

        # Start new workflow
        workflow_id = await workflow_service.start_workflow(
            user_id=user_id,
            domain=domain,
            workflow_name=workflow_def.name,
        )
        
        logger.info(f"Started workflow {workflow_id} for user {user_id}")

        # Get first step
        steps = workflow_def.tool_chain.get("steps", [])
        if steps:
            first_step = steps[0]
            instruction = first_step.get("instruction", "").strip()
            if instruction:
                return _(
                    f"Started {workflow_def.name} workflow!\n\n{instruction}",
                    language=language,
                )

        return _(
            f"Started {workflow_def.name} workflow! Let's begin.",
            language=language,
        )

    except Exception as e:
        logger.error(f"Error in execute_workflow_tool wrapper: {e}", exc_info=True)
        return f"Error executing workflow: {str(e)}"


async def get_support_info_tool(
    question: str,
    user_id: int,
    thread_id: str,
    language: str,
) -> str:
    """Get support information and help resources.

    This wrapper calls the existing pydantic-ai implementation.

    Args:
        question: User's support question
        user_id: User ID
        thread_id: Thread ID
        language: User's language

    Returns:
        Support information and resources
    """
    try:
        from pydantic_ai import RunContext

        from luka_bot.agents.context import ConversationContext
        from luka_bot.agents.tools.support_tools import get_support_info

        conv_ctx = ConversationContext(
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            enabled_tools=["support"],
            platform="telegram",
        )

        # Create RunContext with required parameters (pydantic-ai 1.0+)
        from pydantic_ai.usage import Usage
        run_ctx = RunContext(
            deps=conv_ctx, 
            retry=0,
            model="langgraph",
            usage=Usage()
        )

        result = await get_support_info(ctx=run_ctx, question=question)

        return result

    except Exception as e:
        logger.error(f"Error in get_support_info_tool wrapper: {e}")
        return "Error getting support info. Please contact support@gurunetwork.ai"


async def connect_to_support_tool(
    reason: str,
    user_id: int,
    thread_id: str,
    language: str,
) -> str:
    """Connect user to human support team.

    This wrapper calls the existing pydantic-ai implementation.

    Args:
        reason: Reason for support escalation
        user_id: User ID
        thread_id: Thread ID
        language: User's language

    Returns:
        Support escalation confirmation
    """
    try:
        from pydantic_ai import RunContext

        from luka_bot.agents.context import ConversationContext
        from luka_bot.agents.tools.support_tools import connect_to_support

        conv_ctx = ConversationContext(
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            enabled_tools=["support"],
            platform="telegram",
        )

        # Create RunContext with required parameters (pydantic-ai 1.0+)
        from pydantic_ai.usage import Usage
        run_ctx = RunContext(
            deps=conv_ctx, 
            retry=0,
            model="langgraph",
            usage=Usage()
        )

        result = await connect_to_support(ctx=run_ctx, reason=reason)

        return result

    except Exception as e:
        logger.error(f"Error in connect_to_support_tool wrapper: {e}")
        return "Error connecting to support. Please contact support@gurunetwork.ai"


async def get_youtube_transcript_tool(
    video_url: str,
    language: str,
    user_id: int,
    thread_id: str,
    user_language: str,
) -> str:
    """Get transcript from YouTube video.

    This wrapper calls the existing pydantic-ai implementation.

    Args:
        video_url: YouTube URL or video ID
        language: Preferred transcript language
        user_id: User ID
        thread_id: Thread ID
        user_language: User's interface language

    Returns:
        Video transcript or error message
    """
    try:
        from luka_bot.agents.context import ConversationContext
        from luka_bot.agents.tools.youtube_tools import get_youtube_transcript

        conv_ctx = ConversationContext(
            user_id=user_id,
            thread_id=thread_id,
            language=user_language,
            enabled_tools=["youtube"],
            platform="telegram",
        )

        result = await get_youtube_transcript(
            ctx=conv_ctx, video_url=video_url, language=language
        )

        return result

    except Exception as e:
        logger.error(f"Error in get_youtube_transcript_tool wrapper: {e}")
        return f"Error getting YouTube transcript: {str(e)}"


async def generate_conversation_suggestions_from_state(
    user_id: int,
    thread_id: str,
    language: str,
    messages: list,
) -> str:
    """
    Generate contextual conversation suggestions from current state messages.
    
    This function accepts messages directly from the current state (not from checkpoint).
    This ensures we use the LATEST assistant response for generating aligned suggestions.
    
    Args:
        user_id: User ID
        thread_id: Thread ID
        language: User's language
        messages: List of LangChain messages from current state
        
    Returns:
        String with format "SUGGESTIONS:[...]" or "SUGGESTIONS:[]"
    """
    try:
        from luka_bot.services.prompt_pool_service import get_prompt_pool_service
        from luka_bot.core.config import settings
        from langchain_core.messages import HumanMessage, AIMessage
        import json
        import random
        
        # Extract last 10 messages to ensure we get enough context (we'll filter to last 3 valid pairs)
        # We need to look back further because ToolMessages are filtered out
        recent_messages = messages[-10:] if len(messages) >= 10 else messages
        
        conversation_context = []
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                content = str(msg.content) if msg.content else ""
                if content.strip():
                    conversation_context.append(f"User: {content}")
            elif isinstance(msg, AIMessage):
                content = str(msg.content) if msg.content else ""
                if content.strip():
                    conversation_context.append(f"Assistant: {content}")
            # Note: We don't include ToolMessage in context for suggestions, but we look back further
            # to ensure we get enough HumanMessage/AIMessage pairs
        
        # Take only the last 3 valid message pairs for suggestion generation
        conversation_context = conversation_context[-3:] if len(conversation_context) > 3 else conversation_context
        
        logger.debug(f"📚 Extracted {len(conversation_context)} messages from state (total: {len(messages)})")
        logger.debug(f"📝 Built context with {len(conversation_context)} messages")
        
        if len(conversation_context) < 2:
            # Not enough valid conversation messages - use static prompts as fallback
            logger.info(f"📋 Insufficient conversation context ({len(conversation_context)} messages), using static prompts fallback")
            try:
                prompt_pool_service = get_prompt_pool_service()
                static_prompt_options = await prompt_pool_service.get_quick_prompts(locale=language, count=10)
                
                if static_prompt_options:
                    # Randomize and take 3
                    random.shuffle(static_prompt_options)
                    suggestions = [opt.text for opt in static_prompt_options[:3]]
                    logger.info(f"✅ Using {len(suggestions)} randomized static prompts: {suggestions}")
                    return f"SUGGESTIONS:{json.dumps(suggestions)}"
            except Exception as e:
                logger.warning(f"⚠️ Failed to get static prompts: {e}")
            
            return "SUGGESTIONS:[]"
        
        # Use LLM to generate suggestions
        from langchain_ollama import ChatOllama
        from langchain_openai import ChatOpenAI
        
        # Select LLM provider
        if settings.DEFAULT_LLM_PROVIDER == "ollama":
            ollama_url = settings.OLLAMA_URL.rstrip("/v1").rstrip("/")
            llm = ChatOllama(
                model=settings.OLLAMA_MODEL_NAME,
                base_url=ollama_url,
                temperature=0.7,
            )
        else:
            llm = ChatOpenAI(
                model=settings.DEFAULT_LLM_MODEL or "gpt-4o-mini",
                temperature=0.7,
            )
        
        # Build prompt (single English prompt with language variable)
        # Extract the last assistant message to align suggestions with the current response
        last_assistant_msg = None
        for msg in reversed(conversation_context):
            if msg.startswith("Assistant:"):
                last_assistant_msg = msg.replace("Assistant:", "").strip()
                break
        
        # Get previous conversation context (excluding the last assistant message)
        previous_context = []
        for msg in conversation_context:
            if msg.startswith("Assistant:") and msg.replace("Assistant:", "").strip() != last_assistant_msg:
                previous_context.append(msg)
            elif msg.startswith("User:"):
                previous_context.append(msg)
        
        # Remove the last assistant message from previous context if it's there
        previous_context = [msg for msg in previous_context if not (msg.startswith("Assistant:") and msg.replace("Assistant:", "").strip() == last_assistant_msg)]
        previous_context_text = "\n".join(previous_context[-4:]) if previous_context else "No previous conversation"
        
        context_text = "\n".join(conversation_context)
        
        # Special handling for trip planner responses (extract key locations for focused suggestions)
        import re
        is_trip_plan = False
        trip_locations = []
        trip_instruction = ""
        assistant_summary = last_assistant_msg[:100] if last_assistant_msg else "N/A"
        
        if last_assistant_msg and len(last_assistant_msg) > 1000:
            # Likely a trip plan - check for markers
            if "🗺️" in last_assistant_msg or "Trip Plan" in last_assistant_msg or "📍" in last_assistant_msg:
                is_trip_plan = True
                # Extract location names from POI cards (format: "📍 **LocationName**")
                location_matches = re.findall(r'📍\s+\*\*([^*]+)\*\*', last_assistant_msg)
                trip_locations = location_matches[:5]  # First 5 stops
                if trip_locations:
                    logger.debug(f"🗺️ Detected trip plan with {len(trip_locations)} locations: {trip_locations}")
                    # Create focused instruction for trip-specific suggestions
                    trip_instruction = f"\n🚨 CRITICAL: This is a TRIP PLAN with stops: {', '.join(trip_locations[:3])}. YOU MUST suggest questions about THESE SPECIFIC LOCATIONS (e.g., 'Tell me more about {trip_locations[0]}?', 'What's in {trip_locations[1] if len(trip_locations) > 1 else trip_locations[0]}?'). DO NOT suggest generic trip questions like 'How much will it cost' or 'What about meals'."
                    # Use location summary instead of full plan
                    assistant_summary = f"Trip plan with stops: {', '.join(trip_locations)}"
        
        logger.debug(f"📝 Last assistant message for alignment: '{assistant_summary}...'")
        
        # Get full last assistant message for detailed analysis (limit to 2000 chars to avoid token limits)
        full_last_message = last_assistant_msg[:2000] if last_assistant_msg else "N/A"
        
        prompt = f"""Analyze the conversation and suggest 3 short things the USER might say next to the assistant.

ASSISTANT'S LAST MESSAGE (FOCUS HERE - THIS IS THE PRIMARY SOURCE):
"{full_last_message}"

PREVIOUS CONVERSATION CONTEXT (for reference only):
{previous_context_text}

CRITICAL INSTRUCTIONS - FOCUS ON LAST MESSAGE:
- These suggestions will appear as CLICKABLE BUTTONS that the USER will send to the assistant
- Phrase suggestions as what the USER WOULD SAY (can be questions, statements, or short responses - prefer natural user voice)
- **2 out of 3 suggestions MUST be about the LAST MESSAGE** - they should mention specific main points, features, locations, numbers, or topics from the assistant's last response
- **1 suggestion can be from previous conversation context** (if relevant) or a general follow-up
- Extract and reference SPECIFIC details from the last message (e.g., if the assistant mentioned "Milan", "€100 budget", "3-day trip", "historic sites", etc., mention these in the suggestions)
- Each suggestion should sound like something the user would naturally say (e.g., "Tell me more about Milan 🏛️", "What about the €100 budget?", "Sounds perfect! ✨")
- DO NOT use commands (WRONG: "Show my notes", "Edit the note", "Set a reminder")
- DO NOT phrase as assistant questions to the user (WRONG: "What should I do with 89?", "Did you remember 42?")
- If the assistant gave a GENERIC greeting (like "Hi! How can I help?"), suggest GENERIC user responses (not specific to old context)
- Generate suggestions in {language.upper()} language
- Do NOT repeat what was already discussed
- Suggest NATURAL next things the user might say
- Each suggestion should be short (max 50 characters to accommodate emojis)
- Use emojis where appropriate
- Return only the suggestions, one per line
- No numbering or bullet points

Examples of CORRECT phrasing (mentioning specific points from last message):
- If assistant mentioned "Milan" and "historic sites": "Tell me more about Milan 🏛️"
- If assistant mentioned "€100 budget": "What about the €100 budget?"
- If assistant mentioned "3-day trip": "What's in the 3-day plan?"
- If assistant mentioned specific features: "Tell me about [specific feature]"
- General follow-up: "Sounds perfect! ✨"

Examples of WRONG phrasing (commands):
- "Show my notes" ❌
- "Edit the 42 note" ❌
- "Set a reminder for 89" ❌

Examples of WRONG phrasing (assistant asking user):
- "What should I do with 89?" ❌
- "Did you remember 42?" ❌
- "Need to add 89 to a reminder?" ❌

Suggestions:"""
        
        # Call LLM
        logger.debug(f"🤖 Calling LLM for conversation suggestions (thread {thread_id})")
        response = await llm.ainvoke(prompt)
        suggestions_text = response.content.strip()
        logger.debug(f"🤖 LLM raw response: {suggestions_text}")
        
        # Parse suggestions (one per line)
        raw_lines = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
        logger.debug(f"📝 Parsed {len(raw_lines)} raw lines: {raw_lines}")
        
        suggestions = []
        for s in raw_lines:
            # Remove common prefixes like "1.", "-", "•", etc.
            cleaned = s.lstrip("1234567890.-•* ").strip()
            # Be more permissive with length (3-50 chars to accommodate Russian)
            if cleaned and 3 <= len(cleaned) <= 50:
                suggestions.append(cleaned)
            else:
                logger.debug(f"⏭️ Skipped suggestion (len={len(cleaned) if cleaned else 0}): '{cleaned}'")
        
        # Limit to 3
        suggestions = suggestions[:3]
        
        # If LLM failed to generate valid suggestions, fall back to static prompts
        if not suggestions or len(suggestions) < 2:
            logger.warning(f"⚠️ LLM generated insufficient suggestions ({len(suggestions)}), falling back to static prompts")
            try:
                prompt_pool_service = get_prompt_pool_service()
                static_prompt_options = await prompt_pool_service.get_quick_prompts(locale=language, count=10)
                
                if static_prompt_options:
                    random.shuffle(static_prompt_options)
                    suggestions = [opt.text for opt in static_prompt_options[:3]]
                    logger.info(f"✅ Using {len(suggestions)} randomized static prompts (LLM fallback): {suggestions}")
                    return f"SUGGESTIONS:{json.dumps(suggestions)}"
            except Exception as e:
                logger.warning(f"⚠️ Failed to get static prompts: {e}")
            
            return "SUGGESTIONS:[]"
        
        logger.info(f"✅ Generated {len(suggestions)} conversation suggestions for thread {thread_id}: {suggestions}")
        return f"SUGGESTIONS:{json.dumps(suggestions)}"

    except Exception as e:
        logger.error(f"❌ Error in generate_conversation_suggestions_from_state: {e}", exc_info=True)
        return "SUGGESTIONS:[]"


async def generate_workflow_suggestions_from_state(
    messages: list,
    workflow_domain: str,
    current_step: str,
    workflow_progress: float,
    language: str,
    thread_id: str,
) -> list[str]:
    """
    Generate dynamic workflow suggestions based on workflow progress and conversation context.
    
    Args:
        messages: List of LangChain messages from current state
        workflow_domain: Workflow domain name
        current_step: Current workflow step ID
        workflow_progress: Workflow progress (0-100)
        language: User's language
        thread_id: Thread ID for logging
        
    Returns:
        List of 3 suggestion strings
    """
    try:
        from luka_bot.services.prompt_pool_service import get_prompt_pool_service
        from luka_bot.services.workflow_discovery_service import get_workflow_discovery_service
        from luka_bot.core.config import settings
        from langchain_core.messages import HumanMessage, AIMessage
        import json
        import random
        
        # Get workflow definition for step description context only (not for suggestions)
        discovery_service = get_workflow_discovery_service()
        await discovery_service.initialize()
        workflow_def = discovery_service.get_workflow(workflow_domain)
        
        # Extract last 6 messages for context
        recent_messages = messages[-6:] if len(messages) >= 6 else messages
        
        conversation_context = []
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                content = str(msg.content) if msg.content else ""
                if content.strip():
                    conversation_context.append(f"User: {content}")
            elif isinstance(msg, AIMessage):
                content = str(msg.content) if msg.content else ""
                if content.strip():
                    conversation_context.append(f"Assistant: {content}")
        
        logger.debug(f"📚 Extracted {len(conversation_context)} messages for workflow suggestions (total: {len(messages)})")
        
        # If not enough context, return empty suggestions (no fallback to static)
        # With 3 messages, we need at least 1 valid message pair
        if len(conversation_context) < 1:
            logger.info(f"📋 Insufficient conversation context, returning empty suggestions (no static fallback)")
            return []
        
        # Use LLM to generate dynamic suggestions
        from langchain_ollama import ChatOllama
        from langchain_openai import ChatOpenAI
        
        # Select LLM provider
        if settings.DEFAULT_LLM_PROVIDER == "ollama":
            ollama_url = settings.OLLAMA_URL.rstrip("/v1").rstrip("/")
            llm = ChatOllama(
                model=settings.OLLAMA_MODEL_NAME,
                base_url=ollama_url,
                temperature=0.7,
            )
        else:
            llm = ChatOpenAI(
                model=settings.DEFAULT_LLM_MODEL or "gpt-4o-mini",
                temperature=0.7,
            )
        
        # Extract the last assistant message
        last_assistant_msg = None
        for msg in reversed(conversation_context):
            if msg.startswith("Assistant:"):
                last_assistant_msg = msg.replace("Assistant:", "").strip()
                break
        
        # Get previous conversation context (excluding the last assistant message)
        previous_context = []
        for msg in conversation_context:
            if msg.startswith("Assistant:") and msg.replace("Assistant:", "").strip() != last_assistant_msg:
                previous_context.append(msg)
            elif msg.startswith("User:"):
                previous_context.append(msg)
        
        # Remove the last assistant message from previous context if it's there
        previous_context = [msg for msg in previous_context if not (msg.startswith("Assistant:") and msg.replace("Assistant:", "").strip() == last_assistant_msg)]
        previous_context_text = "\n".join(previous_context[-4:]) if previous_context else "No previous conversation"
        
        context_text = "\n".join(conversation_context)
        
        # Special handling: Detect if bot presented multiple choice options
        import re
        presented_options = []
        option_instruction = ""
        if last_assistant_msg:
            # Look for bullet points with emojis (e.g., "• 🏰 Historic castles and architecture?")
            option_matches = re.findall(r'[•\-\*]\s*([🏰🍷🌄🎨🎭🏛️🌊🗿🎪🎡🎢🎠🎟️🎫🎬🎮🎯🎲🎰🃏🀄🎴🎭🎨🖼️🎪🎢🎡🎠🏛️🏰🏯🏟️⛪🕌🕍🛕🕋⛩️🗿🗽🗼🏭🏗️🏘️🏚️🏠🏡🏢🏣🏤🏥🏦🏨🏩🏪🏫🏬🏭🏯🏰🗻🗼🗽⛲⛺🌁🌃🌄🌅🌆🌇🌉🌌🎆🎇🌠🎑🗾🌋⛰️🏔️🗻🏕️🏖️🏜️🏝️🏞️🏟️🏛️🏗️🧱🏘️🏚️🏠🏡🏢🏣🏤🏥🏦🏨🏩🏪🏫🏬🏭🏯🏰💒🗼🗽⛪🕌🛕🕍⛩️🕋]\s*[^?•\-\*\n]+)', last_assistant_msg)
            if option_matches and len(option_matches) >= 2:
                # Extract and clean options (remove trailing "?")
                presented_options = [opt.strip().rstrip('?').strip() for opt in option_matches[:5]]
                logger.debug(f"🎯 Detected {len(presented_options)} multiple choice options: {presented_options}")
                option_instruction = f"\n🎯 CRITICAL: The assistant just presented MULTIPLE CHOICE OPTIONS: {', '.join(presented_options)}. YOU MUST create suggestions that represent THESE EXACT OPTIONS as USER STATEMENTS (NOT questions). Example: if option is '🏰 Historic castles', suggestion should be '🏰 Historic castles' or 'Historic castles and architecture 🏰' (NOT 'What about historic castles?'). Phrase as the USER SELECTING an option, not asking about it."
        
        # Get step description if available
        step_description = ""
        if workflow_def:
            steps = workflow_def.tool_chain.get("steps", [])
            current_step_def = next(
                (s for s in steps if s.get("id") == current_step),
                None
            )
            if current_step_def:
                step_description = current_step_def.get("description", "") or current_step_def.get("name", "")
        
        logger.debug(f"📝 Generating workflow suggestions for step '{current_step}' (progress: {workflow_progress}%)")
        
        # Get full last assistant message for detailed analysis (limit to 2000 chars to avoid token limits)
        full_last_message = last_assistant_msg[:2000] if last_assistant_msg else "N/A"
        
        prompt = f"""You are helping a user progress through a workflow. Generate 3 short suggestions that the USER might say next to continue the workflow.

Workflow Context:
- Workflow: {workflow_domain}
- Current Step: {current_step}
- Step Description: {step_description if step_description else 'N/A'}
- Progress: {workflow_progress}%
- User Language: {language.upper()}

ASSISTANT'S LAST MESSAGE (FOCUS HERE - THIS IS THE PRIMARY SOURCE):
"{full_last_message}"

PREVIOUS CONVERSATION CONTEXT (for reference only):
{previous_context_text}
{option_instruction}

CRITICAL INSTRUCTIONS - FOCUS ON LAST MESSAGE:
- These suggestions will appear as CLICKABLE BUTTONS that the USER will send to the assistant
- Phrase suggestions as what the USER WOULD SAY (statements or short responses, can be questions but prefer statements)
- **2 out of 3 suggestions MUST be about the LAST MESSAGE** - they should mention specific main points, features, options, numbers, or topics from the assistant's last response
- **1 suggestion can be from previous conversation context** (if relevant) or a general workflow progression question
- Extract and reference SPECIFIC details from the last message (e.g., if the assistant mentioned "€100 budget", "historic sites", "3-day trip", specific options, etc., mention these in the suggestions)
- If the assistant presented MULTIPLE CHOICE OPTIONS, create suggestions that match those options as USER SELECTIONS (these count as the 2 focused suggestions)
- If no options were presented, suggest natural USER RESPONSES or questions that progress the workflow and reference specific points from the last message
- Base suggestions on:
  1. The assistant's last response (especially any options, features, or details mentioned) - PRIMARY FOCUS
  2. The current workflow step and its purpose
  3. The workflow progress (what comes next)
- DO NOT use commands (WRONG: "Show my notes", "Complete this step")
- DO NOT phrase as assistant questions to the user (WRONG: "What should I do?", "Did you complete this?")
- Generate suggestions in {language.upper()} language
- Each suggestion should be short (max 50 characters to accommodate emojis)
- Use emojis where appropriate
- Return only the suggestions, one per line
- No numbering or bullet points

Examples of CORRECT phrasing for MULTIPLE CHOICE OPTIONS (mentioning specific options from last message):
Assistant says: "Are you into: • 🏰 Historic castles? • 🍷 Wine regions? • 🌄 Nature?"
Suggestions should be:
- "🏰 Historic castles"
- "🍷 Wine and cuisine"
- "🌄 Nature and outdoors"

Examples of CORRECT phrasing for OPEN-ENDED questions (mentioning specific points from last message):
Assistant says: "What's your budget? I can plan trips from €50 to €500 per day."
Suggestions could be:
- "Around €100 per day"
- "Budget-friendly options"
- "Tell me about €500 trips"

Examples of WRONG phrasing (commands):
- "Show my notes" ❌
- "Complete this step" ❌
- "Skip to next" ❌

Examples of WRONG phrasing (questions when options were presented):
Assistant presented options but suggestions are generic questions:
- "What about historic castles?" ❌ (should be "Historic castles 🏰")
- "Tell me about wine regions?" ❌ (should be "Wine regions 🍷")

Suggestions:"""
        
        # Call LLM
        logger.debug(f"🤖 Calling LLM for workflow suggestions (thread {thread_id}, step: {current_step})")
        response = await llm.ainvoke(prompt)
        suggestions_text = response.content.strip()
        logger.debug(f"🤖 LLM raw response: {suggestions_text}")
        
        # Parse suggestions (one per line)
        raw_lines = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
        logger.debug(f"📝 Parsed {len(raw_lines)} raw lines: {raw_lines}")
        
        suggestions = []
        for s in raw_lines:
            # Remove common prefixes like "1.", "-", "•", etc.
            cleaned = s.lstrip("1234567890.-•* ").strip()
            # Be more permissive with length (3-50 chars to accommodate Russian)
            if cleaned and 3 <= len(cleaned) <= 50:
                suggestions.append(cleaned)
            else:
                logger.debug(f"⏭️ Skipped suggestion (len={len(cleaned) if cleaned else 0}): '{cleaned}'")
        
        # Limit to 3
        suggestions = suggestions[:3]
        
        # If LLM failed to generate valid suggestions, return empty (no fallback to static)
        if not suggestions or len(suggestions) < 2:
            logger.warning(f"⚠️ LLM generated insufficient suggestions ({len(suggestions)}), returning empty (no static fallback)")
            return []
        
        logger.info(f"✅ Generated {len(suggestions)} dynamic workflow suggestions for step '{current_step}': {suggestions}")
        return suggestions

    except Exception as e:
        logger.error(f"❌ Error in generate_workflow_suggestions_from_state: {e}", exc_info=True)
        # Return empty suggestions instead of falling back to static
        logger.warning(f"⚠️ Returning empty suggestions due to error (no static fallback)")
        return []


async def get_conversation_suggestions_tool(
    user_id: int,
    thread_id: str,
    language: str,
) -> str:
    """
    Generate contextual conversation suggestions based on recent message history.
    
    This tool analyzes the last 5 messages in the conversation and generates
    3 relevant suggestions for continuing the conversation.
    
    Args:
        user_id: User ID (from context)
        thread_id: Thread ID (from context)
        language: User's language (from context)
        
    Returns:
        JSON string with suggestions: '{"suggestions": ["suggestion1", "suggestion2", "suggestion3"]}'
    """
    try:
        from luka_bot.services.prompt_pool_service import get_prompt_pool_service
        from luka_bot.core.config import settings
        from luka_bot.lg_lukabot.checkpointer import get_checkpointer
        from langchain_core.messages import HumanMessage, AIMessage
        import json
        import random
        
        # Load messages from LangGraph checkpoint
        conversation_context = []
        
        try:
            checkpointer = get_checkpointer()
            if checkpointer:
                # Get latest checkpoint state for this thread
                config = {"configurable": {"thread_id": thread_id}}
                
                # Try to get checkpoint (RedisSaver may support async or sync)
                if hasattr(checkpointer, 'aget'):
                    checkpoint = await checkpointer.aget(config)
                elif hasattr(checkpointer, 'get'):
                    checkpoint = checkpointer.get(config)
                else:
                    logger.warning("⚠️ Checkpointer doesn't have get/aget method")
                    checkpoint = None
                
                if checkpoint and checkpoint.get("channel_values"):
                    state = checkpoint["channel_values"]
                    messages = state.get("messages", [])
                    
                    # Extract last 3 messages INCLUDING the assistant's latest response
                    # We need to see what the assistant just said to generate aligned suggestions
                    recent_messages = messages[-3:] if len(messages) >= 3 else messages
                    
                    for msg in recent_messages:
                        if isinstance(msg, HumanMessage):
                            content = str(msg.content) if msg.content else ""
                            if content.strip():
                                conversation_context.append(f"User: {content}")
                        elif isinstance(msg, AIMessage):
                            content = str(msg.content) if msg.content else ""
                            if content.strip():
                                conversation_context.append(f"Assistant: {content}")
                    
                    logger.debug(f"📚 Loaded {len(conversation_context)} messages from LangGraph checkpoint (total: {len(messages)})")
            else:
                logger.debug("⚠️ Checkpointer not available, will use static prompts")
        except Exception as e:
            logger.debug(f"⚠️ Failed to load checkpoint state: {e}")
        
        logger.debug(f"📝 Built context with {len(conversation_context)} messages")
        
        # With 3 messages, we need at least 1 valid message pair
        if len(conversation_context) < 1:
            # Not enough valid conversation messages - use static prompts as fallback
            logger.info(f"📋 Insufficient conversation context ({len(conversation_context)} messages), using static prompts fallback")
            try:
                prompt_pool_service = get_prompt_pool_service()
                static_prompt_options = await prompt_pool_service.get_quick_prompts(locale=language, count=10)
                
                if static_prompt_options:
                    # Randomize and take 3
                    random.shuffle(static_prompt_options)
                    suggestions = [opt.text for opt in static_prompt_options[:3]]
                    logger.info(f"✅ Using {len(suggestions)} randomized static prompts: {suggestions}")
                    return f"SUGGESTIONS:{json.dumps(suggestions)}"
            except Exception as e:
                logger.warning(f"⚠️ Failed to get static prompts: {e}")
            
            return "SUGGESTIONS:[]"
        
        # Use LLM to generate suggestions
        from langchain_ollama import ChatOllama
        from langchain_openai import ChatOpenAI
        
        # Select LLM provider
        if settings.DEFAULT_LLM_PROVIDER == "ollama":
            ollama_url = settings.OLLAMA_URL.rstrip("/v1").rstrip("/")
            llm = ChatOllama(
                model=settings.OLLAMA_MODEL_NAME,
                base_url=ollama_url,
                temperature=0.7,
            )
        else:
            llm = ChatOpenAI(
                model=settings.DEFAULT_LLM_MODEL or "gpt-4o-mini",
                temperature=0.7,
            )
        
        # Build prompt (single English prompt with language variable)
        # Extract the last assistant message to align suggestions with the current response
        last_assistant_msg = None
        for msg in reversed(conversation_context):
            if msg.startswith("Assistant:"):
                last_assistant_msg = msg.replace("Assistant:", "").strip()
                break
        
        # Get previous conversation context (excluding the last assistant message)
        previous_context = []
        for msg in conversation_context:
            if msg.startswith("Assistant:") and msg.replace("Assistant:", "").strip() != last_assistant_msg:
                previous_context.append(msg)
            elif msg.startswith("User:"):
                previous_context.append(msg)
        
        # Remove the last assistant message from previous context if it's there
        previous_context = [msg for msg in previous_context if not (msg.startswith("Assistant:") and msg.replace("Assistant:", "").strip() == last_assistant_msg)]
        previous_context_text = "\n".join(previous_context[-4:]) if previous_context else "No previous conversation"
        
        context_text = "\n".join(conversation_context)
        
        logger.debug(f"📝 Last assistant message for alignment: '{last_assistant_msg[:100] if last_assistant_msg else 'N/A'}...'")
        
        # Get full last assistant message for detailed analysis (limit to 2000 chars to avoid token limits)
        full_last_message = last_assistant_msg[:2000] if last_assistant_msg else "N/A"
        
        prompt = f"""Analyze the conversation and suggest 3 short QUESTIONS that the USER might ask the assistant next.

ASSISTANT'S LAST MESSAGE (FOCUS HERE - THIS IS THE PRIMARY SOURCE):
"{full_last_message}"

PREVIOUS CONVERSATION CONTEXT (for reference only):
{previous_context_text}

CRITICAL INSTRUCTIONS - FOCUS ON LAST MESSAGE:
- These suggestions will appear as CLICKABLE BUTTONS that the USER will send to the assistant
- Phrase suggestions as USER QUESTIONS to the assistant (must be questions, not commands)
- **2 out of 3 suggestions MUST be about the LAST MESSAGE** - they should ask questions about specific main points, features, locations, numbers, or topics from the assistant's last response
- **1 suggestion can be from previous conversation context** (if relevant) or a general follow-up question
- Extract and reference SPECIFIC details from the last message in your questions (e.g., if the assistant mentioned "Milan", "€100 budget", "3-day trip", "historic sites", etc., mention these in the questions)
- Each suggestion should be a QUESTION the user would ask (e.g., "What is X?", "How do I Y?", "Tell me more about Z")
- DO NOT use commands or statements (WRONG: "Show my notes", "Edit the note", "Set a reminder")
- DO NOT phrase as assistant questions to the user (WRONG: "What should I do with 89?", "Did you remember 42?")
- If the assistant gave a GENERIC greeting (like "Hi! How can I help?"), suggest GENERIC user questions (not specific to old context)
- Generate suggestions in {language.upper()} language
- Do NOT repeat what was already discussed
- Suggest NATURAL next questions the user might ask
- Each suggestion should be short (max 40 characters)
- Use emojis where appropriate
- Return only the suggestions, one per line
- No numbering or bullet points

Examples of CORRECT phrasing (mentioning specific points from last message):
- If assistant mentioned "Milan": "Tell me more about Milan 🏛️"
- If assistant mentioned "€100 budget": "What about the €100 budget?"
- If assistant mentioned "3-day trip": "What's in the 3-day plan?"
- If assistant mentioned specific features: "How does [specific feature] work?"
- General follow-up: "What else can you do?"

Examples of WRONG phrasing (commands - NOT questions):
- "Show my notes" ❌
- "Edit the 42 note" ❌
- "Set a reminder for 89" ❌

Examples of WRONG phrasing (assistant asking user):
- "What should I do with 89?" ❌
- "Did you remember 42?" ❌
- "Need to add 89 to a reminder?" ❌

Suggestions:"""
        
        # Call LLM
        logger.debug(f"🤖 Calling LLM for conversation suggestions (thread {thread_id})")
        response = await llm.ainvoke(prompt)
        suggestions_text = response.content.strip()
        logger.debug(f"🤖 LLM raw response: {suggestions_text}")
        
        # Parse suggestions (one per line)
        raw_lines = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
        logger.debug(f"📝 Parsed {len(raw_lines)} raw lines: {raw_lines}")
        
        suggestions = []
        for s in raw_lines:
            # Remove common prefixes like "1.", "-", "•", etc.
            cleaned = s.lstrip("1234567890.-•* ").strip()
            # Be more permissive with length (3-50 chars to accommodate Russian)
            if cleaned and 3 <= len(cleaned) <= 50:
                suggestions.append(cleaned)
            else:
                logger.debug(f"⏭️ Skipped suggestion (len={len(cleaned) if cleaned else 0}): '{cleaned}'")
        
        # Limit to 3
        suggestions = suggestions[:3]
        
        # If LLM failed to generate valid suggestions, fall back to static prompts
        if not suggestions:
            logger.info(f"📋 LLM generated no valid suggestions, using static prompts fallback")
            try:
                prompt_pool_service = get_prompt_pool_service()
                static_prompt_options = await prompt_pool_service.get_quick_prompts(locale=language, count=10)
                
                if static_prompt_options:
                    # Randomize and take 3
                    random.shuffle(static_prompt_options)
                    suggestions = [opt.text for opt in static_prompt_options[:3]]
                    logger.info(f"✅ Using {len(suggestions)} randomized static prompts as fallback: {suggestions}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to get static prompts fallback: {e}")
        else:
            logger.info(f"✅ Generated {len(suggestions)} conversation suggestions for thread {thread_id}: {suggestions}")
        
        return f"SUGGESTIONS:{json.dumps(suggestions)}"
        
    except Exception as e:
        logger.error(f"❌ Error generating conversation suggestions: {e}", exc_info=True)
        # Last resort: try to return static prompts even on fatal error
        try:
            from luka_bot.services.prompt_pool_service import get_prompt_pool_service
            import random
            prompt_pool_service = get_prompt_pool_service()
            static_prompt_options = await prompt_pool_service.get_quick_prompts(locale=language, count=10)
            
            if static_prompt_options:
                random.shuffle(static_prompt_options)
                suggestions = [opt.text for opt in static_prompt_options[:3]]
                logger.info(f"✅ Recovered with {len(suggestions)} static prompts after error")
                return f"SUGGESTIONS:{json.dumps(suggestions)}"
        except Exception as fallback_error:
            logger.error(f"❌ Even fallback failed: {fallback_error}")
        
        return "SUGGESTIONS:[]"


async def access_conversation_history_tool(
    query: str,
    max_messages: int,
    user_id: int,
    thread_id: str,
) -> str:
    """
    Access conversation history from checkpoint to answer questions about past messages.
    
    This tool allows the LLM to search through saved conversation history when needed,
    for example when the user asks "what number did I ask you to remember?" or
    "what did we discuss earlier?"
    
    Args:
        query: What to search for in the conversation history
        max_messages: Maximum number of messages to search (default: 20, max: 100)
        user_id: User ID (from context)
        thread_id: Thread ID (from context)
        
    Returns:
        Formatted conversation history relevant to the query
    """
    try:
        from luka_bot.lg_lukabot.checkpointer import get_checkpointer
        from langchain_core.messages import HumanMessage, AIMessage
        
        checkpointer = get_checkpointer()
        if not checkpointer:
            return "❌ Checkpointer not available. Cannot access conversation history."
        
        # Get latest checkpoint state for this thread
        config = {"configurable": {"thread_id": thread_id}}
        
        # Try to get checkpoint
        if hasattr(checkpointer, 'aget'):
            checkpoint = await checkpointer.aget(config)
        elif hasattr(checkpointer, 'get'):
            checkpoint = checkpointer.get(config)
        else:
            return "❌ Checkpointer doesn't support get/aget method."
        
        if not checkpoint or not checkpoint.get("channel_values"):
            return "❌ No checkpoint found for this conversation."
        
        state = checkpoint["channel_values"]
        messages = state.get("messages", [])
        
        # Limit to max_messages (most recent)
        search_messages = messages[-max_messages:] if len(messages) > max_messages else messages
        
        # Format messages for search
        formatted_history = []
        for i, msg in enumerate(search_messages, start=1):
            if isinstance(msg, HumanMessage):
                content = str(msg.content) if msg.content else ""
                if content.strip():
                    formatted_history.append(f"[Message {i}] User: {content}")
            elif isinstance(msg, AIMessage):
                content = str(msg.content) if msg.content else ""
                if content.strip():
                    formatted_history.append(f"[Message {i}] Assistant: {content}")
        
        if not formatted_history:
            return f"❌ No conversation history found in checkpoint (searched {len(search_messages)} messages)."
        
        # Return formatted history
        history_text = "\n".join(formatted_history)
        total_messages = len(messages)
        searched_count = len(search_messages)
        
        result = f"📚 Conversation History (searched {searched_count} of {total_messages} total messages):\n\n{history_text}"
        
        logger.info(f"✅ Accessed conversation history for query '{query[:50]}...' ({searched_count}/{total_messages} messages)")
        return result

    except Exception as e:
        logger.error(f"❌ Error in access_conversation_history_tool: {e}", exc_info=True)
        return f"❌ Error accessing conversation history: {str(e)}"


# ============================================================================
# TOOL FACTORY FUNCTIONS
# ============================================================================


def create_knowledge_base_tool(
    user_id: int,
    thread_id: str,
    language: str,
    knowledge_bases: list[str],
) -> BaseTool:
    """Create knowledge base search tool with user context (DEPRECATED - use separate tools).

    Args:
        user_id: User ID for access control
        thread_id: Thread ID for context
        language: User's language
        knowledge_bases: List of KB indices user can access

    Returns:
        LangChain StructuredTool for KB search
    """
    return StructuredTool.from_function(
        name="search_knowledge_base",
        description=(
            "Search the user's knowledge base (message history) AND crypto Twitter for relevant information. "
            "FEATURES: "
            "1. Crypto Twitter: Automatically searches crypto-tweets index for crypto/blockchain queries "
            "2. Smart Routing: Detects crypto keywords and searches Twitter KOLs first, synthesizes answers with LLM "
            "3. Message History: Searches user's Telegram message history with advanced filters "
            "4. Digests: Use query='*' with date filters for comprehensive summaries (batched processing for large sets) "
            "5. Deeplinks: Returns formatted cards with clickable links to original messages "
            "USE CASES: "
            "- Crypto queries: 'What is SolAtlas?', 'Solana market updates' → searches crypto tweets + synthesizes answer "
            "- Past conversations: 'What did we discuss about X?' → searches message history "
            "- Digests: 'digest for last week' → query='*', date_from='7d' → comprehensive summary "
            "- User filter: 'What did John say?' → from_user='John' "
            "Supports time-based filters (7d, 1m, YYYY-MM-DD) and user filters."
        ),
        func=lambda query, from_user=None, date_from=None, date_to=None, max_results=5: search_knowledge_base_tool(
            query=query,
            from_user=from_user,
            date_from=date_from,
            date_to=date_to,
            max_results=max_results,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            knowledge_bases=knowledge_bases,
        ),
        args_schema=SearchKnowledgeBaseInput,
        coroutine=lambda query, from_user=None, date_from=None, date_to=None, max_results=5: search_knowledge_base_tool(
            query=query,
            from_user=from_user,
            date_from=date_from,
            date_to=date_to,
            max_results=max_results,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            knowledge_bases=knowledge_bases,
        ),
    )


def create_search_crypto_tweets_tool(
    user_id: int,
    thread_id: str,
    language: str,
) -> BaseTool:
    """Create crypto tweets search tool.
    
    Args:
        user_id: User ID
        thread_id: Thread ID
        language: User's language
        
    Returns:
        LangChain StructuredTool for crypto tweets search
    """
    return StructuredTool.from_function(
        name="search_crypto_tweets",
        description=(
            "Search crypto Twitter for recent tweets from crypto KOLs (Key Opinion Leaders). "
            "This tool searches the crypto-tweets index and synthesizes natural answers using LLM. "
            "USE THIS TOOL when users ask about: "
            "- Crypto projects/tokens: 'What is Solana?', 'Tell me about Bitcoin', 'SOL market updates' "
            "- Blockchain topics: 'DeFi trends', 'NFT news', 'Layer 2 solutions' "
            "- Market updates: 'crypto market today', 'Bitcoin price discussion' "
            "- Twitter/social media: 'what's on crypto twitter?', 'top crypto tweets', 'pull SOL tweets' "
            "The tool returns a complete synthesized answer (not raw tweets) ready to present to the user. "
            "Supports time-based filters (7d, 1m, YYYY-MM-DD) and author filters."
        ),
        func=lambda query, from_user=None, date_from=None, date_to=None, max_results=20: search_crypto_tweets_tool(
            query=query,
            from_user=from_user,
            date_from=date_from,
            date_to=date_to,
            max_results=max_results,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
        ),
        args_schema=SearchCryptoTweetsInput,
        coroutine=lambda query, from_user=None, date_from=None, date_to=None, max_results=20: search_crypto_tweets_tool(
            query=query,
            from_user=from_user,
            date_from=date_from,
            date_to=date_to,
            max_results=max_results,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
        ),
    )


def create_search_message_history_tool(
    user_id: int,
    thread_id: str,
    language: str,
    knowledge_bases: list[str],
) -> BaseTool:
    """Create message history search tool.
    
    Args:
        user_id: User ID
        thread_id: Thread ID
        language: User's language
        knowledge_bases: List of KB indices to search
        
    Returns:
        LangChain StructuredTool for message history search
    """
    return StructuredTool.from_function(
        name="search_message_history",
        description=(
            "Search the user's Telegram message history (personal KB or group KBs). "
            "This tool searches indexed messages from past conversations. "
            "USE THIS TOOL when users ask about: "
            "- Past conversations: 'What did we discuss about X?', 'What did I say about Y?' "
            "- Stored information: 'Find my notes about Z', 'What did John say?' "
            "- Digests: 'digest for last week' → query='*', date_from='7d' "
            "- Message history: 'show messages from last month', 'what happened in the group?' "
            "Supports time-based filters (7d, 1m, YYYY-MM-DD), user filters, and digests with batched processing."
        ),
        func=lambda query, from_user=None, date_from=None, date_to=None, max_results=5: search_message_history_tool(
            query=query,
            from_user=from_user,
            date_from=date_from,
            date_to=date_to,
            max_results=max_results,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            knowledge_bases=knowledge_bases,
        ),
        args_schema=SearchMessageHistoryInput,
        coroutine=lambda query, from_user=None, date_from=None, date_to=None, max_results=5: search_message_history_tool(
            query=query,
            from_user=from_user,
            date_from=date_from,
            date_to=date_to,
            max_results=max_results,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            knowledge_bases=knowledge_bases,
        ),
    )


async def advance_workflow_tool(
    user_response: str,
    user_id: int,
    thread_id: str,
    language: str,
) -> str:
    """Advance workflow to next step based on user's response.
    
    This is called when the user responds to a workflow suggestion.
    
    Args:
        user_response: User's response/choice
        user_id: User ID
        thread_id: Thread ID
        language: User's language
        
    Returns:
        Next step information with new suggestions
    """
    try:
        from luka_bot.services import (
            get_workflow_discovery_service,
            get_workflow_service,
        )
        from luka_bot.utils.i18n_helper import _
        
        workflow_service = get_workflow_service()
        discovery_service = get_workflow_discovery_service()
        await discovery_service.initialize()
        
        # Find active workflow for this user (check all domains)
        available_workflows = await discovery_service.get_available_workflows()
        
        active_workflow = None
        workflow_def = None
        for domain in available_workflows.keys():
            wf_status = await workflow_service.get_active_workflow_for_user(user_id, domain)
            if wf_status:
                active_workflow = wf_status
                workflow_def = discovery_service.get_workflow(domain)
                break
        
        if not active_workflow or not workflow_def:
            return _("No active workflow found.", language=language)
        
        # Find current and next step
        steps = workflow_def.tool_chain.get("steps", [])
        current_step_idx = next(
            (i for i, s in enumerate(steps) if s.get("id") == active_workflow.current_step),
            None
        )
        
        if current_step_idx is None:
            return _("Current step not found in workflow.", language=language)
        
        # Check if already at complete step
        if active_workflow.current_step == "complete":
            # Already completed - terminate and clean up
            await workflow_service.terminate_workflow(
                workflow_id=active_workflow.workflow_id,
                cleanup=True
            )
            logger.info(f"✅ Workflow {active_workflow.workflow_id} already at complete, terminated")
            return _("🎉 Workflow completed! Great job!", language=language)
        
        # Move to next step
        next_step_idx = current_step_idx + 1
        if next_step_idx >= len(steps):
            # Workflow completed - terminate and clean up
            await workflow_service.terminate_workflow(
                workflow_id=active_workflow.workflow_id,
                cleanup=True
            )
            logger.info(f"✅ Workflow {active_workflow.workflow_id} completed and terminated")
            return _("🎉 Workflow completed! Great job!", language=language)
        
        next_step = steps[next_step_idx]
        next_step_id = next_step.get("id")
        
        # Check if next step is 'complete' - terminate immediately
        if next_step_id == "complete":
            # Update to complete step first
            context = {"user_response": user_response}
            await workflow_service.execute_workflow_step(
                workflow_id=active_workflow.workflow_id,
                step_name="complete",
                context=context
            )
            # Then terminate
            await workflow_service.terminate_workflow(
                workflow_id=active_workflow.workflow_id,
                cleanup=True
            )
            logger.info(f"✅ Workflow {active_workflow.workflow_id} reached complete step and terminated")
            return _("🎉 Workflow completed! Great job!", language=language)
        
        # Update workflow state
        context = {"user_response": user_response}
        await workflow_service.execute_workflow_step(
            workflow_id=active_workflow.workflow_id,
            step_name=next_step_id,
            context=context
        )
        
        # Get instruction for new step
        instruction = next_step.get("instruction", "").strip()
        progress = (next_step_idx + 1) / len(steps)
        
        logger.info(f"✅ Advanced workflow to step '{next_step_id}' (progress: {progress:.0%})")
        
        if instruction:
            return _(
                f"Great! Moving forward (progress: {progress:.0%}).\n\n{instruction}",
                language=language,
            )
        else:
            return _(
                f"Moving to next step (progress: {progress:.0%})",
                language=language,
            )
            
    except Exception as e:
        logger.error(f"Error advancing workflow: {e}", exc_info=True)
        return f"Error advancing workflow: {str(e)}"


class AdvanceWorkflowInput(BaseModel):
    """Input for advance_workflow tool."""
    user_response: str = Field(description="User's response or choice from workflow suggestions")


def create_list_recent_messages_tool(
    user_id: int,
    thread_id: str,
    language: str,
    knowledge_bases: list[str],
) -> BaseTool:
    """Create list recent messages tool with user context.

    Args:
        user_id: User ID for access control
        thread_id: Thread ID for context
        language: User's language
        knowledge_bases: List of KB indices user can access

    Returns:
        LangChain StructuredTool for listing recent messages
    """
    return StructuredTool.from_function(
        name="list_recent_messages",
        description=(
            "List the most recent messages from the knowledge base without search query. "
            "Use when users explicitly ask to 'list' or 'show' recent messages WITHOUT time period. "
            "Examples: 'list recent messages', 'show latest messages'. "
            "DO NOT use for digests or overviews with dates - use search_knowledge_base instead!"
        ),
        func=lambda max_results=10: list_recent_messages_tool(
            max_results=max_results,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            knowledge_bases=knowledge_bases,
        ),
        args_schema=ListRecentMessagesInput,
        coroutine=lambda max_results=10: list_recent_messages_tool(
            max_results=max_results,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            knowledge_bases=knowledge_bases,
        ),
    )


def create_knowledge_base_stats_tool(
    user_id: int,
    thread_id: str,
    language: str,
    knowledge_bases: list[str],
) -> BaseTool:
    """Create knowledge base statistics tool with user context.

    Args:
        user_id: User ID for access control
        thread_id: Thread ID for context
        language: User's language
        knowledge_bases: List of KB indices user can access

    Returns:
        LangChain StructuredTool for KB statistics
    """
    return StructuredTool.from_function(
        name="get_knowledge_base_stats",
        description=(
            "Get comprehensive KB statistics with optional advanced analytics. "
            "ALWAYS INCLUDED: Total messages, KB size, active users, top contributors, unanswered questions. "
            "OPTIONAL FLAGS: include_timeline (activity over time), include_hourly_activity (peak hours), "
            "include_hashtags (most used tags), include_media_breakdown (content types), "
            "include_user_engagement (participation levels). "
            "USE CASES: "
            "- Basic stats: 'How much info is in KB?', 'Who's most active?', 'Show me stats' "
            "- Activity patterns: 'When are people most active?' (use include_hourly_activity=True) "
            "- Content analysis: 'What topics are discussed?' (use include_hashtags=True) "
            "- Engagement: 'How many people contribute?' (use include_user_engagement=True) "
            "- Questions: 'What questions are unanswered?' (included by default) "
            "Supports custom date ranges (date_from/date_to) and top_n for number of items."
        ),
        func=lambda date_from="7d", date_to="", include_timeline=False, include_hourly_activity=False, 
                    include_hashtags=False, include_media_breakdown=False, include_user_engagement=False,
                    include_unanswered_questions=True, top_n=5: get_knowledge_base_stats_tool(
            date_from=date_from,
            date_to=date_to,
            include_timeline=include_timeline,
            include_hourly_activity=include_hourly_activity,
            include_hashtags=include_hashtags,
            include_media_breakdown=include_media_breakdown,
            include_user_engagement=include_user_engagement,
            include_unanswered_questions=include_unanswered_questions,
            top_n=top_n,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            knowledge_bases=knowledge_bases,
        ),
        args_schema=GetKnowledgeBaseStatsInput,
        coroutine=lambda date_from="7d", date_to="", include_timeline=False, include_hourly_activity=False,
                        include_hashtags=False, include_media_breakdown=False, include_user_engagement=False,
                        include_unanswered_questions=True, top_n=5: get_knowledge_base_stats_tool(
            date_from=date_from,
            date_to=date_to,
            include_timeline=include_timeline,
            include_hourly_activity=include_hourly_activity,
            include_hashtags=include_hashtags,
            include_media_breakdown=include_media_breakdown,
            include_user_engagement=include_user_engagement,
            include_unanswered_questions=include_unanswered_questions,
            top_n=top_n,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
            knowledge_bases=knowledge_bases,
        ),
    )


def create_workflow_tool(
    user_id: int,
    thread_id: str,
    language: str,
) -> BaseTool:
    """Create workflow execution tool with user context.

    Args:
        user_id: User ID
        thread_id: Thread ID for workflow context
        language: User's language

    Returns:
        LangChain StructuredTool for workflow execution
    """
    return StructuredTool.from_function(
        name="execute_workflow",
        description=(
            "Start, manage, or complete multi-step workflows like onboarding, tutorials, "
            "or guided tasks. Workflows provide step-by-step guidance with suggestions. "
            "Use this when the user wants to begin a structured process or tutorial."
        ),
        func=lambda domain: execute_workflow_tool(
            domain=domain,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
        ),
        args_schema=ExecuteWorkflowInput,
        coroutine=lambda domain: execute_workflow_tool(
            domain=domain,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
        ),
    )


def create_advance_workflow_tool(
    user_id: int,
    thread_id: str,
    language: str,
) -> BaseTool:
    """Create workflow advancement tool with user context.
    
    Args:
        user_id: User ID
        thread_id: Thread ID
        language: User's language
        
    Returns:
        LangChain StructuredTool for advancing workflows
    """
    return StructuredTool.from_function(
        name="advance_workflow",
        description=(
            "Advance the current workflow to the next step based on user's response. "
            "Call this when the user responds to a workflow suggestion or completes a step. "
            "The user's response will be recorded and the workflow will move forward."
        ),
        func=lambda user_response: advance_workflow_tool(
            user_response=user_response,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
        ),
        args_schema=AdvanceWorkflowInput,
        coroutine=lambda user_response: advance_workflow_tool(
            user_response=user_response,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
        ),
    )


def create_support_info_tool(
    user_id: int,
    thread_id: str,
    language: str,
) -> BaseTool:
    """Create support information tool with user context.

    Args:
        user_id: User ID
        thread_id: Thread ID
        language: User's language

    Returns:
        LangChain StructuredTool for support info
    """
    return StructuredTool.from_function(
        name="get_support_info",
        description=(
            "Get support information, help resources, and documentation links. "
            "Use this when the user asks how to use the bot, needs help with features, "
            "or asks general questions about bot capabilities."
        ),
        func=lambda question: get_support_info_tool(
            question=question,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
        ),
        args_schema=GetSupportInfoInput,
        coroutine=lambda question: get_support_info_tool(
            question=question,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
        ),
    )


def create_support_escalation_tool(
    user_id: int,
    thread_id: str,
    language: str,
) -> BaseTool:
    """Create support escalation tool with user context.

    Args:
        user_id: User ID
        thread_id: Thread ID
        language: User's language

    Returns:
        LangChain StructuredTool for support escalation
    """
    return StructuredTool.from_function(
        name="connect_to_support",
        description=(
            "Connect the user to human support team. Use this when the user "
            "explicitly asks to speak with a human, has a complex technical issue, "
            "or when automated help is insufficient."
        ),
        func=lambda reason: connect_to_support_tool(
            reason=reason,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
        ),
        args_schema=ConnectToSupportInput,
        coroutine=lambda reason: connect_to_support_tool(
            reason=reason,
            user_id=user_id,
            thread_id=thread_id,
            language=language,
        ),
    )


def create_youtube_tool(
    user_id: int,
    thread_id: str,
    language: str,
) -> BaseTool:
    """Create YouTube transcript tool.

    Args:
        user_id: User ID
        thread_id: Thread ID
        language: User's interface language

    Returns:
        LangChain StructuredTool for YouTube transcripts
    """
    return StructuredTool.from_function(
        name="get_youtube_transcript",
        description=(
            "Get transcript from a YouTube video. Useful for summarizing videos, "
            "answering questions about video content, or extracting key points. "
            "Use this when the user provides a YouTube URL or asks about video content."
        ),
        func=lambda video_url, language="en": get_youtube_transcript_tool(
            video_url=video_url,
            language=language,
            user_id=user_id,
            thread_id=thread_id,
            user_language=language,
        ),
        args_schema=GetYouTubeTranscriptInput,
        coroutine=lambda video_url, language="en": get_youtube_transcript_tool(
            video_url=video_url,
            language=language,
            user_id=user_id,
            thread_id=thread_id,
            user_language=language,
        ),
    )


def create_conversation_suggestions_tool(
    user_id: int,
    thread_id: str,
    language: str,
) -> BaseTool:
    """Create conversation suggestions tool with user context.
    
    Args:
        user_id: User ID
        thread_id: Thread ID
        language: User's language
        
    Returns:
        LangChain StructuredTool for conversation suggestions
    """
    return StructuredTool.from_function(
        name="get_conversation_suggestions",
        description=(
            "Generate contextual suggestions for continuing the conversation based on recent message history. "
            "Call this tool AFTER providing your response to the user, to suggest relevant next steps or questions. "
            "This helps guide the conversation naturally. Only call this when there is NO active workflow."
        ),
        func=lambda: get_conversation_suggestions_tool(
            user_id=user_id,
            thread_id=thread_id,
            language=language,
        ),
        args_schema=GetConversationSuggestionsInput,
        coroutine=lambda: get_conversation_suggestions_tool(
            user_id=user_id,
            thread_id=thread_id,
            language=language,
        ),
    )


def create_access_conversation_history_tool(
    user_id: int,
    thread_id: str,
) -> BaseTool:
    """Create conversation history access tool.
    
    This tool allows the LLM to access saved conversation history from checkpoints
    when the user asks about past messages (e.g., "what number did I ask you to remember?").
    
    Args:
        user_id: User ID
        thread_id: Thread ID
        
    Returns:
        LangChain StructuredTool for accessing conversation history
    """
    return StructuredTool.from_function(
        name="access_conversation_history",
        description=(
            "Access conversation history from checkpoint to answer questions about past messages. "
            "Use this tool when the user asks about something from earlier in the conversation, "
            "such as 'what number did I ask you to remember?', 'what did we discuss about X?', "
            "or 'find when I mentioned Y'. This searches through saved checkpoint history (up to 100 messages). "
            "Only use this when you need to recall specific information from past messages that isn't in your current context."
        ),
        func=lambda query, max_messages=20: access_conversation_history_tool(
            query=query,
            max_messages=max_messages,
            user_id=user_id,
            thread_id=thread_id,
        ),
        args_schema=AccessConversationHistoryInput,
        coroutine=lambda query, max_messages=20: access_conversation_history_tool(
            query=query,
            max_messages=max_messages,
            user_id=user_id,
            thread_id=thread_id,
        ),
    )


# ============================================================================
# TOOL NAME MAPPING (Config → LangGraph)
# ============================================================================

def map_config_tools_to_langgraph_tools(config_tools: list[str]) -> list[str]:
    """Map config tool names (module names) to LangGraph individual tool names.
    
    This function converts high-level tool module names from config (like "tripplanner")
    to individual LangGraph tool names (like ["search_locations", "plan_trip", ...]).
    
    Args:
        config_tools: List of tool module names from config (e.g., ["knowledge_base", "tripplanner"])
        
    Returns:
        List of individual LangGraph tool names
    """
    # Map of config tool module names to individual LangGraph tool names
    tool_module_map = {
        "knowledge_base": [
            "search_crypto_tweets",  # NEW: separate tool for crypto tweets
            "search_message_history",  # NEW: separate tool for message history
            "list_recent_messages",
            "knowledge_base_stats",
        ],
        "support": ["support", "support_escalation"],
        "youtube": ["youtube"],
        "workflow": ["workflow", "advance_workflow"],
    }
    
    # Always include these LangGraph-specific tools (not in config)
    langgraph_specific_tools = [
        "conversation_suggestions",
        "access_conversation_history",
    ]
    
    # Map config tools to LangGraph tools
    langgraph_tools = []
    for config_tool in config_tools:
        if config_tool in tool_module_map:
            langgraph_tools.extend(tool_module_map[config_tool])
        else:
            # If tool name is already a LangGraph tool name, use it directly
            # This allows backward compatibility and direct tool name usage
            langgraph_tools.append(config_tool)
    
    # Add LangGraph-specific tools
    langgraph_tools.extend(langgraph_specific_tools)
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for tool in langgraph_tools:
        if tool not in seen:
            seen.add(tool)
            result.append(tool)
    
    return result


# ============================================================================
# MASTER TOOL CREATOR
# ============================================================================


def create_tools_for_user(
    user_id: int,
    thread_id: str,
    knowledge_bases: list[str],
    enabled_tools: list[str],
    language: str = "en",
) -> list[BaseTool]:
    """Create all enabled tools for a user.

    This is the main entry point for creating tools. It returns a list of
    LangChain tools based on the user's configuration.

    Args:
        user_id: User ID for context
        thread_id: Thread ID for workflows and context
        knowledge_bases: List of KB indices user can access
        enabled_tools: List of tool names to enable (e.g., ["knowledge_base", "workflow"])
        language: User's preferred language (default: "en")

    Returns:
        List of LangChain BaseTool instances

    Example:
        >>> tools = create_tools_for_user(
        ...     user_id=123,
        ...     thread_id="user_123",
        ...     knowledge_bases=["tg-kb-user-123"],
        ...     enabled_tools=["knowledge_base", "workflow", "support"],
        ...     language="en"
        ... )
        >>> # Use in LangGraph agent
        >>> llm_with_tools = llm.bind_tools(tools)
    """
    tools = []

    # Map of tool names to factory functions
    tool_factories = {
        # Knowledge base tools
        "knowledge_base": lambda: create_knowledge_base_tool(
            user_id, thread_id, language, knowledge_bases
        ),  # Deprecated - kept for backward compatibility
        "search_crypto_tweets": lambda: create_search_crypto_tweets_tool(
            user_id, thread_id, language
        ),
        "search_message_history": lambda: create_search_message_history_tool(
            user_id, thread_id, language, knowledge_bases
        ),
        "list_recent_messages": lambda: create_list_recent_messages_tool(
            user_id, thread_id, language, knowledge_bases
        ),
        "knowledge_base_stats": lambda: create_knowledge_base_stats_tool(
            user_id, thread_id, language, knowledge_bases
        ),
        # Workflow tools
        "workflow": lambda: create_workflow_tool(user_id, thread_id, language),
        "advance_workflow": lambda: create_advance_workflow_tool(user_id, thread_id, language),
        # Support tools
        "support": lambda: create_support_info_tool(user_id, thread_id, language),
        "support_escalation": lambda: create_support_escalation_tool(
            user_id, thread_id, language
        ),
        # Content tools
        "youtube": lambda: create_youtube_tool(user_id, thread_id, language),
        # Conversation tools
        "conversation_suggestions": lambda: create_conversation_suggestions_tool(user_id, thread_id, language),
        "access_conversation_history": lambda: create_access_conversation_history_tool(user_id, thread_id),
    }

    # Create tools based on enabled list
    for tool_name in enabled_tools:
        if tool_name in tool_factories:
            try:
                tool = tool_factories[tool_name]()
                tools.append(tool)
                logger.debug(f"Created tool: {tool_name} for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to create tool {tool_name}: {e}")
        else:
            logger.warning(f"Unknown tool name: {tool_name}")

    logger.info(f"Created {len(tools)} tools for user {user_id}: {enabled_tools}")
    return tools

