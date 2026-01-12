"""
luka_bot entry point
"""
import asyncio
import os
import socket
from contextlib import suppress
from loguru import logger

# luka_bot core
from luka_bot.core.config import settings
from luka_bot.core.loader import app, bot, dp, redis_client

# handlers
from luka_bot.handlers import get_llm_bot_router
from luka_bot.keyboards.default_commands import set_default_commands
from luka_bot.middlewares.i18n_middleware import UserProfileI18nMiddleware
from luka_bot.middlewares.form_input_middleware import FormInputMiddleware

# Webhook setup - import at module level for use in setup_webhook()
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from luka_bot.handlers.metrics import MetricsView
from luka_bot.middlewares.prometheus import prometheus_middleware_factory
from aiogram.exceptions import TelegramConflictError

# Global metrics server state (for polling mode)
_metrics_server_task = None
_metrics_runner = None

# Polling lock state
_POLLING_LOCK_KEY = f"luka_bot:polling_lock:{settings.BOT_TOKEN[:8]}"
_POLLING_LOCK_TTL = 90
_polling_lock_task = None
_polling_lock_owner = None


async def start_metrics_server() -> None:
    """
    Start a separate metrics server for polling mode.
    In webhook mode, metrics are served on the webhook port.
    """
    global _metrics_server_task, _metrics_runner
    
    if not settings.METRICS_ENABLED:
        logger.info("ℹ️  Metrics disabled (METRICS_ENABLED=False)")
        return
    
    if settings.USE_WEBHOOK:
        # In webhook mode, metrics are served on the webhook server
        # No need for separate server
        return
    
    logger.info(f"📊 Starting metrics server on {settings.METRICS_HOST}:{settings.METRICS_PORT}...")
    
    try:
        # Create a simple aiohttp app for metrics only
        metrics_app = web.Application()
        metrics_app.middlewares.append(prometheus_middleware_factory())
        metrics_app.router.add_route("GET", "/metrics", MetricsView)
        
        # Use AppRunner for graceful lifecycle management
        _metrics_runner = web.AppRunner(metrics_app)
        await _metrics_runner.setup()
        
        site = web.TCPSite(
            _metrics_runner,
            host=settings.METRICS_HOST,
            port=settings.METRICS_PORT
        )
        await site.start()
        
        logger.info(f"✅ Metrics server started on http://{settings.METRICS_HOST}:{settings.METRICS_PORT}/metrics")
        logger.info("   Prometheus can scrape metrics from this endpoint")
        
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"❌ Port {settings.METRICS_PORT} is already in use")
            logger.error("   Change METRICS_PORT in .env or disable METRICS_ENABLED")
        else:
            logger.error(f"❌ Failed to start metrics server: {e}")
    except Exception as e:
        logger.error(f"❌ Failed to start metrics server: {e}", exc_info=True)


async def stop_metrics_server() -> None:
    """Stop the metrics server (polling mode only)."""
    global _metrics_runner
    
    if _metrics_runner is None:
        return
    
    logger.info("🛑 Stopping metrics server...")
    
    try:
        await _metrics_runner.cleanup()
        _metrics_runner = None
        logger.info("✅ Metrics server stopped")
    except Exception as e:
        logger.error(f"❌ Failed to stop metrics server: {e}", exc_info=True)


async def _refresh_polling_lock(owner_id: str) -> None:
    """Background task that refreshes the polling lock TTL while the process is alive."""
    try:
        while True:
            await asyncio.sleep(_POLLING_LOCK_TTL / 2)
            current_owner = await redis_client.get(_POLLING_LOCK_KEY)
            if current_owner is None:
                logger.warning("⚠️ Polling lock unexpectedly missing; stopping refresh task")
                return
            current_owner = current_owner.decode() if isinstance(current_owner, (bytes, bytearray)) else str(current_owner)
            if current_owner != owner_id:
                logger.warning("⚠️ Polling lock ownership changed; stopping refresh task")
                return
            await redis_client.expire(_POLLING_LOCK_KEY, _POLLING_LOCK_TTL)
    except asyncio.CancelledError:
        logger.debug("Polling lock refresh task cancelled")
    except Exception as exc:
        logger.error(f"❌ Error refreshing polling lock: {exc}", exc_info=True)


async def acquire_polling_lock() -> bool:
    """Acquire a Redis-backed lock to ensure only one polling instance runs."""
    if not settings.POLLING_LOCK_ENABLED:
        logger.debug("ℹ️  Polling lock disabled (POLLING_LOCK_ENABLED=False), skipping lock acquisition")
        return True
    
    global _polling_lock_task, _polling_lock_owner
    
    owner_id = f"{socket.gethostname()}:{os.getpid()}"
    try:
        acquired = await redis_client.set(
            _POLLING_LOCK_KEY,
            owner_id,
            ex=_POLLING_LOCK_TTL,
            nx=True,
        )
        if not acquired:
            existing = await redis_client.get(_POLLING_LOCK_KEY)
            existing_id = existing.decode() if isinstance(existing, (bytes, bytearray)) else existing
            logger.error(
                "❌ Another luka_bot instance is already polling Telegram updates.\n"
                f"   Current owner: {existing_id}"
            )
            return False
        
        _polling_lock_owner = owner_id
        _polling_lock_task = asyncio.create_task(_refresh_polling_lock(owner_id))
        logger.info(f"🔐 Acquired polling lock as {owner_id}")
        return True
    except Exception as exc:
        logger.error(f"❌ Failed to acquire polling lock: {exc}", exc_info=True)
        return False


async def release_polling_lock() -> None:
    """Release the polling lock if this instance owns it."""
    if not settings.POLLING_LOCK_ENABLED:
        return  # Lock is disabled, nothing to release
    
    global _polling_lock_task, _polling_lock_owner
    
    if _polling_lock_task:
        _polling_lock_task.cancel()
        with suppress(asyncio.CancelledError):
            await _polling_lock_task
        _polling_lock_task = None
    
    if not _polling_lock_owner:
        return
    
    try:
        current_owner = await redis_client.get(_POLLING_LOCK_KEY)
        if current_owner is not None:
            current_owner = current_owner.decode() if isinstance(current_owner, (bytes, bytearray)) else current_owner
            if current_owner == _polling_lock_owner:
                await redis_client.delete(_POLLING_LOCK_KEY)
                logger.info("🔓 Released polling lock")
    except Exception as exc:
        logger.error(f"⚠️ Failed to release polling lock cleanly: {exc}", exc_info=True)
    finally:
        _polling_lock_owner = None


async def on_startup() -> None:
    """Bot startup initialization."""
    logger.info("🚀 luka_bot starting...")
    
    # Clear workflow validation cache to ensure fresh validation on startup
    try:
        from luka_bot.core.loader import redis_client
        # Clear validation cache (correct key pattern without 'luka:' prefix)
        keys = await redis_client.keys("workflow_validation:*")
        if keys:
            deleted_count = await redis_client.delete(*keys)
            logger.info(f"🧹 Cleared {deleted_count} workflow validation cache entries from Redis")
        else:
            logger.debug("🧹 No workflow validation cache entries in Redis to clear")
        
        # Also clear any workflow definition cache keys
        def_keys = await redis_client.keys("workflow_definition:*")
        if def_keys:
            def_deleted = await redis_client.delete(*def_keys)
            logger.info(f"🧹 Cleared {def_deleted} workflow definition cache entries from Redis")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clear workflow validation cache: {e}")
    
    # Force workflow discovery service to reinitialize to pick up config changes
    try:
        from luka_bot.services import workflow_discovery_service
        # Clear the singleton to force reinitialization
        if hasattr(workflow_discovery_service, '_workflow_discovery_service'):
            workflow_discovery_service._workflow_discovery_service = None
            logger.info("🧹 Cleared workflow discovery singleton cache")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clear workflow discovery singleton: {e}")
    
    # Initialize and check LLM providers
    try:
        from luka_bot.services.llm_provider_fallback import get_llm_provider_fallback
        fallback = get_llm_provider_fallback()
        
        # Perform unified startup check
        result = await fallback.initialize_on_startup()
        
        # Log provider availability
        if result["stats"]:
            stats = result["stats"]
            logger.info("")
            logger.info("🔍 LLM Provider Status:")
            logger.info(f"   Primary: {stats['primary_provider']}")
            logger.info(f"   Fallback: {stats['fallback_provider']}")
            logger.info("")
            
            for provider_name in fallback.ALL_PROVIDERS:
                health = stats['provider_health'][provider_name]
                status_emoji = "✅" if health['available'] else "❌"
                config_status = "configured" if health['configured'] else "not configured"
                health_status = "healthy" if health['healthy'] else "unhealthy"
                
                logger.info(
                    f"   {status_emoji} {provider_name.upper()}: "
                    f"{config_status}, {health_status}, "
                    f"available={health['available']}"
                )
        
        # Log active provider
        if result["active_provider"]:
            logger.info(f"✅ Active provider: {result['active_provider'].upper()} (cached for 30 minutes)")
        elif result["error"]:
            logger.error(f"❌ No providers available: {result['error']}")
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize LLM providers: {e}")

    # Validate public KB configuration for guest users
    public_kbs = settings.public_knowledge_bases
    if public_kbs:
        logger.info(f"📚 Public KB configured: {public_kbs} ({len(public_kbs)} indices)")

        if not settings.ELASTICSEARCH_ENABLED:
            logger.warning("⚠️  Public KB configured but Elasticsearch is disabled!")
            logger.warning("   Set ELASTICSEARCH_ENABLED=true to enable KB features")
        else:
            try:
                from luka_bot.services.elasticsearch_service import get_elasticsearch_service
                es_service = await get_elasticsearch_service()

                # Validate each index
                for index_name in public_kbs:
                    exists = await es_service.index_exists(index_name)

                    if exists:
                        logger.info(f"✅ Public KB index verified: {index_name}")
                    else:
                        logger.warning(f"⚠️  Public KB index not found: {index_name}")
                        logger.warning(f"   Guest users won't have access to this KB")
                        logger.warning(f"   Create index: curl -X PUT '{settings.ELASTICSEARCH_URL}/{index_name}'")

            except Exception as e:
                logger.warning(f"⚠️  Failed to validate public KB indices: {e}")
    else:
        logger.info("📚 Public KB not configured (LUKA_PUBLIC_KNOWLEDGE_BASE empty)")
        logger.info("   Guest users will use personal KB indices")


    # Register middlewares (ORDER MATTERS!)
    # 1. Password gate FIRST (if enabled, blocks unauthenticated users)
    if settings.LUKA_PASSWORD_ENABLED:
        from luka_bot.middlewares.password_middleware import PasswordMiddleware
        dp.message.middleware(PasswordMiddleware())
        dp.callback_query.middleware(PasswordMiddleware())
        logger.info("🔒 Password authentication middleware registered")
        logger.info(f"   Password protection: {'ENABLED ✅' if settings.LUKA_PASSWORD else 'ENABLED but NO PASSWORD SET ⚠️'}")
    
    # 2. Flow API auth SECOND (provides user context with Camunda credentials)
    from luka_bot.middlewares.flow_auth_middleware import FlowAuthMiddleware
    dp.message.middleware(FlowAuthMiddleware())
    dp.callback_query.middleware(FlowAuthMiddleware())
    logger.info("🔐 Flow API session-based auth middleware registered")
    
    # 3. Form input guard THIRD (prevents LLM handlers from consuming form messages)
    dp.message.middleware(FormInputMiddleware())
    logger.info("🛡️ Form input guard middleware registered")
    
    # 4. I18n FOURTH (can use user context from auth middleware)
    dp.message.middleware(UserProfileI18nMiddleware())
    dp.callback_query.middleware(UserProfileI18nMiddleware())
    logger.info(f"🌍 I18n middleware registered (default locale: {settings.DEFAULT_LOCALE})")
    
    # Log Privacy Mode status
    privacy_status = "ON (limited visibility)" if settings.BOT_PRIVACY_MODE_ENABLED else "OFF (sees all messages)"
    privacy_emoji = "🔒" if settings.BOT_PRIVACY_MODE_ENABLED else "👁️"
    logger.info(f"{privacy_emoji} Privacy Mode: {privacy_status}")
    
    # Load process definitions from Camunda (if available)
    if settings.CAMUNDA_ENABLED:
        try:
            from luka_bot.services.process_definition_cache import get_process_definition_cache
            from luka_bot.services.user_profile_service import get_user_profile_service

            # Get any valid user ID for loading definitions (use first available user or system user)
            profile_service = get_user_profile_service()

            # Try to get a system/admin user ID from config, or use a default
            # This user ID is only used for authentication, not for filtering
            system_user_id = settings.SYSTEM_USER_ID if hasattr(settings, 'SYSTEM_USER_ID') else 0

            if system_user_id == 0:
                # If no system user configured, get any existing user from profiles
                # This is just for auth - we load ALL process definitions
                logger.debug("No SYSTEM_USER_ID configured, using guest/default user for process definition loading")

            cache = get_process_definition_cache()
            await cache.load_definitions(telegram_user_id=system_user_id)

            if cache.is_loaded():
                chatbot_processes = cache.get_chatbot_processes()
                if chatbot_processes:
                    logger.info(f"✅ Chatbot processes available: {', '.join([p.key for p in chatbot_processes])}")
                else:
                    logger.warning("⚠️  No chatbot_* processes found - chatbot features may be limited")
            else:
                logger.warning("⚠️  Process definitions not loaded - process-based features disabled")

        except Exception as e:
            logger.warning(f"⚠️  Failed to load process definitions: {e}")
            logger.info("   Bot will continue without process-based features")
    else:
        logger.info("ℹ️  Camunda integration disabled (CAMUNDA_ENABLED=False)")
        logger.info("   Process-based features will not be available")

    # Initialize workflow service for dialog workflows
    try:
        from luka_bot.services.workflow_service import get_workflow_service
        workflow_service = get_workflow_service()
        initialized = await workflow_service.initialize()
        
        if initialized:
            available_workflows = await workflow_service.get_available_workflows()
            logger.info(f"✅ WorkflowService initialized with {len(available_workflows)} workflow(s)")
            
            if available_workflows:
                domains = [w.domain for w in available_workflows]
                logger.info(f"   Available workflow domains: {', '.join(domains)}")
            else:
                logger.warning("⚠️  No workflows discovered - check workflow directories and config.yaml files")
            
            # Log validation errors if any
            from luka_bot.services.workflow_discovery_service import get_workflow_discovery_service
            discovery_service = get_workflow_discovery_service()
            validation_errors = discovery_service.get_validation_errors()
            
            if validation_errors:
                logger.warning(f"⚠️  Found validation issues in {len(validation_errors)} workflow(s):")
                for domain, errors in validation_errors.items():
                    error_count = len([e for e in errors if e.get("level") == "error"])
                    warning_count = len([e for e in errors if e.get("level") == "warning"])
                    if error_count > 0:
                        logger.error(f"   ❌ {domain}: {error_count} errors, {warning_count} warnings")
                        # Log detailed errors in debug mode
                        try:
                            log_level = logger._core.min_level if hasattr(logger, '_core') else 20
                            if log_level <= 10:  # DEBUG level
                                formatted = discovery_service._validation_service.format_validation_errors(errors)
                                if formatted:
                                    logger.debug(f"      Validation details for {domain}:\n{formatted}")
                        except (AttributeError, TypeError):
                            # Fallback: try to format anyway
                            try:
                                formatted = discovery_service._validation_service.format_validation_errors(errors)
                                if formatted:
                                    logger.debug(f"      Validation details for {domain}:\n{formatted}")
                            except Exception:
                                pass
                    elif warning_count > 0:
                        logger.warning(f"   ⚠️  {domain}: {warning_count} warnings")
        else:
            logger.warning("⚠️  WorkflowService initialization failed - workflows will not be available")
            
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize workflow service: {e}")
        logger.info("   Bot will continue without dialog workflow features")

    # Register handlers
    dp.include_router(get_llm_bot_router())
    logger.info("📦 Handlers registered")

    # Start metrics server (polling mode only)
    # In webhook mode, metrics are served on the same port as webhook
    await start_metrics_server()
    
    # Start AG-UI Gateway server (if enabled)
    # In webhook mode: AG-UI was already mounted in setup_webhook() before aiogram setup
    # In polling mode: Start separate AG-UI server on port 8000
    if settings.AG_UI_ENABLED and not settings.USE_WEBHOOK:
        from luka_bot.core.ag_ui_integration import start_ag_ui_server
        await start_ag_ui_server(webhook_app=None)  # None = polling mode, separate server
    
    # Set default commands for the command menu
    logger.info("")
    logger.info("📋 Configuring default commands...")
    await set_default_commands(bot)
    
    # Display summary of configured commands
    from luka_bot.keyboards.default_commands import (
        private_commands_by_language,
        group_commands_by_language,
        admin_commands_by_language,
    )
    
    logger.info("")
    logger.info("📋 Command Summary:")
    logger.info("   Private Chats:")
    # Filter by LUKA_COMMANDS_ENABLED
    enabled_commands = settings.LUKA_COMMANDS_ENABLED
    for cmd in private_commands_by_language["en"].keys():
        if cmd in enabled_commands:
            logger.info(f"      /{cmd}")
    
    # Show disabled commands
    disabled_commands = [cmd for cmd in private_commands_by_language["en"].keys() if cmd not in enabled_commands]
    if disabled_commands:
        logger.info("   Disabled Commands:")
        for cmd in disabled_commands:
            logger.info(f"      /{cmd} (configure via LUKA_COMMANDS_ENABLED)")
    
    logger.info("   Groups:")
    for cmd in group_commands_by_language["en"].keys():
        logger.info(f"      /{cmd}")
    
    logger.info("   Group Admins (additional):")
    admin_only = set(admin_commands_by_language["en"].keys()) - set(group_commands_by_language["en"].keys())
    for cmd in admin_only:
        logger.info(f"      /{cmd}")
    
    logger.info("   Supported languages: en, ru")
    
    # Get bot info
    bot_info = await bot.get_me()
    logger.info(f"✅ Bot: {bot_info.full_name} (@{bot_info.username}, ID: {bot_info.id})")
    
    # Initialize WebSocket manager (if enabled)
    if settings.WAREHOUSE_ENABLED:
        try:
            from luka_bot.services.task_websocket_manager import get_websocket_manager
            ws_manager = get_websocket_manager()
            
            logger.info("")
            logger.info("🔌 WebSocket Task Notifications:")
            logger.info(f"   Warehouse URL: {settings.WAREHOUSE_WS_URL}")
            logger.info("   Per-user connections will be established on first interaction")
            logger.info("   Real-time task notifications enabled ✅")
            
        except Exception as e:
            logger.warning(f"⚠️  WebSocket manager initialization failed: {e}")
            logger.warning("   Bot will fall back to polling mode")
    else:
        logger.info("")
        logger.info("ℹ️  WebSocket disabled (WAREHOUSE_ENABLED=False)")
        logger.info("   Using polling mode for task detection")
    
    # Log metrics availability summary
    if settings.METRICS_ENABLED:
        logger.info("")
        logger.info("📊 Prometheus Metrics Summary:")
        if settings.USE_WEBHOOK:
            logger.info(f"   Mode: Webhook (same port as bot)")
            logger.info(f"   URL: http://{settings.WEBHOOK_HOST}:{settings.WEBHOOK_PORT}/metrics")
        else:
            logger.info(f"   Mode: Polling (separate metrics server)")
            logger.info(f"   URL: http://{settings.METRICS_HOST}:{settings.METRICS_PORT}/metrics")
        logger.info("   Status: Ready for Prometheus scraping ✅")
    
    logger.info("")
    logger.info("✅ luka_bot started successfully")


async def on_shutdown() -> None:
    """Bot shutdown cleanup."""
    logger.info("🛑 luka_bot stopping...")
    
    # Cancel all background tasks (Moderation)
    try:
        from luka_bot.utils.background_tasks import cancel_all_background_tasks
        await cancel_all_background_tasks()
    except Exception as e:
        logger.warning(f"⚠️ Error cancelling background tasks: {e}")
    
    # Shutdown WebSocket manager
    if settings.WAREHOUSE_ENABLED:
        try:
            from luka_bot.services.task_websocket_manager import shutdown_websocket_manager
            await shutdown_websocket_manager()
            logger.info("✅ WebSocket manager shut down")
        except Exception as e:
            logger.warning(f"⚠️  Error shutting down WebSocket manager: {e}")
    
    # Stop metrics server (polling mode only)
    await stop_metrics_server()
    
    # Stop AG-UI Gateway server (if enabled)
    if settings.AG_UI_ENABLED:
        try:
            from luka_bot.core.ag_ui_integration import stop_ag_ui_server
            await stop_ag_ui_server()
        except Exception as e:
            logger.warning(f"⚠️  Error stopping AG-UI server: {e}")
    
    # Release polling lock on shutdown
    if not settings.USE_WEBHOOK:
        await release_polling_lock()
    
    # Close bot session
    await bot.session.close()
    
    # Close storage
    await dp.storage.close()
    await dp.fsm.storage.close()
    
    logger.info("✅ luka_bot stopped")


async def setup_webhook() -> None:
    """Setup webhook for production (Phase 8)."""
    webhook_url = settings.webhook_url
    logger.info(f"Setting up webhook: {webhook_url}")

    # CRITICAL: Route registration order matters in aiohttp!
    # Specific routes MUST be registered before catch-all routes.
    #
    # Correct order:
    # 1. Metrics (specific: /metrics)
    # 2. Webhook handler (specific: /webhook or configured path)
    # 3. AG-UI catch-all (last: /* catches everything else)

    # Step 1: Initialize AG-UI (starts internal server, returns proxy handlers)
    ag_ui_handlers = None
    if settings.AG_UI_ENABLED:
        logger.info("🔧 Initializing AG-UI Gateway...")
        from luka_bot.core.ag_ui_integration import mount_ag_ui_on_webhook
        ag_ui_handlers = await mount_ag_ui_on_webhook(app)

    # Step 2: Setup metrics endpoint (specific route)
    if settings.METRICS_ENABLED:
        logger.info("📊 Setting up Prometheus metrics...")
        app.middlewares.append(prometheus_middleware_factory())
        app.router.add_route("GET", "/metrics", MetricsView)
        logger.info("✅ Metrics endpoint registered at /metrics")
    else:
        logger.info("ℹ️  Metrics disabled (METRICS_ENABLED=False)")

    # Step 3: Register webhook handler (specific route)
    await bot.set_webhook(
        url=webhook_url,
        drop_pending_updates=True,
        secret_token=settings.WEBHOOK_SECRET
    )

    webhook_request_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
        secret_token=settings.WEBHOOK_SECRET,
    )
    webhook_request_handler.register(app, path=settings.WEBHOOK_PATH)
    logger.info(f"✅ Telegram webhook registered at {settings.WEBHOOK_PATH}")

    # Step 4: Setup aiogram application lifecycle
    setup_application(app, dp, bot=bot)

    # Step 5: Register AG-UI WebSocket handler (BEFORE catch-all proxy)
    if ag_ui_handlers is not None and isinstance(ag_ui_handlers, dict):
        websocket_proxy = ag_ui_handlers.get("websocket_proxy")
        http_proxy = ag_ui_handlers.get("http_proxy")
        
        if websocket_proxy:
            app.router.add_route("GET", "/ws/{path_info:.*}", websocket_proxy, name="ag_ui_websocket_proxy")
            logger.info("✅ AG-UI WebSocket proxy registered at /ws/*")
        
        # Step 6: Register AG-UI catch-all HTTP proxy LAST (after all specific routes)
        if http_proxy:
            app.router.add_route("*", "/{path_info:.*}", http_proxy, name="ag_ui_proxy")
            logger.info("✅ AG-UI catch-all HTTP proxy registered (handles all non-webhook/metrics/ws routes)")
            logger.info("📌 Route priority: /metrics → /webhook → /ws/* → /* (AG-UI)")
    elif ag_ui_handlers is not None:
        # Backward compatibility: if old format (single handler) is returned
        app.router.add_route("*", "/{path_info:.*}", ag_ui_handlers, name="ag_ui_proxy")
        logger.info("✅ AG-UI catch-all proxy registered (legacy mode)")
        logger.info("📌 Route priority: /metrics → /webhook → /* (AG-UI)")

    # Log all registered routes for verification
    logger.info("")
    logger.info("🗺️  Registered routes on webhook server:")
    for route in app.router.routes():
        route_info = route.resource
        if hasattr(route_info, 'canonical'):
            logger.info(f"   {route.method:6s} {route_info.canonical}")
        else:
            logger.info(f"   {route.method:6s} {route_info}")
    logger.info("")
    
    # Use AppRunner instead of run_app to avoid nested event loop
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=settings.WEBHOOK_HOST, port=settings.WEBHOOK_PORT)
    await site.start()
    
    logger.info(f"🌐 Webhook server started on {settings.WEBHOOK_HOST}:{settings.WEBHOOK_PORT}")
    
    # Keep the server running indefinitely
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Received exit signal")
    finally:
        await runner.cleanup()


async def main() -> None:
    """Main entry point."""
    # Register startup/shutdown handlers
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    
    if settings.USE_WEBHOOK:
        # setup_application() will automatically call registered startup/shutdown handlers
        await setup_webhook()
    else:
        # Delete webhook and start polling (Phase 1)
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("📡 Using polling mode")
        
        if not await acquire_polling_lock():
            logger.error("Exiting because polling lock could not be acquired.")
            return
        
        try:
            await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
        except TelegramConflictError as conflict_error:
            logger.error(f"❌ Telegram conflict while polling: {conflict_error}")
            logger.error("   Another process may still be polling. Exiting.")
        finally:
            await release_polling_lock()


if __name__ == "__main__":
    asyncio.run(main())
