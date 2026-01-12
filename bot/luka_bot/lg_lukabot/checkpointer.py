"""Redis checkpointer for LangGraph state persistence.

Uses the existing Redis connection from luka_bot to persist agent state
across conversations and bot restarts.
"""

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.base import CheckpointTuple
from langchain_core.runnables import RunnableConfig
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

_checkpointer_instance = None
_executor = None


async def create_redis_checkpointer():
    """
    Create or return existing Redis checkpointer instance.

    Uses Redis connection settings from luka_bot configuration.
    Configures TTL for automatic checkpoint expiration (7 days default).

    Returns:
        RedisSaver instance for LangGraph state persistence

    Raises:
        RuntimeError: If Redis configuration is not available or connection fails
    """
    global _checkpointer_instance

    if _checkpointer_instance is not None:
        return _checkpointer_instance

    try:
        from luka_bot.core.config import settings
        import redis

        # Build Redis connection URL from settings
        redis_host = settings.REDIS_HOST
        redis_port = settings.REDIS_PORT
        redis_db = getattr(settings, "REDIS_DB", 0)
        redis_password = getattr(settings, "REDIS_PASSWORD", None)

        if redis_password:
            redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
        else:
            redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

        logger.info(f"🔴 Creating Redis checkpointer: {redis_host}:{redis_port}/{redis_db}")

        # Test Redis connection before creating checkpointer (using sync client)
        try:
            test_client = redis.from_url(redis_url, decode_responses=True)
            test_client.ping()
            test_client.close()
            logger.debug("✅ Redis connection test successful")
        except Exception as conn_error:
            logger.error(f"❌ Redis connection test failed: {conn_error}")
            raise RuntimeError(f"Redis connection failed: {conn_error}") from conn_error

        # TTL: 7 days (matching current thread_history expiry)
        # 7 days * 24 hours * 60 minutes * 60 seconds = 604800 seconds
        checkpoint_ttl_seconds = 7 * 24 * 60 * 60
        
        # RedisSaver expects ttl as a dict with "default_ttl" key, not an integer
        ttl_config = {"default_ttl": checkpoint_ttl_seconds}

        # Create RedisSaver with URL string (using sync Redis client)
        # RedisSaver requires RedisJSON module (JSON.SET command)
        # If RedisJSON is not available, we'll fall back to MemorySaver
        try:
            # Test if RedisJSON is available
            test_redis = redis.from_url(redis_url)
            try:
                test_redis.execute_command("JSON.GET", "test_key")
            except redis.exceptions.ResponseError as e:
                if "unknown command" in str(e).lower() or "json.get" in str(e).lower():
                    logger.warning("⚠️ RedisJSON module not available - falling back to MemorySaver")
                    logger.warning("⚠️ Note: MemorySaver doesn't persist across restarts")
                    logger.info("💡 To enable Redis persistence, install Redis Stack or RedisJSON module:")
                    logger.info("   • Docker: docker run -d -p 6379:6379 redis/redis-stack-server:latest")
                    logger.info("   • Homebrew: brew install redis-stack")
                    test_redis.close()
                    from langgraph.checkpoint.memory import MemorySaver
                    _checkpointer_instance = MemorySaver()
                    logger.info("✅ Using MemorySaver (in-memory only)")
                    return _checkpointer_instance
            except redis.exceptions.ResponseError:
                # Key doesn't exist, but command is recognized - RedisJSON is available
                pass
            finally:
                test_redis.close()
            
            # RedisJSON is available, proceed with RedisSaver
            redis_saver = RedisSaver(redis_url, ttl=ttl_config)
        except TypeError:
            # If RedisSaver doesn't accept ttl parameter, use URL only
            logger.warning("⚠️ RedisSaver doesn't support ttl parameter, using default")
            redis_saver = RedisSaver(redis_url)
        
        # Check if async methods are implemented
        import inspect
        try:
            source = inspect.getsource(redis_saver.aget_tuple)
            # Check if aget_tuple is implemented (not just raises NotImplementedError)
            lines = source.split('\n')
            code_started = False
            code_lines = []
            in_docstring = False
            
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                if '"""' in stripped:
                    in_docstring = not in_docstring
                    continue
                if in_docstring:
                    continue
                if stripped.startswith('async def') or stripped.startswith('def'):
                    continue
                if not in_docstring and not code_started:
                    code_started = True
                if code_started and stripped:
                    code_lines.append(stripped)
            
            executable_code = [l for l in code_lines if l and not l.startswith('#')]
            if len(executable_code) == 1 and executable_code[0] == "raise NotImplementedError":
                # Async not implemented - wrap sync methods with async wrapper
                logger.info("⚠️ RedisSaver.aget_tuple not implemented - creating async wrapper around sync methods")
                _checkpointer_instance = AsyncRedisSaverWrapper(redis_saver)
                logger.info(f"✅ Redis checkpointer created with async wrapper (TTL: {checkpoint_ttl_seconds}s / 7 days)")
                return _checkpointer_instance
        except Exception as check_error:
            logger.warning(f"⚠️ Could not verify RedisSaver async implementation: {check_error}")
            # Assume async not implemented and create wrapper
            logger.info("⚠️ Creating async wrapper around RedisSaver (assuming async not implemented)")
            _checkpointer_instance = AsyncRedisSaverWrapper(redis_saver)
            logger.info(f"✅ Redis checkpointer created with async wrapper (TTL: {checkpoint_ttl_seconds}s / 7 days)")
            return _checkpointer_instance

        # If async is implemented, use RedisSaver directly
        _checkpointer_instance = redis_saver
        logger.info(f"✅ Redis checkpointer created successfully (TTL: {checkpoint_ttl_seconds}s / 7 days)")
        return _checkpointer_instance

    except Exception as e:
        logger.error(f"❌ Failed to create Redis checkpointer: {e}")
        raise RuntimeError(f"Failed to create Redis checkpointer: {e}") from e


def get_checkpointer():
    """
    Get existing checkpointer instance without creating a new one.

    Returns:
        Existing RedisSaver instance or None if not created yet
    """
    return _checkpointer_instance


class AsyncRedisSaverWrapper:
    """
    Async wrapper around sync RedisSaver.
    
    This wrapper allows using RedisSaver (which only implements sync methods)
    with async graph operations by running sync methods in a thread pool.
    """
    
    def __init__(self, redis_saver: RedisSaver):
        """Initialize wrapper with sync RedisSaver instance."""
        self._saver = redis_saver
        global _executor
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="redis_checkpoint")
        self._executor = _executor
    
    def __getattr__(self, name):
        """Delegate all other attributes to wrapped RedisSaver."""
        return getattr(self._saver, name)
    
    async def aget(self, config: RunnableConfig):
        """Async wrapper for sync get method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._saver.get, config)
    
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Async wrapper for sync get_tuple method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._saver.get_tuple, config)
    
    async def aput(self, config: RunnableConfig, checkpoint, metadata, new_versions):
        """Async wrapper for sync put method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            self._saver.put, 
            config, checkpoint, metadata, new_versions
        )
    
    async def aput_writes(self, config: RunnableConfig, writes, task_id):
        """Async wrapper for sync put_writes method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._saver.put_writes,
            config, writes, task_id
        )
    
    async def alist(self, config: RunnableConfig, filter=None, before=None, limit=None):
        """Async wrapper for sync list method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._saver.list,
            config, filter, before, limit
        )
    
    async def adelete_thread(self, config: RunnableConfig):
        """Async wrapper for sync delete_thread method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._saver.delete_thread,
            config
        )

