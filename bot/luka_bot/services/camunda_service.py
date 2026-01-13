"""
Camunda Engine integration service for luka_bot.
Manages Camunda connections, process instances, and tasks.
"""
import asyncio
import hashlib
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID
from loguru import logger

from camunda_client.clients.engine.client import CamundaEngineClient
from camunda_client.clients.dto import AuthData
from camunda_client.clients.engine.schemas.response import (
    TaskSchema,
    ProcessInstanceSchema,
    ProcessVariablesSchema
)
from camunda_client.clients.engine.schemas import GetTasksFilterSchema
from camunda_client.exceptions import CamundaClientError
from luka_bot.core.config import settings
import httpx


@dataclass
class CamundaUserMapping:
    """Maps user (Flow API UUID) to Camunda credentials"""
    user_id: str  # Flow API UUID
    camunda_user_id: str
    camunda_password: str


class CamundaService:
    """Singleton service for Camunda operations"""
    
    _instance: Optional['CamundaService'] = None
    
    def __init__(self):
        self._client: Optional[CamundaEngineClient] = None
        self._transport = httpx.AsyncHTTPTransport(
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self._user_mappings: Dict[str, CamundaUserMapping] = {}
        # Cache clients per user to reuse sessions and prevent resource leaks
        # For web users, use composite key: (telegram_user_id, webapp_user_id, platform)
        # For telegram users, use just telegram_user_id
        self._clients: Dict[tuple, CamundaEngineClient] = {}
        
    @classmethod
    def get_instance(cls) -> 'CamundaService':
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
            logger.info("✅ CamundaService singleton created")
        return cls._instance
    
    def _get_cache_key(
        self,
        user_id: str,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> tuple:
        """
        Generate cache key for client.
        
        For web users, include webapp_user_id in the key to handle cases where
        the same user_id maps to different Camunda credentials.
        For telegram users, just use user_id (Flow API UUID).
        """
        if platform == "web" and webapp_user_id:
            return (user_id, webapp_user_id, platform)
        else:
            return (user_id,)
    
    async def _get_client(
        self, 
        user_id: str,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> CamundaEngineClient:
        """
        Get or create cached Camunda client for user.
        
        Args:
            user_id: Flow API UUID (string) - primary identifier
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform type ("web" or "telegram")
        
        Uses composite cache keys for web users to handle different webapp_user_id mappings.
        Only creates new client if cache key changes or client is closed.
        """
        cache_key = self._get_cache_key(user_id, webapp_user_id, platform)
        
        logger.debug(
            f"🔍 _get_client called: user_id={user_id}, "
            f"webapp_user_id={webapp_user_id}, platform={platform}, "
            f"cache_key={cache_key}, "
            f"cached_client_exists={cache_key in self._clients}, "
            f"mapping_exists={user_id in self._user_mappings}"
        )
        
        # Check if we have a cached client with this key
        cached_client = self._clients.get(cache_key)
        if cached_client:
            # Check if client is still valid (not closed)
            is_closed = False
            if hasattr(cached_client, '_http_client') and cached_client._http_client:
                if hasattr(cached_client._http_client, 'is_closed') and cached_client._http_client.is_closed:
                    is_closed = True
            elif hasattr(cached_client, '_session') and cached_client._session:
                if hasattr(cached_client._session, 'closed') and cached_client._session.closed:
                    is_closed = True
            
            if not is_closed:
                logger.debug(f"🔧 Using cached Camunda client for cache_key={cache_key}")
                return cached_client
            else:
                # Client is closed, remove from cache and create new one
                logger.debug(f"🔄 Cached client for {cache_key} is closed, removing from cache")
                del self._clients[cache_key]
        
        # Get or create user mapping
        mapping = await self._get_or_create_user_mapping(
            user_id, 
            webapp_user_id=webapp_user_id,
            platform=platform
        )
        
        logger.debug(
            f"🔍 After _get_or_create_user_mapping: "
            f"mapping_exists={user_id in self._user_mappings}, "
            f"mapping_camunda_user_id={mapping.camunda_user_id if mapping else None}, "
            f"mapping_keys={list(self._user_mappings.keys())}"
        )
        
        if not mapping:
            raise ValueError(f"Failed to create user mapping for {user_id}")
        
        # Validate credentials before creating client
        if not mapping.camunda_user_id or not mapping.camunda_password:
            raise ValueError(
                f"Missing Camunda credentials for user {user_id}: "
                f"camunda_user_id={mapping.camunda_user_id}, "
                f"camunda_password={'***' if mapping.camunda_password else None}"
            )
        
        logger.info(
            f"🔧 Creating Camunda client for user {user_id} "
            f"(platform={platform}, webapp_user_id={webapp_user_id}) "
            f"with camunda_user_id={mapping.camunda_user_id} "
            f"(password length: {len(mapping.camunda_password) if mapping.camunda_password else 0})"
        )
        
        auth_data = AuthData(
            username=mapping.camunda_user_id,
            password=mapping.camunda_password
        )
        
        # Create and cache new client
        client = CamundaEngineClient(
            base_url=settings.ENGINE_URL,
            auth_data=auth_data,
            transport=self._transport
        )
        
        # Note: We don't verify credentials here because:
        # 1. The /user/{id}/profile endpoint doesn't exist in Camunda Engine REST API
        # 2. Actual operations will fail with proper errors (401) if credentials are invalid
        # 3. This avoids unnecessary API calls and log noise
        
        # Cache client using composite key
        self._clients[cache_key] = client
        
        logger.debug(f"✅ Created and cached Camunda client for cache_key={cache_key}")
        return client
    
    async def close_all_clients(self):
        """Close all cached clients (call on shutdown)"""
        for cache_key, client in list(self._clients.items()):
            try:
                await client.close()
                logger.debug(f"🔒 Closed Camunda client for cache_key={cache_key}")
            except Exception as e:
                logger.warning(f"Failed to close client for cache_key={cache_key}: {e}")
        self._clients.clear()
    
    async def _get_or_create_user_mapping(
        self, 
        user_id: str,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> CamundaUserMapping:
        """
        Get or create Camunda user mapping.
        
        Args:
            user_id: Flow API UUID (string) - primary identifier
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform type ("web" or "telegram")
        
        Priority order:
        1. In-memory cache (fastest)
        2. Session cache (fast, from Flow API)
        3. Flow API lookup (for web users, lookup by webapp_user_id)
        """
        # Check in-memory cache first
        if user_id in self._user_mappings:
            return self._user_mappings[user_id]
        
        # Try session cache (from Flow API auth middleware)
        from luka_bot.services.user_session_cache import get_cached_user_info
        user_info = await get_cached_user_info(user_id)
        
        if user_info:
            camunda_user_id = user_info.get("camunda_user_id")
            camunda_key = user_info.get("camunda_key")
            
            if camunda_user_id and camunda_key:
                mapping = CamundaUserMapping(
                    user_id=user_id,
                    camunda_user_id=camunda_user_id,
                    camunda_password=camunda_key
                )
                self._user_mappings[user_id] = mapping
                
                logger.info(
                    f"🔐 Created Camunda mapping from session cache: "
                    f"User {user_id} → Camunda {camunda_user_id}"
                )
                return mapping
        
        # Last resort: Fetch directly from Flow API
        logger.info(f"🔍 Fetching user {user_id} from Flow API...")
        try:
            from flow_client.clients.flow.client import FlowClient
            
            async with FlowClient(
                base_url=settings.FLOW_API_URL,
                sys_key=settings.FLOW_API_SYS_KEY
            ) as flow_client:
                # Try multiple lookup methods in order of preference
                user_data = None
                
                # Method 1: Try by Flow API UUID (user_id is now the Flow API UUID)
                try:
                    user_data = await flow_client.get_user(user_id=user_id)
                    if user_data:
                        logger.info(f"✅ Found user in Flow API by user_id (UUID): {user_id}, has_camunda={bool(user_data.camunda_user_id)}")
                except Exception as e:
                    logger.debug(f"⚠️ Exception getting user by user_id {user_id}: {e}")
                    user_data = None
                
                # Method 2: For web users, try webapp_user_id if user_id lookup failed
                if not user_data and platform == "web" and webapp_user_id:
                    try:
                        # Ensure webapp_user_id is a valid UUID string format
                        from uuid import UUID as UUIDType
                        try:
                            # Validate UUID format
                            uuid_obj = UUIDType(webapp_user_id)
                            user_data = await flow_client.get_user(webapp_id=str(uuid_obj))
                            if user_data:
                                logger.info(f"✅ Found user in Flow API by webapp_user_id: {webapp_user_id}, has_camunda={bool(user_data.camunda_user_id)}")
                        except ValueError as uuid_error:
                            logger.warning(f"⚠️ webapp_user_id {webapp_user_id} is not a valid UUID format: {uuid_error}")
                    except Exception as e:
                        error_str = str(e)
                        # 422 means invalid UUID format or validation error - log but don't treat as fatal
                        if "422" in error_str or "Unprocessable" in error_str:
                            logger.debug(f"⚠️ webapp_user_id {webapp_user_id} rejected by Flow API (422 - validation error)")
                        else:
                            logger.warning(f"⚠️ Exception getting user by webapp_user_id {webapp_user_id}: {e}")
                
                # If user found but no Camunda credentials, this indicates a Flow API bug
                # Flow API's create_user endpoint should automatically create Camunda credentials
                # If they're missing, it means Flow API's error handling failed to delete the user
                if user_data and not (user_data.camunda_user_id and user_data.camunda_key):
                    logger.error(
                        f"❌ User {user_data.id} exists in Flow API but missing Camunda credentials. "
                        f"camunda_user_id={user_data.camunda_user_id}, camunda_key={'***' if user_data.camunda_key else None}. "
                        "This indicates Flow API's create_user endpoint failed to create Camunda user and didn't delete the user as expected."
                    )
                    raise ValueError(
                        f"User {user_data.id} exists in Flow API but has no Camunda credentials. "
                        "Flow API's create_user endpoint should automatically create Camunda credentials. "
                        "If Camunda creation fails, Flow API should delete the user. "
                        "Please check Flow API logs and ensure create_camunda_user is working correctly."
                    )
                
                # If user found with credentials, use them
                if user_data and user_data.camunda_user_id and user_data.camunda_key:
                    mapping = CamundaUserMapping(
                        user_id=user_id,
                        camunda_user_id=user_data.camunda_user_id,
                        camunda_password=user_data.camunda_key
                    )
                    self._user_mappings[user_id] = mapping
                    
                    logger.info(
                        f"🔐 Created Camunda mapping from Flow API: "
                        f"User {user_id} → Camunda {user_data.camunda_user_id}"
                    )
                    logger.info(f"💾 Cached credentials in memory for future use")
                    return mapping
                
                # User not found - create them if this is a web user
                # Only create if user_data is None (user truly doesn't exist)
                if platform == "web" and not user_data:
                    logger.info(f"🆕 Creating new Flow API user for web user {user_id}...")
                    
                    # Generate stable webapp_user_id if not provided
                    if not webapp_user_id:
                        # Use user_id as webapp_user_id if it's a valid UUID
                        # Otherwise generate from user_id
                        try:
                            from uuid import UUID as UUIDType
                            UUIDType(user_id)  # Validate it's a UUID
                            webapp_user_id = user_id
                        except (ValueError, TypeError):
                            # Generate deterministic UUID4 from user_id
                            seed = f"web_user_{user_id}".encode()
                            hash_bytes = hashlib.md5(seed).digest()
                            
                            # Create UUID from bytes and set version/variant bits for UUID4
                            uuid_bytes = bytearray(hash_bytes[:16])
                            uuid_bytes[6] = (uuid_bytes[6] & 0x0f) | 0x40  # Set version 4
                            uuid_bytes[8] = (uuid_bytes[8] & 0x3f) | 0x80  # Set variant 10
                            webapp_user_id = str(UUID(bytes=bytes(uuid_bytes)))
                    
                    try:
                        # Create user in Flow API (will auto-create Camunda credentials)
                        logger.info(f"🆕 Creating Flow API user with webapp_user_id={webapp_user_id}")
                        created_user = await flow_client.add_user(
                            username=f"web_user_{user_id[:8]}",
                            webapp_user_id=webapp_user_id,  # Pass string directly
                            telegram_user_id=None,  # Web users don't have telegram_user_id
                            language_code="en"
                        )
                        
                        if created_user and created_user.camunda_user_id and created_user.camunda_key:
                            mapping = CamundaUserMapping(
                                user_id=user_id,
                                camunda_user_id=created_user.camunda_user_id,
                                camunda_password=created_user.camunda_key
                            )
                            self._user_mappings[user_id] = mapping

                            logger.info(
                                f"✅ Created new Flow API user with Camunda credentials: "
                                f"User {user_id} → Camunda {created_user.camunda_user_id}"
                            )
                            return mapping
                    except Exception as create_error:
                        error_str = str(create_error)
                        # If user already exists (409 Conflict), fetch them and use their credentials
                        # Flow API should have created Camunda credentials when the user was first created
                        if "409" in error_str or "Conflict" in error_str:
                            logger.info(f"🔄 User already exists in Flow API (409), fetching existing user by webapp_user_id={webapp_user_id}")
                            try:
                                # Try to get user by webapp_user_id, handling potential 422 errors
                                existing_user = None
                                if webapp_user_id:
                                    try:
                                        from uuid import UUID as UUIDType
                                        uuid_obj = UUIDType(webapp_user_id)
                                        existing_user = await flow_client.get_user(webapp_id=str(uuid_obj))
                                    except (ValueError, Exception) as uuid_error:
                                        logger.debug(f"⚠️ Could not query by webapp_user_id after 409: {uuid_error}, trying user_id")
                                
                                # If webapp_user_id lookup failed, try user_id (Flow API UUID) as fallback
                                if not existing_user:
                                    try:
                                        existing_user = await flow_client.get_user(user_id=user_id)
                                    except Exception:
                                        pass
                                
                                # Flow API should have created Camunda credentials when user was created
                                if existing_user and existing_user.camunda_user_id and existing_user.camunda_key:
                                    mapping = CamundaUserMapping(
                                        user_id=user_id,
                                        camunda_user_id=existing_user.camunda_user_id,
                                        camunda_password=existing_user.camunda_key
                                    )
                                    self._user_mappings[user_id] = mapping
                                    
                                    logger.info(
                                        f"🔐 Created Camunda mapping from existing Flow API user: "
                                        f"User {user_id} → Camunda {existing_user.camunda_user_id}"
                                    )
                                    return mapping
                                elif existing_user:
                                    # User exists but missing Camunda credentials - this is a Flow API bug
                                    logger.error(
                                        f"❌ Existing user {existing_user.id} found but missing Camunda credentials. "
                                        f"camunda_user_id={existing_user.camunda_user_id}. "
                                        "Flow API should have created Camunda credentials when user was created."
                                    )
                                    raise ValueError(
                                        f"User {existing_user.id} exists in Flow API but has no Camunda credentials. "
                                        "Flow API's create_user endpoint should automatically create Camunda credentials. "
                                        "Please check Flow API logs and ensure create_camunda_user is working correctly."
                                    )
                            except ValueError:
                                # Re-raise ValueError (missing credentials)
                                raise
                            except Exception as fetch_error:
                                logger.error(f"❌ Failed to fetch existing user after 409: {fetch_error}")
                        logger.error(f"❌ Failed to create Flow API user: {create_error}", exc_info=True)
                        # Fall through to error below
                        
        except Exception as e:
            logger.warning(f"⚠️  Could not fetch/create from Flow API: {e}", exc_info=True)
        
        # No credentials found anywhere
        logger.error(f"❌ No Camunda credentials for user {user_id}")
        raise ValueError(
            f"User {user_id} has no Camunda credentials. "
            "Credentials should be provided via Flow API authentication. "
            "Please ensure:\n"
            "1. User is registered in Flow API\n"
            "2. Flow API has assigned Camunda credentials\n"
            "3. FlowAuthMiddleware is registered and functioning"
        )
    
    async def start_process(
        self,
        user_id: str,
        process_key: str,
        business_key: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> ProcessInstanceSchema:
        """
        Start a BPMN process for user.
        
        Args:
            user_id: Flow API UUID (string) - primary identifier
            process_key: Process definition key
            business_key: Optional business key
            variables: Process variables
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform type ("web" or "telegram")
        """
        client = await self._get_client(
            user_id,
            webapp_user_id=webapp_user_id,
            platform=platform
        )
        
        # Convert variables to Camunda format
        logger.debug(f"📤 Raw variables for process {process_key}: {variables}")
        camunda_vars = self._format_variables(variables or {})
        logger.debug(f"📤 Formatted Camunda variables: {camunda_vars}")
        
        try:
            process_instance = await client.start_process(
                process_key=process_key,
                business_key=business_key,
                variables=camunda_vars
            )
            logger.info(
                f"🚀 Started process {process_key} for user {user_id}: {process_instance.id} "
                f"with {len(variables or {})} variables (business_key={business_key})"
            )
            return process_instance
        except Exception as e:
            logger.error(f"❌ Failed to start process {process_key}: {e}")
            raise
    
    async def correlate_message(
        self,
        user_id: str,
        message_data: Dict[str, Any],
        message_type: str,
        kb_doc_id: str
    ) -> bool:
        """
        Send message to Camunda via correlation with ES document ID.
        
        Args:
            user_id: Flow API UUID (string) - primary identifier
            message_data: Message data with thread context
            message_type: Message type (GROUP_MESSAGE, DM_MESSAGE, ASSISTANT_MESSAGE)
            kb_doc_id: Document ID for ES reference
            
        Returns:
            True if successful
        """
        try:
            client = await self._get_client(user_id)
            camunda_variables = self._format_message_variables(message_data, message_type, kb_doc_id)
            
            from camunda_client.clients.engine.schemas import SendCorrelationMessageSchema
            schema = SendCorrelationMessageSchema(
                message_name=message_type,
                business_key=self._build_business_key(message_data),
                process_variables=camunda_variables
            )
            
            await client.send_correlation_message(schema)
            logger.info(f"📤 Correlated {message_type} message {kb_doc_id} to Camunda")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to correlate {message_type} message {kb_doc_id}: {e}")
            return False
    
    async def get_user_tasks(
        self,
        user_id: str,
        process_definition_key: Optional[str] = None,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> List[TaskSchema]:
        """
        Get tasks assigned to user, optionally filtered by process definition.
        
        When process_definition_key="chatbot_start", includes tasks from direct
        sub-processes where the root is the user's chatbot_start instance.

        Args:
            user_id: Flow API UUID (string) - primary identifier
            process_definition_key: Optional process key (e.g., "chatbot_start")
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform identifier ("web" or "telegram")

        Returns:
            List of tasks matching the criteria
        """
        client = await self._get_client(user_id, webapp_user_id=webapp_user_id, platform=platform)
        
        # Safely get mapping - _get_client should have created it, but check to avoid KeyError
        mapping = self._user_mappings.get(user_id)
        if not mapping:
            logger.error(
                f"❌ No user mapping found for user {user_id} after _get_client. "
                f"This should not happen - _get_client should create the mapping. "
                f"Available mappings: {list(self._user_mappings.keys())}"
            )
            # Try to get mapping again by calling _get_or_create_user_mapping directly
            mapping = await self._get_or_create_user_mapping(
                user_id,
                webapp_user_id=webapp_user_id,
                platform=platform
            )
            if not mapping:
                raise ValueError(f"Failed to create user mapping for {user_id}")

        # Special handling for chatbot_start: include sub-process tasks
        if process_definition_key == "chatbot_start":
            process_instances = await self._get_chatbot_start_process_instances(
                user_id, client
            )
            
            if not process_instances:
                logger.debug(f"📋 No chatbot_start process found for user {user_id}")
                return []
            
            # Collect tasks from all process instances (root + sub-processes)
            all_tasks = []
            for process_instance in process_instances:
                filter_schema = GetTasksFilterSchema(
                    assignee=mapping.camunda_user_id,
                    process_instance_id=process_instance.id
                )
                tasks = await client.get_tasks(schema=filter_schema)
                all_tasks.extend(tasks)
            
            logger.debug(
                f"📋 Retrieved {len(all_tasks)} tasks for user {user_id} "
                f"from chatbot_start and {len(process_instances) - 1} sub-processes"
            )
            return list(all_tasks)
        
        # Standard filtering for other processes or no filter
        filter_schema = GetTasksFilterSchema(
            assignee=mapping.camunda_user_id,
            process_definition_key=process_definition_key
        )
        tasks = await client.get_tasks(schema=filter_schema)

        if process_definition_key:
            logger.debug(
                f"📋 Retrieved {len(tasks)} tasks for user {user_id} "
                f"from process {process_definition_key}"
            )
        else:
            logger.debug(f"📋 Retrieved {len(tasks)} tasks for user {user_id}")

        return list(tasks)
    
    async def _get_chatbot_start_process_instances(
        self, 
        user_id: str, 
        client
    ) -> List[ProcessInstanceSchema]:
        """
        Get chatbot_start root process and its direct sub-processes.
        
        Args:
            user_id: Flow API UUID (string) - primary identifier
            client: Camunda client
        
        Returns list containing: [root_instance, sub_instance_1, sub_instance_2, ...]
        """
        from camunda_client.clients.engine.schemas.query import ProcessInstanceQuerySchema
        
        # Business key format for chatbot_start: {user_id}-chatbot-start
        # Note: user_id is now Flow API UUID (string)
        business_key = f"{user_id}-chatbot-start"
        
        # Get root chatbot_start process
        root_query = ProcessInstanceQuerySchema(
            process_definition_key="chatbot_start",
            business_key=business_key,
            active=True
        )
        root_instances = await client.get_process_instances(root_query)
        
        if not root_instances:
            return []
        
        root_instance = root_instances[0]
        process_instances = [root_instance]
        
        # Get direct sub-processes
        sub_query = ProcessInstanceQuerySchema(
            super_process_instance=str(root_instance.id),
            active=True
        )
        sub_instances = await client.get_process_instances(sub_query)
        process_instances.extend(sub_instances)
        
        logger.debug(
            f"Found {len(process_instances)} process instances: "
            f"1 root + {len(sub_instances)} sub-processes"
        )
        
        return process_instances
    
    async def get_process_instance_by_business_key(
        self,
        user_id: str,
        business_key: str,
        active_only: bool = True
    ) -> Optional[ProcessInstanceSchema]:
        """
        Get process instance by business key.
        
        Args:
            user_id: Flow API UUID (string) - primary identifier (for authentication)
            business_key: Business key to search for (e.g., "import_-1001902150742_922705")
            active_only: Only return active (non-suspended) instances
        
        Returns:
            ProcessInstanceSchema if found, None otherwise
        """
        from camunda_client.clients.engine.schemas.query import ProcessInstanceQuerySchema
        
        client = await self._get_client(user_id)
        
        query = ProcessInstanceQuerySchema(
            business_key=business_key,
            active=active_only
        )
        
        try:
            instances = await client.get_process_instances(params=query)
            
            if instances:
                instance = instances[0]
                logger.info(
                    f"📍 Found existing process instance: {instance.id} "
                    f"(business_key={business_key})"
                )
                return instance
            else:
                logger.debug(f"No active process found for business_key={business_key}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying process instances: {e}")
            return None
    
    async def delete_process_instance(
        self,
        user_id: str,
        process_instance_id: str,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> bool:
        """
        Delete a process instance.
        
        Args:
            user_id: Flow API UUID (string) - primary identifier
            process_instance_id: Process instance ID to delete
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform identifier ("web" or "telegram")
        
        Returns:
            True if deletion successful, False otherwise
        """
        client = await self._get_client(user_id, webapp_user_id=webapp_user_id, platform=platform)
        try:
            await client.delete_process(process_instance_id)
            logger.info(f"✅ Deleted process instance {process_instance_id} for user {user_id}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"⚠️ Process instance {process_instance_id} not found")
                return False
            elif e.response.status_code == 403:
                logger.warning(f"⚠️ Permission denied to delete process instance {process_instance_id}")
                return False
            else:
                logger.error(f"❌ Error deleting process instance {process_instance_id}: {e.response.status_code}")
                raise
        except Exception as e:
            logger.error(f"❌ Unexpected error deleting process instance {process_instance_id}: {e}", exc_info=True)
            raise
    
    async def get_task(
        self, 
        user_id: str, 
        task_id: str,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> Optional[TaskSchema]:
        """
        Get specific task by ID and verify it belongs to the user.
        
        Args:
            user_id: Flow API UUID (string) - primary identifier
            task_id: Task ID to fetch
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform identifier ("web" or "telegram")
        
        This method ensures the task is accessible to the user by:
        1. First trying to fetch the task directly
        2. If found, verifying it's assigned to the user or accessible via candidate groups
        
        Note: If a task was shown in the user's menu, it should be accessible here.
        This handles race conditions where tasks might be newly created.
        """
        client = await self._get_client(user_id, webapp_user_id=webapp_user_id, platform=platform)
        mapping = self._user_mappings.get(user_id)
        
        if not mapping:
            logger.warning(f"No Camunda mapping for user {user_id}")
            return None
        
        try:
            # Try to get task directly
            task = await client.get_task(task_id)
            
            if not task:
                # Task not found - might have been completed/deleted
                logger.warning(
                    f"Task {task_id} not found for user {user_id}. "
                    f"This might mean the task was completed, deleted, or never existed."
                )
                return None
            
            # Verify task belongs to user by checking assignee
            # If task is assigned, it must be assigned to this user
            if task.assignee:
                if task.assignee != mapping.camunda_user_id:
                    logger.warning(
                        f"Task {task_id} is assigned to {task.assignee}, "
                        f"but user {user_id} (camunda_user_id: {mapping.camunda_user_id}) tried to access it. "
                        f"Checking if task is accessible via candidate groups..."
                    )
                    # Fallback: Check if task is in user's task list (might be accessible via candidate groups)
                    # This handles cases where task is in candidate groups
                    user_tasks = await self.get_user_tasks(user_id)
                    task_ids = [str(t.id) for t in user_tasks]
                    if task_id not in task_ids:
                        logger.error(
                            f"Task {task_id} not accessible to user {user_id}. "
                            f"Assignee: {task.assignee}, User's camunda_user_id: {mapping.camunda_user_id}. "
                            f"This is a security issue - task should not be accessible to this user."
                        )
                        return None
                    # Task is accessible to user (via candidate groups or other means)
                    logger.debug(f"Task {task_id} accessible to user {user_id} (not directly assigned)")
            else:
                # Task has no assignee - this is fine, it might be a candidate task
                # or newly created. Since it was found via direct lookup, allow it.
                logger.debug(
                    f"Task {task_id} has no assignee - allowing access for user {user_id}. "
                    f"This might be a candidate task or newly created task."
                )

            return task

        except Exception as e:
            logger.error(f"Error fetching task {task_id} for user {user_id}: {e}")
            return None
    
    async def get_task_variables(
        self,
        user_id: str,
        task_id: str,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> List[Dict[str, Any]]:
        """
        Get form variables for task.
        
        Args:
            user_id: Flow API UUID (string) - primary identifier
            task_id: Task ID
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform identifier ("web" or "telegram")
        
        Returns:
            List of variable dicts with keys: name, value, type, writable, valueInfo
            Returns empty list if task has no form (404 error).
        """
        client = await self._get_client(user_id, webapp_user_id=webapp_user_id, platform=platform)
        
        try:
            # get_task_form_variables returns dict[str, VariableValueSchema]
            variables_dict = await client.get_task_form_variables(task_id)
        except CamundaClientError as e:
            # Check if it's a "no form" error (404)
            if e.status_code == 404:
                # Check error message to confirm it's a "no form" case
                error_str = str(e)
                if "No matching rendered form" in error_str or "rendered form" in error_str.lower():
                    logger.debug(f"📝 Task {task_id} has no form - returning empty variables")
                    return []
            # Re-raise other errors
            raise
        except Exception as e:
            # Check for 404 in string representation as fallback
            error_str = str(e)
            if "404" in error_str and "No matching rendered form" in error_str:
                logger.debug(f"📝 Task {task_id} has no form - returning empty variables")
                return []
            # Re-raise other errors
            raise
        
        # Convert dict to list format expected by TaskService
        # All variables from form are writable (they're input fields)
        variables_list = []
        for var_name, var_data in variables_dict.items():
            var_dict = {
                "name": var_name,
                "value": var_data.value,
                "type": var_data.type,
                "label": var_data.label or "",  # Include label from form parsing
                "writable": True,  # All form inputs are writable
                "valueInfo": var_data.value_info or {}
            }
            variables_list.append(var_dict)
        
        logger.debug(f"📝 Retrieved {len(variables_list)} variables for task {task_id}")
        return variables_list
    
    async def get_process_definition(
        self,
        user_id: str,
        process_key: str,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> Optional[Dict[str, Any]]:
        """
        Get process definition details (name, description, etc.).

        Args:
            user_id: Flow API UUID (string) - primary identifier
            process_key: Process definition key
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform identifier ("web" or "telegram")

        Returns:
            Dict with process definition details or None if not found
        """
        from camunda_client.clients.engine.schemas.query import ProcessDefinitionQuerySchema

        client = await self._get_client(user_id, webapp_user_id=webapp_user_id, platform=platform)
        try:
            # Query for process definition by key (latest version)
            query = ProcessDefinitionQuerySchema(
                key=process_key,
                latest_version=True
            )
            definitions = await client.get_process_definitions(query)
            
            if not definitions:
                raise ValueError(f"No process definition found for key: {process_key}")
            
            # Get the first (and should be only) result
            definition = definitions[0]
            
            # Extract name from definition
            raw_name = definition.name if hasattr(definition, 'name') else None
            
            # If name is the same as key or not set, format the key into a readable name
            if not raw_name or raw_name == process_key:
                # Transform snake_case or kebab-case to Title Case
                # "chatbot_group_import_history" → "Chatbot Group Import History"
                display_name = process_key.replace('_', ' ').replace('-', ' ').title()
            else:
                display_name = raw_name
            
            # Extract relevant fields
            result = {
                "id": definition.id if hasattr(definition, 'id') else None,
                "key": definition.key if hasattr(definition, 'key') else process_key,
                "name": display_name,
                "raw_name": raw_name,  # Keep original for debugging
                "description": definition.description if hasattr(definition, 'description') else None,
                "version": definition.version if hasattr(definition, 'version') else None,
            }
            
            logger.debug(f"📋 Process definition loaded: {display_name}")
            return result
            
        except Exception as e:
            logger.warning(f"Could not fetch process definition for {process_key}: {e}")
            # Fallback: format key into readable name
            display_name = process_key.replace('_', ' ').replace('-', ' ').title()
            return {
                "key": process_key,
                "name": display_name,
                "description": None,
                "version": None
            }
    
    async def get_process_definitions_batch(
        self,
        process_keys: List[str],
        user_id: str,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> List[Dict[str, Any]]:
        """
        Get process definition details for multiple process keys in batch.

        Args:
            process_keys: List of process definition keys
            user_id: Flow API UUID (string) - primary identifier
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform identifier ("web" or "telegram")

        Returns:
            List of process definition dicts with: key, name, description, version
            Missing processes are skipped (not included in result)
        """
        results = []

        for process_key in process_keys:
            try:
                definition = await self.get_process_definition(
                    user_id=user_id,
                    process_key=process_key,
                    webapp_user_id=webapp_user_id,
                    platform=platform
                )
                if definition:
                    results.append(definition)
            except Exception as e:
                logger.warning(f"Could not fetch process definition for {process_key}: {e}")
                # Skip this process, continue with others
                continue
        
        logger.info(f"📋 Fetched {len(results)}/{len(process_keys)} process definitions")
        return results
    
    async def get_start_form_key(
        self,
        user_id: str,
        process_key: str,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> Optional[Dict[str, Any]]:
        """
        Get start form key information for a process definition.

        Args:
            user_id: Flow API UUID (string) - primary identifier
            process_key: Process definition key
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform identifier ("web" or "telegram")

        Returns:
            Dict with keys: formKey, camundaFormRef, contextPath
            - formKey: string or None
              - None: no form (process can be started directly)
              - "embedded:app:path/to/form.html": embedded HTML form
              - other string: form key for deployed form
            - camundaFormRef: dict with binding, key, version (if deployed form)
            - contextPath: string (deployment context path)

            Returns None if error occurs
        """
        client = await self._get_client(user_id, webapp_user_id=webapp_user_id, platform=platform)
        try:
            form_key_info = await client.get_process_definition_start_form_key(process_key)
            logger.info(f"📋 Start form key for {process_key}: {form_key_info} (type: {type(form_key_info)})")
            return form_key_info
        except Exception as e:
            error_str = str(e)
            # Check if it's a 404 (no form) vs other error
            if "404" in error_str or "Not Found" in error_str:
                logger.debug(f"📝 Process {process_key} has no start form")
                return {"formKey": None, "camundaFormRef": None, "contextPath": None}
            else:
                logger.warning(f"⚠️ Error getting start form key for {process_key}: {e}")
                return None
    
    async def get_start_form_variables(
        self,
        user_id: str,
        process_key: str,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Get start form variables for process definition.

        Args:
            user_id: Flow API UUID (string) - primary identifier
            process_key: Process definition key
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform identifier ("web" or "telegram")

        Returns:
            Tuple of (variables_list, error_message)
            - If successful: (variables, None)
            - If no start form: ([], None)
            - If error: ([], error_message)

        Each variable dict contains: name, value, type, label, valueInfo
        """
        client = await self._get_client(
            user_id,
            webapp_user_id=webapp_user_id,
            platform=platform
        )
        try:
            # get_process_definition_start_form returns a dict[str, VariableValueSchema]
            # where VariableValueSchema has attributes: value, type, label, value_info
            # This works for both generated forms (from variables) and embedded forms (Camunda forms)
            variables_dict = await client.get_process_definition_start_form(process_key)
            
            # If the dict is empty, it might mean:
            # 1. No form exists
            # 2. Form exists but has no variables (embedded form with no inputs)
            if not variables_dict:
                logger.debug(f"📝 Process {process_key} has start form but no variables (might be embedded form)")
                # Return empty list with no error - the form exists but has no fields
                return ([], None)
            
            # Convert dict to list format, preserving labels
            # var_data is ALWAYS a VariableValueSchema object (not a dict)
            variables_list = []
            for var_name, var_data in variables_dict.items():
                var_dict = {
                    "name": var_name,
                    "value": var_data.value,
                    "type": var_data.type,
                    "label": var_data.label or '',
                    "valueInfo": var_data.value_info or {}
                }
                variables_list.append(var_dict)
            
            logger.info(f"📝 Retrieved {len(variables_list)} start form variables for {process_key}")
            return (variables_list, None)
        except Exception as e:
            error_str = str(e)
            
            # Extract response data if present (for better debugging)
            response_data = None
            if "RESPONSE DATA:" in error_str:
                try:
                    # Extract the response data part
                    response_part = error_str.split("RESPONSE DATA:")[1].strip()
                    # Try to parse as JSON for better formatting
                    import json
                    if response_part.startswith("b'") or response_part.startswith('b"'):
                        # It's a bytes string representation
                        response_part = response_part[2:-1]  # Remove b' and '
                    response_data = json.loads(response_part)
                except:
                    pass
            
            # Check if it's a real error (500) vs. just no form (404)
            if "500" in error_str or "Internal Server Error" in error_str:
                logger.error(
                    f"❌ Camunda 500 Error fetching start form for process '{process_key}':\n"
                    f"   Error: {e}\n"
                    f"   Response Data: {json.dumps(response_data, indent=2) if response_data else 'N/A'}"
                )
                
                # Create user-friendly error message
                if response_data and isinstance(response_data, dict):
                    error_detail = response_data.get('error', 'Internal Server Error')
                    error_path = response_data.get('path', 'Unknown')
                    user_msg = f"Camunda server error: {error_detail} (path: {error_path})"
                else:
                    user_msg = f"Camunda server error: {error_str[:100]}"
                
                return ([], user_msg)
            else:
                logger.debug(f"No start form for process {process_key}: {e}")
                return ([], None)
    
    async def complete_task(
        self,
        user_id: str,
        task_id: str,
        variables: Optional[Dict[str, Any]] = None,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> bool:
        """
        Complete a task with variables.

        Args:
            user_id: Flow API UUID (string) - primary identifier
            task_id: Task ID to complete
            variables: Optional dict of variables to submit
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform identifier ("web" or "telegram")

        Returns:
            True if task was completed successfully, False otherwise
        """
        client = await self._get_client(user_id, webapp_user_id=webapp_user_id, platform=platform)
        
        logger.debug(f"📤 Raw variables for task {task_id}: {variables}")
        camunda_vars = self._format_variables(variables or {})
        logger.debug(f"📤 Formatted Camunda variables: {camunda_vars}")
        
        # Wrap variables in Camunda's expected format
        # Per OpenAPI spec: CompleteTaskDto requires variables to be wrapped
        payload = {"variables": camunda_vars}
        logger.debug(f"📤 Complete task payload: {payload}")
        
        await client.complete_task(task_id, variables=payload)
        logger.info(f"✅ Completed task {task_id} for user {user_id} with {len(variables or {})} variables")
        return True
    
    async def get_completed_task_details(
        self,
        user_id: str,
        task_id: str,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram"
    ) -> Optional[Dict[str, Any]]:
        """
        Get completed task details from Camunda History API.

        Args:
            user_id: Flow API UUID (string) - primary identifier
            task_id: Task ID to get details for
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform identifier ("web" or "telegram")

        Returns:
            Dict with task details including name, description, completion time, and variables, or None if not found
        """
        from uuid import UUID
        from camunda_client.clients.engine.schemas.body import GetHistoryTasksFilterSchema

        client = await self._get_client(user_id, webapp_user_id=webapp_user_id, platform=platform)
        
        try:
            # Query history API for completed task
            filter_schema = GetHistoryTasksFilterSchema(
                finished=True
            )
            # Note: GetHistoryTasksFilterSchema doesn't have task_id filter directly,
            # so we'll filter after fetching
            history_tasks = await client.get_history_tasks(filter_schema)
            
            # Find the specific task by ID
            task = None
            for t in history_tasks:
                if str(t.id) == task_id:
                    task = t
                    break
            
            if not task:
                logger.warning(f"⚠️ Completed task {task_id} not found in history")
                return None
            
            # Get variables from process instance
            variables = {}
            try:
                variable_instances = await client.get_variable_instances(
                    process_instance_id=task.process_instance_id,
                    deserialize_values=True
                )
                for var in variable_instances:
                    variables[var.name] = var.value
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch variables for completed task {task_id}: {e}")
            
            return {
                "id": str(task.id),
                "name": task.name,
                "description": task.description,
                "completedAt": task.end_time.isoformat() if task.end_time else None,
                "processInstanceId": str(task.process_instance_id),
                "processDefinitionKey": task.process_definition_key,
                "variables": variables
            }
        except Exception as e:
            logger.error(f"❌ Error getting completed task details for {task_id}: {e}", exc_info=True)
            return None
    
    async def get_completed_tasks_for_process(
        self,
        user_id: str,
        process_instance_id: str,
        webapp_user_id: Optional[str] = None,
        platform: str = "telegram",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get all completed tasks for a process instance from Camunda History API.

        Args:
            user_id: Flow API UUID (string) - primary identifier
            process_instance_id: Process instance ID
            webapp_user_id: Optional webapp_user_id for web users
            platform: Platform identifier ("web" or "telegram")
            limit: Maximum number of completed tasks to return (default: 10)

        Returns:
            List of completed task details
        """
        from uuid import UUID
        from camunda_client.clients.engine.schemas.body import GetHistoryTasksFilterSchema

        client = await self._get_client(user_id, webapp_user_id=webapp_user_id, platform=platform)
        
        try:
            # Query history API for completed tasks in this process instance
            filter_schema = GetHistoryTasksFilterSchema(
                process_instance_id=UUID(process_instance_id),
                finished=True
            )
            history_tasks = await client.get_history_tasks(filter_schema)
            
            # Get variables for the process instance (once for all tasks)
            variables_map = {}
            try:
                variable_instances = await client.get_variable_instances(
                    process_instance_id=process_instance_id,
                    deserialize_values=True
                )
                for var in variable_instances:
                    variables_map[var.name] = var.value
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch variables for process {process_instance_id}: {e}")
            
            # Convert to list of dicts, sorted by completion time (most recent first)
            completed_tasks = []
            for task in sorted(history_tasks, key=lambda t: t.end_time if t.end_time else datetime.min, reverse=True)[:limit]:
                completed_tasks.append({
                    "id": str(task.id),
                    "name": task.name,
                    "description": task.description,
                    "completedAt": task.end_time.isoformat() if task.end_time else None,
                    "processInstanceId": str(task.process_instance_id),
                    "processDefinitionKey": task.process_definition_key,
                    "variables": variables_map.copy()  # Share variables across tasks in same process
                })
            
            logger.debug(f"📋 Found {len(completed_tasks)} completed tasks for process {process_instance_id}")
            return completed_tasks
        except Exception as e:
            logger.error(f"❌ Error getting completed tasks for process {process_instance_id}: {e}", exc_info=True)
            return []
    
    def _format_variables(self, variables: Dict[str, Any]) -> Dict:
        """Format variables for Camunda"""
        from camunda_client.types_ import VariableValueSchema
        
        formatted = {}
        for key, value in variables.items():
            if isinstance(value, bool):
                var_schema = VariableValueSchema(value=value, type="Boolean")
            elif isinstance(value, int):
                var_schema = VariableValueSchema(value=value, type="Long")
            elif isinstance(value, float):
                var_schema = VariableValueSchema(value=value, type="Double")
            else:
                var_schema = VariableValueSchema(value=str(value), type="String")
            
            # Convert to dict for JSON serialization
            formatted[key] = var_schema.model_dump()
        
        return formatted
    
    def _format_message_variables(self, message_data: Dict[str, Any], message_type: str, kb_doc_id: str) -> Dict:
        """
        Format message data for Camunda with enhanced fields for async processing, tool selection, and billing.
        
        Args:
            message_data: Enhanced message data with thread context
            message_type: Message type (GROUP_MESSAGE, DM_MESSAGE, ASSISTANT_MESSAGE)
            kb_doc_id: Document ID for ES reference
            
        Returns:
            Formatted Camunda variables
        """
        # Build enhanced message variables for async processing
        variables = {
            # === Core Identity Fields ===
            "form_chatName": self._get_chat_name(message_data),
            "form_question": message_data.get("message_text", ""),
            "config_chatID": str(message_data.get("group_id", message_data.get("user_id", ""))),
            "config_chatHumanName": message_data.get("group_name", message_data.get("sender_name", "")),  # Group name for groups, sender name for DMs
            "config_threadID": message_data.get("thread_id", ""),  # Telegram's message_thread_id for supergroup topics
            "form_replyMessageType": self._get_reply_message_type(message_type),
            
            # === NEW: Document ID for ES Reference ===
            "form_documentId": kb_doc_id,  # KEY FIELD - links to ES document
            
            # === NEW: Enhanced Message Info (JSON string) ===
            "form_tgMessageInfo": self._build_telegram_message_info(message_data, kb_doc_id),
            
            # === NEW: Tool Selection & Processing Hints ===
            "config_enabledTools": ",".join(message_data.get("enabled_tools", [])),
            "config_disabledTools": ",".join(message_data.get("disabled_tools", [])),
            "config_knowledgeBases": ",".join(message_data.get("knowledge_bases", [])),
            
            # === NEW: LLM Configuration ===
            "config_llmProvider": message_data.get("llm_provider", ""),
            "config_modelName": message_data.get("model_name", ""),
            "config_systemPrompt": message_data.get("system_prompt", ""),
            
            # === NEW: Billing & Usage Tracking ===
            "billing_userId": str(message_data.get("user_id", "")),
            "billing_messageType": message_data.get("role", "user"),
            "billing_timestamp": message_data.get("message_date", ""),
            
            # === NEW: Agent Configuration ===
            "config_agentName": message_data.get("agent_name", ""),
            "config_agentDescription": message_data.get("agent_description", ""),
            "config_threadLanguage": message_data.get("thread_language", "en"),
            
            # === NEW: Conversation Context ===
            "context_messageCount": message_data.get("message_count", 0),
            "context_conversationSummary": message_data.get("conversation_summary", ""),
            
            # === NEW: Thread Context ===
            "thread_type": message_data.get("thread_type", "dm"),
            "thread_owner_id": str(message_data.get("thread_owner_id", "")),
            "thread_name": message_data.get("thread_name", ""),
            
            # === NEW: Process Tracking ===
            "process_instance_id": message_data.get("process_instance_id", ""),
            "active_workflows": ",".join(message_data.get("active_workflows", [])),
        }
        
        return self._format_variables(variables)
    
    def _build_business_key(self, message_data: Dict[str, Any]) -> str:
        """
        Build business key for Camunda correlation.
        
        Format: group-{group_id}[-{thread_id}] or dm-{user_id}
        
        Args:
            message_data: Message data
            
        Returns:
            Business key string
        """
        if "group_id" in message_data:
            group_id = message_data["group_id"]
            thread_id = message_data.get("thread_id")
            if thread_id:
                return f"group-{group_id}-{thread_id}"
            else:
                return f"group-{group_id}"
        else:
            user_id = message_data.get("user_id", "")
            return f"dm-{user_id}"
    
    def _build_telegram_message_info(self, message_data: Dict[str, Any], kb_doc_id: str) -> str:
        """
        Build enhanced Telegram message info JSON with kb_doc_id and extended fields.
        
        Args:
            message_data: Enhanced message data with thread context
            kb_doc_id: Document ID for ES reference
            
        Returns:
            JSON string with comprehensive Telegram message info
        """
        import json
        
        # Determine message type based on reply context
        tg_message_type = "tgReply" if message_data.get("reply_to_message_id") else "tgMessage"
        
        info = {
            # === Original Fields ===
            "tg_message_text": message_data.get("message_text", ""),
            "tg_message_author_camunda_user_id": str(message_data.get("user_id", "")),
            "tg_message_type": tg_message_type,
            "tg_parent_message_text": message_data.get("parent_message_text"),
            "tg_parent_message_id": message_data.get("parent_message_id"),
            "tg_parent_message_user_tg_id": message_data.get("parent_message_user_id"),
            "tg_chat_ID": str(message_data.get("group_id", message_data.get("user_id", ""))),
            "tg_chat_human_name": message_data.get("sender_name", ""),
            "tg_chat_topic_id": message_data.get("telegram_topic_id"),  # Telegram's native topic ID
            
            # === NEW: Document References ===
            "kb_doc_id": kb_doc_id,  # KEY: ES document ID for worker reference
            "elasticsearch_index": message_data.get("_index_name"),  # Which ES index
            
            # === NEW: Message Metadata ===
            "telegram_message_id": message_data.get("_telegram_message_id"),  # Original Telegram message ID
            "message_date": message_data.get("message_date"),
            "media_type": message_data.get("media_type", "text"),  # text, photo, voice, etc.
            
            # === NEW: Mentions & Entities ===
            "mentions": message_data.get("mentions", []),  # @mentions
            "hashtags": message_data.get("hashtags", []),  # #hashtags
            "urls": message_data.get("urls", []),  # Extracted URLs
            
            # === NEW: Thread Context ===
            "thread_id": message_data.get("thread_id"),  # Luka's internal thread UUID
            "thread_type": message_data.get("thread_type", "dm"),  # dm, group, supergroup
            "thread_owner_id": str(message_data.get("thread_owner_id", "")),
            "thread_name": message_data.get("thread_name", ""),
            
            # === NEW: Agent & LLM Configuration ===
            "agent_name": message_data.get("agent_name", ""),
            "agent_description": message_data.get("agent_description", ""),
            "llm_provider": message_data.get("llm_provider", ""),
            "model_name": message_data.get("model_name", ""),
            "system_prompt": message_data.get("system_prompt", ""),
            
            # === NEW: Tool Configuration ===
            "enabled_tools": message_data.get("enabled_tools", []),
            "disabled_tools": message_data.get("disabled_tools", []),
            "knowledge_bases": message_data.get("knowledge_bases", []),
            
            # === NEW: Conversation Context ===
            "thread_language": message_data.get("thread_language", "en"),
            "conversation_summary": message_data.get("conversation_summary", ""),
            "message_count": message_data.get("message_count", 0),
            
            # === NEW: Process Tracking ===
            "process_instance_id": message_data.get("process_instance_id", ""),
            "active_workflows": message_data.get("active_workflows", []),
            
            # === NEW: Billing Context ===
            "billing_userId": str(message_data.get("user_id", "")),
            "billing_messageType": message_data.get("role", "user"),
            "billing_timestamp": message_data.get("message_date", ""),
        }
        
        return json.dumps(info)
    
    async def _build_enhanced_message_data(self, message_data: Dict[str, Any], thread: Optional["Thread"]) -> Dict[str, Any]:
        """
        Add thread context to message data.
        
        Args:
            message_data: Base message data
            thread: Optional Thread object with context
            
        Returns:
            Enhanced message data with thread context
        """
        enhanced_data = message_data.copy()
        
        if thread:
            enhanced_data.update({
                "thread_type": thread.thread_type,
                "thread_owner_id": str(thread.owner_id),
                "thread_name": thread.name,
                "agent_name": thread.agent_name or "",
                "agent_description": thread.agent_description or "",
                "llm_provider": thread.llm_provider or "",
                "model_name": thread.model_name or "",
                "system_prompt": thread.system_prompt or "",
                "enabled_tools": thread.enabled_tools or [],
                "disabled_tools": thread.disabled_tools or [],
                "knowledge_bases": thread.knowledge_bases or [],
                "thread_language": thread.language or "",
                "conversation_summary": thread.conversation_summary or "",
                "message_count": thread.message_count or 0,
                "process_instance_id": thread.process_instance_id or "",
                "active_workflows": thread.active_workflows or []
            })
        
        return enhanced_data
    
    def _get_chat_name(self, message_data: Dict[str, Any]) -> str:
        """Get chat name for Camunda form field."""
        if "group_id" in message_data:
            return f"group-{message_data['group_id']}"
        else:
            return "dm"
    
    def _get_reply_message_type(self, message_type: str) -> str:
        """Convert message type to Camunda reply message type."""
        type_mapping = {
            "GROUP_MESSAGE": "MESSAGE",
            "DM_MESSAGE": "DM", 
            "ASSISTANT_MESSAGE": "ASSISTANT"
        }
        return type_mapping.get(message_type, "MESSAGE")


# Singleton accessor
def get_camunda_service() -> CamundaService:
    """Get CamundaService singleton"""
    return CamundaService.get_instance()

