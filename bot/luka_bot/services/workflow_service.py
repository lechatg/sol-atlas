"""
Workflow orchestration service for Luka bot dialog workflows.
Provides in-memory workflow lifecycle management for agent tooling.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger

from luka_bot.services.workflow_discovery_service import get_workflow_discovery_service, WorkflowDiscoveryService
from luka_bot.services.workflow_definition_service import get_workflow_definition_service, WorkflowDefinitionService
from luka_bot.services.workflow_context_service import get_workflow_context_service, WorkflowContextService


class WorkflowMetadata:
    """Workflow metadata for discovery and selection."""

    def __init__(
        self,
        domain: str,
        name: str,
        version: str,
        description: str,
        estimated_duration: str = "",
        tool_count: int = 0,
    ):
        self.domain = domain
        self.name = name
        self.version = version
        self.description = description
        self.estimated_duration = estimated_duration
        self.tool_count = tool_count
        self.created_at = datetime.utcnow()


class WorkflowStatus:
    """Current workflow execution status and progress information."""

    def __init__(
        self,
        workflow_id: str,
        domain: str,
        state: str,
        progress: float = 0.0,
        current_step: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        self.workflow_id = workflow_id
        self.domain = domain
        self.state = state  # 'initialized', 'running', 'paused', 'completed', 'failed'
        self.progress = progress  # 0.0 to 1.0
        self.current_step = current_step
        self.artifacts = artifacts or {}
        self.context = context or {}  # Step data passed between workflow steps
        self.error = error
        self.last_updated = datetime.utcnow()


class WorkflowService:
    """
    Main workflow orchestration service integrating Luka dialog workflows.

    Responsibilities:
    - Orchestrate workflow discovery, execution, and state management
    - Provide unified interface for workflow operations across the bot
    - Manage workflow lifecycle and provide status to tools
    """

    def __init__(
        self,
        discovery_service: Optional[WorkflowDiscoveryService] = None,
        definition_service: Optional[WorkflowDefinitionService] = None,
        context_service: Optional[WorkflowContextService] = None,
    ):
        self._active_workflows: Dict[str, WorkflowStatus] = {}
        self._workflow_counter = 0
        self._initialized = False

        self._discovery_service = discovery_service
        self._definition_service = definition_service
        self._context_service = context_service

    def _get_discovery_service(self) -> WorkflowDiscoveryService:
        if self._discovery_service is None:
            self._discovery_service = get_workflow_discovery_service()
        return self._discovery_service

    def _get_definition_service(self) -> WorkflowDefinitionService:
        if self._definition_service is None:
            self._definition_service = get_workflow_definition_service()
        return self._definition_service

    def _get_context_service(self) -> WorkflowContextService:
        if self._context_service is None:
            self._context_service = get_workflow_context_service()
        return self._context_service

    async def initialize(self) -> bool:
        """Initialize workflow service with all dependencies."""
        try:
            logger.info("Initializing Luka WorkflowService...")

            discovery_service = self._get_discovery_service()
            self._get_definition_service()
            self._get_context_service()

            await discovery_service.initialize()
            discovered_workflows = await discovery_service.get_available_workflows()

            logger.info(f"WorkflowService initialized with {len(discovered_workflows)} workflow(s)")
            logger.debug(f"Available workflow domains: {list(discovered_workflows.keys())}")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize WorkflowService: {e}")
            return False

    async def get_available_workflows(self) -> List[WorkflowMetadata]:
        """Get all discovered and validated workflows."""
        if not self._initialized:
            await self.initialize()

        try:
            discovery_service = self._get_discovery_service()
            workflows_dict = await discovery_service.get_available_workflows()

            metadata_list: List[WorkflowMetadata] = []
            for domain, workflow_def in workflows_dict.items():
                tool_steps = workflow_def.tool_chain.get("steps", [])
                metadata_list.append(
                    WorkflowMetadata(
                        domain=domain,
                        name=workflow_def.name,
                        version=workflow_def.version,
                        description=workflow_def.description,
                        estimated_duration=workflow_def.estimated_duration,
                        tool_count=len(tool_steps),
                    )
                )

            logger.debug(f"Retrieved {len(metadata_list)} workflow metadata entries")
            return metadata_list

        except Exception as e:
            logger.error(f"Failed to get available workflows: {e}")
            return []

    async def start_workflow(self, user_id: int, domain: str, workflow_name: Optional[str] = None) -> str:
        """Initialize workflow execution and return workflow_id."""
        if not self._initialized:
            await self.initialize()

        try:
            self._workflow_counter += 1
            workflow_id = f"wf_{user_id}_{domain}_{self._workflow_counter}_{int(datetime.utcnow().timestamp())}"

            discovery_service = self._get_discovery_service()
            workflows = await discovery_service.get_available_workflows()

            if domain not in workflows:
                raise ValueError(f"Workflow domain '{domain}' not found")

            # Set first step from workflow definition
            steps = workflows[domain].tool_chain.get("steps", [])
            first_step_id = steps[0].get("id", "initialization") if steps else "initialization"
            
            workflow_status = WorkflowStatus(
                workflow_id=workflow_id,
                domain=domain,
                state="running",  # Start in running state, not initialized
                progress=0.1,  # 10% for starting
                current_step=first_step_id,  # Start at first real step
            )

            self._active_workflows[workflow_id] = workflow_status
            logger.info(f"Started workflow {workflow_id} for user {user_id} in domain {domain}")
            return workflow_id

        except Exception as e:
            logger.error(f"Failed to start workflow for user {user_id}, domain {domain}: {e}")
            raise

    async def get_active_workflow_for_user(self, user_id: int, domain: str) -> Optional[WorkflowStatus]:
        """
        Find active workflow for a user in a specific domain.
        
        Returns the most recent active workflow (running or initialized) for the user.
        """
        matching_workflows = []
        for workflow_id, status in self._active_workflows.items():
            # Extract user_id from workflow_id format: wf_{user_id}_{domain}_{counter}_{timestamp}
            if workflow_id.startswith(f"wf_{user_id}_{domain}_"):
                if status.state in ("initialized", "running", "paused"):
                    matching_workflows.append((workflow_id, status))
        
        if not matching_workflows:
            return None
        
        # Return most recent (last updated)
        matching_workflows.sort(key=lambda x: x[1].last_updated, reverse=True)
        return matching_workflows[0][1]

    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowStatus]:
        """Get current workflow progress and artifacts."""
        status = self._active_workflows.get(workflow_id)
        if status:
            status.last_updated = datetime.utcnow()
        return status

    async def execute_workflow_step(self, workflow_id: str, step_name: str, context: Dict[str, Any]) -> bool:
        """
        Execute a single workflow step with context.

        Note: This method updates workflow status. The actual step execution
        (following instructions, using tools, etc.) is done by the LLM agent
        following the step instructions from the workflow definition.
        """
        if workflow_id not in self._active_workflows:
            return False

        status = self._active_workflows[workflow_id]
        status.current_step = step_name
        status.state = "running"

        # Merge provided context into workflow status context
        if context:
            status.context.update(context)

        try:
            discovery_service = self._get_discovery_service()
            workflows = await discovery_service.get_available_workflows()
            workflow_def = workflows.get(status.domain)

            if not workflow_def:
                status.state = "failed"
                status.error = f"Workflow definition not found for domain {status.domain}"
                return False

            logger.info(f"Updating workflow step to '{step_name}' for workflow {workflow_id}")

            # Check if this step has background tasks to launch
            steps = workflow_def.tool_chain.get("steps", [])
            current_step_def = next((s for s in steps if s.get("id") == step_name), None)

            if current_step_def:
                background_tasks = current_step_def.get("background_tasks", [])
                if background_tasks:
                    # Launch background tasks for this step
                    import asyncio
                    from luka_bot.agents.tools.workflow_tools import _send_crypto_insight_followup, _send_group_stats_followup

                    user_id = context.get("user_id")
                    thread_id = context.get("thread_id")
                    language = context.get("language", "en")

                    # Skip if required context is missing
                    if not user_id or not thread_id:
                        logger.warning(f"⚠️ Cannot launch background tasks: missing user_id or thread_id")
                    else:
                        for task in background_tasks:
                            task_type = task.get("type")
                            trigger = task.get("trigger", "on_step_start")

                            if trigger == "on_step_start":
                                if task_type == "crypto_insight":
                                    logger.info(f"🚀 Launching crypto insight background task for step {step_name}")
                                    asyncio.create_task(_send_crypto_insight_followup(user_id, thread_id, language))
                                elif task_type == "group_stats":
                                    logger.info(f"🚀 Launching group stats background task for step {step_name}")
                                    asyncio.create_task(_send_group_stats_followup(user_id, thread_id, language))
                                else:
                                    logger.warning(f"⚠️ Unknown background task type: {task_type}")

            # Calculate progress based on step position
            if steps and step_name != "initialization":
                # Find current step index
                step_index = next((i for i, s in enumerate(steps) if s.get("id") == step_name), -1)
                if step_index >= 0:
                    # Progress = (step_index + 1) / total_steps, with some buffer for completion
                    status.progress = min((step_index + 1) / len(steps), 0.95)  # Leave 5% for final completion
                else:
                    # Step not found, increment slightly
                    status.progress = min(status.progress + 0.1, 0.95)
            else:
                # Initialization step
                status.progress = 0.1
            
            status.last_updated = datetime.utcnow()

            # Mark as completed if we've reached the last step (will be finalized when step completes)
            if status.progress >= 0.95:
                logger.debug(f"Workflow {workflow_id} nearing completion (progress: {status.progress:.0%})")

            return True

        except Exception as e:
            status.state = "failed"
            status.error = str(e)
            logger.error(f"Failed to update workflow step {step_name} for {workflow_id}: {e}")
            return False
    
    async def advance_workflow_to_next_step(self, workflow_id: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Advance workflow to the next step based on current step and workflow definition.
        
        Returns:
            Next step ID if found, None if workflow is complete or has no next step
        """
        if workflow_id not in self._active_workflows:
            return None
        
        status = self._active_workflows[workflow_id]
        
        try:
            discovery_service = self._get_discovery_service()
            workflows = await discovery_service.get_available_workflows()
            workflow_def = workflows.get(status.domain)
            
            if not workflow_def:
                return None
            
            steps = workflow_def.tool_chain.get("steps", [])
            if not steps:
                return None
            
            # Find current step index
            current_step_id = status.current_step or "initialization"
            if current_step_id == "initialization":
                # First step after initialization
                next_step = steps[0] if steps else None
            else:
                current_index = next((i for i, s in enumerate(steps) if s.get("id") == current_step_id), -1)
                if current_index < 0 or current_index >= len(steps) - 1:
                    # Last step or step not found
                    status.state = "completed"
                    status.progress = 1.0
                    logger.info(f"Workflow {workflow_id} completed - no more steps")
                    return None
                
                next_step = steps[current_index + 1]
            
            if next_step:
                next_step_id = next_step.get("id")
                await self.execute_workflow_step(workflow_id, next_step_id, context)
                logger.info(f"Advanced workflow {workflow_id} to step: {next_step_id}")
                return next_step_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to advance workflow {workflow_id}: {e}")
            return None

    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause workflow with state preservation."""
        status = self._active_workflows.get(workflow_id)
        if status and status.state == "running":
            status.state = "paused"
            status.last_updated = datetime.utcnow()
            logger.info(f"Paused workflow {workflow_id}")
            return True
        return False

    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume paused workflow execution."""
        status = self._active_workflows.get(workflow_id)
        if status and status.state == "paused":
            status.state = "running"
            status.last_updated = datetime.utcnow()
            logger.info(f"Resumed workflow {workflow_id}")
            return True
        return False

    async def terminate_workflow(self, workflow_id: str, cleanup: bool = True) -> bool:
        """Stop workflow and optionally clean up state."""
        status = self._active_workflows.get(workflow_id)
        if not status:
            return False

        status.state = "terminated"
        status.last_updated = datetime.utcnow()

        if cleanup:
            status.artifacts.clear()
            del self._active_workflows[workflow_id]
            logger.info(f"Terminated and cleaned up workflow {workflow_id}")
        else:
            logger.info(f"Terminated workflow {workflow_id} (state preserved)")

        return True

    async def get_workflow_context(self, domain: str) -> Optional[str]:
        """Get complete workflow context for LLM integration."""
        try:
            context_service = self._get_context_service()
            return await context_service.get_workflow_context(domain, include_documentation=True)
        except Exception as e:
            logger.error(f"Failed to get workflow context for domain {domain}: {e}")
            return None

    async def cleanup_inactive_workflows(self, max_age_hours: int = 24) -> int:
        """Clean up workflows that have been inactive for specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        workflows_to_remove = [
            workflow_id
            for workflow_id, status in self._active_workflows.items()
            if status.last_updated < cutoff_time and status.state in {"completed", "failed", "terminated"}
        ]

        for workflow_id in workflows_to_remove:
            del self._active_workflows[workflow_id]

        if workflows_to_remove:
            logger.info(f"Cleaned up {len(workflows_to_remove)} inactive workflows")

        return len(workflows_to_remove)

    async def get_service_health(self) -> Dict[str, Any]:
        """Get workflow service health and status information."""
        return {
            "initialized": self._initialized,
            "active_workflows": len(self._active_workflows),
            "workflow_counter": self._workflow_counter,
        }


_workflow_service: Optional[WorkflowService] = None


def get_workflow_service() -> WorkflowService:
    """Get or create workflow service singleton."""
    global _workflow_service

    if _workflow_service is None:
        _workflow_service = WorkflowService()
        logger.info("✅ WorkflowService singleton created")

    return _workflow_service
