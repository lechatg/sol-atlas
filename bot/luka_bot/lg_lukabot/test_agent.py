#!/usr/bin/env python
"""Test script for LangGraph Telegram agent.

This script tests the end-to-end functionality of the LangGraph agent:
1. Creates the agent graph with checkpointing
2. Sends test messages
3. Verifies tool execution
4. Verifies state persistence

Usage:
    python -m luka_bot.lg_lukabot.test_agent
"""

import asyncio
from langchain_core.messages import HumanMessage
from loguru import logger

from luka_bot.lg_lukabot.graph import get_telegram_agent_graph
from luka_bot.lg_lukabot.state import AgentState


async def test_basic_conversation():
    """Test basic conversation without tools."""
    logger.info("=" * 80)
    logger.info("TEST 1: Basic Conversation (no tools)")
    logger.info("=" * 80)

    # Get the agent graph
    graph = await get_telegram_agent_graph()

    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content="Hello! Who are you?")],
        "user_id": 12345,
        "thread_id": "test_thread_1",
        "language": "en",
        "platform": "telegram",
        "is_guest": False,
        "knowledge_bases": ["tg-kb-user-12345"],
        "enabled_tools": ["support"],  # Only support tool, won't trigger
        "active_workflow": None,
        "workflow_step": None,
        "workflow_progress": 0.0,
        "workflow_suggestions": [],
        "tool_results": {},
        "next_action": None,
        "ui_context": {},
        "metadata": {},
    }

    # Configure thread for checkpointing
    config = {"configurable": {"thread_id": "test_thread_1"}}

    try:
        # Run the graph
        logger.info("🚀 Invoking agent...")
        result = await graph.ainvoke(initial_state, config)

        # Display results
        logger.info("✅ Agent completed successfully!")
        logger.info(f"Total messages in conversation: {len(result['messages'])}")
        logger.info(f"Final message: {result['messages'][-1].content[:200]}...")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_tool_usage():
    """Test conversation that triggers tool usage."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Tool Usage (support tool)")
    logger.info("=" * 80)

    # Get the agent graph
    graph = await get_telegram_agent_graph()

    # Create initial state asking for help
    initial_state = {
        "messages": [
            HumanMessage(content="I need help with using the bot. Can you help me?")
        ],
        "user_id": 12346,
        "thread_id": "test_thread_2",
        "language": "en",
        "platform": "telegram",
        "is_guest": False,
        "knowledge_bases": ["tg-kb-user-12346"],
        "enabled_tools": ["support", "knowledge_base"],
        "active_workflow": None,
        "workflow_step": None,
        "workflow_progress": 0.0,
        "workflow_suggestions": [],
        "tool_results": {},
        "next_action": None,
        "ui_context": {},
        "metadata": {},
    }

    config = {"configurable": {"thread_id": "test_thread_2"}}

    try:
        logger.info("🚀 Invoking agent...")
        result = await graph.ainvoke(initial_state, config)

        logger.info("✅ Agent completed successfully!")
        logger.info(f"Total messages: {len(result['messages'])}")

        # Check if tools were called
        has_tool_calls = any(
            hasattr(msg, "tool_calls") and msg.tool_calls for msg in result["messages"]
        )
        logger.info(f"Tools called: {'Yes' if has_tool_calls else 'No'}")

        if has_tool_calls:
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        logger.info(f"  - Tool: {tool_call['name']}")

        logger.info(f"Final response: {result['messages'][-1].content[:200]}...")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_state_persistence():
    """Test that state persists across invocations."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: State Persistence")
    logger.info("=" * 80)

    graph = await get_telegram_agent_graph()

    thread_id = "test_thread_3"
    config = {"configurable": {"thread_id": thread_id}}

    # First message
    logger.info("📨 Sending first message...")
    state1 = {
        "messages": [HumanMessage(content="My name is Alice.")],
        "user_id": 12347,
        "thread_id": thread_id,
        "language": "en",
        "platform": "telegram",
        "is_guest": False,
        "knowledge_bases": ["tg-kb-user-12347"],
        "enabled_tools": ["support"],
        "active_workflow": None,
        "workflow_step": None,
        "workflow_progress": 0.0,
        "workflow_suggestions": [],
        "tool_results": {},
        "next_action": None,
        "ui_context": {},
        "metadata": {},
    }

    try:
        result1 = await graph.ainvoke(state1, config)
        logger.info(
            f"✅ First response: {result1['messages'][-1].content[:100]}..."
        )

        # Second message - should remember the name
        logger.info("\n📨 Sending second message (testing memory)...")
        state2 = {
            "messages": result1["messages"]
            + [HumanMessage(content="What's my name?")],
            "user_id": 12347,
            "thread_id": thread_id,
            "language": "en",
            "platform": "telegram",
            "is_guest": False,
            "knowledge_bases": ["tg-kb-user-12347"],
            "enabled_tools": ["support"],
            "active_workflow": None,
            "workflow_step": None,
            "workflow_progress": 0.0,
            "workflow_suggestions": [],
            "tool_results": {},
            "next_action": None,
            "ui_context": {},
            "metadata": {},
        }

        result2 = await graph.ainvoke(state2, config)
        final_response = result2["messages"][-1].content
        logger.info(f"✅ Second response: {final_response[:200]}...")

        # Check if it remembers the name
        remembers_name = "alice" in final_response.lower()
        logger.info(f"Remembers name: {'✅ Yes' if remembers_name else '❌ No'}")

        return remembers_name

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    logger.info("🧪 Starting LangGraph Agent Tests")
    logger.info("=" * 80)

    results = {}

    # Test 1: Basic conversation
    results["basic_conversation"] = await test_basic_conversation()
    await asyncio.sleep(1)

    # Test 2: Tool usage
    results["tool_usage"] = await test_tool_usage()
    await asyncio.sleep(1)

    # Test 3: State persistence
    results["state_persistence"] = await test_state_persistence()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        logger.info(f"{status} - {test_name}")

    logger.info("-" * 80)
    logger.info(f"Total: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

