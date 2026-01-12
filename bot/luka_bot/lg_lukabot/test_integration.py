#!/usr/bin/env python
"""Quick integration test without OpenAI API key.

This test uses a mock LLM to verify the integration layer works correctly.
It tests the full flow from Telegram message to bot response.

Usage:
    python -m luka_bot.lg_lukabot.test_integration
"""

import asyncio

from langchain_core.messages import AIMessage
from loguru import logger

from luka_bot.lg_lukabot.integration import (
    create_state_from_telegram_message,
    extract_response_from_state,
)


class MockChatModel:
    """Mock LLM that returns canned responses."""

    def __init__(self, response="Hello! I'm a test response from the mock LLM.", **kwargs):
        # Accept any kwargs to match ChatOpenAI interface
        self.response = response

    def bind_tools(self, tools):
        """Mock tool binding."""
        return self

    async def ainvoke(self, messages):
        """Return mock response."""
        logger.info(f"📝 Mock LLM received {len(messages)} messages")
        return AIMessage(content=self.response, tool_calls=[])


async def test_state_creation():
    """Test creating state from Telegram message."""
    logger.info("=" * 80)
    logger.info("TEST 1: State Creation from Telegram Message")
    logger.info("=" * 80)

    try:
        state = await create_state_from_telegram_message(
            message_text="Hello bot!",
            user_id=999,
            thread_id="test_thread",
            language="en",
            enabled_tools=["support", "knowledge_base"],
        )

        # Validate state
        assert state["user_id"] == 999
        assert state["thread_id"] == "test_thread"
        assert state["platform"] == "telegram"
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "Hello bot!"
        assert "support" in state["enabled_tools"]
        assert "knowledge_base" in state["enabled_tools"]

        logger.info("✅ State creation successful")
        logger.info(f"   User ID: {state['user_id']}")
        logger.info(f"   Thread ID: {state['thread_id']}")
        logger.info(f"   Messages: {len(state['messages'])}")
        logger.info(f"   Enabled tools: {state['enabled_tools']}")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_response_extraction():
    """Test extracting response from completed state."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Response Extraction from State")
    logger.info("=" * 80)

    try:
        # Create mock state with AI response
        from langchain_core.messages import HumanMessage

        state = await create_state_from_telegram_message(
            message_text="Test",
            user_id=999,
            thread_id="test_thread",
        )

        # Add AI response
        state["messages"].append(
            AIMessage(content="This is the bot's response!", tool_calls=[])
        )

        # Add workflow suggestions
        state["workflow_suggestions"] = ["Option 1", "Option 2", "Option 3"]
        state["ui_context"] = {"reply_keyboard_needs_update": True}

        # Extract response
        response = await extract_response_from_state(state)

        # Validate
        assert response["text"] == "This is the bot's response!"
        assert len(response["suggestions"]) == 3
        assert response["keyboard_update_needed"] is True

        logger.info("✅ Response extraction successful")
        logger.info(f"   Text: {response['text'][:50]}...")
        logger.info(f"   Suggestions: {response['suggestions']}")
        logger.info(f"   Keyboard update needed: {response['keyboard_update_needed']}")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_mock_agent_flow():
    """Test complete flow with mock LLM."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Complete Flow with Mock LLM")
    logger.info("=" * 80)

    try:
        from luka_bot.lg_lukabot.graph import get_telegram_agent_graph

        # Patch the global variables in nodes module to use MockChatModel
        import luka_bot.lg_lukabot.nodes as nodes_module
        nodes_module.ChatOpenAI = MockChatModel
        nodes_module.ChatOllama = MockChatModel
        
        # Get agent (will use mocked LLM)
        graph = await get_telegram_agent_graph()

        # Create state
        state = await create_state_from_telegram_message(
            message_text="Hello! Can you help me?",
            user_id=999,
            thread_id="mock_test_thread",
            enabled_tools=["support"],
        )

        # Run agent
        config = {"configurable": {"thread_id": "mock_test_thread"}}
        logger.info("🚀 Running mock agent...")

        result = await graph.ainvoke(state, config)

        # Extract response
        response = await extract_response_from_state(result)

        logger.info("✅ Mock agent flow completed")
        logger.info(f"   Response: {response['text'][:100]}...")
        logger.info(f"   Suggestions: {response['suggestions']}")

        # Validate
        assert len(response["text"]) > 0
        logger.info("✅ All validations passed")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_high_level_integration():
    """Test high-level invoke_langgraph_agent function."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: High-Level Integration Function")
    logger.info("=" * 80)

    try:
        from luka_bot.lg_lukabot.integration import invoke_langgraph_agent

        # Patch the global variables in nodes module to use MockChatModel
        import luka_bot.lg_lukabot.nodes as nodes_module
        nodes_module.ChatOpenAI = MockChatModel
        nodes_module.ChatOllama = MockChatModel

        # Call high-level function
        logger.info("🚀 Calling invoke_langgraph_agent...")

        response = await invoke_langgraph_agent(
            message_text="What's the weather like?",
            user_id=999,
            thread_id="integration_test",
            language="en",
            enabled_tools=["support"],
        )

        logger.info("✅ High-level integration completed")
        logger.info(f"   Response: {response['text'][:100]}...")
        logger.info(f"   Suggestions: {response['suggestions']}")
        logger.info(
            f"   Keyboard update: {response.get('keyboard_update_needed', False)}"
        )

        # Validate
        assert "text" in response
        assert len(response["text"]) > 0

        logger.info("✅ All validations passed")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all integration tests."""
    logger.info("🧪 Starting LangGraph Integration Tests (Mock Mode)")
    logger.info("=" * 80)
    logger.info("These tests use a mock LLM and don't require an OpenAI API key")
    logger.info("=" * 80)

    results = {}

    # Test 1: State creation
    results["state_creation"] = await test_state_creation()
    await asyncio.sleep(0.5)

    # Test 2: Response extraction
    results["response_extraction"] = await test_response_extraction()
    await asyncio.sleep(0.5)

    # Test 3: Mock agent flow
    results["mock_agent_flow"] = await test_mock_agent_flow()
    await asyncio.sleep(0.5)

    # Test 4: High-level integration
    results["high_level_integration"] = await test_high_level_integration()

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
        logger.info("🎉 All integration tests passed!")
        logger.info("\nNext steps:")
        logger.info("1. Set OPENAI_API_KEY to test with real LLM")
        logger.info("2. Run: python -m luka_bot.lg_lukabot.test_agent")
        logger.info("3. Enable LANGGRAPH_ENABLED=true in .env")
        logger.info("4. Test with real Telegram messages")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

