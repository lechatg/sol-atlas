# SOL Atlas Luka Bot

Luka Bot is the conversational agent that powers the SOL Atlas ecosystem. It delivers LLM-assisted workflows, knowledge base search, and group operations for Discord and Telegram communities while integrating with the rest of the SOL Atlas stack (Camunda process automation, Flow API, and the Atlas UI gateway).

## Core Capabilities
- Streaming conversations with provider failover (Ollama → OpenAI) and thread persistence.
- Knowledge-base ingestion and retrieval backed by Elasticsearch and S3 media storage.
- Group management with reputation, moderation toggles, and localized onboarding.
- Workflow execution through Camunda BPM and Flow API real-time task updates.
- Observability via Prometheus metrics, structured Loguru logging, and Sentry hooks.

## Repository Structure
- `luka_bot/` – production Telegram agent (handlers, services, middlewares, locales, scripts, docs).
- `ag_ui_gateway/` – Flask-based admin and Atlas web gateway for operators.
- `flow_client/`, `camunda_client/` – shared protocol and client libraries used by the bot and UI.
- `docker-compose.yml`, `Dockerfile`, `.env.example` – reference infrastructure for local and staging environments.
- `AGENTS.md` – contributor guidelines for agents and supporting services.

## Quick Start
```bash
# Python 3.11+
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # populate tokens and service URLs (minimally: BOT_TOKEN + OPENAI_API_KEY)
python -m luka_bot          # starts Luka Bot in polling mode
```

Optional services:
- `docker compose up -d redis` – Redis for FSM storage (required).
- `docker compose up -d elasticsearch` – enable KB indexing/search.
- `docker compose up -d` – bring up the full stack (bot, Redis, monitoring scaffolding).

## Use Cases

### 1. Personal AI Assistant in Telegram
The simplest use case: get your own ChatGPT directly in Telegram. Start a direct message with the bot and have natural conversations. The bot maintains conversation history, supports streaming responses, and can access tools like knowledge base search (by default, the KB is your own message history with the bot). No special setup needed—just message the bot and start chatting.

**Getting started:**
- Send `/start` to the bot in a private message
- Start chatting naturally—the bot responds with context from your conversation history
- Use `/profile` to adjust your language preferences and settings

### 2. Group Admin: Control Bot Behavior & Moderation
As a group administrator, you have fine-grained control over how the bot operates in your community. Switch between different AI assistant modes and enable powerful moderation features to keep your group healthy.

**Key features:**
- **AI Assistant Modes:**
  - *Answer All*: Bot responds to every message (great for small teams or support channels)
  - *Mentions Only*: Bot only responds when @mentioned or replied to (default, respectful mode)
  - *Silent*: Bot indexes messages but doesn't respond (pure knowledge base mode)
  
- **Moderation Tools:**
  - Auto-delete join/leave notifications to reduce noise
  - Spam detection and filtering
  - Message indexing for searchable group history

**How to configure:**
1. Add the bot to your group
2. Send `/groups` in a DM with the bot
3. Select your group → "⚙️ Group Settings"
4. Navigate to "🤖 AI Assistant" to toggle modes and moderation features

### 3. Group Persona: Customize Bot Personality
Give your bot a unique personality tailored to your group's purpose. Whether you're running an event promotion channel, a technical support group, or a casual community, you can define exactly how the bot should communicate. **The best part: it's incredibly fast and easy to change the prompt on the fly**—no redeployment needed! Experiment with different personalities and instantly see how your group members respond. Tweak the tone, adjust the style, iterate in real-time.

**Example personas:**
- **Event Promoter**: *"You're an enthusiastic community manager promoting our Web3 hackathon on March 15th. Encourage participation, highlight the $50k prize pool, and remind users to register at solhackathon.io. Be energetic and use rocket 🚀 emojis!"*
- **Product Tech Support**: *"You're the technical support agent for our SaaS platform 'CloudFlow'. Help users troubleshoot issues with API integration, authentication, and deployment workflows. Reference our documentation when needed, be patient with beginners, and escalate complex bugs to the engineering team. Keep responses clear and actionable."*
- **Casual & Fun**: *"You're a chill community member who loves memes and jokes. Keep responses short, use slang, and throw in some 😎 vibes. Don't be too formal!"*

**Pro tip:** For bots that need deep product knowledge, you can paste entire FAQs, product documentation, or website content directly into the personality prompt—not just 1-2 lines. The bot will use this context to provide accurate, informed responses about your product or service.

**How to set up:**
1. Send `/groups` to the bot (in DM)
2. Select your group → "⚙️ Group Settings"
3. Go to "🤖 AI Assistant"
4. Select "✏️ Group Personality Prompt"
5. Enter your custom personality instructions
6. The bot will now respond according to your defined style while maintaining safety guardrails

**Note:** The bot's core identity and safety guidelines remain unchanged—personality prompts are style suggestions layered on top.

### 4. Turn Your Group Chat into a Searchable Knowledge Base
Every group member can search through indexed group messages using natural language queries. The bot automatically builds a searchable knowledge base of all group conversations, making it easy to find that important link someone shared last week or recall a past discussion. **Say goodbye to endless scrolling—find the gems buried in your chat history.**

**Features:**
- Natural language search: "Where was that coffee shop everyone recommended?"
- Time-based filters: "What new exhibitions were discussed in our city chat last week?"
- User filters: "What bugs did Sarah report?"
- Multi-group search: Search across multiple groups you're in simultaneously

**How to use:**
1. Send `/search` to the bot in a DM (not in the group chat)
2. The bot shows which knowledge bases (groups) you have access to
3. Click "🔍 Search" and enter your query
4. Use "⚙️ Settings" to enable/disable specific group knowledge bases
5. Get formatted results with links to original messages

**Advanced:** Combine search with workflows by asking the bot "Can you search my messages about..." and it will automatically use the search tool to find relevant information.

## Documentation
- Updated guides live in `luka_bot/docs/` (start with `README.md` for the table of contents).
- Deployment, operations, and API notes are organized by topic; legacy notes remain under `luka_bot/docs/archive/`.
- For contributor workflow and coding conventions, see `AGENTS.md`.

## License
Released under the LGPL-3.0 license (`LICENSE.md`). Contributions are welcome—open an issue or follow the workflow described in `AGENTS.md`.
