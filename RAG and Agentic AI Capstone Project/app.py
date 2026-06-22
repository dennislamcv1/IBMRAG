# Libraries to create our MCP host application
import os
import gradio as gr
from pathlib import Path
from fastmcp.client import Client, PythonStdioTransport
from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Configuration
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
SYSTEM_PROMPT = """
You are Connoisseur Companion, a warm and knowledgeable AI guide to
California's restaurant scene. Your job is to help users discover restaurants
by name, cuisine, location, vibe, signature dishes, price range, ratings, and
review details using the MCP tools available to you.

Use the tools to ground your answers:
- Use get_restaurant_info when the user asks about a specific restaurant.
- Use recommend_by_vibe when the user asks for restaurants matching an
  atmosphere, mood, style, neighborhood feel, or dining occasion.
- Use get_review when the user asks for review details, user impressions,
  visit notes, image captions, or a more subjective take.

Guidelines:
- Do not invent restaurants, ratings, dishes, locations, reviews, or prices.
- If a tool returns no match, say so clearly and suggest a broader or partial
  search term the user can try.
- When recommending restaurants, summarize the strongest matches with the
  restaurant name, location, food style, vibe, rating, price range, and why it
  fits the request when those fields are available.
- Keep responses concise, conversational, and useful. Mention trade-offs like
  high price, loud atmosphere, or special-occasion fit when the data supports it.
- Ask a brief clarifying question only when the request is too ambiguous to
  answer well; otherwise, make the best grounded recommendation from the tools.
"""

project_id = (
    os.environ.get("WATSONX_AI_PROJECT_ID")
    or os.environ.get("WATSONX_PROJECT_ID")
)
# Initializing the WatsonX LLM
def make_model():
    return ChatWatsonx(
        model_id="ibm/granite-4-h-small",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params={"temperature": 0.7},
    )

# MCP Host — ReAct Agent Loop
async def chat_with_agent(user_message: str, history: list) -> str:
    """Connect to the MCP server, discover tools, and run a ReAct loop.
    The LLM decides which tools to call, calls them via the MCP server,
    and repeats until it produces a final text response."""
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)

    async with Client(transport) as client:
        # Discover available tools from the MCP server
        mcp_tools = await client.list_tools()

        # Convert MCP tool schemas to OpenAI-style tool definitions for the LLM
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in mcp_tools
        ]

        model = make_model().bind_tools(openai_tools)

        # Build the message list from chat history and the new user message
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))

        # ReAct loop — call tools until the LLM returns a plain text reply
        for _ in range(10):
            response = await model.ainvoke(messages)
            messages.append(response)

            # No tool calls means the LLM is done — return the final response
            if not response.tool_calls:
                raw = response.content
                if isinstance(raw, list):
                    return " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in raw
                    )
                return str(raw)

            # Execute each tool call via the MCP server and feed results back
            for tool_call in response.tool_calls:
                result = await client.call_tool(tool_call["name"], tool_call["args"])
                tool_output = " ".join(
                    item.text if hasattr(item, "text") else str(item)
                    for item in result.content
                ) if result.content else "(no result)"
                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

        return "I wasn't able to complete that request. Please try again."

# Gradio Event Handler
async def handle_chat(user_message, history):
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        yield history
        return

    # Show a thinking placeholder while the agent runs
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Thinking..."},
    ]
    yield history

    response_text = await chat_with_agent(user_message, history[:-2])
    history[-1] = {"role": "assistant", "content": response_text}
    yield history

# Gradio Interface
with gr.Blocks(title="Connoisseur Companion") as demo:
    gr.Markdown("# Connoisseur Companion\nYour AI guide to California's restaurant scene. Ask me about restaurants by name, cuisine, or vibe!")

    chatbot = gr.Chatbot(height=500, type="messages")
    msg_input = gr.Textbox(
        label="Ask about restaurants",
        placeholder='e.g., "Find me a moody spot in DTLA" or "Tell me about Sakura Garden"',
    )

    with gr.Row():
        btn1 = gr.Button("Find moody restaurants", size="sm")
        btn2 = gr.Button("Tell me about Iron & Embers", size="sm")
        btn3 = gr.Button("Zen dining in Little Tokyo?", size="sm")

    msg_input.submit(handle_chat, [msg_input, chatbot], [chatbot])
    msg_input.submit(lambda: "", None, msg_input)

    btn1.click(handle_chat, [gr.State("Find me some moody restaurants"), chatbot], [chatbot])
    btn2.click(handle_chat, [gr.State("Tell me about Iron & Embers"), chatbot], [chatbot])
    btn3.click(handle_chat, [gr.State("What's a zen dining experience in Little Tokyo?"), chatbot], [chatbot])

# Launch the App
if __name__ == "__main__":
    print("Starting Connoisseur Companion...")
    demo.launch(
        share=True,
        theme=gr.themes.Soft(),
    )
