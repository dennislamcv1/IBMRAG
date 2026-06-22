# Libraries for MCP client, LLM handling, and async operations
import asyncio
import json
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    Root,
    TextContent,
    CreateMessageResult,
    CreateMessageRequestParams,
)

# Configuration
SERVER_SCRIPT = Path(__file__).parent / "server.py"
PROJECT_DIR = Path(__file__).parent.resolve()

# StdioServerParameters launches this same Python interpreter with server.py
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(SERVER_SCRIPT)],
)

# ROOTS — Tell the server which directories it can access
def list_roots(*_: object) -> list[Root]:
    """Limit the server's file access to this project directory."""
    return [Root(uri=PROJECT_DIR.as_uri(), name=PROJECT_DIR.name)]

# SAMPLING — Handle LLM requests delegated from the server
async def handle_sampling(params: CreateMessageRequestParams) -> CreateMessageResult:
    """Run a Claude LLM call on behalf of the server and return the result."""
    from anthropic import Anthropic

    # Extract the prompt text from the first sampling message
    prompt = params.messages[0].content.text

    print(f"\n[Sampling] Server requested LLM task:")
    print(f"  Prompt preview: {prompt[:150]}...")

    anthropic_client = Anthropic()
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=params.maxTokens or 200,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.content[0].text
    print(f"  LLM Response: {response_text[:100]}...")

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=response_text),
        model="claude-sonnet-4-20250514",
    )

# HELPER — Open a session, call a tool, and return the parsed JSON result
async def call_tool(tool_name: str, arguments: dict) -> dict:
    """Connect to the server, call a tool, and return the parsed JSON result."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=handle_sampling,
            list_roots_callback=list_roots,
        ) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            # The server always returns a single TextContent item
            response_text = result.content[0].text
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Tool '{tool_name}' returned non-JSON output:\n{response_text}"
                ) from exc

# CONNECTION & DISCOVERY
async def verify_connection():
    """Connect to the server and verify all expected tools and resources exist."""
    print("=" * 60)
    print("MCP Connection Verification")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=handle_sampling,
            list_roots_callback=list_roots,
        ) as session:
            await session.initialize()

            # list_tools() sends a "tools/list" JSON-RPC request to the server
            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]
            print("--- START SCREENSHOT ---")
            print(f"\nDiscovered {len(tool_names)} tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description[:80]}...")

            assert "get_restaurant_info" in tool_names, "FAIL: get_restaurant_info not found!"
            assert "recommend_by_vibe" in tool_names, "FAIL: recommend_by_vibe not found!"
            assert "get_review" in tool_names, "FAIL: get_review not found!"
            print("\nAll required tools verified!")

            # list_resources() discovers data endpoints the server exposes
            resources_result = await session.list_resources()
            print(f"\nDiscovered {len(resources_result.resources)} resources:")
            for resource in resources_result.resources:
                print(f"  - {resource.uri}: {resource.name}")

            roots = list_roots()
            print(f"\nConfigured {len(roots)} roots:")
            for root in roots:
                print(f"  - {root.name}: {root.uri}")

            print("--- END SCREENSHOT ---")

# DEMOS — Call each tool through the MCP protocol
async def demo_get_restaurant_info():
    """Demo: Look up a restaurant by name."""
    print("\n" + "-" * 60)
    print("Demo: get_restaurant_info('Iron & Embers')")
    print("-" * 60)

    data = await call_tool("get_restaurant_info", {"restaurant_name": "Iron & Embers"})
    print(json.dumps(data, indent=2))

async def demo_recommend_by_vibe():
    """Demo: Find restaurants by vibe keyword."""
    print("\n" + "-" * 60)
    print("Demo: recommend_by_vibe('moody')")
    print("-" * 60)

    data = await call_tool("recommend_by_vibe", {"vibe": "moody"})
    print(f"Vibe: {data['vibe_searched']}")
    print(f"Structured matches: {len(data['structured_matches'])}")
    for match in data["structured_matches"]:
        food_style = match.get("food_style", match.get("cuisine", "N/A"))
        print(f"  - {match['name']} ({food_style}) - {match['rating']}/5")
    print(f"Raw text excerpts: {len(data['raw_text_excerpts'])}")

async def demo_get_review():
    """Demo: Retrieve a restaurant review."""
    print("\n" + "-" * 60)
    print("Demo: get_review('Iron & Embers')")
    print("-" * 60)

    data = await call_tool("get_review", {"restaurant_name": "Iron & Embers"})
    print(json.dumps(data, indent=2))

# Main Entry Point
async def main():
    """Run all demos sequentially."""
    await verify_connection()
    await demo_get_restaurant_info()
    await demo_recommend_by_vibe()
    await demo_get_review()

if __name__ == "__main__":
    asyncio.run(main())
