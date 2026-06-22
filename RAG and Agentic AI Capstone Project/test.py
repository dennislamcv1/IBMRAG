import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_test():

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_restaurant_info", arguments={"restaurant_name": "Iron"})

            payload = json.loads(result.content[0].text)
            assert payload["status"] == "found"
            assert payload["count"] >= 1
            assert any("iron" in restaurant["name"].lower() for restaurant in payload["results"])

            print("\n--- START SCREENSHOT ---")
            print(result.content[0].text)
            print("--- END SCREENSHOT ---\n")

if __name__ == "__main__":
    asyncio.run(run_test())
