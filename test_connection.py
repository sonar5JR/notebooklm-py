"""Quick test to verify NotebookLM connection works."""
import asyncio
import sys
sys.path.insert(0, "src")

from notebooklm import NotebookLMClient

async def test():
    async with await NotebookLMClient.from_storage() as client:
        notebooks = await client.notebooks.list()
        print(f"Connected! Found {len(notebooks)} notebooks:")
        for nb in notebooks:
            print(f"  - {nb.title} (ID: {nb.id[:12]}...)")

asyncio.run(test())
