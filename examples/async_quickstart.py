"""The same API, async. Every method has an async twin with the same signature.

    export PICTOGRAPH_API_KEY=pk_live_...
    python examples/async_quickstart.py

Use ``AsyncClient`` as an async context manager so its connection pool closes
cleanly. Concurrent reads with ``asyncio.gather`` are the reason to reach for it.
"""

from __future__ import annotations

import asyncio

from pictograph import AsyncClient


async def main() -> None:
    async with AsyncClient() as client:  # reads PICTOGRAPH_API_KEY from the environment
        datasets = await client.datasets.list(limit=5)
        print(f"{len(datasets)} dataset(s):")

        # Fan out: fetch each dataset's full record concurrently with gather.
        details = await asyncio.gather(*(client.datasets.get(name=d.name) for d in datasets))
        for dataset in details:
            print(f"  {dataset.name}: {dataset.image_count} image(s)")


if __name__ == "__main__":
    asyncio.run(main())
