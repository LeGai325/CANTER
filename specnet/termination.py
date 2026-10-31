"""Wait for the actual local operation before releasing caller-owned resources.

Repeated caller cancellation cannot release a local HTTP slot while its task
still runs. Local termination does not confirm a provider's backend job ended.
"""
import asyncio


async def await_terminal(task):
    interrupted=False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted=True
    value=task.result()
    if interrupted:raise asyncio.CancelledError
    return value
