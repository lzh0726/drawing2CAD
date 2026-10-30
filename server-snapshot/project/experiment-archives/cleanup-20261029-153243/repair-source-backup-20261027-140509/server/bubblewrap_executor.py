"""Explicit Linux executor; never falls back to host execution."""
import asyncio
from .policy import validate_code
from .research_sandbox import run
from .sandbox import CADFailure,SandboxUnavailable

class BubblewrapExecutor:
    async def available(self):
        try:
            checks=await asyncio.to_thread(run,{'probe':True},15)
            return bool(checks) and all(checks.values())
        except (OSError,RuntimeError,ValueError):return False

    async def run(self,code):
        validate_code(code)
        try:
            return await asyncio.to_thread(run,{'code':code},90)
        except OSError as e:
            raise SandboxUnavailable('Isolated CAD executor unavailable; no host fallback.') from e
        except RuntimeError as e:
            raise CADFailure(str(e)[:500]) from e
