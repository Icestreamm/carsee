# =============================================================================
# Drop-in auth for CarSee cloud damage API (FastAPI / Starlette on Render)
# =============================================================================
# 1. Copy this file into your Render service repo (same folder as main.py is fine).
# 2. Set env on Render: PROCESSING_API_EXPECTED_TOKEN=<same as Flutter PROCESSING_API_KEY>
# 3. After creating your FastAPI `app`, add:
#        from bearer_middleware import ProcessingBearerAuthMiddleware
#        app.add_middleware(ProcessingBearerAuthMiddleware)
#
# Protects only POST /run_single_model and POST /process. Leaves /health etc. open.
# =============================================================================

from __future__ import annotations

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_PROTECTED: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/run_single_model"),
        ("POST", "/process"),
    }
)


def _normalize_path(path: str) -> str:
    if len(path) > 1 and path.endswith("/"):
        return path.rstrip("/")
    return path or "/"


class ProcessingBearerAuthMiddleware(BaseHTTPMiddleware):
    """Require Authorization: Bearer <PROCESSING_API_EXPECTED_TOKEN> on protected routes."""

    async def dispatch(self, request: Request, call_next) -> Response:
        method = request.method.upper()
        path = _normalize_path(request.url.path)

        if method == "OPTIONS":
            return await call_next(request)

        key = (method, path)
        if key not in _PROTECTED:
            return await call_next(request)

        expected = os.environ.get("PROCESSING_API_EXPECTED_TOKEN", "").strip()
        if not expected:
            return JSONResponse(
                {"detail": "Server misconfigured: PROCESSING_API_EXPECTED_TOKEN not set"},
                status_code=500,
            )

        auth = request.headers.get("authorization") or ""
        if not auth.startswith("Bearer "):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        token = auth[7:].strip()
        if token != expected:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        return await call_next(request)
