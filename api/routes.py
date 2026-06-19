"""FastAPI route handlers."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from httpx import AsyncClient
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count
from core.trace import trace_event
from providers.registry import ProviderRegistry

from . import dependencies
from .dependencies import get_settings, require_api_key
from .model_catalog import build_models_list_response
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.openai_responses import OpenAIResponsesRequest
from .models.responses import ModelsListResponse
from .request_pipeline import ApiRequestPipeline

_REAL_ANTHROPIC_API = "https://api.anthropic.com"

# ── Plugin marketplace cache ─────────────────────────────────────────────────
_PLUGINS_DIR = (
    Path.home() / ".claude" / "plugins" / "marketplaces" / "claude-plugins-official" / ".claude-plugin"
)
_PLUGIN_MARKETPLACE: dict | None = None
_PLUGINS_LIST: list[dict] | None = None


def _load_plugin_marketplace() -> dict:
    """Load the official plugin marketplace data (cached after first call)."""
    global _PLUGIN_MARKETPLACE
    if _PLUGIN_MARKETPLACE is not None:
        return _PLUGIN_MARKETPLACE
    path = _PLUGINS_DIR / "marketplace.json"
    if not path.exists():
        logger.warning("Plugin marketplace not found at {}", path)
        _PLUGIN_MARKETPLACE = {"plugins": []}
        return _PLUGIN_MARKETPLACE
    try:
        with open(path, encoding="utf-8") as f:
            _PLUGIN_MARKETPLACE = json.load(f)
            logger.info("Loaded {} plugins from marketplace", len(_PLUGIN_MARKETPLACE.get("plugins", [])))
    except Exception:
        logger.exception("Failed to load plugin marketplace")
        _PLUGIN_MARKETPLACE = {"plugins": []}
    return _PLUGIN_MARKETPLACE


def _get_plugin_catalog() -> list[dict]:
    """Return a formatted list of all available plugins from the official marketplace."""
    global _PLUGINS_LIST
    if _PLUGINS_LIST is not None:
        return _PLUGINS_LIST
    data = _load_plugin_marketplace()
    plugins = []
    for p in data.get("plugins", []):
        plugins.append({
            "id": p["name"],
            "name": p.get("title", p["name"]),
            "description": p.get("description", ""),
            "author": p.get("author", {}).get("name", "") if isinstance(p.get("author"), dict) else "",
            "category": p.get("category", "") if isinstance(p.get("category"), str) else (p["category"][0] if p.get("category") else ""),
            "homepage": p.get("homepage", ""),
        })
    _PLUGINS_LIST = plugins
    return _PLUGINS_LIST
# Headers NOT forwarded to real Anthropic (set by httpx or proxy-specific).
_FORWARD_EXCLUDED_HEADERS = frozenset({
    "host", "content-length", "connection", "keep-alive",
    "transfer-encoding", "upgrade", "proxy-connection",
})

router = APIRouter()


def get_request_pipeline(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ApiRequestPipeline:
    """Build the API request pipeline for route handlers."""
    return ApiRequestPipeline(
        settings,
        provider_getter=lambda provider_type: dependencies.resolve_provider(
            provider_type, app=request.app, settings=settings
        ),
        token_counter=get_token_count,
    )


def _probe_response(allow: str) -> Response:
    """Return an empty success response for compatibility probes."""
    return Response(status_code=204, headers={"Allow": allow})


# =============================================================================
# Routes
# =============================================================================
@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    pipeline: ApiRequestPipeline = Depends(get_request_pipeline),
    _auth=Depends(require_api_key),
):
    """Create a message (always streaming)."""
    return pipeline.create_message(request_data)


@router.api_route("/v1/messages", methods=["HEAD", "OPTIONS"])
async def probe_messages(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the messages endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/responses")
async def create_response(
    request_data: OpenAIResponsesRequest,
    pipeline: ApiRequestPipeline = Depends(get_request_pipeline),
    _auth=Depends(require_api_key),
):
    """Create an OpenAI Responses-compatible response through this proxy."""
    return await pipeline.create_response(request_data)


@router.api_route("/v1/responses", methods=["HEAD", "OPTIONS"])
async def probe_responses(_auth=Depends(require_api_key)):
    """Respond to OpenAI Responses compatibility probes."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request_data: TokenCountRequest,
    pipeline: ApiRequestPipeline = Depends(get_request_pipeline),
    _auth=Depends(require_api_key),
):
    """Count tokens for a request."""
    return pipeline.count_tokens(request_data)


@router.api_route("/v1/messages/count_tokens", methods=["HEAD", "OPTIONS"])
async def probe_count_tokens(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the token count endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.get("/")
async def root(
    settings: Settings = Depends(get_settings), _auth=Depends(require_api_key)
):
    """Root endpoint."""
    return {
        "status": "ok",
        "provider": settings.provider_type,
        "model": settings.model,
    }


@router.api_route("/", methods=["HEAD", "OPTIONS"])
async def probe_root():
    """Respond to unauthenticated local compatibility probes for the root endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.api_route("/health", methods=["HEAD", "OPTIONS"])
async def probe_health():
    """Respond to compatibility probes for the health endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models(
    request: Request,
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """List the model ids this proxy advertises to Claude-compatible clients."""
    trace_event(stage="ingress", event="api.models.list", source="api")
    registry = getattr(request.app.state, "provider_registry", None)
    provider_registry = registry if isinstance(registry, ProviderRegistry) else None
    return build_models_list_response(settings, provider_registry)


@router.post("/stop")
async def stop_cli(request: Request, _auth=Depends(require_api_key)):
    """Stop all CLI sessions and pending tasks."""
    workflow = getattr(request.app.state, "messaging_workflow", None)
    if not workflow:
        # Fallback if messaging not initialized
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            logger.info("STOP_CLI: source=cli_manager cancelled_count=N/A")
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await workflow.stop_all_tasks()
    trace_event(
        stage="ingress",
        event="api.cli.stop_via_messaging_workflow",
        source="api",
        cancelled_nodes=count,
    )
    logger.info("STOP_CLI: source=messaging_workflow cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}


def _forward_headers(request: Request) -> dict[str, str]:
    """Extract headers to forward, excluding hop-by-hop and proxy-specific ones."""
    headers = {}
    for name, value in request.headers.items():
        if name.lower() not in _FORWARD_EXCLUDED_HEADERS:
            headers[name] = value
    return headers


@router.api_route("/v1/plugin", methods=["GET", "POST"])
@router.api_route("/v1/plugin/", methods=["GET", "POST"])
async def list_plugins(request: Request):
    """Return the official plugin marketplace catalog rather than forwarding to
    the real Anthropic API. This makes all 200+ marketplace plugins visible in
    the Claude Code Desktop even when using an organization account that has no
    custom plugins configured.
    """
    plugins = _get_plugin_catalog()
    return {
        "data": [
            {
                "type": "plugin",
                "id": p["id"],
                "attributes": {
                    "name": p["name"],
                    "description": p["description"],
                    "author": p["author"],
                    "category": p["category"],
                    "homepage": p["homepage"],
                },
            }
            for p in plugins
        ],
        "total": len(plugins),
    }


@router.api_route("/v1/organization", methods=["GET", "POST"])
@router.api_route("/v1/organization/", methods=["GET", "POST"])
async def get_organization():
    """Return a minimal org stub so the client doesn't show warnings about
    missing organization configuration.
    """
    return {
        "type": "organization",
        "id": "org_fcc_proxy",
        "attributes": {
            "name": "Free Claude Code Proxy",
            "plugins_enabled": True,
        },
    }


@router.api_route("/v1/teams", methods=["GET", "POST"])
@router.api_route("/v1/teams/", methods=["GET", "POST"])
async def list_teams():
    """Return an empty team list so the client treats this as a personal-style
    account rather than one with restricted org plugin access.
    """
    return {"data": []}


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_to_anthropic(request: Request, path: str):
    """Forward unknown requests to the real Anthropic API.

    The Claude Code client sends ALL `api.anthropic.com` traffic through the proxy
    when ``ANTHROPIC_BASE_URL`` is set. Endpoints the proxy doesn't handle
    explicitly (plugin catalog, organization info, etc.) are forwarded here so
    they work transparently.
    """
    if not path.startswith("v1/"):
        raise HTTPException(status_code=404, detail="Not found")

    url = f"{_REAL_ANTHROPIC_API}/{path}"
    if request.url.query_string:
        url = f"{url}?{request.url.query_string.decode()}"

    body = await request.body()
    headers = _forward_headers(request)

    async with AsyncClient() as client:
        resp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body or None,
            timeout=30.0,
        )

    # Strip hop-by-hop and encoding headers that Starlette/httpx handle internally.
    response_headers = {
        name: value
        for name, value in resp.headers.items()
        if name.lower() not in _FORWARD_EXCLUDED_HEADERS
        and name.lower() not in ("content-encoding",)
    }

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=response_headers,
    )
