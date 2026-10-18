"""Browser Use Cloud API v2 tasks. Checked against docs.browser-use.com.

The key for this project was verified on ``GET /api/v2/billing/account``.
Catalog collection uses the v2 task API:

``POST /api/v2/tasks`` with ``structuredOutput`` as a stringified JSON schema,
poll ``GET /api/v2/tasks/{task_id}/status`` until ``finished``, ``failed``, or
``stopped``, then ``GET /api/v2/tasks/{task_id}``. Auth header is
``X-Browser-Use-API-Key`` with no Bearer prefix. This client never places an
order or types a payment.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from geneai_core.domains.shopping import products_from_browser_output
from geneai_core.providers.base import CredentialsError, ProviderError

API_ORIGIN = "https://api.browser-use.com/api/v2"
TERMINAL = frozenset({"finished", "failed", "stopped"})

CATALOG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["products"],
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "title",
                    "price",
                    "product_url",
                    "image_url",
                    "retailer",
                    "availability",
                    "role",
                    "colors",
                    "style",
                ],
                "properties": {
                    "title": {"type": "string"},
                    "price": {"type": "number"},
                    "currency": {"type": "string"},
                    "product_url": {"type": "string"},
                    "image_url": {"type": "string"},
                    "retailer": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["top", "bottom", "shoes", "layer", "accessory"],
                    },
                    "colors": {"type": "array", "items": {"type": "string"}},
                    "style": {"type": "array", "items": {"type": "string"}},
                    "occasions": {"type": "array", "items": {"type": "string"}},
                    "sizes": {"type": "array", "items": {"type": "string"}},
                    "availability": {"type": "string"},
                },
            },
        }
    },
}


def browser_use_blocker_message() -> str:
    return "\n".join(
        [
            "BLOCKED: Browser Use catalog collection was not executed. No browser call was made.",
            "Missing environment variable: BROWSER_USE_API_KEY.",
            "Live shopping also needs --catalog-url pointing at a public page you are allowed to read.",
            "This client does not default to a retailer, does not invent product URLs, and does not check out.",
            "Setup:",
            "1. Create a key at https://cloud.browser-use.com/settings (documented prefix bu_).",
            "2. Export BROWSER_USE_API_KEY.",
            "3. Re-run: uv run python scripts/run_domain.py --domain shopping --mode live --catalog-url https://example.com/catalog",
            "API: POST https://api.browser-use.com/api/v2/tasks",
            "     GET  https://api.browser-use.com/api/v2/tasks/{task_id}/status",
            "     GET  https://api.browser-use.com/api/v2/tasks/{task_id}",
            "     GET  https://api.browser-use.com/api/v2/billing/account",
            "Header: X-Browser-Use-API-Key (not Bearer). structuredOutput is a JSON string.",
        ]
    )


class BrowserUseClient:
    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 180.0,
        poll_s: float = 2.0,
        sleeper: Any | None = None,
    ) -> None:
        if not api_key:
            raise CredentialsError("BROWSER_USE_API_KEY is missing. No request was sent.")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._client = client
        self._sleeper = sleeper

    @classmethod
    def from_env(cls) -> BrowserUseClient:
        key = os.environ.get("BROWSER_USE_API_KEY", "").strip()
        if not key:
            raise CredentialsError(browser_use_blocker_message())
        return cls(api_key=key)

    def _headers(self) -> dict[str, str]:
        return {
            "X-Browser-Use-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def collect_catalog(self, *, catalog_url: str, query: str) -> dict[str, Any]:
        if not catalog_url.startswith("https://") and not catalog_url.startswith("http://"):
            raise ProviderError("catalog_url must be an http(s) URL. No request was sent.")
        task = (
            "Read the public catalog page. Collect 30 to 40 products spread across roles. "
            "Each role must be one of top, bottom, shoes, layer, accessory. "
            "For every product return title, price, product_url, image_url, retailer, availability, "
            "role, colors, and style tags. Include occasion tags when the page states them. "
            "Do not sign in, do not add to cart, and do not enter payment. "
            "product_url and image_url must be the URLs you opened. "
            f"Page: {catalog_url}. Request: {query}"
        )
        created = await self._request(
            "POST",
            f"{API_ORIGIN}/tasks",
            {
                "task": task,
                "startUrl": catalog_url,
                "structuredOutput": json.dumps(CATALOG_SCHEMA),
                "maxSteps": 40,
            },
        )
        task_id = created.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ProviderError("Browser Use create response did not include id")
        session_id = created.get("sessionId") if isinstance(created.get("sessionId"), str) else None
        document = await self._wait(task_id)
        status = str(document.get("status") or "")
        if status != "finished":
            raise ProviderError(
                f"Browser Use task {task_id} ended {status}",
                provider_run_id=task_id,
                code=str(document.get("error") or status or ""),
            )
        output = document.get("output")
        catalog = products_from_browser_output(
            output,
            run_id=task_id,
            session_id=session_id or (document.get("sessionId") if isinstance(document.get("sessionId"), str) else None),
            model=document.get("llm") if isinstance(document.get("llm"), str) else document.get("model") if isinstance(document.get("model"), str) else None,
        )
        catalog["captured_at"] = document.get("finishedAt")
        catalog["total_cost_usd"] = document.get("cost")
        catalog["browser_use_run_id"] = task_id
        if not catalog["products"]:
            raise ProviderError(
                f"Browser Use task {task_id} returned no products with an http(s) product_url. "
                "The fixture catalog was not substituted.",
                provider_run_id=task_id,
                code="EMPTY_CATALOG",
            )
        return catalog

    async def _wait(self, task_id: str) -> dict[str, Any]:
        import time

        deadline = time.monotonic() + self.timeout_s
        status_url = f"{API_ORIGIN}/tasks/{task_id}/status"
        while True:
            status_doc = await self._request("GET", status_url, None)
            status = str(status_doc.get("status") or "")
            if status in TERMINAL:
                return await self._request("GET", f"{API_ORIGIN}/tasks/{task_id}", None)
            if time.monotonic() >= deadline:
                raise ProviderError(
                    f"Browser Use task {task_id} did not finish within {self.timeout_s:.0f}s",
                    provider_run_id=task_id,
                    code="TIMEOUT",
                    retryable=True,
                )
            await self._sleep(self.poll_s)

    async def _request(self, method: str, url: str, body: dict[str, Any] | None) -> dict[str, Any]:
        if not url.startswith(API_ORIGIN + "/") and url != API_ORIGIN:
            raise ProviderError("refusing a non Browser Use v2 URL")
        owns = self._client is None
        client = self._client or httpx.AsyncClient(timeout=60.0)
        try:
            response = await client.request(method, url, headers=self._headers(), json=body)
        finally:
            if owns:
                await client.aclose()
        if response.status_code >= 400:
            raise ProviderError(
                f"Browser Use {method} {response.status_code}",
                status_code=response.status_code,
                retryable=response.status_code in {429, 503},
            )
        document = response.json()
        if not isinstance(document, dict):
            raise ProviderError("Browser Use response was not an object")
        return document

    async def _sleep(self, seconds: float) -> None:
        if self._sleeper is not None:
            await self._sleeper(seconds)
            return
        import asyncio

        await asyncio.sleep(seconds)
