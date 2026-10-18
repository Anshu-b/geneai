"""Browser Use Cloud API v4. Checked 2026-10-18 against docs.browser-use.com.

Create ``POST /api/v4/runs``, poll ``GET /api/v4/runs/{id}/status``, then
``GET /api/v4/runs/{id}``. Auth header is ``X-Browser-Use-API-Key`` with no
Bearer prefix. This client never places an order or types a payment.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from geneai_core.domains.shopping import products_from_browser_output
from geneai_core.providers.base import CredentialsError, ProviderError

API_ORIGIN = "https://api.browser-use.com/api/v4"
TERMINAL = frozenset({"completed", "failed", "cancelled"})

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
                "required": ["title", "price", "currency", "product_url", "role"],
                "properties": {
                    "title": {"type": "string"},
                    "price": {"type": "number"},
                    "currency": {"type": "string"},
                    "product_url": {"type": "string"},
                    "image_url": {"type": "string"},
                    "role": {"type": "string"},
                    "colors": {"type": "array", "items": {"type": "string"}},
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
            "API: POST https://api.browser-use.com/api/v4/runs",
            "     GET  https://api.browser-use.com/api/v4/runs/{id}/status",
            "     GET  https://api.browser-use.com/api/v4/runs/{id}",
            "Header: X-Browser-Use-API-Key (not Bearer).",
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
            "Read the public catalog page and list apparel products you can actually open. "
            "Do not sign in, do not add to cart, and do not enter payment. "
            "Return only products whose product_url is the page you opened. "
            f"Page: {catalog_url}. Request: {query}"
        )
        created = await self._request("POST", f"{API_ORIGIN}/runs", {"task": task, "outputSchema": CATALOG_SCHEMA})
        run_id = created.get("id")
        if not isinstance(run_id, str) or not run_id:
            raise ProviderError("Browser Use create response did not include id")
        document = await self._wait(run_id)
        if document.get("status") != "completed":
            raise ProviderError(
                f"Browser Use run {run_id} ended {document.get('status')}",
                provider_run_id=run_id,
                code=str(document.get("error") or document.get("status") or ""),
            )
        output = document.get("output")
        if output is None:
            output = document.get("result")
        catalog = products_from_browser_output(
            output,
            run_id=run_id,
            session_id=created.get("sessionId") if isinstance(created.get("sessionId"), str) else document.get("sessionId"),
            model=document.get("model") if isinstance(document.get("model"), str) else created.get("model"),
        )
        catalog["captured_at"] = document.get("updatedAt")
        catalog["total_cost_usd"] = document.get("totalCostUsd")
        catalog["browser_use_run_id"] = run_id
        if not catalog["products"]:
            raise ProviderError(
                f"Browser Use run {run_id} returned no products with an http(s) product_url. "
                "The fixture catalog was not substituted.",
                provider_run_id=run_id,
                code="EMPTY_CATALOG",
            )
        return catalog

    async def _wait(self, run_id: str) -> dict[str, Any]:
        import time

        deadline = time.monotonic() + self.timeout_s
        status_url = f"{API_ORIGIN}/runs/{run_id}/status"
        while True:
            status_doc = await self._request("GET", status_url, None)
            status = str(status_doc.get("status") or "")
            if status in TERMINAL:
                return await self._request("GET", f"{API_ORIGIN}/runs/{run_id}", None)
            if time.monotonic() >= deadline:
                raise ProviderError(
                    f"Browser Use run {run_id} did not finish within {self.timeout_s:.0f}s",
                    provider_run_id=run_id,
                    code="TIMEOUT",
                    retryable=True,
                )
            await self._sleep(self.poll_s)

    async def _request(self, method: str, url: str, body: dict[str, Any] | None) -> dict[str, Any]:
        if not url.startswith(API_ORIGIN):
            raise ProviderError("refusing a non Browser Use v4 URL")
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
