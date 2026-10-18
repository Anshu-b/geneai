"""In-process domain registry. Registration happens outside the controller."""

from __future__ import annotations

from geneai_core.domains.base import DomainAdapter


class DomainRegistry:
    _adapters: dict[str, DomainAdapter] = {}

    @classmethod
    def register(cls, adapter: DomainAdapter) -> None:
        if not getattr(adapter, "name", None):
            raise ValueError("adapter.name is required")
        cls._adapters[adapter.name] = adapter

    @classmethod
    def get(cls, name: str) -> DomainAdapter:
        try:
            return cls._adapters[name]
        except KeyError as exc:
            known = ", ".join(sorted(cls._adapters)) or "(none)"
            raise KeyError(f"unknown domain {name!r}; registered: {known}") from exc

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._adapters)

    @classmethod
    def clear(cls) -> None:
        cls._adapters = {}
