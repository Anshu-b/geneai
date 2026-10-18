"""Domain plugins.

Concrete adapters are imported by entrypoints (CLI, tests). The evolution
controller does not import this package's siblings.
"""

from geneai_core.domains.base import DomainAdapter

__all__ = ["DomainAdapter"]
