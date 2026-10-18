"""GeneAI evolution core.

The controller is domain-agnostic. Concrete domains register themselves from
application entrypoints; this package does not import them.
"""

from geneai_core.schemas import (
    Evaluation,
    EvolutionConfig,
    Genome,
    ObjectiveSpec,
    RunMode,
)

__all__ = [
    "Evaluation",
    "EvolutionConfig",
    "Genome",
    "ObjectiveSpec",
    "RunMode",
]

__version__ = "0.1.0"
