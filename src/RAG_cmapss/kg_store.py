from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import networkx as nx

from .config import KG_TRIPLE_FILES


class KGStore:
    def __init__(self, kg_dir: str | Path):
        self.kg_dir = Path(kg_dir)
        self.graph = nx.MultiDiGraph()
        self.load_all()

    def load_triples(self, filename: str) -> None:
        path = self.kg_dir / filename
        if not path.exists():
            return

        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            required = {"head", "relation", "tail"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")

            for row in reader:
                head = str(row["head"]).strip()
                relation = str(row["relation"]).strip()
                tail = str(row["tail"]).strip()
                if not head or not relation or not tail:
                    continue
                weight = _safe_float(row.get("weight"), default=1.0)
                self.graph.add_node(head)
                self.graph.add_node(tail)
                self.graph.add_edge(
                    head,
                    tail,
                    key=relation,
                    relation=relation,
                    weight=weight,
                    source=str(row.get("source", "unknown")),
                    notes=str(row.get("notes", "")),
                    file=filename,
                )

    def load_all(self) -> None:
        for filename in KG_TRIPLE_FILES:
            self.load_triples(filename)

    def outgoing(
        self,
        node: str,
        relation: str | None = None,
        min_weight: float = 0.0,
        include_zero: bool = False,
    ) -> list[dict[str, Any]]:
        if node not in self.graph:
            return []

        results: list[dict[str, Any]] = []
        for _, tail, _key, data in self.graph.out_edges(node, keys=True, data=True):
            edge_relation = str(data.get("relation", ""))
            weight = float(data.get("weight", 1.0))
            if relation is not None and edge_relation != relation:
                continue
            if include_zero:
                if weight < min_weight:
                    continue
            elif weight <= min_weight:
                continue
            results.append(
                {
                    "head": node,
                    "relation": edge_relation,
                    "tail": tail,
                    "weight": weight,
                    "source": data.get("source", "unknown"),
                    "notes": data.get("notes", ""),
                    "file": data.get("file", ""),
                }
            )
        return results


def _safe_float(value: object, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

