"""Hierarchical forecasting and reconciliation.

The problem, which is easy to state and easy to get wrong:

    total = 1000        <- forecast independently
      north = 400       <- forecast independently
      south = 550       <- forecast independently

400 + 550 = 950, not 1000. The regional plan and the national plan disagree by 50
units, and procurement, staffing and cash are now planned against two numbers that
cannot both be right. **Reconciliation** adjusts the forecasts so they add up.

Every method here takes `base` as either one number per node (a single step) or one
equal-length list per node (a multi-step forecast, reconciled step by step), and
returns the same shape. Base forecasts are validated: an unknown node name (a typo),
a missing required node, NaN or a mix of scalars and lists raises HierarchyError
instead of being read as zero.

  bottom_up       Sum the leaves upward. Coherent by construction; ignores every
                  forecast above leaf level.
  top_down        Split the root by per-leaf proportions (e.g. historical shares).
                  Cannot represent a leaf whose share is changing.
  mint            MinT-family least squares (Wickramasuriya et al. 2019) with a
                  diagonal covariance: `ols` (identity) or `wls_struct` (each node
                  weighted by its leaf count). The closest-to-base coherent forecast
                  in that metric. No covariance estimate from residuals (`mint_shrink`
                  needs numpy-scale linear algebra), so it is not the full MinT.
  weighted_blend  Heuristic: blend each parent with its children's sum bottom-up, then
                  push the disagreement down top-down. Coherent, uses every level, not
                  optimal in any statistical sense. `optimal` is its old name.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

Values = Mapping[str, Any]


class HierarchyError(ValueError):
    pass


@dataclass
class Node:
    name: str
    children: list[str] = field(default_factory=list)
    parent: str | None = None


@dataclass
class Hierarchy:
    """A tree of series. Leaves hold data; internal nodes are sums of their children."""

    nodes: dict[str, Node] = field(default_factory=dict)
    root: str = ""

    def add(self, name: str, *, parent: str | None = None) -> Node:
        if not isinstance(name, str) or not name:
            raise HierarchyError(f"node name must be a non-empty string, got {name!r}")
        if name in self.nodes:
            raise HierarchyError(f"{name!r} already in the hierarchy")
        if parent is None:
            if self.root:
                raise HierarchyError(f"hierarchy already has root {self.root!r}")
        elif parent not in self.nodes:
            raise HierarchyError(f"unknown parent {parent!r}")
        node = Node(name=name, parent=parent)
        self.nodes[name] = node
        if parent is None:
            self.root = name
        else:
            self.nodes[parent].children.append(name)
        return node

    @classmethod
    def from_paths(
        cls, paths: Iterable[Sequence[str]], *, root: str = "total", sep: str = "/"
    ) -> Hierarchy:
        """Build a tree from level paths such as `("north", "lahore")`.

        Node names are the path joined by `sep` ("north", "north/lahore"), so two
        stores with the same name in different regions stay distinct. An empty path
        list gives a single-node hierarchy (just the root).
        """
        h = cls()
        h.add(root)
        for path in paths:
            parent = root
            for depth in range(len(path)):
                part = str(path[depth])
                if not part:
                    raise HierarchyError(f"empty level value in path {tuple(path)!r}")
                name = sep.join(str(p) for p in path[: depth + 1])
                if name == root:
                    raise HierarchyError(f"level value {name!r} collides with the root name")
                if name not in h.nodes:
                    h.add(name, parent=parent)
                elif h.nodes[name].parent != parent:
                    raise HierarchyError(f"{name!r} appears under two parents")
                parent = name
        return h

    def _require_root(self) -> None:
        if not self.root:
            raise HierarchyError("the hierarchy is empty; add a root node first")

    def leaves(self) -> list[str]:
        return [n.name for n in self.nodes.values() if not n.children]

    def descendant_leaves(self, name: str) -> list[str]:
        if name not in self.nodes:
            raise HierarchyError(f"unknown node {name!r}")
        node = self.nodes[name]
        if not node.children:
            return [name]
        out: list[str] = []
        for child in node.children:
            out.extend(self.descendant_leaves(child))
        return out

    def levels(self) -> list[list[str]]:
        """Nodes grouped by depth, root first."""
        out: list[list[str]] = []
        current = [self.root] if self.root else []
        while current:
            out.append(current)
            nxt: list[str] = []
            for name in current:
                nxt.extend(self.nodes[name].children)
            current = nxt
        return out

    def aggregate(self, leaf_values: Values) -> dict[str, Any]:
        """Sum leaf values (scalars or equal-length lists) up to every node."""
        return bottom_up(self, leaf_values)

    def is_coherent(self, values: Values, *, tolerance: float = 1e-6) -> bool:
        """Does every parent equal the sum of its children, at every step?

        `tolerance` is relative to the parent's magnitude (absolute below 1.0). NaN
        anywhere is incoherent. Unknown or missing node names raise HierarchyError: a
        coherence check on the wrong keys is not a check.
        """
        self._require_root()
        table, _, _ = _normalize(self, values, required=set(self.nodes), allow_nan=True)
        for name, node in self.nodes.items():
            for step, own in enumerate(table[name]):
                if math.isnan(own):
                    return False
                if not node.children:
                    continue
                total = sum(table[c][step] for c in node.children)
                if math.isnan(total) or abs(own - total) > tolerance * max(1.0, abs(own)):
                    return False
        return True


# --- input validation ---------------------------------------------------------------


def _suggest(name: str, choices: Iterable[str]) -> str:
    choices = list(choices)
    folded = [c for c in choices if c.casefold() == name.casefold()]
    close = folded or difflib.get_close_matches(name, choices, n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _number(value: Any, where: str, *, allow_nan: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HierarchyError(f"{where} is {value!r}, not a number")
    f = float(value)
    if math.isnan(f) and allow_nan:
        return f
    if not math.isfinite(f):
        raise HierarchyError(f"{where} is {value!r}; forecasts must be finite numbers")
    return f


def _normalize(
    h: Hierarchy, values: Values, *, required: set[str], allow_nan: bool = False
) -> tuple[dict[str, list[float]], int, bool]:
    """Validate `values` against `h` and return (per-node lists, horizon, was_scalar)."""
    if not isinstance(values, Mapping):
        raise HierarchyError(f"expected a dict of node -> forecast, got {type(values).__name__}")
    unknown = [k for k in values if k not in h.nodes]
    if unknown:
        k = unknown[0]
        raise HierarchyError(f"{k!r} is not a node in the hierarchy{_suggest(str(k), h.nodes)}")
    missing = sorted(required - set(values))
    if missing:
        shown = ", ".join(repr(m) for m in missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise HierarchyError(f"no forecast for {shown}{more}")

    kinds = {isinstance(v, Sequence) and not isinstance(v, (str, bytes)) for v in values.values()}
    if len(kinds) > 1:
        raise HierarchyError("mix of single numbers and lists; give every node the same shape")
    scalar = not kinds or kinds == {False}
    table: dict[str, list[float]] = {}
    horizon = 1
    if scalar:
        for k, v in values.items():
            table[k] = [_number(v, f"forecast for {k!r}", allow_nan=allow_nan)]
    else:
        lengths = {len(v) for v in values.values()}
        if len(lengths) != 1 or 0 in lengths:
            raise HierarchyError(
                f"every node needs the same number of steps (>= 1); got lengths {sorted(lengths)}"
            )
        horizon = lengths.pop()
        for k, v in values.items():
            table[k] = [
                _number(x, f"forecast for {k!r} step {i}", allow_nan=allow_nan)
                for i, x in enumerate(v)
            ]
    return table, horizon, scalar


def _shape(table: dict[str, list[float]], scalar: bool) -> dict[str, Any]:
    return {k: (v[0] if scalar else v) for k, v in table.items()}


def _per_step(h: Hierarchy, table: dict[str, list[float]], horizon: int, fn) -> dict:
    out: dict[str, list[float]] = {name: [] for name in h.nodes}
    for step in range(horizon):
        one = {k: v[step] for k, v in table.items()}
        for k, v in fn(one).items():
            out[k].append(v)
    return out


# --- reconciliation -----------------------------------------------------------------


def _bottom_up_1(h: Hierarchy, leaf: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}

    def value(name: str) -> float:
        node = h.nodes[name]
        out[name] = leaf[name] if not node.children else sum(value(c) for c in node.children)
        return out[name]

    value(h.root)
    return out


def bottom_up(hierarchy: Hierarchy, base: Values) -> dict[str, Any]:
    """Sum the leaves upward. Needs a forecast for every leaf; forecasts given for
    internal nodes are accepted and ignored."""
    hierarchy._require_root()
    table, horizon, scalar = _normalize(hierarchy, base, required=set(hierarchy.leaves()))
    out = _per_step(hierarchy, table, horizon, lambda one: _bottom_up_1(hierarchy, one))
    return _shape(out, scalar)


def top_down(hierarchy: Hierarchy, base: Values, proportions: Mapping[str, float]) -> dict:
    """Split the root forecast by per-leaf proportions of the root.

    `proportions` must name every leaf and nothing else, be non-negative, and sum to
    a positive number (they are renormalised to 1). `historical_proportions` builds
    them from history. Only the root forecast is used.
    """
    hierarchy._require_root()
    leaves = hierarchy.leaves()
    leaf_set = set(leaves)
    bad = [k for k in proportions if k not in leaf_set]
    if bad:
        raise HierarchyError(
            f"proportion for {bad[0]!r}, which is not a leaf{_suggest(str(bad[0]), leaves)}"
        )
    missing = [leaf for leaf in leaves if leaf not in proportions]
    if missing:
        raise HierarchyError(f"no proportion for leaf {missing[0]!r}")
    props = {k: _number(v, f"proportion for {k!r}") for k, v in proportions.items()}
    negative = [k for k, v in props.items() if v < 0]
    if negative:
        raise HierarchyError(f"proportion for {negative[0]!r} is negative")
    total_share = sum(props.values())
    if total_share <= 0:
        raise HierarchyError("proportions must sum to a positive number")

    table, horizon, scalar = _normalize(hierarchy, base, required={hierarchy.root})
    root = table[hierarchy.root]
    leaf_table = {leaf: [r * props[leaf] / total_share for r in root] for leaf in leaves}
    out = _per_step(hierarchy, leaf_table, horizon, lambda one: _bottom_up_1(hierarchy, one))
    return _shape(out, scalar)


def historical_proportions(
    hierarchy: Hierarchy, history: Mapping[str, Sequence[float]]
) -> dict[str, float]:
    """Each leaf's share of the summed history. Needs every leaf's history.

    With no history at all (every sum zero) the split is equal: the only defensible
    default. A leaf whose history sums negative raises, since a negative share of the
    total is not a proportion.
    """
    from .models import clean_series

    hierarchy._require_root()
    leaves = hierarchy.leaves()
    unknown = [k for k in history if k not in hierarchy.nodes]
    if unknown:
        raise HierarchyError(
            f"{unknown[0]!r} is not a node in the hierarchy{_suggest(str(unknown[0]), leaves)}"
        )
    missing = [leaf for leaf in leaves if leaf not in history]
    if missing:
        raise HierarchyError(f"no history for leaf {missing[0]!r}")
    totals = {leaf: sum(clean_series(history[leaf], name=f"history[{leaf!r}]")) for leaf in leaves}
    negative = [k for k, v in totals.items() if v < 0]
    if negative:
        raise HierarchyError(f"history for {negative[0]!r} sums to {totals[negative[0]]}")
    grand = sum(totals.values())
    if grand <= 0:
        return {leaf: 1.0 / len(leaves) for leaf in leaves}
    return {leaf: totals[leaf] / grand for leaf in leaves}


def _cholesky_solver(a: list[list[float]]):
    """Factor a symmetric positive-definite matrix; return a solve(b) function."""
    n = len(a)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise HierarchyError("reconciliation matrix is not positive definite")
                lower[i][i] = math.sqrt(s)
            else:
                lower[i][j] = s / lower[j][j]

    def solve(b: list[float]) -> list[float]:
        y = [0.0] * n
        for i in range(n):
            y[i] = (b[i] - sum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]
        x = [0.0] * n
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(lower[k][i] * x[k] for k in range(i + 1, n))) / lower[i][i]
        return x

    return solve


def mint(hierarchy: Hierarchy, base: Values, *, method: str = "wls_struct") -> dict[str, Any]:
    """Least-squares reconciliation, MinT with a diagonal covariance.

    Finds leaf values `b` minimising `sum_i (base_i - (S b)_i)^2 / w_i` over every
    node `i`, where `(S b)_i` is node i's sum of leaves; the result `S b` is coherent.

      method="ols"         w_i = 1 for every node
      method="wls_struct"  w_i = number of leaves under node i, so an aggregate's
                           error is expected to be as large as the sum of its parts'

    Needs a forecast for every node. Solves an m-by-m system (m = number of leaves)
    once, then each step is a triangular solve: fine for hundreds of leaves, slow for
    tens of thousands. Like every MinT variant it can return negative values for a
    non-negative series when the base forecasts disagree badly.
    """
    hierarchy._require_root()
    if method not in ("ols", "wls_struct"):
        raise HierarchyError(f"method must be 'ols' or 'wls_struct', got {method!r}")
    table, horizon, scalar = _normalize(hierarchy, base, required=set(hierarchy.nodes))
    leaves = hierarchy.leaves()
    index = {leaf: j for j, leaf in enumerate(leaves)}
    under = {
        name: [index[x] for x in hierarchy.descendant_leaves(name)] for name in hierarchy.nodes
    }
    weight = {name: (1.0 if method == "ols" else float(len(under[name]))) for name in under}

    m = len(leaves)
    a = [[0.0] * m for _ in range(m)]
    for name, cols in under.items():
        inv = 1.0 / weight[name]
        for j in cols:
            row = a[j]
            for k in cols:
                row[k] += inv
    solve = _cholesky_solver(a)

    out: dict[str, list[float]] = {name: [] for name in hierarchy.nodes}
    for step in range(horizon):
        b = [0.0] * m
        for name, cols in under.items():
            contribution = table[name][step] / weight[name]
            for j in cols:
                b[j] += contribution
        x = solve(b)
        for name, cols in under.items():
            out[name].append(sum(x[j] for j in cols))
    return _shape(out, scalar)


def _blend_1(
    h: Hierarchy, base: Mapping[str, float], weights: Mapping[str, float]
) -> dict[str, float]:
    blended: dict[str, float] = {}

    def blend(name: str) -> float:
        node = h.nodes[name]
        if not node.children:
            blended[name] = base[name]
            return blended[name]
        child_total = sum(blend(c) for c in node.children)
        w = weights.get(name, 1.0)
        # Weight 0 means the node's own forecast is ignored: it becomes its children.
        blended[name] = child_total if w == 0 else (w * base[name] + child_total) / (w + 1)
        return blended[name]

    blend(h.root)
    out: dict[str, float] = {h.root: blended[h.root]}

    def distribute(name: str) -> None:
        children = h.nodes[name].children
        if not children:
            return
        values = [blended[c] for c in children]
        child_total = sum(values)
        same_sign = all(v >= 0 for v in values) or all(v <= 0 for v in values)
        if same_sign and child_total != 0:
            # Proportional: a large child absorbs more of the adjustment than a small one.
            scale = out[name] / child_total
            for c in children:
                out[c] = blended[c] * scale
        else:
            # Mixed signs (net sales with a store in net returns) or all zero: a
            # proportional scale would divide by a near-zero net total and explode.
            # Spread the difference additively, in proportion to each child's size.
            diff = out[name] - child_total
            magnitude = sum(abs(v) for v in values)
            for c, v in zip(children, values, strict=True):
                share = abs(v) / magnitude if magnitude else 1 / len(children)
                out[c] = v + diff * share
        for c in children:
            distribute(c)

    distribute(h.root)
    return out


def weighted_blend(
    hierarchy: Hierarchy, base: Values, *, weights: Mapping[str, float] | None = None
) -> dict[str, Any]:
    """Blend every level's forecast, then distribute the disagreement. A heuristic.

    Pass 1 (bottom-up): each internal node becomes `(w * own + children_sum) / (w + 1)`,
    so information from every level reaches the root. Pass 2 (top-down): fix the root,
    then scale each node's children to match their parent, preserving the blend's
    proportions (additively when the children have mixed signs). The order matters: a
    single top-down pass overwrites the value each node's parent just fixed.

    `weights` (default 1.0 per internal node) says how much to trust a node's own
    forecast; 0 ignores it, which makes the whole method exactly `bottom_up`. Needs a
    forecast for every leaf and every internal node with non-zero weight.
    """
    hierarchy._require_root()
    weights = dict(weights or {})
    for k, v in weights.items():
        if k not in hierarchy.nodes:
            raise HierarchyError(
                f"weight for unknown node {k!r}{_suggest(str(k), hierarchy.nodes)}"
            )
        if _number(v, f"weight for {k!r}") < 0:
            raise HierarchyError(f"weight for {k!r} is negative")
    required = {
        name
        for name, node in hierarchy.nodes.items()
        if not node.children or weights.get(name, 1.0) > 0
    }
    table, horizon, scalar = _normalize(hierarchy, base, required=required)
    for name in hierarchy.nodes:
        table.setdefault(name, [0.0] * horizon)  # weight-0 nodes: value is never read
    out = _per_step(hierarchy, table, horizon, lambda one: _blend_1(hierarchy, one, weights))
    return _shape(out, scalar)


# Kept for code written against 0.1.0. The method is a heuristic blend, not MinT.
optimal = weighted_blend

METHODS = ("bottom_up", "top_down", "mint_ols", "mint_wls", "weighted_blend")


def reconcile(
    hierarchy: Hierarchy,
    base: Values,
    method: str,
    *,
    history: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Dispatch by name, for configs and the CLI. `top_down` needs `history`."""
    if method == "bottom_up":
        return bottom_up(hierarchy, base)
    if method == "top_down":
        if history is None:
            raise HierarchyError("top_down needs history to compute proportions")
        leaf_history = {leaf: history[leaf] for leaf in hierarchy.leaves() if leaf in history}
        return top_down(hierarchy, base, historical_proportions(hierarchy, leaf_history))
    if method == "mint_ols":
        return mint(hierarchy, base, method="ols")
    if method == "mint_wls":
        return mint(hierarchy, base, method="wls_struct")
    if method in ("weighted_blend", "optimal"):
        return weighted_blend(hierarchy, base)
    raise HierarchyError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
