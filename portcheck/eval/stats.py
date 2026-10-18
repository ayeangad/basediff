"""Paired statistics used everywhere (PORT_PLAN v2 §5.4). All models answer the same items."""
import math
import random

from scipy.stats import binomtest

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts
    (b = old right / new wrong, c = old wrong / new right)."""
    if b + c == 0:
        return 1.0
    return binomtest(b, b + c, 0.5).pvalue


def paired_bootstrap(x: list[bool], y: list[bool], n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean difference y - x over paired items with a percentile 95% CI."""
    assert len(x) == len(y)
    n = len(x)
    rng = random.Random(seed)
    diffs = [int(b) - int(a) for a, b in zip(x, y)]
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"diff": sum(diffs) / n, "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot) - 1]}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (monotone)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[k] = running
    return out
