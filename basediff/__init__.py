"""basediff: what breaks when a fine-tuned specialist moves to a new base model.

Rebuild recipes, a per-example migration diff (fixes, negative flips, copied errors,
per-slice changes against a seed-noise floor) and a SHIP / REVIEW / BLOCK verdict.
"""
__version__ = "0.1.0"
