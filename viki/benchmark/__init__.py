"""
viki.benchmark
--------------
External-dataset ingestion for validating ViKi's perception tract against a
ground-truthed source. Nothing here is part of ``viki run``; these modules only
read a published dataset and emit standard capture-time episodes plus
ground-truth arrays that the existing ``extract -> prepare -> hand-fit`` tract
can consume unmodified.

Currently one importer: :mod:`viki.benchmark.hocap` (HO-Cap, arXiv:2406.06843,
CC BY 4.0).
"""

from __future__ import annotations

from viki.benchmark.hocap import ImportResult, import_sequence, render_report

__all__ = ["ImportResult", "import_sequence", "render_report"]
