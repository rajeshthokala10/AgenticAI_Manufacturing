"""RAGAS-inspired offline evaluation harness for the Hybrid GraphRAG pipeline.

Use ``python -m comparison.eval.run`` to score the three pipelines against
the bundled golden Q&A set on:

* ``faithfulness``       — claims in the answer that are entailed by the evidence.
* ``answer_relevancy``   — how well the answer addresses the query.
* ``context_precision``  — fraction of retrieved chunks that look relevant.
* ``citation_accuracy``  — cited chunk ids that actually appear in the evidence.
* ``guardrail_pass_rate``— share of answers passing the deterministic guardrails.

See :mod:`comparison.eval.metrics` for the implementations and
:mod:`comparison.eval.golden` for the curated dataset.
"""

from comparison.eval.harness import EvalHarness, EvalResult  # noqa: F401
from comparison.eval.golden import GoldenItem, load_golden_set  # noqa: F401
from comparison.eval.metrics import score_record  # noqa: F401
