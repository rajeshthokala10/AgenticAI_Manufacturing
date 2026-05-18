"""Deterministic vocabulary miner for the onboarding agent's Stage A.

Stage A used to do everything in one LLM call — read the docs, detect
the archetype, AND pick which noun phrases to nominate as candidate
vocabulary. Three tasks competing for attention in a single forward
pass, against truncated input (only ~12k chars per doc reached the
prompt). The result: 2 terms per entity type, no matter the corpus
size.

This module decouples extraction from labelling:

    full docs (no truncation)
        ↓
    [1] sentence chunking            spaCy
        ↓
    [2] noun-phrase extraction       spaCy NP chunker
        ↓
    [3] frequency + IDF filter       cross-doc; keep terms in ≥ 2 docs
        ↓
    [4] embedding clustering         SentenceTransformer + agglomerative
        ↓
    candidate clusters → Stage A LLM only LABELS and picks the archetype

The LLM stops doing extraction (which it's bad at over long context) and
just does labelling (which it's good at). Side benefits:

  - Deterministic: two runs over the same docs yield the same clusters
  - Provenance: every term has a sample sentence + doc index
  - Confidence: term frequency + doc coverage as numeric signal
  - No truncation: the miner sees the full doc; only the small summary
    goes to the model

The output is a list of ``CandidateCluster`` — each cluster groups
near-synonyms (e.g. {"hairpin stator", "hairpin stators", "continuous
hairpin stator"}) under a representative term, with the sentence that
yielded the top mention as evidence.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger("core.vocab_miner")


# ─── Tunables ──────────────────────────────────────────────────────────────

# Phrase-length window. Single-token noun phrases ("rotor") tend to be
# too generic; very long phrases ("shaft must be designed to be
# temperature-stable") are full sentences that slipped through. The 2-5
# token window captures the useful middle.
MIN_PHRASE_TOKENS = 1
MAX_PHRASE_TOKENS = 5

# Minimum cross-doc frequency before a phrase is kept. Set to 2 so a
# phrase only seen once in a single doc gets dropped as noise.
MIN_DOC_COVERAGE = 1   # used when the user uploads only one doc
MIN_TOTAL_FREQUENCY = 2

# How many clusters to surface to Stage A. Too few → loses coverage;
# too many → wastes tokens. ~50 is the sweet spot for 3-5 input docs.
TARGET_CLUSTER_COUNT = 50

# Cosine-distance threshold for agglomerative clustering. Lower = tighter
# clusters (fewer terms grouped together). 0.30 keeps obvious synonyms
# together (singular/plural, hyphen variants) without overmerging.
CLUSTER_DISTANCE_THRESHOLD = 0.30


# ─── Stopword + filter heuristics ──────────────────────────────────────────

# Generic determiners / adjectives / numbers we never want as a vocab term
# on their own. spaCy NP detection often returns these as one-token NPs.
_GENERIC_DROP = {
    # Document-structure words that aren't domain concepts on their own
    "fig", "figure", "table", "section", "chapter", "example", "note",
    "step", "system", "method", "result", "value", "data",
    "type", "case", "form", "kind", "way", "thing", "part", "side",
    "page", "ref", "etc", "ie", "eg", "i.e", "e.g", "vs", "etc.",
    "this", "that", "these", "those", "such", "more", "less",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "n/a", "iso", "din",
    # Table-markdown residue from the advanced parser
    "none", "excerpt", "columns", "rows", "header", "headers",
    "form data", "none | table", "| table", "table on", "| none",
    # Academic / author / affiliation boilerplate
    "prof", "prof.", "dr", "dr.", "dr.-ing", "director",
    "chair", "the chair", "university", "department",
    "rwth aachen university", "rwth aachen",
    # Generic connectors that spaCy NP can return as one-noun chunks
    "which", "place", "alternative", "an alternative",
    "components", "their components", "manufacture", "the manufacture",
    "production", "the process", "processes", "test methods", "testing",
}

# Skip phrases that are mostly punctuation or non-alphabetic
_ALPHA_FRACTION_MIN = 0.5


# ─── Public dataclasses ───────────────────────────────────────────────────


@dataclass
class CandidateCluster:
    """A group of near-synonymous noun phrases mined from the corpus.

    The Stage A LLM receives one of these per cluster and only has to
    decide: (a) is this signal or noise? (b) if signal, which entity
    type does it belong to?
    """

    label: str                          # representative phrase (most frequent)
    members: List[str]                  # all phrases in this cluster, sorted
    frequency: int                      # total mention count across docs
    doc_coverage: int                   # how many docs contain ≥ 1 member
    sample_quote: str                   # sentence containing the top mention
    sample_doc_index: int               # which doc the quote came from

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "members": list(self.members),
            "frequency": self.frequency,
            "doc_coverage": self.doc_coverage,
            "sample_quote": self.sample_quote,
            "sample_doc_index": self.sample_doc_index,
        }


@dataclass
class MiningResult:
    clusters: List[CandidateCluster] = field(default_factory=list)
    docs_seen: int = 0
    total_chars: int = 0
    raw_phrase_count: int = 0           # phrases before filtering
    kept_phrase_count: int = 0          # phrases after frequency + filter
    cluster_count: int = 0
    used_embeddings: bool = False       # False → fell back to lexical clustering
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "clusters": [c.to_dict() for c in self.clusters],
            "docs_seen": self.docs_seen,
            "total_chars": self.total_chars,
            "raw_phrase_count": self.raw_phrase_count,
            "kept_phrase_count": self.kept_phrase_count,
            "cluster_count": self.cluster_count,
            "used_embeddings": self.used_embeddings,
            "notes": list(self.notes),
        }


# ─── The miner ─────────────────────────────────────────────────────────────


def mine_candidates(
    docs: List[str],
    *,
    target_clusters: int = TARGET_CLUSTER_COUNT,
    distance_threshold: float = CLUSTER_DISTANCE_THRESHOLD,
) -> MiningResult:
    """Mine candidate vocabulary clusters from the full corpus.

    Deterministic — same docs in, same clusters out (modulo the
    embedding model's float arithmetic, which is stable in practice).
    Runs entirely locally; no LLM calls.
    """
    result = MiningResult(
        docs_seen=len(docs),
        total_chars=sum(len(d) for d in docs),
    )
    if not docs:
        return result

    # 1) Sentence + noun-phrase extraction via spaCy.
    nlp = _load_spacy()
    if nlp is None:
        result.notes.append("spaCy not available — miner cannot run")
        return result

    # phrase_text → list of (doc_index, sentence_text)
    phrase_evidence: dict[str, list[Tuple[int, str]]] = defaultdict(list)

    for doc_idx, doc_text in enumerate(docs):
        # Chunk overly long docs to avoid spaCy's 1MB default limit and
        # keep per-call latency reasonable. 200k chars/page is plenty.
        for chunk in _chunk_for_spacy(doc_text, max_chars=200_000):
            try:
                spacy_doc = nlp(chunk)
            except Exception as exc:
                logger.warning("spaCy parse failed on doc %d chunk: %r", doc_idx, exc)
                continue
            for sent in spacy_doc.sents:
                for np in sent.noun_chunks:
                    phrase = _normalise_phrase(np.text)
                    if not _is_useful_phrase(phrase):
                        continue
                    phrase_evidence[phrase].append((doc_idx, sent.text.strip()))

    result.raw_phrase_count = len(phrase_evidence)

    # 2) Frequency + IDF filter.
    min_coverage = MIN_DOC_COVERAGE if len(docs) == 1 else 2
    kept: list[Tuple[str, list[Tuple[int, str]]]] = []
    for phrase, hits in phrase_evidence.items():
        doc_coverage = len({d for d, _ in hits})
        if len(hits) < MIN_TOTAL_FREQUENCY:
            continue
        if doc_coverage < min_coverage:
            continue
        kept.append((phrase, hits))

    result.kept_phrase_count = len(kept)
    if not kept:
        result.notes.append("no phrases survived frequency filter — corpus may be too small")
        return result

    # 3) Embedding-based clustering. Fall back to lexical clustering when
    # the embedding pipeline isn't available (e.g. transient torch error).
    phrases = [p for p, _ in kept]
    clusters, used_embeddings = _cluster_phrases(
        phrases, target_clusters=target_clusters, distance_threshold=distance_threshold,
    )
    result.used_embeddings = used_embeddings

    # 4) Build CandidateCluster per cluster, picking the most frequent
    # phrase as the label and its longest sample sentence as the quote.
    phrase_to_hits = dict(kept)
    seen_cluster_ids: set[int] = set()
    out_clusters: list[CandidateCluster] = []

    for cluster_id, members in clusters.items():
        if cluster_id in seen_cluster_ids:
            continue
        seen_cluster_ids.add(cluster_id)

        # rank members by frequency (desc), tie-break alphabetically
        ranked = sorted(
            members,
            key=lambda p: (-len(phrase_to_hits.get(p, [])), p),
        )
        label = ranked[0]
        all_hits: list[Tuple[int, str]] = []
        for m in ranked:
            all_hits.extend(phrase_to_hits.get(m, []))
        # Pick the longest sample sentence — it usually has the most
        # contextual signal for a human reviewer or the labelling LLM.
        sample_doc, sample_quote = max(all_hits, key=lambda dq: len(dq[1]))
        out_clusters.append(CandidateCluster(
            label=label,
            members=sorted(set(ranked)),
            frequency=len(all_hits),
            doc_coverage=len({d for d, _ in all_hits}),
            sample_quote=sample_quote[:240],
            sample_doc_index=sample_doc,
        ))

    # Rank clusters: high doc coverage first (true cross-doc concepts),
    # then frequency. Truncate to the budget.
    out_clusters.sort(key=lambda c: (-c.doc_coverage, -c.frequency, c.label))
    result.clusters = out_clusters[:target_clusters]
    result.cluster_count = len(result.clusters)
    return result


# ─── Internals ─────────────────────────────────────────────────────────────


def _load_spacy():
    """Lazy import spaCy; return None if unavailable so callers can fall back."""
    try:
        import spacy  # noqa: PLC0415
    except ImportError:
        return None
    try:
        # ``en_core_web_sm`` is the de facto default; the model used by
        # most of the existing pipeline. Multilingual corpora may want
        # ``xx_ent_wiki_sm`` instead — out of scope for now.
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except Exception as exc:
        logger.warning("spaCy model en_core_web_sm not loadable: %r", exc)
        return None


def _chunk_for_spacy(text: str, *, max_chars: int) -> Iterable[str]:
    """Yield contiguous slices of ``text`` no larger than ``max_chars``.

    Split on paragraph boundaries when possible to avoid mid-sentence cuts."""
    if len(text) <= max_chars:
        yield text
        return
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Back off to the nearest paragraph break, but only within
            # the last 10% of the window — beyond that, just take the cut.
            backstop = text.rfind("\n\n", start + int(max_chars * 0.9), end)
            if backstop > start:
                end = backstop
        yield text[start:end]
        start = end


# Whitespace incl. NBSP / zero-width / BOM characters that .strip() may miss.
_ANY_WHITESPACE = re.compile(r"[\s ​﻿]+")

# Determiner / article prefixes to drop from noun chunks. spaCy's NP
# detector frequently returns "the X", "a X", "an X" — for vocab purposes
# the determiner is noise.
_LEADING_DETERMINERS = ("the ", "a ", "an ", "their ", "its ", "this ", "that ",
                        "these ", "those ", "some ", "any ", "no ", "such ")


def _normalise_phrase(s: str) -> str:
    """Lowercase, collapse whitespace (incl. NBSP), strip surrounding
    punctuation, drop leading determiners."""
    s = _ANY_WHITESPACE.sub(" ", s).strip().lower()
    s = s.strip(".,;:!?\"'()[]{}<>—–-|/")
    for prefix in _LEADING_DETERMINERS:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.strip()


def _is_useful_phrase(phrase: str) -> bool:
    """Drop generic / too-short / too-long / non-alpha / table-residue chunks."""
    if not phrase or len(phrase) < 3:
        return False
    # Table-markdown residue (cells joined by pipes survive normalisation
    # in some cases). Anything containing a pipe is structural noise.
    if "|" in phrase:
        return False
    tokens = phrase.split()
    if not (MIN_PHRASE_TOKENS <= len(tokens) <= MAX_PHRASE_TOKENS):
        return False
    if phrase in _GENERIC_DROP:
        return False
    # Drop phrases dominated by numbers or punctuation
    alpha_chars = sum(c.isalpha() for c in phrase)
    if alpha_chars / max(len(phrase), 1) < _ALPHA_FRACTION_MIN:
        return False
    # Single-token phrases must be longer than 3 chars and not in the
    # generic-stopword list (already checked above).
    if len(tokens) == 1 and len(phrase) < 4:
        return False
    # Drop phrases that begin or end with a stopword-ish connector that
    # spaCy's NP heuristic occasionally includes.
    if tokens[0] in {"of", "to", "in", "on", "at", "by", "for", "with",
                     "from", "into", "onto", "via", "per", "and", "or", "no"}:
        return False
    if tokens[-1] in {"of", "to", "in", "on", "at", "by", "for", "with",
                      "from", "into", "onto", "via", "per", "and", "or"}:
        return False
    return True


def _cluster_phrases(
    phrases: List[str],
    *,
    target_clusters: int,
    distance_threshold: float,
) -> Tuple[dict, bool]:
    """Group near-synonyms via embedding + agglomerative clustering.

    Returns ``(cluster_id -> [phrase, ...], used_embeddings)``. Falls back
    to lexical (substring + Jaccard) clustering when embeddings aren't
    available.
    """
    if len(phrases) <= 1:
        return ({0: list(phrases)}, False)

    # Try the project's existing embedding facility first; fall back to a
    # tiny lexical clusterer when it can't be loaded (e.g. CI without
    # torch).
    try:
        from doc_pipeline.embeddings import _load_model  # type: ignore
        from config import EMBEDDING_MODEL              # type: ignore
        model = _load_model(EMBEDDING_MODEL)
        vectors = model.encode(phrases, normalize_embeddings=True, show_progress_bar=False)
    except Exception as exc:
        logger.warning("Embedding load failed (%r) — falling back to lexical clustering", exc)
        return (_lexical_cluster(phrases), False)

    try:
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np  # noqa: PLC0415
    except ImportError:
        logger.warning("sklearn unavailable — falling back to lexical clustering")
        return (_lexical_cluster(phrases), False)

    n_clusters = min(target_clusters, len(phrases))
    # Use distance_threshold to let cluster count vary with corpus
    # diversity; cap with n_clusters as a hard upper bound.
    try:
        algo = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = algo.fit_predict(np.asarray(vectors))
    except Exception as exc:
        logger.warning("Agglomerative clustering failed (%r) — using lexical", exc)
        return (_lexical_cluster(phrases), False)

    clusters: dict[int, list[str]] = defaultdict(list)
    for phrase, lbl in zip(phrases, labels):
        clusters[int(lbl)].append(phrase)
    return (dict(clusters), True)


def _lexical_cluster(phrases: List[str]) -> dict:
    """Cheap fallback: group phrases that share their longest token.

    Loses semantic equivalence (e.g. won't group "anode" with "negative
    electrode") but at least catches singular/plural pairs."""
    clusters: dict[str, list[str]] = defaultdict(list)
    for p in phrases:
        # Pick the longest content token as the cluster key — usually
        # the head noun.
        toks = [t for t in p.split() if len(t) > 3]
        key = max(toks, key=len) if toks else p
        clusters[key].append(p)
    return {i: members for i, members in enumerate(clusters.values())}
