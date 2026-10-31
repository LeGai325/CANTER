"""Question-and-page NQ adapter with gold-independent, page-local passages.

This is an explicit *derived passage* workload, not native whole-HTML NQ and not
open-domain Web search.  The native adapter and its annotation consensus remain
the source of truth for answers.  Gold spans are only copied into references
for offline audits; they never choose passage boundaries or retrieval order.
"""
from collections import Counter
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re

from .benchmarks import document, html_blocks, load_nq, normalize_answer
from .scoped_corpus import TaskScopedCorpus
from .nq_structure import STRUCTURE_VERSION, structured_passages


ADAPTER_VERSION = "nq-given-page-token-passages-v2"
BOUNDED_ADAPTER_VERSION = "nq-given-page-complete-bounded-passages-v2"
STRUCTURED_ADAPTER_VERSION = "nq-given-page-structured-complete-passages-v3"
RETRIEVER_VERSION = "nq-page-local-bm25-v1"
ANSWER_TEXT_RENDERING = "official-document-token-join-v2"
# Generation.answer remains a string for runtime/artifact compatibility. Only
# the explicit v3 generation contract emits this tagged, lossless item array;
# historical plain answers retain their original semicolon interpretation.
NQ_ANSWER_ITEMS_PREFIX = "nq-answer-items-v1:"


class _ContentRegions(HTMLParser):
    """Locate boilerplate/main regions using HTML structure, never labels."""
    excluded = {"head", "script", "style", "noscript", "nav", "footer"}
    void = {"area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.source = source
        self.lines = [0]
        self.lines.extend(m.end() for m in re.finditer("\n", source))
        self.stack, self.blocked, self.main = [], [], []

    def position(self):
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self.void:
            return
        self.stack.append((tag, self.position(),
                           tag in self.excluded or attrs.get("id") in {"mw-navigation", "footer"},
                           attrs.get("id") == "mw-content-text"))

    def handle_startendtag(self, tag, attrs):
        return

    def handle_endtag(self, tag):
        matched = next((i for i in range(len(self.stack) - 1, -1, -1)
                        if self.stack[i][0] == tag), None)
        if matched is None:
            return
        end = self.source.find(">", self.position())
        end = len(self.source) if end < 0 else end + 1
        for _, start, blocked, main in self.stack[matched:]:
            if blocked:
                self.blocked.append((start, end))
            if main:
                self.main.append((start, end))
        del self.stack[matched:]

    def finish(self):
        for _, start, blocked, main in self.stack:
            if blocked:
                self.blocked.append((start, len(self.source)))
            if main:
                self.main.append((start, len(self.source)))
        # Convert to the byte coordinate system used by official annotations.
        points = {0, len(self.source)}
        for a, b in self.blocked + self.main:
            points.update((a, b))
        byte_at, previous, offset = {}, 0, 0
        for point in sorted(points):
            offset += len(self.source[previous:point].encode("utf-8"))
            byte_at[point], previous = offset, point
        convert = lambda ranges: [(byte_at[a], byte_at[b]) for a, b in ranges]
        return convert(self.main), convert(self.blocked)


def _visible_tokens(row):
    """Return (official token index, token text), keeping source coordinates."""
    tokens = row.get("document_tokens")
    if not tokens:
        raise ValueError("NQ page passages require original document_tokens with byte offsets")
    parser = _ContentRegions(row["document_html"])
    parser.feed(row["document_html"])
    parser.close()
    main, excluded = parser.finish()
    result = []
    for index, token in enumerate(tokens):
        if token["html_token"]:
            continue
        start, end = token.get("start_byte", -1), token.get("end_byte", -1)
        if start < 0 or end < start:
            raise ValueError("NQ page passages require valid official token byte offsets")
        if main and not any(a <= start and end <= b for a, b in main):
            continue
        if any(start < b and end > a for a, b in excluded):
            continue
        text = token["token"]
        if text.strip():
            result.append((index, text))
    if not result:
        raise ValueError(f"NQ page {row['example_id']} has no visible content tokens")
    return result


def _span_token_bounds(row, span):
    """Validate native half-open offsets; absent offsets permit an explicit fallback."""
    start, end = span.get("start_token"), span.get("end_token")
    if start in (None, -1) and end in (None, -1):
        return None
    if (type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(row["document_tokens"])):
        raise ValueError("NQ short answer has invalid official token coordinates")
    return start, end


def _span_text(row, span):
    """Render gold with the passage's native-token spelling and whitespace.

    The native token strings can differ from HTML bytes (e.g. ``--`` versus an
    em dash). Byte-HTML rendering is only a compatibility fallback when both
    token offsets are absent; invalid or out-of-range offsets never fall back.
    """
    bounds = _span_token_bounds(row, span)
    if bounds is not None:
        start, end = bounds
        return " ".join(t["token"] for t in row["document_tokens"][start:end]
                        if not t["html_token"] and t["token"].strip())
    start, end = span.get("start_byte"), span.get("end_byte")
    body = row["document_html"].encode("utf-8")
    if type(start) is int and type(end) is int and 0 <= start < end <= len(body):
        return " ".join(html_blocks(body[start:end].decode("utf-8")))
    raise ValueError("NQ short answer is missing valid byte and token coordinates")


def load_nq_pages(path: Path, limit=None, *, passage_words=300, overlap_words=50,
                  max_passages=None, structured=False):
    """Load provided-page NQ, exposing only that question's derived passages.

    ``passage_words`` counts non-HTML official tokens.  Span coordinates remain
    official token offsets despite removal of script/style/navigation content.
    With ``max_passages``, the window grows based on page length alone so the
    whole visible page fits within that many candidate objects.  This is a
    provided-page partition, not a claim of improved open retrieval recall.
    Candidate truncation is left to the caller and must be audited separately.
    ``structured=True`` instead preserves source layout and bounds individual
    passage size without limiting the number of passages. It requires
    ``max_passages=None``; source context replaces fixed token-window overlap.
    """
    if passage_words < 1 or overlap_words < 0 or (not structured and overlap_words >= passage_words):
        raise ValueError("require passage_words > overlap_words >= 0")
    if max_passages is not None and (not isinstance(max_passages, int)
                                     or isinstance(max_passages, bool) or max_passages < 1):
        raise ValueError("max_passages must be a positive integer or None")
    if structured and max_passages is not None:
        raise ValueError("structured NQ requires max_passages=None; complete pages cannot be capped silently")
    adapter_version = BOUNDED_ADAPTER_VERSION if max_passages is not None else ADAPTER_VERSION
    if structured:
        adapter_version = STRUCTURED_ADAPTER_VERSION
    original, tasks, references = load_nq(Path(path), limit)
    documents, task_documents = {}, {}
    with Path(path).open() as source:
        for row_index, line in enumerate(source):
            if limit is not None and row_index >= limit:
                break
            row = json.loads(line)
            task_id = "nq-" + str(row["example_id"])
            reference = references[task_id]
            page = original.documents[reference["original_document_id"]]
            visible = _visible_tokens(row)
            effective_words = passage_words
            if max_passages is not None:
                effective_words = max(passage_words,
                                      math.ceil((len(visible) - overlap_words) / max_passages)
                                      + overlap_words)
            passages = []
            if structured:
                chunks = structured_passages(row, visible, passage_words=passage_words)
            else:
                chunks = []
                for start in range(0, len(visible), effective_words - overlap_words):
                    selected = visible[start:start + effective_words]
                    chunks.append(dict(text=" ".join(word for _, word in selected),
                                       token_indices=[index for index, _ in selected]))
                    # Avoid a final passage consisting solely of overlap words.
                    if start + effective_words >= len(visible):
                        break
            for passage_index, chunk in enumerate(chunks):
                selected = [(index, row["document_tokens"][index]["token"])
                            for index in chunk["token_indices"]]
                token_indices = [index for index, _ in selected]
                text = chunk["text"]
                primary_indices = chunk.get("primary_token_indices", token_indices)
                payload = dict(title=page.title, text=text, adapter_version=adapter_version,
                               source_document_id=page.id, source_document_version=page.version,
                               source_url=page.original_url, passage_index=passage_index,
                               start_token=primary_indices[0], end_token=primary_indices[-1] + 1)
                if max_passages is not None:
                    payload.update(max_passages=max_passages, effective_words=effective_words,
                                   overlap_words=overlap_words)
                if structured:
                    payload.update({key: chunk[key] for key in (
                        "structure_version", "structure_features", "section_path", "section_kind",
                        "primary_word_count", "visible_word_count", "repeated_context_word_count",
                        "context_truncated", "continued_structural_block")})
                    payload.update(passage_words=passage_words, overlap_words=0)
                body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
                # Search source words, not layout tags or synthetic position labels.
                search_text = " ".join(word for _, word in selected) if structured else text
                doc = document(page.title, body, "nq-passage", search_text,
                               page.original_url + "#specnet-passage-" + str(passage_index))
                documents[doc.id] = doc
                passage = dict(document_id=doc.id, start_token=primary_indices[0],
                               end_token=primary_indices[-1] + 1, token_indices=token_indices)
                if structured:
                    passage.update({key: value for key, value in chunk.items() if key != "text"})
                passages.append(passage)
            task_documents[task_id] = [p["document_id"] for p in passages]
            positive = long_votes = 0
            answer_sets, spans, long_spans = [], [], []
            for annotation_index, annotation in enumerate(row["annotations"]):
                short = annotation["short_answers"]
                yesno = annotation["yes_no_answer"].lower()
                if yesno in {"yes", "no"} or short:
                    positive += 1
                    answer_sets.append([yesno] if yesno in {"yes", "no"}
                                       else [_span_text(row, span) for span in short])
                long_answer = annotation["long_answer"]
                if long_answer.get("start_token", -1) >= 0 or long_answer.get("start_byte", -1) >= 0:
                    long_votes += 1
                    if structured:
                        bounds = _span_token_bounds(row, long_answer)
                        long_spans.append(dict(
                            annotation_index=annotation_index,
                            start_token=long_answer.get("start_token", -1),
                            end_token=long_answer.get("end_token", -1),
                            token_indices=([i for i in range(*bounds)
                                            if not row["document_tokens"][i]["html_token"]
                                            and row["document_tokens"][i]["token"].strip()]
                                           if bounds is not None else [])))
                for span in short:
                    bounds = _span_token_bounds(row, span)
                    indices = ([i for i in range(*bounds)
                                if not row["document_tokens"][i]["html_token"]
                                and row["document_tokens"][i]["token"].strip()]
                               if bounds is not None else [])
                    spans.append(dict(annotation_index=annotation_index, text=_span_text(row, span),
                                      start_token=span.get("start_token", -1), end_token=span.get("end_token", -1),
                                      token_indices=indices,
                                      start_byte=span.get("start_byte", -1), end_byte=span.get("end_byte", -1),
                                      text_rendering=(ANSWER_TEXT_RENDERING if bounds is not None
                                                      else "byte-html-fallback-token-coordinates-missing")))
            required = 2 if len(row["annotations"]) > 1 else 1
            stratum = ("short_consensus" if positive >= required else "short_disputed" if positive
                       else "long_only" if long_votes else "null")
            canonical_sets = answer_sets if positive >= required else [["noanswer"]]
            reference.update(adapter=adapter_version, stratum=stratum, positive_votes=positive,
                             long_votes=long_votes, required_positive_votes=required,
                             native_html_answers=list(reference["answers"]),
                             answers=list(dict.fromkeys(" ; ".join(items) for items in canonical_sets)),
                             short_answer_sets=canonical_sets, answer_text_rendering=ANSWER_TEXT_RENDERING,
                             answer_span_texts=list(dict.fromkeys(span["text"] for span in spans)),
                             answer_spans=spans, passage_document_ids=task_documents[task_id],
                             passages=passages, original_document_version=page.version,
                             source_url=page.original_url, visible_tokens=len(visible),
                             passage_words=passage_words, effective_words=effective_words,
                             overlap_words=overlap_words, max_passages=max_passages,
                             task_semantics="given-original-page; generated short answer or noanswer; "
                                            "derived passage retrieval; not official span leaderboard")
            if structured:
                agreement_counts = Counter(tuple(sorted(set(normalize_answer(item) for item in items)))
                                           for items in answer_sets)
                reference.update(
                    structure_version=STRUCTURE_VERSION, visible_token_indices=[i for i, _ in visible],
                    structure_features=sorted({feature for chunk in chunks for feature in chunk["structure_features"]}),
                    long_answer_spans=long_spans, overlap_words=0,
                    passage_size_semantics="bounded visible token occurrences including repeated source context",
                    annotation_agreement=dict(
                        positive_annotation_count=positive,
                        distinct_positive_answer_sets=len(agreement_counts),
                        max_matching_positive_annotations=max(agreement_counts.values(), default=0),
                        max_positive_agreement_fraction=(max(agreement_counts.values()) / positive if positive else None),
                        short_consensus_semantics="positive-vote threshold, not content agreement"))
    corpus = TaskScopedCorpus(documents.values(), task_documents,
                              scope=("nq-given-page-structured-passages" if structured else "nq-given-page-passages"),
                              retriever_version=RETRIEVER_VERSION)
    if structured:
        corpus.require_complete_candidates = True
    return corpus, tasks, references


def _validated_answer_items(items):
    if (not isinstance(items, list) or not items
            or any(not isinstance(item, str) or not item.strip() for item in items)):
        raise ValueError("NQ answer_items must be a nonempty array of nonempty strings")
    return [item.strip() for item in items]


def encode_nq_answer_items(items):
    """Preserve answer boundaries, including punctuation within each span."""
    return NQ_ANSWER_ITEMS_PREFIX + json.dumps(
        _validated_answer_items(items), ensure_ascii=False, separators=(",", ":"))


def nq_prediction_items(prediction):
    """Read v3 item arrays without changing historical plain-text scoring."""
    if isinstance(prediction, str) and prediction.startswith(NQ_ANSWER_ITEMS_PREFIX):
        return _validated_answer_items(json.loads(prediction[len(NQ_ANSWER_ITEMS_PREFIX):]))
    if isinstance(prediction, (list, tuple)):
        return [str(item) for item in prediction]
    return re.split(r"\s*;\s*", str(prediction).strip())


def nq_answer_scores(prediction, reference):
    """Order-independent short-answer-set text EM and token F1.

    This remains a text adapter metric: it does not claim official span EM or
    long-answer evaluation.  Consensus/abstention labels are unchanged.
    """
    predicted = sorted(set(normalize_answer(item) for item in nq_prediction_items(prediction)))
    gold_sets = reference.get("short_answer_sets")
    if gold_sets is None:
        gold_sets = [nq_prediction_items(answer) for answer in reference["answers"]]
    scores = []
    for items in gold_sets:
        gold = sorted(set(normalize_answer(item) for item in items))
        em = float(predicted == gold)
        if predicted != gold and (set(predicted + gold) & {"yes", "no", "noanswer"}):
            scores.append((em, 0.0))
            continue
        a, b = " ".join(predicted).split(), " ".join(gold).split()
        common = sum((Counter(a) & Counter(b)).values())
        scores.append((em, 2 * common / (len(a) + len(b)) if common else float(not a and not b)))
    return max((score[0] for score in scores), default=0.0), max((score[1] for score in scores), default=0.0)
