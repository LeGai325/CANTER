"""Offline dataset and actual-input audits; never imported by admission logic.

Gold annotations measure coverage here. They do not repair candidate pools,
change rankings, or select questions. NQ text retention is a source-coordinate
proxy, not a claim that a passage entails an answer.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from .benchmarks import WebExtractor, answer_scores, normalize_answer
from .inputs import packed_documents_by_job
from .nq_page import nq_answer_scores, nq_prediction_items


VERSION = "provided-context-offline-source-span-audit-v4.1-runtime-whitespace"


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _space(text):
    return " ".join(text.split())


def _read(directory, name, default):
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else default


def _scope_ids(corpus, task_id):
    return list(getattr(corpus, "task_documents", {}).get(task_id, corpus.documents))


def _hotpot_coverage(corpus, reference, pool_ids, selected_ids, packed=None):
    by_title = defaultdict(list)
    for doc_id in pool_ids:
        by_title[corpus.documents[doc_id].title].append(doc_id)
    selected = set(selected_ids)
    facts = []
    for title, index in reference["supporting_facts"]:
        alternatives = []
        for doc_id in by_title[title]:
            row = json.loads(corpus.documents[doc_id].body)
            sentences = row["sentences"]
            if 0 <= index < len(sentences):
                alternatives.append((doc_id, sentences[index]))
        candidate_present = any(doc_id in selected for doc_id, _ in alternatives)
        if packed is None:
            present = candidate_present
        else:
            # The original sentence and its original index must both survive.
            # A lone [index] or a sentence from another title is not coverage.
            present = any(doc_id in selected and any(
                _space(f"[{index}] {sentence}") in _space(text)
                for text in packed.get(doc_id, [])) for doc_id, sentence in alternatives)
        facts.append(dict(title=title, sentence_index=index, source_available=bool(alternatives),
                          candidate_present=candidate_present, complete_sentence_present=present))
    titles = set(title for title, _ in reference["supporting_facts"])
    present_titles = {corpus.documents[doc_id].title for doc_id in selected if doc_id in corpus.documents}
    return dict(supporting_titles=len(titles), supporting_titles_present=len(titles & present_titles),
                all_supporting_titles_present=titles <= present_titles,
                supporting_facts=len(facts), supporting_facts_present=sum(x["complete_sentence_present"] for x in facts),
                all_supporting_facts_present=bool(facts) and all(x["complete_sentence_present"] for x in facts),
                facts=facts)


def _nq_coverage(corpus, reference, selected_ids, packed=None):
    selected = set(selected_ids)
    passages = {p["document_id"]: p for p in reference.get("passages", [])}
    allowed = set(reference.get("passage_document_ids", passages))
    foreign = []
    retained, unknown, complete_renderings = {}, [], []
    for doc_id in selected:
        if doc_id not in corpus.documents:
            foreign.append(doc_id)
            continue
        doc = corpus.documents[doc_id]
        payload = json.loads(doc.body)
        if (doc_id not in allowed or payload.get("source_document_id") != reference["original_document_id"]
                or payload.get("source_document_version") != reference.get("original_document_version")):
            foreign.append(doc_id)
            continue
        indices = passages[doc_id]["token_indices"]
        if packed is None:
            retained[doc_id] = set(indices)
            complete_renderings.append(doc_id)
            continue
        # The adapter joins each official visible token with one space. Verify
        # the correspondence instead of assuming a same-name answer occurrence
        # comes from the annotated location.
        source_text = doc.title + "\n" + payload["text"]
        # Coordinator.finish_extract normalizes only outer whitespace before
        # accepting evidence. Match that public execution contract exactly;
        # missing words or closing structure still fail the prefix/full checks.
        expected = source_text.strip()
        offsets = passages[doc_id].get('rendered_token_offsets')
        if offsets is None:
            words = list(re.finditer(r"\S+", payload["text"]))
            if len(words) != len(indices):
                unknown.append(dict(document_id=doc_id, reason="source_token_text_not_one_to_one"))
                retained[doc_id] = set()
                continue
            offsets = [dict(token_index=index, start=word.start(), end=word.end())
                       for index, word in zip(indices, words)]
        elif (set(p['token_index'] for p in offsets) != set(indices) or
              any(not 0 <= p['start'] < p['end'] <= len(payload['text']) for p in offsets)):
            unknown.append(dict(document_id=doc_id, reason='invalid_rendered_source_mapping'))
            retained[doc_id] = set()
            continue
        visible = set()
        for text in packed.get(doc_id, []):
            if not expected.startswith(text):
                unknown.append(dict(document_id=doc_id, reason="packed_text_not_source_prefix"))
                continue
            offset = len(doc.title) + 1 - (len(source_text) - len(source_text.lstrip()))
            visible.update(p['token_index'] for p in offsets if offset + p['end'] <= len(text))
            if text == expected:
                complete_renderings.append(doc_id)
        retained[doc_id] = visible
    union = set().union(*retained.values()) if retained else set()
    spans, annotations = [], defaultdict(list)
    for span in reference.get("answer_spans", []):
        required = set(span.get("token_indices", []))
        complete = bool(required) and required <= union
        single = [doc_id for doc_id, indices in retained.items() if required and required <= indices]
        row = dict(annotation_index=span["annotation_index"], start_token=span["start_token"],
                   end_token=span["end_token"], text=span["text"], required_visible_tokens=len(required),
                   retained_visible_tokens=len(required & union), complete=complete,
                   complete_in_one_passage=bool(single), complete_passage_ids=sorted(single))
        spans.append(row)
        annotations[span["annotation_index"]].append(complete)
    annotation_rows = [dict(annotation_index=index, span_count=len(values), all_spans_present=all(values))
                       for index, values in sorted(annotations.items())]
    result = dict(same_original_page_only=not foreign, foreign_document_ids=sorted(foreign),
                annotated_span_count=len(spans), complete_span_count=sum(s["complete"] for s in spans),
                any_annotation_all_spans_present=(any(x["all_spans_present"] for x in annotation_rows)
                                                  if annotation_rows else None),
                all_annotated_spans_present=all(s["complete"] for s in spans) if spans else None,
                span_coverage_applicable=bool(spans),
                coverage_note="official source-token coverage; packed retention is a coordinate/prefix proxy, not semantic sufficiency; yes/no has no short span",
                retention_unknown_documents=unknown, spans=spans, annotations=annotation_rows)
    if reference.get('structure_version'):
        required_page = set(reference['visible_token_indices'])
        contexts = []
        for span in reference.get('long_answer_spans', []):
            required = set(span['token_indices'])
            contexts.append(dict(annotation_index=span['annotation_index'],
                required_visible_tokens=len(required), retained_visible_tokens=len(required & union),
                complete=bool(required) and required <= union,
                complete_in_one_passage=bool(required) and any(required <= v for v in retained.values())))
        result.update(structure_version=reference['structure_version'],
            original_visible_tokens=len(required_page), retained_original_visible_tokens=len(required_page & union),
            structured_page_complete=bool(required_page) and required_page <= union and not unknown and not foreign,
            complete_rendered_passage_count=len(set(complete_renderings)),
            selected_passage_count=len(selected),
            all_selected_renderings_complete=selected <= set(complete_renderings),
            long_contexts=contexts,
            all_annotated_long_contexts_present=all(c['complete'] for c in contexts) if contexts else None,
            structure_note='safe source structure and repeated table context; token/context retention does not establish semantic sufficiency')
    return result


def _coverage(corpus, reference, pool_ids, selected_ids, packed=None):
    if reference["benchmark"] == "hotpotqa":
        return _hotpot_coverage(corpus, reference, pool_ids, selected_ids, packed)
    return _nq_coverage(corpus, reference, selected_ids, packed)


def audit_candidates(workload, tasks, config):
    """Audit the complete task-local pool separately from the returned top N."""
    corpus, _, references = workload
    rows = []
    for task in tasks:
        pool_ids = _scope_ids(corpus, task.id)
        limit = config["candidate_limit"]
        search = getattr(corpus, "search_for_task", None)
        found = search(task.id, task.question, limit) if search else corpus.search(task.question, limit)
        ids = [doc_id for doc_id, _ in found]
        ref = references[task.id]
        rows.append(dict(task_id=task.id, benchmark=task.benchmark, stratum=ref.get("stratum", task.benchmark),
                         question=task.question, corpus_pool_count=len(pool_ids), retrieved_count=len(ids),
                         corpus_pool_ids=pool_ids, retrieved_ids=ids,
                         retrieved_payload_bytes=sum(len(corpus.documents[i].body) for i in ids),
                         corpus_pool=_coverage(corpus, ref, pool_ids, pool_ids),
                         retrieved=_coverage(corpus, ref, pool_ids, ids)))
    return dict(version=VERSION, corpus_sha256=corpus.version, candidate_limit=config["candidate_limit"],
                task_count=len(rows), strata=dict(Counter(r["stratum"] for r in rows)),
                scope=getattr(corpus, "scope", "global"), rows=rows,
                aggregate=_coverage_summary(rows, ("corpus_pool", "retrieved")))


def audit_full_input_capacity(workload, tasks, config, tokenizer):
    """Dry-pack the full candidate set before spending API calls.

    This checks input feasibility only. Real runs still audit the actual
    delivered subset independently; no reference annotations reach the model.
    """
    from .contracts import CandidateMetadata, Evidence, Role
    from .experiment import make_contract
    from .inputs import pack_inputs
    corpus, _, references = workload
    rows = []
    for task in tasks:
        found = corpus.search_for_task(task.id, task.question, config['candidate_limit'])
        evidence = []
        for rank, (doc_id, score) in enumerate(found):
            doc = corpus.documents[doc_id]
            position = (json.loads(doc.body)['passage_index']
                        if getattr(corpus, 'require_complete_candidates', False) else None)
            candidate = CandidateMetadata(doc.id, 'question', '/documents/' + doc.id,
                doc.version, doc.title, doc.text[:160], -score, rank, doc.source,
                len(doc.body), query_text=task.question, source_position=position)
            evidence.append(Evidence(candidate, WebExtractor()._extract(candidate, doc.body).strip(),
                Role.BASE if rank < config['base_minimum'] else Role.OPTIONAL, 0, f'preflight-{rank}'))
        contract = make_contract(task, config, tokenizer, SimpleNamespace(version='offline-input-audit'), reserve=60)
        packed = pack_inputs(contract, tuple(evidence), tokenizer)
        by_candidate = {d.candidate_id: [d.text] for d in packed.documents}
        included = {d.candidate_id: d.text for d in packed.documents}
        intact = len(packed.documents) == len(evidence) and all(included.get(e.candidate.id) == e.text for e in evidence)
        coverage = _coverage(corpus, references[task.id], _scope_ids(corpus, task.id), list(by_candidate), by_candidate)
        pool_complete = len(found) == len(_scope_ids(corpus, task.id))
        limits_fit = (len(evidence) - config['base_minimum'] <= config['optional_limit'] and
            all(e.candidate.length_hint <= config['per_document_bytes'] for e in evidence) and
            sum(e.candidate.length_hint for e in evidence if e.role == Role.OPTIONAL) <= config['optional_payload_budget'])
        rows.append(dict(task_id=task.id, benchmark=task.benchmark, candidate_count=len(evidence),
            input_tokens=packed.token_count, input_sha256=packed.sha256, full_evidence_intact=intact,
            complete_candidate_pool=pool_complete, hard_limits_fit=limits_fit, actual_input=coverage,
            passed=intact and pool_complete and limits_fit and coverage.get('structured_page_complete', True)
                   and not coverage.get('retention_unknown_documents')))
    return dict(version=VERSION, model_calls=0, rows=rows, task_count=len(rows),
                passed=all(r['passed'] for r in rows),
                note='offline feasibility of full evidence; actual fetch/packing/model outcome remains a separate gate')


def _coverage_summary(rows, stages):
    result = {}
    for stage in stages:
        groups = defaultdict(list)
        for row in rows:
            groups[row["benchmark"]].append(row[stage])
        result[stage] = {}
        for benchmark, values in groups.items():
            key = "all_supporting_facts_present" if benchmark == "hotpotqa" else "any_annotation_all_spans_present"
            applicable = [v[key] for v in values if v[key] is not None]
            result[stage][benchmark] = dict(task_count=len(values), applicable_tasks=len(applicable),
                                           complete_tasks=sum(applicable),
                                           complete_rate=sum(applicable)/len(applicable) if applicable else None)
    return result


def _tokenizer(manifest, contracts):
    versions = {c.get("tokenizer_version") for c in contracts}
    if versions == {"fixture-codepoint-v1"}:
        from .inputs import FixtureTokenizer
        return FixtureTokenizer(), None
    root = Path(manifest.get("config", {}).get("data_root", "data"))
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root
    try:
        from .provider import ModelTokenizer
        tokenizer = ModelTokenizer(root / "tokenizer_glm52")
        if versions and versions != {tokenizer.version}:
            return None, "recorded tokenizer version differs from current files"
        return tokenizer, None
    except (FileNotFoundError, ValueError, ImportError, OSError) as error:
        return None, type(error).__name__


def _full_text(corpus, job):
    candidate = SimpleNamespace(**job["candidate"])
    doc = corpus.documents[candidate.id]
    if doc.source == "nq-passage":
        return (doc.title + "\n" + json.loads(doc.body)["text"]).strip()
    return WebExtractor()._extract(candidate, doc.body).strip()


def audit_run(run_dir, workload):
    """Read completed/partial artifacts without changing results or the runtime."""
    directory = Path(run_dir)
    corpus, tasks, references = workload
    manifest = _read(directory, "MANIFEST.json", {})
    contracts = _read(directory, "contracts.json", [])
    tokenizer, token_error = _tokenizer(manifest, contracts)
    results = {r["task_id"]: r for r in _read(directory, "results.json", [])}
    packed_all = _read(directory, "packed_inputs.json", {})
    jobs = _read(directory, "jobs.json", [])
    searches = _read(directory, "search_records.json", [])
    task_ids = manifest.get("task_ids", list(results))
    question_by_id = {t.id: t.question for t in tasks}
    rows = []
    for task_id in task_ids:
        ref = references[task_id]
        result = results.get(task_id, {})
        own_jobs = [j for j in jobs if j["task_id"] == task_id]
        by_job = {j["id"]: j for j in own_jobs}
        payload = packed_all.get(task_id, {})
        prompt = json.loads(payload.get("prompt", "{}"))
        docs = prompt.get("documents", [])
        packed = defaultdict(list)
        for doc in docs:
            packed[doc["candidate_id"]].append(doc["text"])
        accepted = result.get("accepted_ids", [])
        normalized_prompt = {**prompt, "documents": [{k: v for k, v in d.items() if k != "id"} for d in docs]}
        truncations = []
        omitted_tokens = 0 if tokenizer else None
        doc_by_job = packed_documents_by_job(payload)
        for job_id in accepted:
            job = by_job.get(job_id)
            if job is None:
                truncations.append(dict(job_id=job_id, error="accepted_job_missing"))
                continue
            full = _full_text(corpus, job)
            included = doc_by_job.get(job_id)
            text = included["text"] if included else ""
            removed = max(0, tokenizer.count(full) - tokenizer.count(text)) if tokenizer else None
            if not included and tokenizer:
                omitted_tokens += removed
            truncations.append(dict(job_id=job_id, candidate_id=job["candidate"]["id"],
                                    included=bool(included), truncated=bool(included) and text != full,
                                    packed_is_source_prefix=full.startswith(text),
                                    full_text_characters=len(full), packed_text_characters=len(text),
                                    removed_evidence_tokens=removed))
        answer = result.get("answer")
        em, f1 = (nq_answer_scores(answer or "", ref) if ref["benchmark"] == "nq"
                  else answer_scores(answer or "", ref["answers"]))
        # A late answer still has semantic quality and abstention behavior;
        # only timely_correct applies the service deadline.
        delivered = result.get("status") in {"delivered", "late"}
        # A missing delivery must not receive credit, including for empty refs.
        if not delivered:
            em = f1 = 0.0
        answer_items = (nq_prediction_items(answer or "") if ref['benchmark'] == 'nq' else [answer or ''])
        abstained = delivered and [normalize_answer(item) for item in answer_items] == ['noanswer']
        gold_abstain = ref["answers"] == ["noanswer"]
        retrieved_ids = list(dict.fromkeys(i for s in searches if s["task_id"] == task_id for i in s["candidates"]))
        accepted_ids = [by_job[j]["candidate"]["id"] for j in accepted if j in by_job]
        row = dict(task_id=task_id, benchmark=ref["benchmark"], stratum=ref.get("stratum", ref["benchmark"]),
                   question=question_by_id.get(task_id), status=result.get("status", "missing_result"),
                   answer=answer, answer_items=answer_items, references=ref["answers"], answer_em=em, answer_f1=f1,
                   delivered=delivered, abstained=abstained, gold_noanswer=gold_abstain,
                   false_answer_on_noanswer=gold_abstain and delivered and not abstained,
                   timely=delivered and result.get("delivered_at", float("inf")) <= result.get("deadline", -1),
                   admitted_count=len(own_jobs), optional_admitted=result.get("optional_admitted", 0),
                   accepted_count=len(accepted), included_count=len(docs),
                   optional_accepted=result.get("optional_accepted", 0),
                   job_outcomes=dict(Counter(j.get("outcome", "unfinished") for j in own_jobs)),
                   retrieved_ids=retrieved_ids, accepted_candidate_ids=accepted_ids,
                   included_candidate_ids=list(packed),
                   normalized_input_sha256=_digest(normalized_prompt) if payload else None,
                   input_sha256=payload.get("sha256"), input_tokens=result.get("input_tokens"),
                   wire_bytes=result.get('wire_bytes'),
                   duration_seconds=(result['delivered_at'] - result['arrived_at']
                                     if delivered and result.get('delivered_at') is not None
                                     and result.get('arrived_at') is not None else None),
                   annotation_agreement=ref.get('annotation_agreement'),
                   truncated_document_count=sum(bool(d.get("truncated")) for d in truncations),
                   accepted_but_omitted_document_count=sum(not d.get("included", False) for d in truncations),
                   removed_evidence_tokens=sum(d.get("removed_evidence_tokens", 0) or 0 for d in truncations) if tokenizer else None,
                   omitted_evidence_tokens=omitted_tokens, document_truncation=truncations,
                   actual_input=_coverage(corpus, ref, _scope_ids(corpus, task_id), list(packed), packed),
                   retrieved=_coverage(corpus, ref, _scope_ids(corpus, task_id), retrieved_ids))
        row["timely_correct"] = row["timely"] and em == 1
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[row["stratum"]].append(row)
    strata = {}
    for stratum, values in groups.items():
        negatives = [r for r in values if r["gold_noanswer"]]
        strata[stratum] = dict(tasks=len(values), answer_em=sum(r["answer_em"] for r in values)/len(values),
                              answer_f1=sum(r["answer_f1"] for r in values)/len(values),
                              abstention_rate=sum(r["abstained"] for r in values)/len(values),
                              delivery_rate=sum(r["delivered"] for r in values)/len(values),
                              timely_correct_rate=sum(r["timely_correct"] for r in values)/len(values),
                              noanswer_tasks=len(negatives),
                              false_answer_on_noanswer_rate=(sum(r["false_answer_on_noanswer"] for r in negatives)/len(negatives)
                                                             if negatives else None))
    return dict(version=VERSION, run_directory=str(directory), policy=manifest.get("policy"), mode=manifest.get("mode"),
                task_count=len(rows), rows=rows, strata=strata,
                aggregate=_coverage_summary(rows, ("retrieved", "actual_input")),
                token_accounting_error=token_error,
                token_accounting_note="removed source-evidence tokens before prompt overhead; includes accepted evidence omitted entirely",
                nq_metric='derived normalized short-answer-set text EM / token F1; not official coordinate-span NQ leaderboard scores',
                nq_noanswer_note='noanswer is the native short-answer vote label, not a claim that no semantic answer exists on the page',
                normalization_note="input hash removes per-run job IDs; preserves order, candidate IDs, roles, versions, and actual text")
