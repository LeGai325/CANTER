"""Frozen paragraph and original-HTML adapters; gold stays evaluation-only."""
from collections import Counter
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sqlite3
import string


class PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.current, self.skip = [], [], 0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}: self.skip += 1
        if tag in {"p", "div", "table", "tr", "li", "h1", "h2", "h3", "br"}: self.flush()
    def handle_endtag(self, tag):
        if tag in {"script", "style"}: self.skip = max(0, self.skip-1)
        if tag in {"p", "div", "table", "tr", "li", "h1", "h2", "h3"}: self.flush()
    def handle_data(self, data):
        if not self.skip: self.current.append(data)
    def flush(self):
        text = " ".join(" ".join(self.current).split())
        if text: self.parts.append(text)
        self.current = []


def html_blocks(html):
    parser = PageText()
    parser.feed(html)
    parser.flush()
    return parser.parts


def terms(text):
    stop = {"the","a","an","of","in","on","to","is","was","were","and","or","who","what","which","when","where","how","did","does","do","for","by","with"}
    return [t for t in re.findall(r"[\w]+", text.lower()) if len(t)>1 and t not in stop]


@dataclass(frozen=True)
class WebDocument:
    id: str
    title: str
    body: bytes
    version: str
    source: str
    text: str
    original_url: str


@dataclass(frozen=True)
class BenchTask:
    id: str
    question: str
    benchmark: str


class Corpus:
    def __init__(self, documents):
        self.documents = {d.id: d for d in documents}
        self.index = sqlite3.connect(":memory:")
        self.index.execute("CREATE VIRTUAL TABLE pages USING fts5(id UNINDEXED, title, body)")
        self.index.executemany("INSERT INTO pages VALUES(?,?,?)", [(d.id,d.title,d.text) for d in self.documents.values()])
        self.index.commit()
        self.version = hashlib.sha256(json.dumps(sorted((d.id,d.version) for d in self.documents.values())).encode()).hexdigest()

    def search(self, query, limit):
        expression = " OR ".join('"'+term.replace('"','')+'"' for term in dict.fromkeys(terms(query)))
        if not expression: return []
        return self.index.execute("SELECT id, bm25(pages,0,3,1) AS score FROM pages WHERE pages MATCH ? ORDER BY score,id LIMIT ?",
                                  (expression,limit)).fetchall()


def document(title, body, source, text, url):
    version = hashlib.sha256(body).hexdigest()
    identity = hashlib.sha256((url+version).encode()).hexdigest()[:24]
    return WebDocument(identity,title,body,version,source,text,url)


def load_hotpot(path: Path, limit=None, *, corpus_path=None):
    rows = json.loads(path.read_text())
    docs, tasks, references = {}, [], {}
    for index,row in enumerate(rows):
        task_id = "hotpot-" + row["_id"]
        if limit is None or index < limit:
            tasks.append(BenchTask(task_id,row["question"],"hotpotqa"))
        references[task_id] = dict(answers=[row["answer"]], supporting_facts=row["supporting_facts"],
                                  benchmark="hotpotqa", original_id=row["_id"])
        for title,sentences in ([] if corpus_path else row["context"]):
            body = json.dumps(dict(title=title,sentences=sentences),ensure_ascii=False).encode()
            doc = document(title,body,"hotpotqa", " ".join(sentences),"hotpot-intro:"+title)
            docs[doc.id] = doc
    if corpus_path:
        # Official processed intro JSONL (.bz2 files) or normalized JSONL directory.
        import bz2
        paths=sorted(Path(corpus_path).rglob('*.bz2')) if Path(corpus_path).is_dir() else [Path(corpus_path)]
        for file in paths:
            opener=bz2.open if file.suffix=='.bz2' else open
            with opener(file,'rt') as source:
                for line in source:
                    row=json.loads(line);title=row['title'];sentences=row.get('text',row.get('sentences'))
                    if not isinstance(sentences,list):raise ValueError('expected official introduction sentences')
                    body=json.dumps(dict(title=title,sentences=sentences),ensure_ascii=False).encode()
                    doc=document(title,body,'hotpotqa',' '.join(sentences),'hotpot-intro:'+title);docs[doc.id]=doc
    return Corpus(docs.values()), tasks, references


def load_nq(path: Path, limit=None):
    docs, tasks, references = {}, [], {}
    with path.open() as source:
        for index,line in enumerate(source):
            if limit is not None and index >= limit: break
            row = json.loads(line)
            body = row["document_html"].encode("utf-8")
            blocks = html_blocks(row["document_html"])
            title = row.get("document_title") or row["document_url"].split("/")[-1]
            doc = document(title,body,"nq", "\n".join(blocks),row["document_url"])
            docs[doc.id] = doc
            task_id = "nq-"+str(row["example_id"])
            tasks.append(BenchTask(task_id,row["question_text"],"nq"))
            answers, positive = [], 0
            for annotation in row["annotations"]:
                yesno=annotation["yes_no_answer"].lower()
                spans=annotation["short_answers"]
                if yesno in {"yes","no"}:
                    answers.append(yesno);positive+=1
                elif spans:
                    pieces=[]
                    for span in spans:
                        if span.get("start_byte",-1)>=0:
                            pieces.append(" ".join(html_blocks(body[span["start_byte"]:span["end_byte"]].decode())))
                        else:
                            pieces.append(" ".join(t["token"] for t in row["document_tokens"][span["start_token"]:span["end_token"]] if not t["html_token"]))
                    answers.append(" ; ".join(pieces));positive+=1
            required = 2 if len(row["annotations"]) > 1 else 1
            references[task_id]=dict(answers=list(dict.fromkeys(answers)) if positive>=required else ["noanswer"],
                benchmark="nq", original_id=str(row["example_id"]), original_document_id=doc.id,
                annotations=row["annotations"], scoring="retrieval-generated-short-answer-text-em-f1; not official span leaderboard")
    return Corpus(docs.values()),tasks,references


class WebExtractor:
    version = "hotpot-numbered-sentences-nq-token-passages-v2"
    async def extract(self,candidate,body):
        import asyncio
        return await asyncio.to_thread(self._extract,candidate,body)

    def _extract(self,candidate,body):
        if candidate.source == 'nq-passage':
            row = json.loads(body)
            return row['title'] + '\n' + row['text']
        if candidate.source in {"hotpotqa","heldout-hotpotqa"}:
            row=json.loads(body)
            return row["title"]+"\n"+"\n".join(f"[{i}] {s}" for i,s in enumerate(row["sentences"]))
        blocks=html_blocks(body.decode("utf-8"))
        query=set(terms(candidate.query_text))
        ranked=sorted(range(len(blocks)),key=lambda i:(-sum(t in query for t in terms(blocks[i])),i))[:16]
        return candidate.title+"\n"+"\n".join(blocks[i] for i in sorted(ranked))


def normalize_answer(text):
    text="".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b"," ",text).split())


def answer_scores(prediction, references):
    pred=normalize_answer(prediction)
    scores=[]
    for answer in references:
        gold=normalize_answer(answer)
        em=float(pred==gold)
        if pred!=gold and (pred in {"yes","no","noanswer"} or gold in {"yes","no","noanswer"}):
            scores.append((em,0.0));continue
        a,b=pred.split(),gold.split()
        common=sum((Counter(a)&Counter(b)).values())
        f1=2*common/(len(a)+len(b)) if common else float(not a and not b)
        scores.append((em,f1))
    return max((x[0] for x in scores),default=0.0),max((x[1] for x in scores),default=0.0)


class AnswerEvaluator:
    version="hotpot-official-3635853-nq-short-answer-set-v2"
    def score(self,answer,reference):return reference_answer_scores(answer,reference)[0]


def reference_answer_scores(answer, reference):
    if 'short_answer_sets' in reference:
        from .nq_page import nq_answer_scores
        return nq_answer_scores(answer, reference)
    if reference.get('benchmark') == 'hotpotqa':
        from .vendor.hotpot_evaluate_v1 import exact_match_score, f1_score
        answers = reference['answers']
        return (max((float(exact_match_score(answer, gold)) for gold in answers), default=0.0),
                max((f1_score(answer, gold)[0] for gold in answers), default=0.0))
    return answer_scores(answer, reference['answers'])


def supporting_scores(predicted,reference,answer):
    gold={tuple(x) for x in reference['supporting_facts']};pred={tuple(x) for x in predicted}
    tp=len(pred&gold);precision=tp/len(pred) if pred else 0.;recall=tp/len(gold) if gold else 0.
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    # Official joint F1 combines answer precision/recall with supporting P/R.
    a=normalize_answer(answer or '').split();b=normalize_answer(reference['answers'][0]).split()
    common=sum((Counter(a)&Counter(b)).values())
    ap=common/len(a) if a else 0.;ar=common/len(b) if b else 0.
    em,af=answer_scores(answer or '',reference['answers'])
    if af==0:ap=ar=0.
    jp,jr=ap*precision,ar*recall
    return dict(supporting_em=float(pred==gold),supporting_f1=f1,joint_em=em*float(pred==gold),
                joint_f1=2*jp*jr/(jp+jr) if jp+jr else 0.)
