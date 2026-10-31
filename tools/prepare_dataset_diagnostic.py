"""Convert pinned source assets without changing historical experiment files."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from prepare_assets import unzip_columns


def write_once(path, value):
    data = value.encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('existing converted asset differs: ' + str(path))
    else:
        path.write_bytes(data)


def main():
    root = Path('data')
    hotpot = []
    for batch in pq.ParquetFile(root / 'hotpot_distractor.parquet').iter_batches(batch_size=128):
        for row in batch.to_pylist():
            row['_id'] = row.pop('id')
            row['context'] = list(zip(row['context']['title'], row['context']['sentences']))
            row['supporting_facts'] = list(zip(row['supporting_facts']['title'], row['supporting_facts']['sent_id']))
            hotpot.append(row)
    hpath = root / 'hotpot_dev_distractor_v1.json'
    write_once(hpath, json.dumps(hotpot, ensure_ascii=False))
    lines = []
    for batch in pq.ParquetFile(root / 'natural_questions.parquet').iter_batches(batch_size=8):
        for row in batch.to_pylist():
            if len(lines) >= 200:
                break
            doc = row['document']
            annotations = unzip_columns(row['annotations'])
            for ann in annotations:
                ann['annotation_id'] = ann.pop('id')
                ann['yes_no_answer'] = {-1: 'NONE', 0: 'NO', 1: 'YES'}[ann['yes_no_answer']]
                ann['short_answers'] = unzip_columns(ann['short_answers'])
            tokens = unzip_columns(doc['tokens'])
            for token in tokens:
                token['html_token'] = token.pop('is_html')
            native = dict(example_id=row['id'], question_text=row['question']['text'],
                          document_title=doc['title'], document_url=doc['url'], document_html=doc['html'],
                          document_tokens=tokens, annotations=annotations,
                          long_answer_candidates=unzip_columns(row['long_answer_candidates']))
            lines.append(json.dumps(native, ensure_ascii=False))
        if len(lines) >= 200:
            break
    npath = root / 'nq_dev_00_200.jsonl'
    write_once(npath, '\n'.join(lines) + '\n')
    files = [root / 'hotpot_distractor.parquet', root / 'natural_questions.parquet', hpath, npath]
    manifest = dict(
        hotpot_repository='hotpotqa/hotpot_qa', hotpot_revision='1908d6afbbead072334abe2965f91bd2709910ab',
        hotpot_file='distractor/validation-00000-of-00001.parquet',
        nq_repository='google-research-datasets/natural_questions', nq_revision='e8103d566bef4154c2c12b17c6095ec5275840cc',
        nq_selection='first 200 records in existing pinned validation shard; no model-outcome filtering',
        hotpot_records=len(hotpot), nq_records=len(lines),
        files={p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files})
    write_once(root / 'DATASET_DIAGNOSTIC_SOURCES.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
