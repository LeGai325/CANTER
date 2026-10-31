"""Download pinned maintainer snapshots and preserve benchmark-native bodies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

SOURCES = {
 'hotpotqa': ('hotpotqa/hotpot_qa','1908d6afbbead072334abe2965f91bd2709910ab','fullwiki/validation-00000-of-00001.parquet','hotpot_qa.parquet'),
 'nq': ('google-research-datasets/natural_questions','e8103d566bef4154c2c12b17c6095ec5275840cc','default/validation-00000-of-00007.parquet','natural_questions.parquet'),
}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as source:
  for chunk in iter(lambda:source.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def download(url,path):
 if path.exists():return
 path.parent.mkdir(parents=True,exist_ok=True)
 partial=path.with_suffix(path.suffix+'.partial')
 with urllib.request.urlopen(url,timeout=180) as source,partial.open('wb') as out:shutil.copyfileobj(source,out)
 partial.rename(path)

def unzip_columns(columns):
 return [dict(zip(columns,values)) for values in zip(*columns.values())]

def main():
 import pyarrow.parquet as pq
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('data'));p.add_argument('--nq-records',type=int,default=40)
 args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
 tokenizer=args.root/'tokenizer_glm52';model='zai-org/GLM-5.2';revision='cf457fa734ab149ffef225f80893eb38c6ff5cdc'
 files={}
 for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
  url=f'https://huggingface.co/{model}/resolve/{revision}/{name}';download(url,tokenizer/name)
  files[name]=dict(url=url,sha256=sha(tokenizer/name))
 manifest=tokenizer/'MANIFEST.json'
 if not manifest.exists():manifest.write_text(json.dumps(dict(model=model,revision=revision,files=files),indent=2))
 provenance={}
 for bench,(repo,rev,filename,local) in SOURCES.items():
  url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{filename}';path=args.root/local
  download(url,path);provenance[bench]=dict(repository=repo,revision=rev,url=url,sha256=sha(path))
 hotpot=args.root/'hotpot_dev_fullwiki_v1.json'
 if not hotpot.exists():
  rows=[]
  for batch in pq.ParquetFile(args.root/'hotpot_qa.parquet').iter_batches(batch_size=128):
   for row in batch.to_pylist():
    row['_id']=row.pop('id');row['context']=list(zip(row['context']['title'],row['context']['sentences']))
    row['supporting_facts']=list(zip(row['supporting_facts']['title'],row['supporting_facts']['sent_id']));rows.append(row)
  hotpot.write_text(json.dumps(rows,ensure_ascii=False))
 nq=args.root/'nq_dev_00_prefix.jsonl'
 if not nq.exists():
  count=0
  with nq.with_suffix('.partial').open('w') as out:
   for batch in pq.ParquetFile(args.root/'natural_questions.parquet').iter_batches(batch_size=8):
    for row in batch.to_pylist():
     if count>=args.nq_records:break
     d=row['document'];annotations=unzip_columns(row['annotations'])
     for ann in annotations:
      ann['annotation_id']=ann.pop('id');ann['yes_no_answer']={-1:'NONE',0:'NO',1:'YES'}[ann['yes_no_answer']]
      ann['short_answers']=unzip_columns(ann['short_answers'])
     tokens=unzip_columns(d['tokens'])
     for token in tokens:token['html_token']=token.pop('is_html')
     native=dict(example_id=row['id'],question_text=row['question']['text'],document_title=d['title'],document_url=d['url'],
       document_html=d['html'],document_tokens=tokens,annotations=annotations,long_answer_candidates=unzip_columns(row['long_answer_candidates']))
     if not native['document_html'].lstrip().startswith('<'):raise ValueError('original HTML required')
     out.write(json.dumps(native,ensure_ascii=False)+'\n');count+=1
    if count>=args.nq_records:break
  nq.with_suffix('.partial').rename(nq)
 provenance['selection']='Hotpot full dev records; NQ first records in pinned validation shard, no answer filtering'
 provenance['converted_files']={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in (hotpot,nq)}
 output=args.root/'BENCHMARK_SOURCES.json'
 if output.exists() and json.loads(output.read_text())!=provenance:raise ValueError('existing manifest differs; use new data root')
 output.write_text(json.dumps(provenance,indent=2));print(json.dumps(provenance['converted_files'],indent=2))
if __name__=='__main__':main()
