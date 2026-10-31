"""Validate already-scored terminal output-limit failures without changing scores."""
import json
from pathlib import Path


def verify_output_limit_failure(call, directory, expected_model, results):
    """Accept only a confirmed, capped response already recorded as a failed task."""
    directory = Path(directory)
    task = call.get('purpose')
    contracts = json.loads((directory/'contracts.json').read_text())
    metrics = json.loads((directory/'METRICS.json').read_text())
    owned = [r for r in results if r['task_id'] == task]
    contract_rows = [r for r in contracts if r['task_id'] == task]
    scores = [r for r in metrics['rows'] if r['task_id'] == task]
    if len(owned) != 1 or len(contract_rows) != 1 or len(scores) != 1:
        raise ValueError('output-limit failure is missing unique task/contract/score records')
    result, contract, score = owned[0], contract_rows[0], scores[0]
    usage = call.get('usage') or {}
    if not (
        call.get('status') == 'failed'
        and call.get('error_type') == 'ProviderError'
        and call.get('reason') == 'response_not_complete'
        and call.get('finish_reason') == 'length'
        and call.get('http_status') == 200
        and call.get('requested_model') == expected_model
        and call.get('response_model') == expected_model
        and call.get('remote_terminal_confirmed') is True
        and call.get('max_tokens') == contract['output_tokens']
        and usage.get('completion_tokens') == contract['output_tokens']
        and type(usage.get('prompt_tokens')) is int
        and 0 <= usage['prompt_tokens'] <= contract['input_tokens']
        and result['status'] == 'failed'
        and result.get('error') == 'ProviderError'
        and result['delivered_at'] is None
        and result['answer'] is None
        and result.get('input_sha256') is not None
        and result['input_sha256'] == call.get('prompt_sha256')
        and score['quality'] == 0
        and score['timely_quality'] == 0
        and score['delivered'] is False
        and metrics['arrivals'] == len(results) == len(metrics['rows'])
    ):
        raise ValueError('output-limit failure is not a confirmed zero-score task')
