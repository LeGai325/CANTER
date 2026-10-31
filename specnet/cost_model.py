"""Offline selected, independently calibrated quantile trees and bucket baseline."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

from .profiles import Prediction


FEATURES = ["log2_length", "length_missing", "base_inflight", "optional_inflight",
            "base_extract_waiting", "optional_extract_waiting", "extracting",
            "service_kbit", "queue_bytes", "tcp_rtt_seconds", "rtt_missing", "telemetry_age", "qos_mode"]


def features(candidate, snapshot, blind=False):
    local = [math.log2(max(1,candidate.length_hint or 1)), float(candidate.length_hint is None),
             snapshot.base_active, snapshot.optional_active,snapshot.base_extract_waiting,
             snapshot.optional_extract_waiting,snapshot.extracting]
    if blind: return local
    values = snapshot.network_features
    if len(values) != 4: raise ValueError("missing network feature vector")
    rate, backlog, rtt, qos = values
    age = max(0,snapshot.at-snapshot.telemetry_at) if snapshot.telemetry_at is not None else -1
    # Keep non-finite telemetry visible to validation; max(0, nan) hides it.
    return local + [rate,backlog,max(0,rtt) if math.isfinite(rtt) else rtt,float(rtt<0),age,qos]


def wilson(successes, total):
    if not total: return [0.0,1.0]
    z=1.96;p=successes/total;d=1+z*z/total
    center=(p+z*z/(2*total))/d;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0,center-half),min(1,center+half)]


def quantile(values,q=.95):
    return sorted(values)[min(len(values)-1, max(0,math.ceil(q*len(values))-1))] if values else None


def _condition(vector, blind=False):
    # Capacity is a controlled, measured regime, not an interpolated guarantee.
    return 'blind' if blind else json.dumps([float(vector[7]), int(vector[12])], separators=(',', ':'))


def _window_state(vector):
    return 'base_busy' if vector[2] > 0 else 'base_idle'


def _bucket_key(source, vector, blind=False):
    return json.dumps([source, _condition(vector, blind), int(vector[0]//2)], separators=(',', ':'))


def _support(rows, blind=False):
    vectors = [s['blind_features' if blind else 'features'] for s in rows]
    support = dict(max_base_active=max(v[2] for v in vectors),
                   max_optional_active=max(v[3] for v in vectors),
                   max_base_extract_waiting=max(v[4] for v in vectors),
                   max_optional_extract_waiting=max(v[5] for v in vectors),
                   max_extracting=max(v[6] for v in vectors),
                   length_missing_observed=any(v[1] == 1 for v in vectors))
    if not blind:
        observed_rtt = [v[9] for v in vectors if v[10] == 0]
        support.update(max_queue_bytes=max(v[8] for v in vectors),
                       max_rtt_seconds=max(observed_rtt) if observed_rtt else None,
                       rtt_missing_observed=any(v[10] == 1 for v in vectors))
    return support


def _support_reason(vector, support, blind=False):
    if vector[1] and not support['length_missing_observed']:
        return 'length_measurement_missing', False
    if vector[2] > support['max_base_active'] or vector[3] > support['max_optional_active']:
        return 'load_out_of_support', True
    if any(vector[i] > support[name] for i, name in (
            (4, 'max_base_extract_waiting'), (5, 'max_optional_extract_waiting'), (6, 'max_extracting'))):
        return 'extraction_load_out_of_support', True
    if not blind:
        if vector[8] > support['max_queue_bytes']:
            return 'queue_out_of_support', True
        if vector[10] and not support['rtt_missing_observed']:
            return 'rtt_measurement_missing', True
        if not vector[10] and (support['max_rtt_seconds'] is None or vector[9] > support['max_rtt_seconds']):
            return 'rtt_out_of_support', True
    return None


class ReadyProfile:
    def __init__(self, model, metadata, *, blind=False, bucket=False):
        self.model,self.metadata,self.blind,self.bucket=model,metadata,blind,bucket
        self.version=metadata["version"]+(":blind" if blind else ":bucket" if bucket else ":gbt")
        # Old frozen profiles remain readable and retain their original decisions.
        # Only newly fitted v2 profiles claim conditional state support checking.
        schema = metadata.get('support_schema_version')
        self.support_validation = (f'conditional_state_v{schema}' if schema in (2, 3)
                                   else 'legacy_partial_support')
        if bucket and schema == 3:
            raise ValueError('window-state profiles require the quantile model, not legacy buckets')

    def predict(self,candidate,snapshot):
        m=self.metadata
        if snapshot.execution_version!=m["execution_version"]:
            return Prediction(None,None,"execution_version_mismatch")
        if candidate.source not in m["sources"]:return Prediction(None,None,"source_unknown")
        if not 0<=time.time()-m["created_unix"]<=m["max_profile_age_seconds"]:return Prediction(None,None,"profile_stale")
        if m["successful_samples"]<m["minimum_successes"]:return Prediction(None,None,"insufficient_samples")
        conditional = m.get('support_schema_version') in (2, 3)
        windowed = m.get('support_schema_version') == 3
        if not conditional and m["failure_fraction"]>m["maximum_failure_fraction"]:
            return Prediction(None,None,"failure_risk_out_of_support")
        if not self.blind:
            if snapshot.domain!=m["domain"]:return Prediction(None,None,"measurement_domain_mismatch")
            if snapshot.telemetry_at is None or not 0<=snapshot.at-snapshot.telemetry_at<=m["telemetry_max_age"]:
                return Prediction(None,None,"telemetry_stale",True)
            if len(snapshot.network_features)!=4:return Prediction(None,None,"network_features_missing",True)
            if snapshot.network_features[1]<0:return Prediction(None,None,"queue_measurement_missing",True)
        vector=features(candidate,snapshot,self.blind)
        if any(not math.isfinite(v) or v < 0 for v in vector):
            return Prediction(None,None,"invalid_profile_features",True)
        if not self.blind and vector[12] not in (0, 1):
            return Prediction(None,None,"invalid_profile_features",True)
        if vector[0]>m["max_log_size"]:return Prediction(None,None,"size_out_of_support")
        if vector[2]>m["max_base_active"] or vector[3]>m["max_optional_active"]:
            return Prediction(None,None,"load_out_of_support",True)
        if not self.blind and not m["service_kbit_range"][0]<=vector[7]<=m["service_kbit_range"][1]:
            return Prediction(None,None,"service_out_of_support",True)
        condition = None
        state = None
        if conditional:
            condition = m['conditions'].get(_condition(vector, self.blind))
            if condition is None:return Prediction(None,None,'network_condition_out_of_support',True)
            if condition['calibration_samples'] < m['minimum_condition_calibration_successes']:
                return Prediction(None,None,'condition_insufficient_calibration_samples',windowed)
            if condition['failure_fraction'] > m['maximum_failure_fraction']:
                return Prediction(None,None,'condition_failure_risk_out_of_support')
            unsupported = _support_reason(vector, condition['support'], self.blind)
            if unsupported:return Prediction(None,None,*unsupported)
            if windowed:
                state = condition.get('window_states', {}).get(_window_state(vector))
                if state is None:
                    return Prediction(None,None,'window_state_missing',True)
                if state['calibration_samples'] < m['minimum_state_calibration_successes']:
                    return Prediction(None,None,'state_insufficient_calibration_samples',True)
                if state['calibration_task_count'] < m['minimum_state_calibration_tasks']:
                    return Prediction(None,None,'state_insufficient_calibration_tasks',True)
                if state['failure_fraction'] > m['maximum_failure_fraction']:
                    return Prediction(None,None,'state_failure_risk_out_of_support',True)
                unsupported = _support_reason(vector, state['support'], self.blind)
                if unsupported:return Prediction(None,None,*unsupported)
        if self.bucket:
            # Frozen source x mode x size bucket; no labels or other-policy outcomes.
            key=(_bucket_key(candidate.source, vector, self.blind) if conditional
                 else str(int(vector[0]//2))+":"+str(int(vector[-1])))
            value=m["buckets"].get(key)
            if value is None:return Prediction(None,None,"bucket_insufficient_samples")
        else:value=float(self.model.predict([vector])[0])+(state['residual_margin'] if state is not None
                    else condition['residual_margin'] if condition else m["residual_margin"])
        reserve=candidate.length_hint or m["size_quantile"]
        return Prediction(max(0,value),max(1,reserve))


def fit_profile(samples,directory,*,execution_version,domain,blind=False,q=.95,
                window_samples=None,window_protocol=None,
                minimum_state_calibration_successes=8,minimum_state_calibration_tasks=3):
    """Splits are assigned by independent task before collection, never by outcome."""
    import joblib
    import numpy as np
    import sklearn
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_pinball_loss
    if not 0 < q <= 1:raise ValueError('quantile must be in (0, 1]')
    if window_samples is not None:
        if any(type(value) is not int or value <= 0 for value in
               (minimum_state_calibration_successes, minimum_state_calibration_tasks)):
            raise ValueError('positive integer window-state calibration thresholds required')
        # The caller selects by observed admission age, never outcome or answer.
        # Prove that no independent mechanism/main observations were injected,
        # relabelled, duplicated, or added only for residual calibration.
        canonical = lambda row: json.dumps(row, sort_keys=True, allow_nan=False)
        available = Counter(canonical(row) for row in samples)
        chosen = Counter(canonical(row) for row in window_samples)
        if not window_samples or chosen - available:
            raise ValueError('window samples must be a nonempty unchanged subset of transport samples')
        if {row['split'] for row in window_samples} != {'fit', 'validation', 'calibration'}:
            raise ValueError('window samples must retain all three independent splits')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    sets={part:[s for s in samples if s["split"]==part and s["outcome"]=="success"] for part in ("fit","validation","calibration")}
    ids={part:{s["task_id"] for s in samples if s["split"]==part} for part in sets}
    if any(ids[a]&ids[b] for a,b in (("fit","validation"),("fit","calibration"),("validation","calibration"))):
        raise ValueError("calibration task overlap")
    if min(map(len,sets.values()))<2:raise ValueError("too few successful samples in independent splits")
    for sample in samples:
        vector = sample['blind_features' if blind else 'features']
        if len(vector) != (7 if blind else len(FEATURES)) or any(not math.isfinite(v) or v < 0 for v in vector):
            raise ValueError('invalid calibration feature vector')
        if vector[1] not in (0, 1) or (not blind and (vector[10] not in (0, 1) or vector[12] not in (0, 1))):
            raise ValueError('invalid calibration missingness or QoS flag')
        if sample['outcome'] == 'success' and (not math.isfinite(sample['ready_seconds']) or sample['ready_seconds'] < 0):
            raise ValueError('invalid successful ready time')
    def xy(part):
        rows=sets[part];return np.array([s["blind_features" if blind else "features"] for s in rows]),np.array([s["ready_seconds"] for s in rows])
    x,y=xy('fit');vx,vy=xy('validation');cx,cy=xy('calibration')
    candidates=[]
    for depth in (1,2):
        model=GradientBoostingRegressor(loss='quantile',alpha=q,n_estimators=60,max_depth=depth,
            min_samples_leaf=2,learning_rate=.05,random_state=17).fit(x,y)
        candidates.append((float(mean_pinball_loss(vy,model.predict(vx),alpha=q)),depth,model))
    _,depth,model=min(candidates,key=lambda row:(row[0],row[1]))
    residual=cy-model.predict(cx)
    # Finite-sample upper rank; tiny pilots remain explicitly empirical.
    margin=max(0,float(quantile(list(residual),min(1,math.ceil((len(cy)+1)*q)/len(cy)))))
    successful=[s for s in samples if s['outcome']=='success'];counts=Counter(s['outcome'] for s in samples)
    # The pooled correction used to let fast/high-capacity samples mask errors in
    # the congested regime. Calibrate each measured capacity x QoS condition on
    # its own disjoint calibration tasks, including a condition-specific failure
    # rate and state support envelope. Tiny groups are marked unsupported.
    grouped = {}
    for sample in samples:
        vector = sample['blind_features' if blind else 'features']
        grouped.setdefault(_condition(vector, blind), []).append(sample)
    conditions = {}
    for key, rows in grouped.items():
        group_counts = Counter(s['outcome'] for s in rows)
        group_successes = group_counts['success']
        calibrated = [s for s in rows if s['split'] == 'calibration' and s['outcome'] == 'success']
        entry = dict(sample_count=len(rows), successful_samples=group_successes,
                     outcome_counts=dict(group_counts), failure_fraction=1-group_successes/len(rows),
                     success_fraction_ci95=wilson(group_successes, len(rows)),
                     calibration_samples=len(calibrated),
                     calibration_task_count=len({s['task_id'] for s in calibrated}),
                     support=_support(rows, blind), residual_margin=None)
        if calibrated:
            gx=np.array([s['blind_features' if blind else 'features'] for s in calibrated])
            gy=np.array([s['ready_seconds'] for s in calibrated])
            base=model.predict(gx)
            correction=max(0,float(quantile(list(gy-base),min(1,math.ceil((len(gy)+1)*q)/len(gy)))))
            gp=base+correction
            entry.update(residual_margin=correction,
                         calibration_underestimate_rate=float(np.mean(gy > gp)),
                         calibration_mean_absolute_error=float(np.mean(abs(gy-gp))),
                         calibration_mean_prediction=float(np.mean(gp)),
                         calibration_max_overrun=float(max(0,max(gy-gp))))
        conditions[key]=entry
    if window_samples is not None:
        window_groups = {}
        for sample in window_samples:
            vector = sample['blind_features' if blind else 'features']
            window_groups.setdefault((_condition(vector, blind), _window_state(vector)), []).append(sample)
        for condition in conditions.values():
            condition['window_states'] = {}
        for (key, state_name), rows in window_groups.items():
            state_counts = Counter(row['outcome'] for row in rows)
            successes = state_counts['success']
            calibrated = [row for row in rows if row['split'] == 'calibration' and row['outcome'] == 'success']
            state = dict(sample_count=len(rows), successful_samples=successes,
                         outcome_counts=dict(state_counts), failure_fraction=1-successes/len(rows),
                         success_fraction_ci95=wilson(successes, len(rows)),
                         independent_task_count=len({row['task_id'] for row in rows}),
                         independent_network_runs=len({row.get('run_id') for row in rows}),
                         calibration_samples=len(calibrated),
                         calibration_task_count=len({row['task_id'] for row in calibrated}),
                         calibration_run_count=len({row.get('run_id') for row in calibrated}),
                         calibration_task_ids=sorted({row['task_id'] for row in calibrated}),
                         split_counts=dict(Counter(row['split'] for row in rows)),
                         support=_support(rows, blind), residual_margin=None,
                         calibration_underestimate_rate=None,
                         calibration_mean_absolute_error=None,
                         calibration_mean_prediction=None, calibration_max_overrun=None)
            if calibrated:
                sx=np.array([row['blind_features' if blind else 'features'] for row in calibrated])
                sy=np.array([row['ready_seconds'] for row in calibrated])
                base=model.predict(sx)
                correction=max(0,float(quantile(list(sy-base),min(1,math.ceil((len(sy)+1)*q)/len(sy)))))
                predicted_state=base+correction
                state.update(residual_margin=correction,
                    calibration_underestimate_rate=float(np.mean(sy > predicted_state)),
                    calibration_mean_absolute_error=float(np.mean(abs(sy-predicted_state))),
                    calibration_mean_prediction=float(np.mean(predicted_state)),
                    calibration_max_overrun=float(max(0,max(sy-predicted_state))))
            conditions[key]['window_states'][state_name] = state
    predicted=model.predict(cx)+np.array([conditions[_condition(v, blind)]['residual_margin'] for v in cx])
    buckets={}
    for s in sets['fit']+sets['calibration']:
        vector=s['blind_features' if blind else 'features']
        key=_bucket_key(s['source'],vector,blind);buckets.setdefault(key,[]).append(s['ready_seconds'])
    bucket_values={k:quantile(v,q) for k,v in buckets.items() if len(v)>=2}
    metadata=dict(version='ready-quantile-gbt-v2',support_schema_version=2,
        condition_definition='local only' if blind else 'exact measured service_kbit x qos_mode',
        conditions=conditions,minimum_condition_calibration_successes=2,
        support_note='Observed upper envelopes; no invented headroom. Zero load is allowed. Recalibrate unseen capacity regimes or out-of-support states before main evaluation.',
        calibration_note='Finite-rank empirical correction per condition; candidate samples within a task/run are correlated, so this is not an independent per-request coverage guarantee.',
        execution_version=execution_version,domain=domain,
        created_unix=time.time(),max_profile_age_seconds=7*86400,telemetry_max_age=2.0,
        feature_names=FEATURES[:7] if blind else FEATURES,blind=blind,quantile=q,selected_depth=depth,
        residual_margin=margin,residual_margin_note='Pooled diagnostic only; v2 predictions use condition-specific residual margins.',
        successful_samples=len(successful),minimum_successes=8,
        maximum_failure_fraction=.20,failure_fraction=1-len(successful)/len(samples),
        outcome_counts=dict(counts),success_fraction_ci95=wilson(len(successful),len(samples)),
        independent_task_count=len({s['task_id'] for s in samples}),
        independent_network_runs=len({s.get('run_id') for s in samples}),
        outcome_ci_note='candidate-level descriptive Wilson interval; correlated candidates are not independent task repetitions',
        sources=sorted({s['source'] for s in samples}),split_task_ids={k:sorted(v) for k,v in ids.items()},
        max_log_size=max(s['features'][0] for s in samples)+1,
        max_base_active=max(s['features'][2] for s in samples),max_optional_active=max(s['features'][3] for s in samples),
        service_kbit_range=[min(s['features'][7] for s in samples),max(s['features'][7] for s in samples)],
        size_quantile=int(quantile([s['payload_bytes'] for s in successful],q)),buckets=bucket_values,
        validation_candidates=[dict(pinball_loss=a,max_depth=b) for a,b,_ in candidates],
        calibration_underestimate_rate=float(np.mean(cy>predicted)),
        calibration_mean_absolute_error=float(np.mean(abs(cy-predicted))),
        calibration_mean_prediction=float(np.mean(predicted)),
        calibration_max_overrun=float(max(0,max(cy-predicted))),
        sklearn_version=sklearn.__version__,scope='pilot empirical support; no per-request deadline guarantee')
    if window_samples is not None:
        window_path=directory/'window_samples.json'
        window_path.write_text(json.dumps(window_samples,indent=2,allow_nan=False))
        metadata.update(version='ready-quantile-gbt-v3',support_schema_version=3,
            minimum_state_calibration_successes=minimum_state_calibration_successes,
            minimum_state_calibration_tasks=minimum_state_calibration_tasks,
            window_state_definition='base_busy iff pre-admission base_inflight > 0; otherwise base_idle',
            window_protocol=window_protocol if window_protocol is not None else {
                'selection':'caller-provided admission window; unchanged subset of full transport samples'},
            window_samples_file='window_samples.json',
            window_samples_sha256=hashlib.sha256(window_path.read_bytes()).hexdigest(),
            window_sample_count=len(window_samples),
            window_outcome_counts=dict(Counter(row['outcome'] for row in window_samples)),
            residual_margin_note='Full-run pooled and condition margins are diagnostics only; v3 predictions use '
                'the independent calibration split within the declared admission window and current base state.',
            calibration_note='Same full fit/validation samples and selected tree as v2; only residual calibration '
                'and state support use the window subset. Correlated candidates/repeated arrival orders are '
                'not additional independent questions or a per-request coverage guarantee.')
    joblib.dump(model,directory/'model.joblib')
    metadata['model_sha256']=hashlib.sha256((directory/'model.joblib').read_bytes()).hexdigest()
    (directory/'PROFILE.json').write_text(json.dumps(metadata,indent=2))
    (directory/'samples.json').write_text(json.dumps(samples,indent=2))
    return ReadyProfile(model,metadata,blind=blind)


def load_profile(directory,*,bucket=False):
    import joblib
    directory=Path(directory);m=json.loads((directory/'PROFILE.json').read_text())
    if hashlib.sha256((directory/'model.joblib').read_bytes()).hexdigest()!=m['model_sha256']:
        raise ValueError('profile model hash mismatch')
    if m.get('support_schema_version') == 3:
        if (m.get('window_samples_file') != 'window_samples.json' or
                hashlib.sha256((directory/'window_samples.json').read_bytes()).hexdigest()!=m['window_samples_sha256']):
            raise ValueError('profile window samples hash mismatch')
    return ReadyProfile(joblib.load(directory/'model.joblib'),m,blind=m['blind'],bucket=bucket)


class ReferenceDecision:
    """Declared reference controllers; common hard guards remain in Coordinator."""
    def __init__(self,kind,thresholds=(1,2,4)):
        self.kind,self.thresholds,self.initial=kind,thresholds,{}
    def __call__(self,request,candidate,snapshot,prediction):
        from .contracts import Policy
        if self.kind=='time_gate_off':
            return Policy.FULL,None,prediction
        if self.kind=='initial_only':
            if request.contract.task_id not in self.initial:
                # Freeze a ranked prefix budget using the entry prediction; no subsequent telemetry.
                time_left=max(0,request.cutoff-snapshot.at)
                k=0 if prediction.ready_seconds is None else min(request.contract.optional_limit,
                    int(time_left/max(.001,prediction.ready_seconds)))
                self.initial[request.contract.task_id]=k
            return Policy.FIXED_K,self.initial[request.contract.task_id],prediction
        if self.kind=='four_mode':
            if prediction.ready_seconds is None:return Policy.PROGRESSIVE,None,prediction
            ratio=(request.cutoff-snapshot.at)/max(.001,prediction.ready_seconds)
            a,b,c=self.thresholds
            k=0 if ratio<a else 1 if ratio<b else 2 if ratio<c else request.contract.optional_limit
            fraction=0 if ratio<a else .25 if ratio<b else .5 if ratio<c else 1.0
            byte_budget=int(request.contract.optional_payload_budget*fraction)
            reserve=min(prediction.reserve_bytes or request.contract.per_document_bytes,request.contract.per_document_bytes)
            if request.admitted>=k or request.spent+request.reserved+reserve>byte_budget:
                # A lower mode can recover while bounded idle wait remains.
                return Policy.PROGRESSIVE,None,Prediction(None,None,'four_mode_hold',True)
        return Policy.PROGRESSIVE,None,prediction
