"""End-to-end batch=1 harness. Caller supplies preloaded models and gate rows."""
import hashlib
import math
import random
import statistics
import time
from .core import infer


def fixed_sample(records, maximum=1000):
    return sorted(records,key=lambda r:(hashlib.sha256(str(r['input_id']).encode()).hexdigest(),
                                        str(r['input_id'])))[:maximum]


def sequential_benchmark(records, *, t1, t2, thresholds, clock=time.perf_counter):
    sample = fixed_sample(records)
    if not sample:
        raise ValueError('empty_speed_sample')
    paths = [('always_normal',None),('t1',None),('t2',None)] + [('cascade',t) for t in thresholds]
    for mode,tau in paths:
        for row in sample[:100]:
            result = infer([row],mode=mode,t1=t1,t2=t2,tau=tau or 0)[0]
            if result['status'] != 'ok':
                raise ValueError('invalid_speed_sample')
    measurements = []
    rng = random.Random(42)
    for repetition in range(5):
        order = list(paths)
        rng.shuffle(order)
        for order_index,(mode,tau) in enumerate(order):
            latencies, calls = [], 0
            for row in sample:
                start = clock()
                result = infer([row],mode=mode,t1=t1,t2=t2,tau=tau or 0)[0]
                elapsed = clock()-start
                if result['status'] != 'ok':
                    raise ValueError('inference_failed_in_benchmark')
                latencies.append(elapsed)
                calls += int(result['t2_called'])
            ordered = sorted(latencies)
            measurements.append(dict(repetition=repetition,order_index=order_index,mode=mode,tau=tau,
                n=len(sample),t2_calls=calls,mean_seconds=statistics.mean(latencies),
                median_seconds=statistics.median(latencies),p95_seconds=ordered[math.ceil(.95*len(ordered))-1],
                throughput_per_second=len(sample)/sum(latencies),raw_seconds=latencies))
    return dict(batch_size=1,sample_ids=[r['input_id'] for r in sample],
                timing='raw URL through feature extraction, routing, model, host result and restoration',
                measurements=measurements)
