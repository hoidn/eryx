"""Compare synthetic hot paths with the pre-audit revision.

Run from the repository root: PYTHONPATH=. python benchmarks/benchmark_pytorch_audit.py
Requires git history containing e3333dd and the test dependencies.
Timings include autograd graph construction, but not backward.
"""

import gc
import json
import statistics
import subprocess
import time
import torch
from tests.test_pytorch_audit import phonon_model, intensity_model
from eryx.models_torch import OnePhonon

namespace = {'__name__': 'audit_baseline'}
exec(compile(subprocess.check_output(['git', 'show', 'e3333dd:eryx/models_torch.py'], text=True), '<baseline>', 'exec'), namespace)
Baseline = namespace['OnePhonon']
torch.set_num_threads(1)

def elapsed(fn, device):
    for _ in range(2):
        fn()
    samples = []
    for _ in range(5):
        gc.collect()
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1000

for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
    model = phonon_model(torch.diag(torch.arange(1, 7, dtype=torch.float64, device=device)).to(torch.complex128))
    model.kvec = torch.arange(128 * 3, dtype=torch.float64, device=device).reshape(-1, 3) / 1000
    intensity = intensity_model(device, 4096)
    for name, m, method in [('phonons_128x6', model, 'compute_gnm_phonons'), ('intensity_4096x3', intensity, 'apply_disorder')]:
        old = lambda: getattr(Baseline, method)(m)
        new = lambda: getattr(OnePhonon, method)(m)
        old_result = old()
        old_values = m.Winv.detach().clone() if old_result is None else old_result.detach().clone()
        new_result = new()
        new_values = m.Winv if new_result is None else new_result
        torch.testing.assert_close(old_values, new_values, equal_nan=True)
        before = elapsed(old, device)
        after = elapsed(new, device)
        print(json.dumps(dict(device=device, case=name, before_ms=before, after_ms=after, speedup=before/after)), flush=True)
