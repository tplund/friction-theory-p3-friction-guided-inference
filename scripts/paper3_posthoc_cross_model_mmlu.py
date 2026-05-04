"""Paper 3 post-hoc #2: cross-model consistency on MMLU-Pro STEM.

Cell 3 (Qwen3-235B) and Cell 4 (LFM2) both ran MMLU-Pro STEM with the
same adaptive pipeline. Are the same questions lifted by both models,
or does each rescue different subsets?

- Same: lift is benchmark-intrinsic (some questions benefit from iteration
  regardless of model)
- Different: lift is model-specific (each model rescues its own failure mode)

Key for generalizability claims in Paper 3.
"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from collections import defaultdict
import numpy as np

def load(fp):
    return [json.loads(l) for l in open(fp, encoding='utf-8') if l.strip()]

QWEN = load('data/results/adaptive_mmlu_pro_qwen3_235b.jsonl')
LFM2 = load('data/results/adaptive_mmlu_pro_liquid_lfm2.jsonl')

def extract_lift(rows):
    """Return dict {idx: (vanilla_correct, committed_correct)}"""
    out = {}
    for r in rows:
        idx = r.get('idx')
        v = r.get('vanilla_correct')
        c = r.get('committed_correct')
        if idx is not None and v is not None and c is not None:
            out[idx] = (bool(v), bool(c))
    return out

q_map = extract_lift(QWEN)
l_map = extract_lift(LFM2)

shared = set(q_map.keys()) & set(l_map.keys())
print(f'Qwen3-235B: {len(q_map)} questions')
print(f'LFM2:       {len(l_map)} questions')
print(f'Shared:     {len(shared)} questions')

if len(shared) < 50:
    print('Too few shared questions for meaningful comparison — different question sets')
    sys.exit(0)

# For shared questions: classify lift type
categories = defaultdict(int)
rescue_qwen = set()
rescue_lfm2 = set()
rescue_both = set()
lose_qwen = set()
lose_lfm2 = set()

for idx in shared:
    qv, qc = q_map[idx]
    lv, lc = l_map[idx]

    q_rescued = (not qv) and qc
    l_rescued = (not lv) and lc
    q_lost = qv and (not qc)
    l_lost = lv and (not lc)

    if q_rescued: rescue_qwen.add(idx)
    if l_rescued: rescue_lfm2.add(idx)
    if q_rescued and l_rescued: rescue_both.add(idx)
    if q_lost: lose_qwen.add(idx)
    if l_lost: lose_lfm2.add(idx)

print(f'\n=== RESCUE ANALYSIS (shared n={len(shared)}) ===')
print(f'Rescued by Qwen:  {len(rescue_qwen)} ({len(rescue_qwen)/len(shared)*100:.1f}%)')
print(f'Rescued by LFM2:  {len(rescue_lfm2)} ({len(rescue_lfm2)/len(shared)*100:.1f}%)')
print(f'Rescued by BOTH:  {len(rescue_both)} ({len(rescue_both)/len(shared)*100:.1f}%)')
print(f'Rescued by ONE only: Qwen-only={len(rescue_qwen-rescue_lfm2)}, LFM2-only={len(rescue_lfm2-rescue_qwen)}')

# Jaccard similarity of rescue sets
intersect = len(rescue_qwen & rescue_lfm2)
union = len(rescue_qwen | rescue_lfm2)
jaccard = intersect / union if union else 0
print(f'\nJaccard(rescued-by-Qwen, rescued-by-LFM2) = {jaccard:.3f}')
print(f'  jaccard < 0.2  → different rescues (model-specific lift)')
print(f'  jaccard > 0.5  → shared rescues (benchmark-intrinsic)')
print(f'  jaccard ~0.3-0.4 → partial overlap (mixed)')

# Expected overlap if independent
if len(shared) > 0:
    p_qwen = len(rescue_qwen) / len(shared)
    p_lfm2 = len(rescue_lfm2) / len(shared)
    expected_both = p_qwen * p_lfm2 * len(shared)
    print(f'\nExpected both-rescue under independence: {expected_both:.1f}')
    print(f'Observed both-rescue: {len(rescue_both)}')
    ratio = len(rescue_both) / expected_both if expected_both > 0 else 0
    print(f'Observed / Expected: {ratio:.2f}x')
    print(f'  ~1.0 = independent (model-specific); >1.5 = correlated (benchmark-intrinsic)')

# Fisher exact test
try:
    from scipy.stats import fisher_exact
    # 2x2 table: (rescued by LFM2? | rescued by Qwen?)
    a = len(rescue_both)
    b = len(rescue_qwen - rescue_lfm2)
    c = len(rescue_lfm2 - rescue_qwen)
    d = len(shared - rescue_qwen - rescue_lfm2)
    odds, p = fisher_exact([[a, b], [c, d]])
    print(f'\nFisher exact test:')
    print(f'  Odds ratio: {odds:.2f}')
    print(f'  p-value:    {p:.4f}')
    print(f'  odds >> 1 and p < 0.05 = significantly correlated rescues')
except ImportError: pass

import os
os.makedirs('data/results', exist_ok=True)
with open('data/results/paper3_posthoc_cross_model_mmlu.json', 'w') as f:
    json.dump({
        'n_shared': len(shared),
        'rescued_qwen': len(rescue_qwen),
        'rescued_lfm2': len(rescue_lfm2),
        'rescued_both': len(rescue_both),
        'jaccard': jaccard,
    }, f, indent=2)
print('\nSaved: data/results/paper3_posthoc_cross_model_mmlu.json')
