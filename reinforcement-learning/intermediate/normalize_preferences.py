#!/usr/bin/env python3
import csv
import re
from pathlib import Path
from collections import Counter

IN = Path(__file__).parent / 'preference_data.csv'
OUT = Path(__file__).parent / 'preference_data_normalized.csv'

# simple canonical mapping via substring checks
def normalize_label(text: str) -> str:
    if not text:
        return 'other'
    s = text.lower()
    s = re.sub(r"[^a-z0-9 ]+", ' ', s)
    s = re.sub(r"\s+", ' ', s).strip()
    if any(k in s for k in ('no acute', 'no evidence', 'negative for', 'normal', 'no pneumothorax', 'no focal', 'no focal airspace')):
        return 'normal'
    if 'cardiomeg' in s:
        return 'cardiomegaly'
    if 'atelect' in s or 'collapse' in s:
        return 'atelectasis'
    if 'pneumonia' in s or 'consolidat' in s:
        return 'pneumonia'
    if 'edema' in s or 'pulmonary edema' in s:
        return 'pulmonary_edema'
    if 'emphysem' in s or 'copd' in s:
        return 'emphysema'
    if 'fibros' in s:
        return 'pulmonary_fibrosis'
    if 'effusion' in s or 'pleural effusion' in s:
        return 'pleural_effusion'
    if 'pneumothorax' in s:
        return 'pneumothorax'
    if 'granuloma' in s or 'calcinos' in s:
        return 'granuloma'
    if 'nodule' in s:
        return 'nodule'
    if 'scar' in s:
        return 'scar'
    if 'opacity' in s:
        return 'opacity'
    if 'indwelling' in s or 'catheter' in s or 'tube' in s:
        return 'device'
    if 'fract' in s or 'rib' in s:
        return 'fracture'
    # fallback: return cleaned short text (max 40 chars) or 'other'
    return s[:40] if len(s)>=3 else 'other'

# clean judge explanation: collapse extremely long repeated substrings and trim
_repeat_re = re.compile(r'(.{10,200}?)\\1{2,}', re.DOTALL)
_nonprint = re.compile(r'[^\x09\x0A\x0D\x20-\x7E]+')

def clean_explanation(text: str) -> str:
    if not text:
        return ''
    t = text.strip()
    # replace non-printable chars
    t = _nonprint.sub(' ', t)
    # collapse repeated blocks
    t = _repeat_re.sub(r'\1', t)
    # trim long explanations
    if len(t) > 1000:
        t = t[:1000] + '...'
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def main():
    if not IN.exists():
        print('Input file not found:', IN)
        return
    rows_out = []
    stats_gt = Counter()
    stats_ch = Counter()
    total = 0
    with IN.open(newline='', encoding='utf-8', errors='replace') as fin:
        reader = csv.DictReader(fin)
        for r in reader:
            total += 1
            doctor_gt = (r.get('doctor_gt') or '').strip()
            chosen = (r.get('chosen') or '').strip()
            judge_expl = (r.get('judge_explanation') or '').strip()

            doctor_gt_n = normalize_label(doctor_gt)
            chosen_n = normalize_label(chosen)
            cleaned_expl = clean_explanation(judge_expl)

            stats_gt[doctor_gt_n] += 1
            stats_ch[chosen_n] += 1

            # simple garbage filter: skip if explanation is extremely long and contains many repeats
            if len(judge_expl) > 3000 and (judge_expl.count('Diagnosis:')>3 or judge_expl.count('Calcinosis')>10):
                continue

            out = {
                'instruction': r.get('instruction',''),
                'doctor_gt_raw': doctor_gt,
                'doctor_gt': doctor_gt_n,
                'chosen_raw': chosen,
                'chosen': chosen_n,
                'score_a': r.get('score_a',''),
                'score_b': r.get('score_b',''),
                'judge_explanation_raw': judge_expl,
                'judge_explanation': cleaned_expl,
                'rejected': r.get('rejected','')
            }
            rows_out.append(out)

    # write normalized CSV
    with OUT.open('w', newline='', encoding='utf-8') as fout:
        fieldnames = ['instruction','doctor_gt_raw','doctor_gt','chosen_raw','chosen','score_a','score_b','judge_explanation_raw','judge_explanation','rejected']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f'Read {total} rows, wrote {len(rows_out)} normalized rows to {OUT}')
    print('\nTop doctor_gt canonical counts:')
    for k,v in stats_gt.most_common(30):
        print(v, k)
    print('\nTop chosen canonical counts:')
    for k,v in stats_ch.most_common(30):
        print(v, k)

if __name__ == '__main__':
    main()
