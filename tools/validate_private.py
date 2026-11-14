"""Read-only local evidence validation. Output only anonymous aggregate errors.

No source files, personal values, identifiers or decrypted material are copied.
This checks existing evidence; it does not restart hardware/legacy-version research.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from neurocore.data.decoded_bq import parse_decoded_bfn
from neurocore.assessment.bqrequest import compute_report
from neurocore.assessment.bqtester import evaluate_assessment


def main():
    base = Path(r'Z:\Work\BQNF\NHS data collection tool\analysis\bq')
    errors = {}
    extras = dict(LogBand=0., Slope=0., AbsoluteStrength=0., RelativeStrength=0.)
    activity_matches = emotion_matches = 0
    total = 0
    for ban in sorted((base/'local/decoded').glob('*.ban.decoded')):
        raw = parse_decoded_bfn(ban.with_name(ban.name.replace('.ban.', '.bfn.')).read_bytes().decode('cp949'))
        result = compute_report(raw.stages[:3], raw.age, raw.percentages, raw.weights)
        text = ban.read_bytes().decode('cp949')
        def values(label):
            line = next(l for l in text.splitlines() if l.startswith(label+':'))
            return np.fromstring(line.split(':', 1)[1], sep=' ')
        labels = {'BRQ': '기초율동', 'SRQ': 'SRQ', 'ATQ': '주의지수', 'ACQ': '활성지수',
                  'EQ': '정서지수', 'AntiStress': '스트레스', 'Synchrony': '상관성',
                  'Symmetry': '세기대칭', 'CQ': '균형지수', 'BQ': 'BQ', 'Rate': '비율'}
        for key, label in labels.items():
            delta = float(np.max(np.abs(result[key]-values(label))))
            errors[key] = max(errors.get(key, 0), delta)
            assert delta <= (5.1e-7 if key == 'Synchrony' else 5.1e-5), key
        records = [np.fromstring(line, sep=' ') for line in text.splitlines() if re.match(r'^[63] ', line)]
        targets = dict(LogBand=np.array([r[1:] for r in records[-9:-3]]).reshape(3,2,6),
                       Slope=np.array([r[1:] for r in records[-3:]]),
                       AbsoluteStrength=values('절대세기'), RelativeStrength=values('상대세기'))
        for key, target in targets.items():
            delta = float(np.max(np.abs(np.asarray(result[key]).reshape(target.shape)-target)))
            extras[key] = max(extras[key], delta)
            assert delta <= 5.1e-6, key
        def stored_label(section):
            section_lines = text.splitlines()
            start = section_lines.index(section)
            return next(line.split(':',1)[1].strip() for line in section_lines[start:] if line.startswith('성향:'))
        activity_matches += stored_label('<활성지수2>') == ('부정적' if result['activity_label']=='Negative' else '긍정적')
        emotion_matches += stored_label('<정서지수>') == result['emotion_label']
        total += 1
    assert total == 20, 'Expected 20 known report pairs'
    # Validate the known-version 2026-11-11 session only; do not fit old P/W.
    sessions = list((base/'local/live-captures').glob('*/session.bfn.decoded'))
    assert len(sessions) == 1, 'Expected exactly one known live session'
    raw = parse_decoded_bfn(sessions[0].read_bytes().decode('cp949'))
    result = evaluate_assessment(raw.stages[3:])
    assert [r['percentage'] for r in result['stages']] == raw.percentages
    assert [r['weight'] for r in result['stages']] == raw.weights
    spec = importlib.util.spec_from_file_location('known_bq_reference', base/'local/verify_live_bq.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    for stage, mode, packaged in zip(raw.stages[3:], (1,4,2), result['stages']):
        assert reference.replay(stage, mode)['reward_sequence'] == packaged['rewards']
    assert activity_matches == emotion_matches == 20
    summary = dict(report_pairs=total, maximum_errors=errors, extras_maximum_errors=extras,
                   activity_label_matches=activity_matches, emotion_label_matches=emotion_matches,
                   known_training_stages_matched=3, known_training_rewards_matched=180,
                   private_data_copied=False)
    (ROOT/'docs/private-validation-summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
