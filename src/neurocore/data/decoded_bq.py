"""Read already-decoded research BFN text. Does not decrypt or validate signatures."""
import re
from dataclasses import dataclass
import numpy as np
from . import blocks


@dataclass
class DecodedAssessment:
    stages: list
    age: int
    percentages: list
    weights: list


def parse_decoded_bfn(text):
    """Parse supplied decoded text only; stages are [seconds, A/B, 256]."""
    lines = text.splitlines()
    heads = [i for i, line in enumerate(lines) if 'Information' in line and i > 24]
    if len(heads) != 6:
        raise ValueError('Expected six BFN stage sections')
    stages = []
    for start, end in zip(heads, heads[1:] + [len(lines)]):
        channel_rows = {'A': [], 'B': []}
        for k in range(start, end - 1):
            match = re.fullmatch(r'([AB]) +([0-9]+)', lines[k])
            if match:
                values = np.fromstring(lines[k+1], sep=',')
                channel_rows[match[1]].append(blocks(values, integer=True))
        if not channel_rows['A'] or len(channel_rows['A']) != len(channel_rows['B']):
            raise ValueError('Missing or mismatched BFN channels')
        stages.append(np.stack([channel_rows['A'], channel_rows['B']], axis=1))
    age = re.search(r'^Age: (\d+)\s*$', text, re.M)
    p = [int(v) for v in re.findall(r'^Percent: (\d+)', text, re.M)]
    w = [int(v) for v in re.findall(r'^Weight: (\d+)', text, re.M)][-3:]
    if age is None or len(p) != 3 or len(w) != 3:
        raise ValueError('Missing age or three P/W pairs')
    return DecodedAssessment(stages, int(age.group(1)), p, w)
