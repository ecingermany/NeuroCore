"""TXT format EEG data loader.

Parses the hardware output TXT format:
    Raw Data File
    {date} Start Time
    {date} End Time

    Channel Number: 2
    Sample Per Line: 256
    Total Sample: 30720

    A  0
    {256 comma-separated integers}
    B  0
    {256 comma-separated integers}
    A  256
    ...
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EEGData:
    A_data: np.ndarray   # [n_seconds, 256]
    B_data: np.ndarray   # [n_seconds, 256]
    sample_rate: int
    total_sample: int
    n_seconds: int
    file_path: str


def load_eeg_txt(filepath: str) -> EEGData:
    """Load an EEG TXT file and return parsed data.

    Raises ValueError if the file format is invalid or data is insufficient.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"EEG file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    sample_rate = None
    total_sample = None
    data_start = 0

    for i, line in enumerate(lines):
        if line.startswith("Sample Per Line:"):
            sample_rate = int(line.split(":")[1].strip())
        elif line.startswith("Total Sample:"):
            total_sample = int(line.split(":")[1].strip())
        elif line.startswith("A  0"):
            data_start = i
            break

    if sample_rate is None:
        raise ValueError("Missing 'Sample Per Line' header")
    if total_sample is None:
        raise ValueError("Missing 'Total Sample' header")
    if data_start == 0:
        raise ValueError("No data found (missing 'A  0' marker)")

    data_lines = lines[data_start:]
    A_blocks = []
    B_blocks = []

    i = 0
    while i < len(data_lines) - 1:
        line = data_lines[i].strip()
        if line.startswith("A  "):
            vals = np.fromstring(data_lines[i + 1].strip(), sep=",", dtype=float)
            if len(vals) == sample_rate:
                A_blocks.append(vals)
            i += 2
        elif line.startswith("B  "):
            vals = np.fromstring(data_lines[i + 1].strip(), sep=",", dtype=float)
            if len(vals) == sample_rate:
                B_blocks.append(vals)
            i += 2
        else:
            i += 1

    if len(A_blocks) == 0 or len(B_blocks) == 0:
        raise ValueError("No valid data blocks found")

    A_data = np.array(A_blocks)
    B_data = np.array(B_blocks)

    if A_data.shape[0] != B_data.shape[0]:
        raise ValueError(
            f"Channel A/B frame count mismatch: A={A_data.shape[0]}, B={B_data.shape[0]}"
        )

    n_seconds = A_data.shape[0]

    return EEGData(
        A_data=A_data,
        B_data=B_data,
        sample_rate=sample_rate,
        total_sample=total_sample,
        n_seconds=n_seconds,
        file_path=str(filepath),
    )
