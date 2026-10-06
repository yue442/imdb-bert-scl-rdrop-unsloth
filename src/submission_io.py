"""Kaggle binary-classification CSV export and independent format checks."""
import csv
import hashlib
import math
from pathlib import Path


def canonical_id(value):
    value = str(value)
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    if not value or '"' in value or value != value.strip():
        raise ValueError(f"Invalid sample ID: {value!r}")
    return value


def validate_submission(path, sample_file=None, expected_ids=None):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['id', 'sentiment']:
            raise ValueError("Expected exactly id,sentiment columns")
        rows = list(reader)
    ids = [row['id'] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Submission is empty or contains duplicate IDs")
    if any(canonical_id(value) != value for value in ids):
        raise ValueError("Submission IDs must not contain literal surrounding quotes")
    for row in rows:
        if None in row or None in row.values():
            raise ValueError("Malformed submission row")
        number = float(row['sentiment'])
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("sentiment must be a finite positive-class probability")
    if sample_file:
        with Path(sample_file).open(newline='', encoding='utf-8-sig') as handle:
            official = [row['id'] for row in csv.DictReader(handle)]
        if ids != official:
            raise ValueError("Submission IDs/order differ from official sampleSubmission.csv")
    if expected_ids is not None and ids != [canonical_id(x) for x in expected_ids]:
        raise ValueError("Submission IDs/order differ from prediction inputs")
    return {'file': Path(path).name, 'rows': len(rows), 'unique_ids': True,
            'official_id_order': True if sample_file else None,
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def write_submission(path, ids, probabilities, sample_file=None):
    ids = [canonical_id(x) for x in ids]
    probabilities = list(probabilities)
    if len(ids) != len(probabilities) or not ids or len(ids) != len(set(ids)):
        raise ValueError("ID/probability lengths must match and IDs must be nonempty/unique")
    if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in probabilities):
        raise ValueError("Invalid positive-class probability")
    if sample_file:
        with Path(sample_file).open(newline='', encoding='utf-8-sig') as handle:
            official = [row['id'] for row in csv.DictReader(handle)]
        if ids != official:
            raise ValueError("IDs/order differ from the official sample")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'sentiment'])
        writer.writerows(zip(ids, (float(value) for value in probabilities)))
    return validate_submission(path, sample_file, ids)
