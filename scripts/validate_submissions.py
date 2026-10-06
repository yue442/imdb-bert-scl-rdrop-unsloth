import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from submission_io import validate_submission

parser = argparse.ArgumentParser()
parser.add_argument('--sample-file', required=True)
parser.add_argument('--directory', default='submission')
args = parser.parse_args()
paths = sorted(Path(args.directory).glob('*.csv'))
if not paths:
    raise SystemExit('No submission CSVs found')
reports = [validate_submission(path, args.sample_file) for path in paths]
if any(row['rows'] != 25000 for row in reports):
    raise SystemExit('Expected 25000 competition test rows')
print(json.dumps(reports, ensure_ascii=False, indent=2))
