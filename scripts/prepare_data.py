"""Extract official nested TSV ZIPs and sample CSV; reviews stay outside Git."""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', help='Official word2vec-nlp-tutorial.zip')
    parser.add_argument('--destination', default='data')
    args = parser.parse_args()
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        for name in ['labeledTrainData.tsv.zip', 'testData.tsv.zip', 'sampleSubmission.csv']:
            target = destination / name
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite {target}')
            with archive.open(name) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from data_utils import read_data, shared_split
    train, val, fingerprint = shared_split(read_data(destination / 'labeledTrainData.tsv.zip'),
                                         'configs/split_seed42.json', 42)
    print(f'train={len(train)} validation={len(val)} split_sha256={fingerprint}')

if __name__ == '__main__':
    main()
