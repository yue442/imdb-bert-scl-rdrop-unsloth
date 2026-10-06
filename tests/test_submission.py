import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from submission_io import canonical_id, validate_submission, write_submission

class SubmissionTests(unittest.TestCase):
    def test_canonical_ids_and_probability_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'submission.csv'
            sample = Path(tmp) / 'sample.csv'
            sample.write_text('id,sentiment\n123_10,0\n9_1,0\n', encoding='utf-8')
            values = [.00123456789, .987654321]
            report = write_submission(output, ['"123_10"', '"9_1"'], values, sample)
            self.assertEqual(report['rows'], 2)
            with output.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([float(x['sentiment']) for x in rows], values)
            self.assertEqual([x['id'] for x in rows], ['123_10', '9_1'])
            with self.assertRaises(FileExistsError):
                write_submission(output, ['123_10', '9_1'], values)

    def test_reject_bad_ids_values_headers_and_order(self):
        self.assertEqual(canonical_id('"00012_10"'), '00012_10')
        for value in ['', 'x"y', ' x']:
            with self.assertRaises(ValueError):
                canonical_id(value)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'out.csv'
            for ids, values in [(['a', 'a'], [.1, .2]), (['a'], [float('nan')]),
                                (['a'], [1.1]), (['a'], [])]:
                with self.assertRaises(ValueError):
                    write_submission(output, ids, values)
            output.write_text('id,sentiment\n"""a""",0.2\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_submission(output)
            output.write_text('id,sentiment\nb,0.2\na,0.3\n', encoding='utf-8')
            sample = Path(tmp) / 'sample.csv'
            sample.write_text('id,sentiment\na,0\nb,0\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_submission(output, sample)

if __name__ == '__main__':
    unittest.main()
