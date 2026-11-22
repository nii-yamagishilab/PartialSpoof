import math
import os
import pickle
import re
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.detection import DetectionCostFunction

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "metric" / "RangeEER.py"
VENDORED = ROOT / "project-NN-Pytorch-scripts.202102"


def make_fixture(root, spoof_scores, bona_fide_scores, scale):
    score_path = root / "dev_score_ali_test.pkl"
    ref_path = root / "spoof.rttm"
    duration_path = root / "reco2dur"
    frame = scale / 100
    scores = {"CON_E_0000001": spoof_scores, "LA_E_0000002": bona_fide_scores}
    arrays = [
        np.array([(uid, -score, score, label) for score in values], dtype=object)
        for label, (uid, values) in enumerate(scores.items())
    ]
    score_path.write_bytes(pickle.dumps(arrays))
    ref_path.write_text(
        f"SPEAKER CON_E_0000001 1 0.0 {len(spoof_scores) * frame:.3f} "
        "<NA> <NA> spoof <NA> <NA>\n", encoding="ascii",
    )
    duration_path.write_text(
        "".join(f"{uid} {len(values) * frame:.3f}\n" for uid, values in scores.items()),
        encoding="ascii",
    )
    args = [
        "--ref_rttm", str(ref_path), "--hyp_sco_ali", str(score_path),
        "--reco2dur_file", str(duration_path), "--scale", str(scale),
        "--save_dir", str(root / "output"),
    ]
    return args, scores, load_rttm(ref_path), (score_path, ref_path, duration_path)


def sweep_eer(scores, references, frame):
    results = []
    for threshold in sorted({score for values in scores.values() for score in values}):
        metric = DetectionCostFunction()
        for uid, values in scores.items():
            hypothesis = Annotation()
            for index, score in enumerate(values):
                if score <= threshold:
                    hypothesis[Segment(index * frame, (index + 1) * frame)] = "spoof"
            metric(
                references.get(uid, Annotation()), hypothesis,
                uem=Segment(0, len(values) * frame),
            )
        totals = metric.accumulated_
        fpr = totals["false alarm"] / totals["negative class total"]
        fnr = totals["miss"] / totals["positive class total"]
        results.append((abs(fpr - fnr), (fpr + fnr) / 2))
    return min(results)


class RangeEERTests(unittest.TestCase):
    def check_cli(self, spoof_scores, bona_fide_scores, scale, bounds=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, scores, references, inputs = make_fixture(
                root, spoof_scores, bona_fide_scores, scale,
            )
            original = [path.read_bytes() for path in inputs]
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(
                [str(VENDORED), str(ROOT), os.environ.get("PYTHONPATH", "")]
            )}
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), *args, *bounds],
                cwd=root, env=env, capture_output=True, text=True, timeout=35, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            match = re.search(r"RangeEER=([^,]+)", result.stdout)
            self.assertIsNotNone(match, result.stdout)
            actual = float(match.group(1))
            self.assertTrue(math.isfinite(actual))
            det_path = root / "output" / "RangeEER_det"
            det = np.loadtxt(det_path, ndmin=2)
            self.assertEqual(det.shape[1], 4)
            self.assertLess(len(det), 100)
            self.assertEqual(set(root.rglob("RangeEER_det")), {det_path})
            self.assertEqual([path.read_bytes() for path in inputs], original)
            gap, expected = sweep_eer(scores, references, scale / 100)
            self.assertAlmostEqual(actual, expected, delta=1e-6)
            return gap, actual

    def test_discrete_scores_terminate_even_when_tolerance_is_unattainable(self):
        for _ in range(2):
            gap, result = self.check_cli([0, 1, 3], [2, 4, 4, 5, 6], 2)
            self.assertGreater(gap, 1e-5)
            self.assertAlmostEqual(result, 4 / 15)

    def test_convergent_unique_scores_at_two_resolutions(self):
        for scale in (2, 4):
            with self.subTest(scale=scale):
                gap, result = self.check_cli(
                    [0.01, 1.02, 2.03], [3.04, 4.05, 5.06, 6.07, 7.08], scale,
                )
                self.assertEqual((gap, result), (0, 0))

    def test_explicit_bounds_with_the_same_percentile(self):
        _, result = self.check_cli(
            [0, 1, 3], [2, 4, 4, 5, 6], 2, ("--th_left", "2.01", "--th_right", "2.02"),
        )
        self.assertAlmostEqual(result, 4 / 15)

    def test_segment_eer_helper_returns_frr_far_and_selected_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _, _ = make_fixture(Path(directory), [1, 3, 4], [0, 2], 2)
            with patch.object(sys, "argv", [str(SCRIPT), *args]), patch.object(
                sys, "path", [str(VENDORED), *sys.path],
            ):
                helper = runpy.run_path(str(SCRIPT))["cal_eer_by_scolab"]

            frame = pd.DataFrame({
                "True_Class": [0, 0, 0, 1, 1],
                "Predict_Score_1": [1, 3, 4, 0, 2],
            })
            self.assertEqual(helper(frame, return_fpr_fnr=True), (0.5, 2 / 3, 1))
            self.assertEqual(helper(frame), ((0.5 + 2 / 3) / 2, 1))


if __name__ == "__main__":
    unittest.main()
