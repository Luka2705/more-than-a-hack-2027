"""Validate the submission, provenance, and optional byte-for-byte reproduction.

Never opens a ground-truth file or a video. Prints checks, not match records.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from evaluation.possession_metrics import read_label_csv, validate_submission
from games import GAME, GAMES, half_video_blocks
from solution.infer import build, sha256


def verify(submission, reproduce=False, data_dir=None):
    path = Path(submission).resolve()
    table = read_label_csv(path)
    reason = validate_submission(table, GAMES[GAME]["n_video_frames"])
    if reason:
        raise ValueError(reason)
    active = np.zeros(len(table), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        active[first:last + 1] = True
    labels = table["label"].to_numpy()
    if not np.all(labels[~active] == "X") or not np.isin(labels[active], ["H", "A", "D"]).all():
        raise ValueError("Labels do not respect the published half boundaries.")
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if sha256(path) != manifest["submission_sha256"]:
        raise ValueError("Submission differs from its generation manifest.")
    root = Path(__file__).resolve().parents[1]
    for source, expected in manifest["source_sha256"].items():
        if sha256(root / source) != expected:
            raise ValueError("Source changed since generation: " + source)
    print("PASS: official CSV contract (199,319 frames), half boundaries, source/output hashes.")
    if reproduce:
        import builtins
        original_open = builtins.open

        def guarded_open(file, *args, **kwargs):
            if isinstance(file, (str, bytes, Path)):
                p = Path(file).resolve()
                if ("gt" in p.parts and "evaluation" in p.parts) or p.suffix.lower() in {".mp4", ".mov"}:
                    raise AssertionError("Inference attempted forbidden ground-truth/video access.")
            return original_open(file, *args, **kwargs)

        print("Reproducing locally with network and ground-truth/video access blocked...", flush=True)
        with tempfile.TemporaryDirectory() as folder:
            with patch("builtins.open", guarded_open), \
                    patch("socket.socket", side_effect=AssertionError("Unexpected network access")), \
                    contextlib.redirect_stdout(io.StringIO()):
                repeated = build(data_dir, str(Path(folder) / "repeat.csv"))
            if sha256(repeated) != manifest["submission_sha256"]:
                raise ValueError("Reproduction is not byte-identical.")
            repeated_manifest = json.loads(repeated.with_suffix(".manifest.json").read_text())
            if repeated_manifest["inputs_sha256"] != manifest["inputs_sha256"]:
                raise ValueError("Reproduction used different input files.")
        print("PASS: byte-identical reproduction; no video, ground truth, or network accessed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", nargs="?", default="artifacts/Import-Export.csv")
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    try:
        verify(args.submission, args.reproduce, args.data_dir)
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print("FAIL: {}".format(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
