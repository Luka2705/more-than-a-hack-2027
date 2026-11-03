"""Create a source-only release from an explicit allowlist, never a repository dump."""

from pathlib import Path
import zipfile


def package():
    root = Path(__file__).resolve().parents[1]
    names = [
        ".gitignore", "LICENSE", "games.py", "requirements-solution.txt",
        "SOLUTION.md", "Import-Export.md", "run_solution.command",
        "SUBMISSION.md", "RULES.md", "DATA.md", "CHALLENGE.md", "EVALUATION.md", "FAQ.md",
        "starter/baseline.py", "starter/load_smart_tagging.py",
        "evaluation/possession_metrics.py", "evaluation/validate_submission.py",
        "evaluation/run_evaluation.py",
    ]
    names += [str(p.relative_to(root)) for folder in ("solution", "tests")
              for p in sorted((root / folder).glob("*.py"))]
    target = root / "dist" / "Import-Export-code.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ["README.md"] + sorted(names):
            source = root / ("SOLUTION.md" if name == "README.md" else name)
            info = zipfile.ZipInfo("Import-Export/" + name, date_time=(2026, 11, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith(".command") else 0o100644) << 16
            archive.writestr(info, source.read_bytes())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("Release archive integrity check failed.")
        for name in archive.namelist():
            if any(part in Path(name).parts for part in ("data", "artifacts", "gt", ".venv", ".git")):
                raise ValueError("Unexpected data directory in release.")
    print("PASS: source-only archive created: {}".format(target))
    return target


if __name__ == "__main__":
    package()
