"""Export a tracked Git tree for review without publishing or modifying remotes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath


def selected(name):
    p = PurePosixPath(name)
    if name in {
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "requirements-dev.txt",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "CITATION.cff",
        ".github/workflows/validate.yml",
        "scripts/export_public.py",
    }:
        return True
    if name.startswith("licenses/") and p.suffix == ".txt":
        return True
    if not name.startswith("ArchesClimate/"):
        return False
    if any(part.startswith("ac_paper") for part in p.parts):
        return False
    if any(
        part
        in {
            "ac_paper",
            "ac_paper_old",
            "native_repro",
            "pcmdi",
            "pmcdi",
            "wandb",
            "__pycache__",
            ".ipynb_checkpoints",
            "outputs",
        }
        for part in p.parts
    ):
        return False
    if p.name in {
        "public_release.md",
        "reviewer_comments.txt",
        "reviewer_response_status.md",
        "upload_stats_to_hub.py",
        "last_two_years_avg_metrics.txt",
        "last_two_years_yearly_metrics.txt",
    }:
        return False
    if "configs/cluster/" in name and p.name not in {"local.yaml", "slurm.yaml"}:
        return False
    if p.suffix in {".ipynb", ".sh", ".slurm", ".pt", ".pth", ".ckpt", ".npy", ".nc"}:
        return False
    if p.suffix == ".npz":
        return any(
            part.startswith("bootstrap") and "results" in part for part in p.parts
        )
    return True


def export(root, ref, output):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    tree = git("rev-parse", "--verify", ref + "^{tree}").decode().strip()
    if output.exists():
        raise FileExistsError(f"Destination must not exist: {output}")
    entries = []
    excluded = 0
    for record in git("ls-tree", "-rz", tree).split(b"\0"):
        if not record:
            continue
        metadata, raw_name = record.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        name = raw_name.decode()
        if not selected(name):
            excluded += 1
            continue
        if (
            mode not in {"100644", "100755"}
            or kind != "blob"
            or ".." in PurePosixPath(name).parts
        ):
            raise ValueError(f"Unsupported export entry: {name}")
        entries.append((name, oid, mode))
    output.mkdir(parents=True)
    files = {}
    for name, oid, mode in entries:
        data = git("cat-file", "blob", oid)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o755 if mode == "100755" else 0o644)
        files[name] = hashlib.sha256(data).hexdigest()
    manifest = {"source_tree": tree, "excluded_files": excluded, "sha256": files}
    (output / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD", help="Commit or staged tree to export")
    args = parser.parse_args()
    root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode()
        .strip()
    )
    manifest = export(root, args.ref, args.output.resolve())
    print(f"Exported {len(manifest['sha256'])} tracked files to {args.output}")


if __name__ == "__main__":
    main()
