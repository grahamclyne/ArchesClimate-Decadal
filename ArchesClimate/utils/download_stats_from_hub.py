"""Download external normalization statistics into the configured data directory."""

import argparse

from ArchesClimate.paths import path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="grahamclyne/ArchesClimate-Decadal-stats")
    parser.add_argument("--directory", default=str(path("stats")))
    parser.add_argument("--revision", default=None, help="Optional artifact revision")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    destination = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.directory,
        revision=args.revision,
    )
    print(f"Downloaded statistics into {destination}")


if __name__ == "__main__":
    main()
