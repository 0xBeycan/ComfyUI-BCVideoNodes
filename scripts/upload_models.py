"""Upload the converted model files (scripts/convert_models.py) and the model card to the
Hugging Face repository the nodes download from.

    HF_TOKEN=... python scripts/upload_models.py --dir /path/to/out [--card README.md] [--create]

The token is read from the HF_TOKEN environment variable at upload time and never written
anywhere. Only the files the nodes load are uploaded, each after checking that it is a
model file this package can load (its architecture and format version in the metadata).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_models import MODELS  # noqa: E402

from preprocess.models import checkpoint  # noqa: E402

REPO_ID = "beycanai/BCVideoNodes-models"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", required=True, help="folder holding the converted .safetensors files")
    parser.add_argument("--card", default=os.path.join(ROOT, "scripts", "model_card.md"), help="model card to upload as the repository README.md")
    parser.add_argument("--create", action="store_true", help="create the repository if it does not exist")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; export a write token for the repository first")

    uploads = []
    for name in args.models:
        path = os.path.join(args.dir, MODELS[name]["file"])
        if not os.path.isfile(path):
            raise SystemExit(f"{path} is missing; run scripts/convert_models.py first")
        meta = checkpoint.read_metadata(path)
        if meta.get("architecture") != name or meta.get("format_version") != str(checkpoint.FORMAT_VERSION):
            raise SystemExit(f"{path}: expected a {name} file of format {checkpoint.FORMAT_VERSION}, found "
                             f"{meta.get('architecture')!r} format {meta.get('format_version')!r}")
        uploads.append((path, MODELS[name]["file"]))
    if args.card:
        if not os.path.isfile(args.card):
            raise SystemExit(f"{args.card} is missing")
        uploads.append((args.card, "README.md"))

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if args.create:
        api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    for path, target in uploads:
        print(f"uploading {path} -> {REPO_ID}/{target}", flush=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=target, repo_id=REPO_ID, repo_type="model",
                        commit_message=f"Update {target}")


if __name__ == "__main__":
    main()
