"""Launch FaceArt Studio with the project's CUDA runtime configuration."""

import isolated_runtime  # noqa: F401 - isolate before importing runtime dependencies.
import argparse
from runtime_bootstrap import prepare_runtime

prepare_runtime()
from modules.studio.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceArt Studio")
    parser.add_argument(
        "--source", help="Initial source image; later selections are remembered"
    )
    args = parser.parse_args()
    main(args.source)
