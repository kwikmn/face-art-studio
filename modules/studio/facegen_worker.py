"""JSON-lines worker. A separate process makes cancellation release CUDA memory."""

import json
import os
import sys
import traceback

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def emit(kind, **data):
    print(
        "FACELAB:" + json.dumps(dict(kind=kind, **data), ensure_ascii=True), flush=True
    )


def main():
    emit("status", message="Preparing the local generation runtime…")
    from modules.studio.facegen import Generator

    import torch

    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    generator = Generator()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            generator.generate(request, request["output_dir"], emit)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            message = str(error)
            if isinstance(error, ImportError):
                message = (
                    "Face Lab dependencies are missing or incompatible. Install requirements-facegen.txt in the Studio environment. "
                    + message
                )
            if "out of memory" in message.lower():
                message = "GPU memory is full. Select RAM offload or a smaller image size, then retry."
            emit("error", message=message)
            # Drop all GPU state after a failed load/run before accepting a retry.
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
