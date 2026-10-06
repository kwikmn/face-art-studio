"""Local FLUX.2 Klein generation; imported only in the isolated Face Lab worker."""

import gc
import hashlib
import json
import secrets
import time
import uuid
from pathlib import Path

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "FLUX.2-klein-4B"
PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
    "LICENSE.md",
    "README.md",
]


def validate_request(request):
    prompt = str(request.get("prompt", "")).strip()
    if not prompt or len(prompt) > 6000:
        raise ValueError("Enter a prompt of 1–6000 characters.")
    size = int(request.get("size", 768))
    count = int(request.get("count", 1))
    memory = request.get("memory", "auto")
    if size not in (512, 768, 1024) or count not in range(1, 5):
        raise ValueError("Choose a supported size and 1–4 candidates.")
    if memory not in ("auto", "gpu", "model", "sequential"):
        raise ValueError("Unknown memory mode.")
    seed = request.get("seed")
    seed = secrets.randbelow(2**32) if seed is None else int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be between 0 and 4294967295.")
    reference = request.get("reference")
    if reference and not Path(reference).is_file():
        raise ValueError("The reference photo is missing. Choose it again.")
    return dict(
        prompt=prompt,
        size=size,
        count=count,
        memory=memory,
        seed=seed,
        reference=reference,
    )


class Generator:
    def __init__(self):
        self.pipe = None
        self.mode = None

    def load(self, mode, emit):
        import torch
        from diffusers import Flux2KleinPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("Face Lab currently needs an NVIDIA CUDA GPU.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This build needs a GPU with BF16 support.")
        if self.pipe is not None and mode == self.mode:
            return
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        from huggingface_hub import snapshot_download

        emit(
            "status", message="Checking model files; the first download is about 16 GB…"
        )
        marker = MODEL_DIR / ".faceart-revision"
        if not marker.exists() or marker.read_text().strip() != REVISION:
            snapshot_download(
                MODEL_ID,
                revision=REVISION,
                local_dir=str(MODEL_DIR),
                allow_patterns=PATTERNS,
                max_workers=2,
            )
            marker.write_text(REVISION)
        emit("status", message="Loading FLUX.2 Klein 4B…")
        pipe = Flux2KleinPipeline.from_pretrained(
            str(MODEL_DIR), torch_dtype=torch.bfloat16, local_files_only=True
        )
        free, _ = torch.cuda.mem_get_info()
        # Include the text encoder and VAE, not just the 4B image transformer.
        weights = sum(path.stat().st_size for path in MODEL_DIR.rglob("*.safetensors"))
        required = weights + 2 * 1024**3  # activation/workspace headroom
        resolved = ("gpu" if free >= required else "model") if mode == "auto" else mode
        if resolved == "gpu":
            pipe.to("cuda")
        elif resolved == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        self.pipe, self.mode = pipe, mode
        self.resolved = resolved
        emit("status", message=f"Model ready · memory mode: {resolved}")

    def generate(self, request, output_dir, emit):
        import torch
        from PIL import Image, ImageOps, PngImagePlugin

        args = validate_request(request)
        self.load(args["memory"], emit)
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        reference = None
        reference_hash = None
        if args["reference"]:
            path = Path(args["reference"])
            reference_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as original:
                reference = ImageOps.exif_transpose(original).convert("RGB")
                reference.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        for index in range(args["count"]):
            seed = (args["seed"] + index) % 2**32

            def progress(pipe, step, timestep, values):
                emit(
                    "progress",
                    value=round(100 * (index + (step + 1) / 4) / args["count"]),
                    message=f"Candidate {index + 1}/{args['count']} · step {step + 1}/4",
                )
                return values

            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                result = self.pipe(
                    prompt=args["prompt"],
                    image=reference,
                    height=args["size"],
                    width=args["size"],
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                    callback_on_step_end=progress,
                ).images[0]
            elapsed = time.perf_counter() - started
            metadata = dict(
                model=MODEL_ID,
                revision=REVISION,
                prompt=args["prompt"],
                seed=seed,
                size=args["size"],
                steps=4,
                guidance_scale=1.0,
                mode="img2img" if reference else "txt2img",
                reference_sha256=reference_hash,
                memory=self.resolved,
                seconds=elapsed,
                peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            )
            png = PngImagePlugin.PngInfo()
            png.add_text("FaceArt Studio", json.dumps(metadata, ensure_ascii=False))
            destination = (
                folder
                / f"face-{time.strftime('%Y%m%d-%H%M%S')}-{seed}-{uuid.uuid4().hex[:6]}.png"
            )
            temporary = destination.with_suffix(".tmp")
            try:
                result.save(temporary, format="PNG", pnginfo=png)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            emit("result", path=str(destination), metadata=metadata)
        emit("done", message="Ready. Review a candidate and choose Use this face.")
