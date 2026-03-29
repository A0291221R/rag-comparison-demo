"""
finetuning/train.py — Fine-tune embedding model using sentence-transformers
with contrastive loss on (anchor, positive, negative) triplets.

Registers the fine-tuned model with a version tag for rollback support.
"""
from __future__ import annotations

import json
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DATASET_DIR = Path(__file__).parent.parent / "data" / "ft_dataset"
MODEL_OUTPUT_DIR = Path(__file__).parent.parent / "models"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VERSION_TAG = "embed-ft-v1"


def load_triplets(split: str = "train") -> list[dict]:
    path = DATASET_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run finetuning/dataset_prep.py first")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def train(
    base_model: str = BASE_MODEL,
    version_tag: str = VERSION_TAG,
    epochs: int = 3,
    batch_size: int = 32,
    warmup_steps: int = 100,
) -> Path:
    from sentence_transformers import SentenceTransformer, InputExample, losses  # type: ignore
    from torch.utils.data import DataLoader

    logger.info("finetuning_start", base_model=base_model, version=version_tag)

    triplets = load_triplets("train")
    examples = [
        InputExample(texts=[t["anchor"], t["positive"], t["negative"]])
        for t in triplets
    ]

    model = SentenceTransformer(base_model)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.TripletLoss(model=model)

    output_path = MODEL_OUTPUT_DIR / version_tag
    output_path.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        show_progress_bar=True,
        save_best_model=True,
    )

    # Write version manifest for rollback
    manifest = {
        "version": version_tag,
        "base_model": base_model,
        "epochs": epochs,
        "training_samples": len(triplets),
        "output_path": str(output_path),
    }
    manifest_path = output_path / "version_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("finetuning_complete", output=str(output_path), version=version_tag)
    return output_path


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging(json_output=False)
    train()
