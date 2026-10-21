"""Two-GPU BF16 LoRA training entrypoint for MiniCPM-o 4.5 audio-to-text."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .config import Config, load_config
from .data import ManifestDataset, MiniCPMODataCollator
from .model import (
    enable_gradient_checkpointing,
    load_processor,
    load_trainable_model,
    parameter_groups,
    trainable_summary,
)


def _manifest_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _move_data(data: object, device: torch.device) -> object:
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {key: _move_data(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [_move_data(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(_move_data(value, device) for value in data)
    return data


def _forward(model, batch: dict[str, object], device: torch.device):
    data = _move_data(batch["data"], device)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    return model(
        data=data,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )


@torch.no_grad()
def _evaluate(model, loader, accelerator: Accelerator, max_batches: int = 32) -> float:
    model.eval()
    loss_sum = torch.zeros((), device=accelerator.device)
    count = torch.zeros((), device=accelerator.device)
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        output = _forward(model, batch, accelerator.device)
        loss_sum += output.loss.detach().float()
        count += 1
    totals = accelerator.reduce(torch.stack((loss_sum, count)), reduction="sum")
    model.train()
    return float((totals[0] / totals[1].clamp_min(1)).item())


def _save_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    output_dir: Path,
    step: int,
    metadata: dict[str, object],
) -> None:
    checkpoint = output_dir / f"checkpoint-{step:08d}"
    if checkpoint.exists() and (checkpoint / "COMPLETE").is_file():
        accelerator.wait_for_everyone()
        return
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete checkpoint: {checkpoint}")
    if accelerator.is_main_process:
        (checkpoint / "state").mkdir(parents=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(checkpoint / "adapter", safe_serialization=True)
        torch.save(
            {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
            checkpoint / "state" / "optimizer_scheduler.pt",
        )
        (checkpoint / "metadata.json").write_text(
            json.dumps({**metadata, "step": step}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(accelerator.device),
        },
        checkpoint / "state" / f"rng-rank-{accelerator.process_index:05d}.pt",
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (checkpoint / "COMPLETE").write_text("ok\n", encoding="utf-8")
    accelerator.wait_for_everyone()


def _resume_metadata(resume: str | None, expected_manifest_hash: str) -> tuple[int, str | None]:
    if resume is None:
        return 0, None
    checkpoint = Path(resume)
    if not (checkpoint / "COMPLETE").is_file():
        raise ValueError(f"Checkpoint is incomplete: {checkpoint}")
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("train_manifest_sha256") != expected_manifest_hash:
        raise ValueError("Resume manifest hash does not match the current training manifest")
    return int(metadata.get("step", 0)), str(checkpoint / "adapter")


def _load_compact_state(accelerator: Accelerator, optimizer, scheduler, resume: str) -> None:
    checkpoint = Path(resume)
    state = torch.load(
        checkpoint / "state" / "optimizer_scheduler.pt",
        map_location="cpu",
        weights_only=False,
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng = torch.load(
        checkpoint / "state" / f"rng-rank-{accelerator.process_index:05d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"], accelerator.device)


def run(config: Config, *, resume: str | None = None, max_steps_override: int | None = None) -> None:
    max_steps = max_steps_override or config.train.max_steps
    accelerator = Accelerator(
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    set_seed(config.train.seed, device_specific=True)
    torch.set_float32_matmul_precision("highest")

    output_dir = Path(config.train.output_dir)
    if accelerator.is_main_process:
        if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --resume or a new directory."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    manifest_hash = _manifest_sha256(config.data.train_manifest)
    initial_step, adapter_path = _resume_metadata(resume, manifest_hash)
    processor = load_processor(config.model)
    model = load_trainable_model(config.model, adapter_path=adapter_path)
    if config.train.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    train_dataset = ManifestDataset(config.data.train_manifest)
    collator = MiniCPMODataCollator(
        processor=processor,
        max_length=config.model.max_length,
        sample_rate=config.data.sample_rate,
        max_audio_seconds=config.data.max_audio_seconds,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.per_device_batch_size,
        shuffle=False,
        sampler=train_dataset.weighted_sampler(
            config.data.task_weights,
            seed=config.train.seed,
        ),
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        collate_fn=collator,
        drop_last=False,
    )
    eval_loader = None
    if config.data.eval_manifest and Path(config.data.eval_manifest).is_file():
        eval_dataset = ManifestDataset(config.data.eval_manifest)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.train.per_device_batch_size,
            shuffle=False,
            num_workers=max(0, config.data.num_workers // 2),
            pin_memory=True,
            collate_fn=collator,
        )

    optimizer = torch.optim.AdamW(
        parameter_groups(
            model,
            lora_lr=config.train.learning_rate_lora,
            projector_lr=config.train.learning_rate_projector,
            weight_decay=config.train.weight_decay,
        ),
        betas=(0.9, 0.95),
        eps=1.0e-8,
        fused=torch.cuda.is_available(),
    )
    warmup_steps = round(max_steps * config.train.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    if eval_loader is None:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    else:
        model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, eval_loader, scheduler
        )

    metadata: dict[str, object] = {
        "config": {"model": asdict(config.model), "data": asdict(config.data), "train": asdict(config.train)},
        "train_manifest_sha256": manifest_hash,
        "world_size": accelerator.num_processes,
        "parameters": trainable_summary(accelerator.unwrap_model(model)),
        "torch": torch.__version__,
    }
    if accelerator.is_main_process:
        (output_dir / "run.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))

    if resume:
        _load_compact_state(accelerator, optimizer, scheduler, resume)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = initial_step
    started = time.monotonic()
    while step < max_steps:
        made_progress = False
        for batch in train_loader:
            made_progress = True
            with accelerator.accumulate(model):
                output = _forward(model, batch, accelerator.device)
                loss = output.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at optimizer step {step}: {loss.item()}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            step += 1
            if step % config.train.logging_steps == 0:
                mean_loss = accelerator.reduce(loss.detach().float(), reduction="mean").item()
                elapsed = time.monotonic() - started
                if accelerator.is_main_process:
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "loss": round(mean_loss, 6),
                                "lr": [group["lr"] for group in optimizer.param_groups],
                                "steps_per_second": round(step / max(elapsed, 1e-6), 4),
                            }
                        ),
                        flush=True,
                    )
            if eval_loader is not None and step % config.train.eval_steps == 0:
                eval_loss = _evaluate(model, eval_loader, accelerator)
                if accelerator.is_main_process:
                    print(json.dumps({"step": step, "eval_loss": round(eval_loss, 6)}), flush=True)
            if step % config.train.save_steps == 0:
                _save_checkpoint(accelerator, model, optimizer, scheduler, output_dir, step, metadata)
            if step >= max_steps:
                break
        if not made_progress:
            raise RuntimeError("Training dataloader produced no batches")

    _save_checkpoint(accelerator, model, optimizer, scheduler, output_dir, step, metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(load_config(args.config), resume=args.resume, max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
