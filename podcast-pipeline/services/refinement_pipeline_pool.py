"""Two-stage pipeline-parallel generation for Qwen refinement models.

The regular Accelerate ``device_map`` path saves memory but executes the two
GPU partitions serially.  This module keeps the same single copy of the model
and feeds several micro-batches through two fixed stages so GPU 0 can start the
next micro-batch while GPU 1 finishes the previous one.

Only Qwen2/Qwen3-style causal language models are accepted.  Both expose the
same stable boundary in transformers 4.53: ``embed_tokens -> layers -> norm``.
Keeping this adapter narrow is intentional; silently guessing another model's
forward contract would risk changing transcript text.
"""

from dataclasses import dataclass
import importlib
import queue
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class _MicroBatch:
    order: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    cache: object
    generated: List[torch.Tensor]
    finished: torch.Tensor
    steps: int = 0


@dataclass
class _StagePayload:
    job: _MicroBatch
    hidden_states: torch.Tensor
    masks: Dict[str, Optional[torch.Tensor]]
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    position_embeddings: Tuple[torch.Tensor, torch.Tensor]


class RefinementPipelinePool:
    """Drive two layer partitions with a queue of independent micro-batches."""

    def __init__(self, model, devices: Sequence[int], split_layer: int,
                 micro_batch_size: int, logger=None):
        if len(devices) != 2 or devices[0] == devices[1]:
            raise ValueError("pipeline refinement requires two distinct GPUs")

        self.model = model
        self.base = getattr(model, "model", None)
        if self.base is None or not all(hasattr(self.base, name) for name in (
                "embed_tokens", "layers", "norm", "rotary_emb")):
            raise TypeError(
                f"{type(model).__name__} is not a supported Qwen-style causal LM")
        if not hasattr(model, "lm_head"):
            raise TypeError(f"{type(model).__name__} has no causal LM head")

        layer_count = len(self.base.layers)
        if not 0 < split_layer < layer_count:
            raise ValueError(
                f"split_layer must be inside 1..{layer_count - 1}, got {split_layer}")

        self.devices = (torch.device(f"cuda:{devices[0]}"),
                        torch.device(f"cuda:{devices[1]}"))
        self.split_layer = int(split_layer)
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.logger = logger

        model_module = importlib.import_module(self.base.__class__.__module__)
        self._create_causal_mask = getattr(model_module, "create_causal_mask", None)
        self._create_sliding_mask = getattr(
            model_module, "create_sliding_window_causal_mask", None)
        if self._create_causal_mask is None:
            raise TypeError(
                f"{type(self.base).__name__} does not expose its causal-mask helper")

    @staticmethod
    def build_device_map(layer_count: int, devices: Sequence[int],
                         split_layer: int) -> Dict[str, int]:
        """Return a complete device map without duplicating any model weights."""
        first, second = (int(devices[0]), int(devices[1]))
        mapping = {
            "model.embed_tokens": first,
            "model.rotary_emb": first,
            "model.norm": second,
            "lm_head": second,
        }
        for index in range(layer_count):
            mapping[f"model.layers.{index}"] = (
                first if index < split_layer else second)
        return mapping

    @staticmethod
    def _move(value, device):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, tuple):
            return tuple(RefinementPipelinePool._move(item, device)
                         for item in value)
        if isinstance(value, dict):
            return {key: RefinementPipelinePool._move(item, device)
                    for key, item in value.items()}
        return value

    def _stage_zero(self, job: _MicroBatch) -> _StagePayload:
        device = self.devices[0]
        input_ids = job.input_ids.to(device, non_blocking=True)
        attention_mask = job.attention_mask.to(device, non_blocking=True)

        hidden_states = self.base.embed_tokens(input_ids)
        past_seen = int(job.cache.get_seq_length())
        cache_position = torch.arange(
            past_seen, past_seen + input_ids.shape[1], device=device)

        # ``generate`` derives positions from the padding mask.  Reproduce that
        # behaviour because the refinement tokenizer uses left padding.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, -input_ids.shape[1]:]

        mask_args = {
            "config": self.base.config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": job.cache,
        }
        masks = {"full_attention": self._create_causal_mask(**mask_args)}
        if (getattr(self.base, "has_sliding_layers", False)
                and self._create_sliding_mask is not None):
            masks["sliding_attention"] = self._create_sliding_mask(**mask_args)

        position_embeddings = self.base.rotary_emb(hidden_states, position_ids)
        for layer in self.base.layers[:self.split_layer]:
            hidden_states = layer(
                hidden_states,
                attention_mask=masks[layer.attention_type],
                position_ids=position_ids,
                past_key_value=job.cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]

        return _StagePayload(
            job=job,
            hidden_states=hidden_states.to(self.devices[1], non_blocking=True),
            masks=self._move(masks, self.devices[1]),
            position_ids=position_ids.to(self.devices[1], non_blocking=True),
            cache_position=cache_position.to(self.devices[1], non_blocking=True),
            position_embeddings=self._move(position_embeddings, self.devices[1]),
        )

    def _stage_one(self, payload: _StagePayload, eos_ids: torch.Tensor,
                   pad_token_id: int, max_new_tokens: int) -> bool:
        job = payload.job
        hidden_states = payload.hidden_states
        for layer in self.base.layers[self.split_layer:]:
            hidden_states = layer(
                hidden_states,
                attention_mask=payload.masks[layer.attention_type],
                position_ids=payload.position_ids,
                past_key_value=job.cache,
                output_attentions=False,
                use_cache=True,
                cache_position=payload.cache_position,
                position_embeddings=payload.position_embeddings,
            )[0]

        # Generation only needs the final position. Avoid materialising logits
        # for every prompt token during the expensive prefill pass.
        hidden_states = self.base.norm(hidden_states[:, -1:, :])
        logits = self.model.lm_head(hidden_states)[:, -1, :]
        next_tokens = torch.argmax(logits, dim=-1).to("cpu")

        if job.finished.any():
            next_tokens = torch.where(
                job.finished,
                torch.full_like(next_tokens, pad_token_id),
                next_tokens,
            )
        job.generated.append(next_tokens)
        job.steps += 1

        if eos_ids.numel():
            just_finished = (next_tokens[:, None] == eos_ids[None, :]).any(dim=1)
            job.finished |= just_finished

        if bool(job.finished.all()) or job.steps >= max_new_tokens:
            return True

        job.input_ids = next_tokens[:, None]
        # Decoder-only generation keeps a rectangular cache. Finished rows feed
        # pad tokens while unfinished rows continue; their decoded pads are
        # discarded later.
        job.attention_mask = torch.cat([
            job.attention_mask,
            torch.ones((job.attention_mask.shape[0], 1), dtype=job.attention_mask.dtype),
        ], dim=1)
        return False

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 *, eos_token_ids: Sequence[int], pad_token_id: int,
                 max_new_tokens: int = 512) -> List[torch.Tensor]:
        """Generate greedily while overlapping the two fixed GPU stages."""
        from transformers.cache_utils import DynamicCache

        jobs = []
        for order, start in enumerate(range(0, input_ids.shape[0],
                                            self.micro_batch_size)):
            stop = min(input_ids.shape[0], start + self.micro_batch_size)
            size = stop - start
            jobs.append(_MicroBatch(
                order=order,
                input_ids=input_ids[start:stop].contiguous(),
                attention_mask=attention_mask[start:stop].contiguous(),
                cache=DynamicCache(),
                generated=[],
                finished=torch.zeros(size, dtype=torch.bool),
            ))

        if not jobs:
            return []

        stage_zero_queue = queue.Queue()
        stage_one_queue = queue.Queue()
        result_queue = queue.Queue()
        sentinel = object()
        eos_ids = torch.tensor(list(eos_token_ids), dtype=torch.long)

        def stage_zero_loop():
            torch.cuda.set_device(self.devices[0])
            with torch.inference_mode():
                while True:
                    item = stage_zero_queue.get()
                    if item is sentinel:
                        return
                    try:
                        stage_one_queue.put(self._stage_zero(item))
                    except BaseException as exc:
                        result_queue.put((item.order, None, exc))

        def stage_one_loop():
            torch.cuda.set_device(self.devices[1])
            with torch.inference_mode():
                while True:
                    payload = stage_one_queue.get()
                    if payload is sentinel:
                        return
                    try:
                        done = self._stage_one(
                            payload, eos_ids, pad_token_id, max_new_tokens)
                        if done:
                            generated = torch.stack(payload.job.generated, dim=1)
                            result_queue.put((payload.job.order, generated, None))
                        else:
                            stage_zero_queue.put(payload.job)
                    except BaseException as exc:
                        result_queue.put((payload.job.order, None, exc))

        stage_zero_thread = threading.Thread(
            target=stage_zero_loop, name="refinement-gpu-stage-0", daemon=True)
        stage_one_thread = threading.Thread(
            target=stage_one_loop, name="refinement-gpu-stage-1", daemon=True)
        stage_zero_thread.start()
        stage_one_thread.start()
        for job in jobs:
            stage_zero_queue.put(job)

        results = {}
        errors = []
        for _ in jobs:
            order, generated, error = result_queue.get()
            if error is not None:
                errors.append(error)
            else:
                results[order] = generated

        stage_zero_queue.put(sentinel)
        stage_one_queue.put(sentinel)
        stage_zero_thread.join()
        stage_one_thread.join()

        if errors:
            if any(isinstance(error, torch.cuda.OutOfMemoryError)
                   for error in errors):
                torch.cuda.empty_cache()
            raise errors[0]

        return [row for order in range(len(jobs)) for row in results[order]]
