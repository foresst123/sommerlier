"""Load a trained adapter and run Vietnamese audio-to-text inference."""

from __future__ import annotations

import argparse

import librosa
import torch
from peft import PeftModel
from transformers import AutoModel


DEFAULT_PROMPT = (
    "Hãy nghe lời người dùng và trả lời tự nhiên, đúng ngữ cảnh bằng tiếng Việt."
)
DEFAULT_REVISION = "1f761131fa83f5ed3cd6f2f22b225c4501d154fa"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Checkpoint adapter/ directory")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    base = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        init_vision=False,
        init_audio=True,
        init_tts=False,
        low_cpu_mem_usage=True,
    )
    base.config.stream_input = False
    model = PeftModel.from_pretrained(base, args.adapter).eval().cuda()
    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    messages = [{"role": "user", "content": [args.prompt, audio]}]
    result = model.chat(
        msgs=messages,
        max_new_tokens=args.max_new_tokens,
        use_tts_template=False,
        generate_audio=False,
        enable_thinking=False,
        stream=args.stream,
        do_sample=args.stream,
        num_beams=1,
    )
    if args.stream:
        for chunk in result:
            print(chunk, end="", flush=True)
        print()
    else:
        print(result)


if __name__ == "__main__":
    main()
