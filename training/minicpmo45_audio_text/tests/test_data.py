import torch

from minicpmo_train.data import MiniCPMODataCollator, find_last_subsequence
from minicpmo_train.manifest import ManifestRow


class _Tokenizer:
    @staticmethod
    def apply_chat_template(messages, **_kwargs):
        return "".join(f"<{item['role']}>{item['content']}" for item in messages)

    @staticmethod
    def encode(text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]


class _Processor:
    tokenizer = _Tokenizer()

    def __call__(self, *, text, **_kwargs):
        encoded = [[ord(character) for character in item] for item in text]
        width = max(map(len, encoded))
        input_ids = torch.zeros((len(encoded), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for index, item in enumerate(encoded):
            input_ids[index, -len(item) :] = torch.tensor(item)
            attention_mask[index, -len(item) :] = True
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_sizes": [[] for _ in encoded],
            "audio_features": [],
            "audio_feature_lens": [[] for _ in encoded],
            "audio_bounds": [torch.empty((0, 2), dtype=torch.long) for _ in encoded],
            "image_bound": [torch.empty((0, 2), dtype=torch.long) for _ in encoded],
            "spk_bounds": [torch.empty((0, 2), dtype=torch.long) for _ in encoded],
        }


def test_find_last_subsequence_prefers_final_answer_occurrence() -> None:
    assert find_last_subsequence([1, 2, 3, 1, 2, 4], [1, 2]) == 3


def test_find_last_subsequence_reports_missing_or_empty() -> None:
    assert find_last_subsequence([1, 2, 3], [2, 4]) == -1
    assert find_last_subsequence([1, 2, 3], []) == -1


def test_collator_masks_prompt_and_keeps_only_final_answer_loss() -> None:
    row = ManifestRow(
        id="text-1",
        task="text",
        messages=[{"role": "user", "content": "Câu hỏi"}],
        response="Câu trả lời",
        audio_path=None,
        audio_start=None,
        audio_end=None,
        source_group="text",
        metadata={},
    )
    batch = MiniCPMODataCollator(processor=_Processor(), max_length=256)([row])
    labels = batch["labels"][0]
    supervised = labels[labels != -100].tolist()
    assert supervised == _Tokenizer.encode(row.response)
    assert batch["data"]["vision_hidden_states"] == [[]]
