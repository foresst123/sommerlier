import json
import os

import pytest

from utils.conversation_selection import ConversationSelectionConfig


def test_render_only_settings_do_not_reach_the_selection_config():
    cfg = ConversationSelectionConfig.from_settings(
        {"render_workers": 4, "reuse_render": True, "min_seconds": 45})
    assert cfg.min_seconds == 45


def test_a_misspelt_setting_still_fails():
    with pytest.raises(TypeError):
        ConversationSelectionConfig.from_settings({"min_secondz": 45})


def test_every_profile_in_config_json_builds_a_selection_config():
    path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(path) as handle:
        environments = json.load(handle)["environments"]
    for name, profile in environments.items():
        settings = profile.get("models", {}).get("conversation_selection", {})
        ConversationSelectionConfig.from_settings(settings), name
