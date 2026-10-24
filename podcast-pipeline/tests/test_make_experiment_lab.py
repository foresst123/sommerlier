import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import make_experiment_lab as lab  # noqa: E402


def test_single_screen_audio_workbench_contract():
    page = lab.PAGE
    assert "height:100dvh" in page
    assert 'id="wave"' in page
    assert 'id="setupBtn"' in page and 'id="variantsBtn"' in page
    assert "No audio loaded" in page
    assert "Choose input audio" in page
    assert "emptyLoadAudio" in page
    assert "Review export" in page
    assert "Export experiment" not in page
    assert "Spectrogram · soon" in page
    assert "@media(max-width:760px)" not in page


def test_variant_selection_updates_center_metrics():
    page = lab.PAGE
    for metric in ("metricDns", "metricWer", "metricDer", "metricEcapa", "metricArtifacts"):
        assert f"$('{metric}').textContent=m." in page
    assert "aria-pressed" in page
    assert "audioDuration.A||audioDuration.B||30" in page


def test_all_direct_dom_ids_exist():
    """Catch startup crashes caused by code targeting removed markup."""
    ids = set(re.findall(r'id="([^"]+)"', lab.PAGE))
    direct_lookups = set(re.findall(r"\$\('([^']+)'\)", lab.PAGE))
    assert direct_lookups <= ids, f"missing DOM ids: {sorted(direct_lookups - ids)}"


def test_prepare_is_honest_and_keeps_per_trial_settings():
    page = lab.PAGE
    assert "function simulateRun" not in page
    assert "function prepareRun" in page
    assert "status:'prepared'" in page
    assert "art:'awaiting import'" in page
    assert "Math.min(.65" not in page  # no fabricated DNSMOS metrics
    assert "settings:{...(settings||{})}" in page
    assert "settings:{...(c.settings||{})}" in page
    assert "Import output" in page


def test_waveform_supports_pan_and_responsive_layout():
    page = lab.PAGE
    assert "viewStart" in page and "viewSpan" in page
    assert "panByPixels" in page and "e.altKey" in page
    assert "shiftKey" in page and "Alt+drag to pan" in page
    assert "touch-action:none" in page
    assert "grid-template-columns:minmax(180px,22%) minmax(0,1fr) minmax(220px,22%)" in page


def test_lab_has_ordered_model_chain_and_history_view():
    page = lab.PAGE
    assert "historyBtn" in page
    assert "modelChainEditor" in page
    assert "addModel" in page
    assert "modelModal" in page
    assert "historyPage" in page
    assert "scope: needs ? 'model' : 'postprocess'" in page
    assert "runMain" in page


def test_model_add_and_edit_actions_are_separate():
    page = lab.PAGE
    assert "Thêm model vào lượt chạy" in page
    assert "Sửa model ${index+1}" in page
    assert "window.__openModelForAdd = model => open(-1, model)" in page
    assert "top.id='variantsBtn'" in page
    assert "window.__openModelEditor = open" in page


def test_run_button_submits_backend_without_export_step():
    page = lab.PAGE
    assert "Run trực tiếp backend" in page
    assert "direct.dataset.action = 'backend-run'" in page
    assert "fetch('/api/experiment/run'" in page
    assert "export it to execute on GPU" not in page


def test_model_catalog_splits_current_chain_and_quick_add():
    page = lab.PAGE
    assert "model-split" in page
    assert "modelCatalog" in page
    assert "data-add-model" in page
    assert "__openModelForAdd" in page
    assert "choice.parentElement.style.display = index < 0 ? 'none' : 'block'" not in page
    assert "const choiceField = choice.closest('.field')" in page
    for model in ("mpsenet_dns", "dbs_tiger", "mambattention", "rf_vocals_3005", "rf_denoise_aggr", "rf_dereverb", "uvr_deecho", "rf_crowd", "re_use_laughter"):
        assert model in page


def test_model_modal_has_one_authoritative_add_controller():
    page = lab.PAGE
    assert "dataset.authoritativeModal" in page
    assert "stopImmediatePropagation" in page
    assert "window.__openModelForAdd" in page
    assert "model-picker{display:grid" in page
    assert "model-search" in page


def test_dnsmos_gate_is_a_global_control_outside_model_popup():
    page = lab.PAGE
    assert "globalDnsThreshold" in page
    assert "Ngưỡng chung của lượt chạy, không thuộc riêng model nào." in page
    assert "modelThreshold')?.closest('.field')?.setAttribute('style','display:none!important')" in page


NODE = shutil.which("node")


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_inline_javascript_parses(tmp_path):
    scripts = re.findall(r"<script>(.*?)</script>", lab.PAGE, re.S)
    assert scripts
    script = "\n".join(scripts)
    js = tmp_path / "experiment-lab.js"
    js.write_text(script, encoding="utf-8")
    result = subprocess.run([NODE, "--check", str(js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
