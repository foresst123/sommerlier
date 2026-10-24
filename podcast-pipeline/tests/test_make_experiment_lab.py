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


def test_medium_breakpoint_keeps_stage_grid_when_drawers_hide():
    """The 761-1400px layout hides side panels without collapsing the canvas."""
    medium = re.search(r"@media \(min-width:761px\)\{(.*?)</style>", lab.PAGE, re.S)
    assert medium, "missing medium-width layout override"
    rules = medium.group(1)
    assert ".stage{display:grid!important" in rules
    assert ".stage" not in re.search(r"\.rail,\.inspector\{([^}]*)\}", rules).group(1)

    medium_wide = re.search(
        r"@media \(min-width:761px\) and \(max-width:1500px\)\{(.*?)</style>",
        lab.PAGE,
        re.S,
    )
    assert medium_wide, "missing medium-wide layout override"
    assert ".stage" not in re.search(r"\.rail,\.inspector\{([^}]*)\}", medium_wide.group(1)).group(1)


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


def test_backend_run_failure_is_truthful_persistent_and_retryable():
    page = lab.PAGE
    assert "runFailureStatus" in page
    assert "r.status=runFailureStatus(error)" in page
    assert "HTTP ${http[1]} — server rejected the run." in page
    assert "Network error — backend unavailable; request was not sent." in page
    assert "r.errorMessage=runErrorMessage(error)" in page
    assert "const retry = $l('runNow')" in page
    assert "retry.textContent = failed ? 'Gửi lại' : 'Chạy'" in page
    assert "delete r.errorMessage" in page


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


def test_model_rows_are_compact_edit_targets_with_delete_in_editor():
    page = lab.PAGE
    assert '<div class="model-current-row" data-current-model="${i}">' in page
    assert 'role="button" tabindex="0" data-current-model="${i}"' not in page
    assert 'class="btn model-edit-affordance"' in page
    assert 'aria-label="Sửa model ${i+1}: ${m}"' in page
    # Destructive deletion is deliberately kept in the modal editor so a row
    # click can never delete a model by accident.
    assert 'data-delete-model="${i}"' not in page
    assert "id='deleteModelPopup'" in page
    assert "remove.disabled = index < 0 || (v.chain||[]).length <= 1" in page
    assert "undoSnapshot={type:'model'" in page
    assert "@media(max-width:420px)" in page


def test_model_modal_focus_contract_traps_and_restores_background():
    page = lab.PAGE
    assert "modal-focus-inert-contract" in page
    assert "const focusSelector =" in page
    assert "event.key !== 'Tab'" in page
    assert "event.key === 'Escape'" in page
    assert "node.inert = true" in page
    assert "restoreBackground(state.background)" in page
    assert "opener.focus()" in page


def test_focus_contract_covers_both_model_and_history_surfaces():
    page = lab.PAGE
    assert "const modalIds = ['modelModal', 'historyPage']" in page
    assert "finalAddClose,#finalAddCancel,#modelClose,#cancelModel,#closeHistory" in page


def test_final_locale_pass_covers_dynamic_ui_copy():
    page = lab.PAGE
    assert "locale-vi-final-pass" in page
    assert "Lỗi mạng — backend không khả dụng; yêu cầu chưa được gửi." in page
    assert "Cần giữ model cuối để chuỗi có thể chạy" in page
    assert "aria-label','Tìm model'" in page
    assert "Chưa có lượt chạy đã lưu" in page


def test_final_locale_pass_batches_scoped_mutations_without_polling():
    page = lab.PAGE
    assert "new MutationObserver(records => records.forEach(record =>" in page
    assert "const pending = new Set()" in page
    assert "if (window.requestAnimationFrame) window.requestAnimationFrame(flush)" in page
    assert "setInterval(run, 300)" not in page


def test_compatibility_sync_is_event_and_scoped_observer_driven():
    page = lab.PAGE
    assert "lab:audio-state" in page
    assert "const roots = [q('canvasWrap'), q('resultRows'), q('historyRows'), q('modelModal'), q('historyPage')]" in page
    assert "roots.forEach(root => observer.observe(root, {childList:true, subtree:true}))" in page
    assert "setInterval" not in page


def test_pinned_reference_ab_surface_defaults_to_source_and_requires_imported_run():
    page = lab.PAGE
    assert "denoise-lab-v4-reference" in page
    assert "id='abSource'" in page or 'id=\"abSource\"' in page
    assert "Nguồn" in page and "Tham chiếu · chưa có" in page
    assert "const imported = entry => Boolean(entry?.r?.audioArtifact?.sessionOnly)" in page
    assert "button.disabled = !imported(entry)" in page
    assert "Bỏ ghim" in page


def test_reference_id_is_a_local_hint_and_candidate_follows_selected_run_only():
    page = lab.PAGE
    assert "localStorage.setItem(key, state.referenceId)" in page
    assert "Nhập lại audio của tham chiếu sau khi tải lại." in page
    assert "const candidate = selectedRun()" in page
    assert "window.addEventListener('lab:audio-state', render)" in page


def test_ab_transport_keeps_one_clock_and_loop_state():
    page = lab.PAGE
    assert "A/B dùng chung thời gian và vòng lặp." in page
    assert "p.addEventListener('timeupdate'" in page
    assert "p.addEventListener('ended'" in page
    assert "const selection=window.__labSelection?.()" in page
    assert "p.currentTime = selection[0] || 0" in page


def test_pinned_ab_has_playback_only_loudness_match():
    page = lab.PAGE
    assert 'id="abLoudnessMatch"' in page
    assert "Cân loudness" in page
    assert "aria-describedby=\"abLoudnessState\"" in page
    assert "const rms = (pcm, seconds, range)" in page
    assert "clamp(20 * Math.log10(a / b), -12, 12)" in page
    assert "player.volume = 1" in page
    assert "Reference im lặng" in page and "Ứng viên im lặng" in page
    assert "Cần đủ 2 audio" in page and "Chọn một đoạn" in page
    # The control is deliberately limited to the player element; it must not
    # write variant settings, recipes, or export artifacts.
    start = page.index('id=\"abLoudnessMatch\"')
    assert "v.settings" not in page[start:page.index('</script>', start)]


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


def test_add_step_is_a_two_step_dialog_with_blend_and_dry():
    page = lab.PAGE
    # model list first, parameters second; blend/dry live in the same catalog
    assert "makeDialog('addDialog'" in page and "const showModelList" in page and "const showParams" in page
    assert "window.__labCatalog = catalog" in page and "blend:['Blend (blendN)'" in page and "dry:['Dry (dryN)'" in page
    # opened from a window-level capture handler, so it never depends on a timer winning a race with the legacy popups
    assert "window.addEventListener('click', e => {\n    if (!e.target.closest?.('#addModelControl,#variantsBtn')) return;" in page


def test_locale_pass_never_rewrites_style_script_or_raw_text():
    page = lab.PAGE
    assert "/^(STYLE|SCRIPT)$/.test(node.tagName) || node.closest?.('[data-raw]')" in page


def test_dry_run_log_message_does_not_pretend_a_job_was_started():
    page = lab.PAGE
    assert "T.state === 'dry-run' ? 'DRY-RUN:" in page and "KHÔNG có cờ --dry-run" in page


def test_status_dialog_warns_early_about_a_skipped_variant_and_does_not_flag_pending_baseline():
    page = lab.PAGE
    assert "'CẢNH BÁO: ' + st.warning" in page and "' · cảnh báo'" in page
    assert "(st.state !== 'running' && v.status !== 'ok')" in page  # 'missing' while running is not an error


def test_stop_after_is_a_visible_control_defaulting_to_denoise():
    page = lab.PAGE
    assert 'id="stopAfter"' in page and "OPTS.some(([v]) => v === saved) ? saved : 'denoise'" in page
    assert "'denoise-lab-stop'" in page and "stop.insertBefore(o, stop.firstChild)" in page


def test_music_removal_is_a_visible_toggle_that_defaults_to_off_and_travels_with_the_recipe():
    page = lab.PAGE
    assert 'id="musicRemoval"' in page and "musicBox.checked = musicSaved === 'on'" in page
    assert "r.settings.music_removal = mb.checked" in page
