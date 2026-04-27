# ISR Submission Package File Organization Script
# Run this from PowerShell to copy (not move) all relevant files into a clean folder structure.
# Original files are preserved in place 鈥?safe for concurrent agent work.

$projectRoot = Split-Path -Parent $PSScriptRoot
$targetRoot = "$projectRoot\isr_submission_package"

# Create folder structure
$folders = @(
    "$targetRoot\01_manuscript",
    "$targetRoot\02_appendix",
    "$targetRoot\03_cover_letter",
    "$targetRoot\04_evidence_control",
    "$targetRoot\05_robustness",
    "$targetRoot\06_red_team",
    "$targetRoot\07_positioning",
    "$targetRoot\08_process_docs"
)
foreach ($f in $folders) { New-Item -ItemType Directory -Force -Path $f | Out-Null }

# === 01_manuscript: Main paper ===
Copy-Item "$projectRoot\paper\who_still_answers_after_chatgpt_manuscript.md" "$targetRoot\01_manuscript\"

# === 02_appendix: Online appendix + supporting tables ===
@(
    "who_still_answers_after_chatgpt_online_appendix.md",   # Original full appendix
    "who_still_answers_online_appendix.md",                  # Phase 3 rebuilt appendix
    "who_still_answers_sample_flow.md",
    "who_still_answers_construct_validity_defense.md",
    "who_still_answers_influence_diagnostics.md",
    "who_still_answers_display_pack.md",
    "who_still_answers_main_text_display_packet.md",
    "who_still_answers_table_figure_roadmap.md"
) | ForEach-Object { 
    $src = "$projectRoot\paper\$_"
    if (Test-Path $src) { Copy-Item $src "$targetRoot\02_appendix\" }
}

# === 03_cover_letter: Editor-facing materials ===
@(
    "who_still_answers_cover_letter.md",
    "who_still_answers_isr_cover_letter.md",
    "who_still_answers_editor_positioning_note.md",
    "who_still_answers_submission_package.md"
) | ForEach-Object {
    $src = "$projectRoot\paper\$_"
    if (Test-Path $src) { Copy-Item $src "$targetRoot\03_cover_letter\" }
}

# === 04_evidence_control: Canonical numbers, claims, bans ===
@(
    "who_still_answers_claim_ledger.csv",
    "who_still_answers_canonical_numbers_table.md",
    "who_still_answers_canonical_evidence_sheet.md",
    "who_still_answers_banned_phrases.md",
    "who_still_answers_forbidden_claim_list.md",
    "who_still_answers_source_chronology.md",
    "who_still_answers_claim_scan_report.md",
    "who_still_answers_replication_report.md",
    "who_still_answers_theory_to_estimand_map.md",
    "who_still_answers_sample_source_bridge.md"
) | ForEach-Object {
    $src = "$projectRoot\paper\$_"
    if (Test-Path $src) { Copy-Item $src "$targetRoot\04_evidence_control\" }
}

# === 05_robustness: Empirical robustness and extension memos ===
@(
    "who_still_answers_few_cluster_extension_memo.md",
    "who_still_answers_timing_extension_memo.md",
    "who_still_answers_alternative_explanations_memo.md",
    "who_still_answers_week2_evidence_gate.md",
    "who_still_answers_contribution_memo.md",
    "who_still_answers_mechanism_decision_memo.md"
) | ForEach-Object {
    $src = "$projectRoot\paper\$_"
    if (Test-Path $src) { Copy-Item $src "$targetRoot\05_robustness\" }
}

# === 06_red_team: All red-team and hostile review docs ===
@(
    "who_still_answers_hostile_memo_phase2.md",
    "who_still_answers_hostile_referee_memo.md",
    "who_still_answers_reviewer_memo.md",
    "who_still_answers_reviewer_question_bank.md",
    "who_still_answers_editor_kill_shots.md",
    "who_still_answers_empirical_kill_shots.md",
    "who_still_answers_isr_editor_red_team.md",
    "who_still_answers_red_team_issue_tracker.md",
    "who_still_answers_final_empirical_red_team_reaffirmation.md"
) | ForEach-Object {
    $src = "$projectRoot\paper\$_"
    if (Test-Path $src) { Copy-Item $src "$targetRoot\06_red_team\" }
}

# === 07_positioning: UTD24 benchmarking and novelty ===
@(
    "who_still_answers_utd24_delta_map.md",
    "who_still_answers_utd24_benchmark_gap_memo.md",
    "who_still_answers_utd24_readiness_memo.md",
    "who_still_answers_literature_matrix.md",
    "who_still_answers_novelty_memo.md",
    "who_still_answers_design_figure_scaffold.md"
) | ForEach-Object {
    $src = "$projectRoot\paper\$_"
    if (Test-Path $src) { Copy-Item $src "$targetRoot\07_positioning\" }
}

# === 08_process_docs: Gate sheets, scorecards, call ledger, decision memos ===
@(
    "who_still_answers_gate_sheet.md",
    "who_still_answers_gate_assessment_phase2.md",
    "who_still_answers_go_no_go_memo.md",
    "who_still_answers_final_scorecard.md",
    "who_still_answers_call_ledger.csv",
    "who_still_answers_render_check.md",
    "who_still_answers_render_check.json",
    "who_still_answers_render_length_audit_2026-04-04.md",
    "who_still_answers_cross_document_discrepancy_table.md",
    "who_still_answers_cross_document_discrepancy_table_2026-04-04.md",
    "who_still_answers_final_core_set_diff_2026-04-04.md",
    "who_still_answers_isr_recovery_plan.md",
    "who_still_answers_reuse_map.md",
    "who_still_answers_week1_agent_roster.md",
    "arendt_results_evidence_integration_2026-04-03.md"
) | ForEach-Object {
    $src = "$projectRoot\paper\$_"
    if (Test-Path $src) { Copy-Item $src "$targetRoot\08_process_docs\" }
}

# === Summary ===
Write-Host "`n=== ISR Submission Package Organized ===" -ForegroundColor Green
Write-Host "Target: $targetRoot" -ForegroundColor Cyan
foreach ($f in $folders) {
    $count = (Get-ChildItem $f -File).Count
    $name = Split-Path $f -Leaf
    Write-Host "  $name : $count files" -ForegroundColor Yellow
}
$totalFiles = (Get-ChildItem $targetRoot -Recurse -File).Count
Write-Host "`nTotal: $totalFiles files copied (originals preserved)" -ForegroundColor Green
