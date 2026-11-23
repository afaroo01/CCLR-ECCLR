@echo off
REM Re-evaluate all trained supervised models
REM (exp1 to exp8, seeds 0 to 4) on PlantDoc WITHOUT the images that have a near-duplicate in the
REM scraped web pool. No retraining: the existing checkpoints in outputs\ are reused.
REM Results go to outputs_cleanpd\ so the original results are untouched.
REM
REM Usage (from the repository root):  scripts\eval_clean_plantdoc.bat
REM Re-running skips models already evaluated.

setlocal EnableDelayedExpansion
cd /d "%~dp0.."
set MPLBACKEND=Agg

set "DATA=%CD%/data"
set "FLAGGED=outputs\near_duplicates_bg\flagged_pairs.csv"
set "CLEAN=%DATA%/plantdoc_clean"
set LOCAL="paths.data_root=%DATA%" datasets.plantvillage.class_map=null "datasets.plantdoc.roots=[%CLEAN%]" datasets.plantdoc.class_map=null output_dir=outputs_cleanpd xai.enabled=false profile_inference=false data.num_workers=2

REM Rebuild the clean folder every run (takes seconds), so it always matches FLAGGED.
python scripts\make_clean_plantdoc.py --flagged "%FLAGGED%" --plantdoc "%DATA%/plantdoc" --out "%CLEAN%"
if !errorlevel! neq 0 goto :failed

for %%E in (exp1_efficientnet_autoaugment exp2_efficientnet_cutmix exp3_efficientnet_cclr exp4_efficientnet_ecclr exp5_vit_autoaugment exp6_vit_cutmix exp7_vit_cclr exp8_vit_ecclr) do (
    for %%S in (0 1 2 3 4) do (
        if not exist "outputs_cleanpd\%%E\seed_%%S\metrics.json" (
            echo ===== %%E seed %%S
            plantdisease evaluate configs\experiments\supervised\%%E.yaml --seed %%S --checkpoint "outputs\%%E\seed_%%S\checkpoints\best.pt" --set %LOCAL%
            if !errorlevel! neq 0 goto :failed
        )
    )
)

python scripts\clean_pd_compare.py outputs outputs_cleanpd
echo All done. Results: outputs_cleanpd\comparison\clean_plantdoc.md
exit /b 0

:failed
echo A step failed (see the message above). Fix it, then re-run: finished models are skipped.
exit /b 1
