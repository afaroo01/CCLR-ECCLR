@echo off
REM Background-swap analysis and size-matched ECCLR.
REM   1. Size-matched ECCLR: ECCLR cut to the CCLR size, exp4/exp8 recipe.
REM   2. Background-swap analysis: CCLR rebuilt with u2net in the original CCLR layout,
REM      with class consistent, class agnostic and cross class swapped backgrounds; exp3/exp7 recipe.
REM      Backgrounds that near duplicate PlantDoc test images are excluded from the pool.
REM All runs are tested on the clean PlantDoc set (511 images).
REM
REM Usage (from the repository root):   scripts\run_background_swap.bat
REM Re-running skips finished datasets and seeds (--resume).
REM To save time, set SEEDS below to 0 1 2 (three seeds) before starting.

setlocal EnableDelayedExpansion
cd /d "%~dp0.."
set MPLBACKEND=Agg

set "DATA=%CD%/data"
set "BG=%CD%/data/backgrounds"
set "FLAGGED=outputs\near_duplicates_bg\flagged_pairs.csv"
set "SEEDS=0 1 2 3 4"
set "MODES=consistent agnostic swap"
set LOCAL="paths.data_root=%DATA%" datasets.plantvillage.class_map=null "datasets.plantdoc.roots=[%DATA%/plantdoc_clean]" datasets.plantdoc.class_map=null xai.enabled=false profile_inference=false data.num_workers=2

REM ---- 0. Clean PlantDoc set (needed for testing) ----------------------------------------
if not exist "%DATA%/plantdoc_clean" (
    python scripts\make_clean_plantdoc.py --flagged "%FLAGGED%" --plantdoc "%DATA%/plantdoc" --out "%DATA%/plantdoc_clean"
    if !errorlevel! neq 0 goto :failed
)

REM ---- 1. Size-matched ECCLR -------------------------------------------------------------
if not exist "%DATA%/ecclr_sizematched" (
    python scripts\make_sizematched_ecclr.py --cclr "%DATA%/plantvillage_cclr" --ecclr "%DATA%/plant_village_ecclr" --out "%DATA%/ecclr_sizematched"
    if !errorlevel! neq 0 goto :failed
)
echo ===== size-matched ECCLR, EfficientNet-B0
plantdisease run configs\experiments\supervised\exp4_efficientnet_ecclr.yaml --seeds %SEEDS% --resume --set %LOCAL% "datasets.plantvillage_ecclr.roots=[%DATA%/ecclr_sizematched]" name=ecclr_sizematched_effnet
if !errorlevel! neq 0 goto :failed
echo ===== size-matched ECCLR, ViT-Tiny
plantdisease run configs\experiments\supervised\exp8_vit_ecclr.yaml --seeds %SEEDS% --resume --set %LOCAL% "datasets.plantvillage_ecclr.roots=[%DATA%/ecclr_sizematched]" name=ecclr_sizematched_vit
if !errorlevel! neq 0 goto :failed

REM ---- 2. Control datasets (u2net segmentation is cached; about 1 to 2 hours the first time)
if not exist "%DATA%/controls/cclr_swap" (
    python scripts\make_control_datasets.py --pv "%DATA%/plantvillage" --backgrounds "%BG%" --out "%DATA%/controls" --exclude "%FLAGGED%" --modes %MODES%
    if !errorlevel! neq 0 goto :failed
)

REM ---- 3. Control trainings ----------------------------------------------------------------
for %%M in (%MODES%) do (
    echo ===== control %%M, EfficientNet-B0
    plantdisease run configs\experiments\supervised\exp3_efficientnet_cclr.yaml --seeds %SEEDS% --resume --set %LOCAL% "datasets.plantvillage_cclr.roots=[%DATA%/controls/cclr_%%M]" name=ctrl_effnet_%%M
    if !errorlevel! neq 0 goto :failed
    echo ===== control %%M, ViT-Tiny
    plantdisease run configs\experiments\supervised\exp7_vit_cclr.yaml --seeds %SEEDS% --resume --set %LOCAL% "datasets.plantvillage_cclr.roots=[%DATA%/controls/cclr_%%M]" name=ctrl_vit_%%M
    if !errorlevel! neq 0 goto :failed
)

REM ---- 4. Tables and tests -----------------------------------------------------------------
python scripts\controls_stats.py outputs outputs_cleanpd
echo All done. Results: outputs\comparison\controls.md
exit /b 0

:failed
echo A step failed (see the message above). Fix it, then re-run: finished steps are skipped.
exit /b 1
