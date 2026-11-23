@echo off
REM ECCLR component ablation.
REM Chain, all with the rebuilt u2net pipeline and the filtered background pool, 9,360 images each:
REM   controls\cclr_consistent  (already trained: ctrl_effnet_consistent, ctrl_vit_consistent)
REM   controls\ecclr_1leaf      one leaf + ECCLR augmentation and random placement
REM   controls\ecclr_full       1 to 5 leaves + ECCLR augmentation and random placement
REM Trained with the ECCLR recipes (exp4/exp8), tested on the clean PlantDoc set.
REM
REM Usage (from the repository root):   scripts\run_ecclr_ablation.bat
REM Re-running skips finished images and seeds.

setlocal EnableDelayedExpansion
cd /d "%~dp0.."
set MPLBACKEND=Agg

set "DATA=%CD%/data"
set "BG=%CD%/data/backgrounds"
set "FLAGGED=outputs\near_duplicates_bg\flagged_pairs.csv"
set "SEEDS=0 1 2 3 4"
set LOCAL="paths.data_root=%DATA%" datasets.plantvillage.class_map=null "datasets.plantdoc.roots=[%DATA%/plantdoc_clean]" datasets.plantdoc.class_map=null xai.enabled=false profile_inference=false data.num_workers=2

python scripts\make_ecclr_variants.py --leaves "%DATA%/controls/leaves_rgba" --backgrounds "%BG%" --exclude "%FLAGGED%" --out "%DATA%/controls"
if !errorlevel! neq 0 goto :failed

for %%V in (ecclr_1leaf ecclr_full) do (
    echo ===== %%V, EfficientNet-B0
    plantdisease run configs\experiments\supervised\exp4_efficientnet_ecclr.yaml --seeds %SEEDS% --resume --set %LOCAL% "datasets.plantvillage_ecclr.roots=[%DATA%/controls/%%V]" name=abl_effnet_%%V
    if !errorlevel! neq 0 goto :failed
    echo ===== %%V, ViT-Tiny
    plantdisease run configs\experiments\supervised\exp8_vit_ecclr.yaml --seeds %SEEDS% --resume --set %LOCAL% "datasets.plantvillage_ecclr.roots=[%DATA%/controls/%%V]" name=abl_vit_%%V
    if !errorlevel! neq 0 goto :failed
)

python scripts\ablation_stats.py outputs
echo All done. Results: outputs\comparison\ecclr_ablation.md
exit /b 0

:failed
echo A step failed (see the message above). Fix it, then re-run: finished steps are skipped.
exit /b 1
