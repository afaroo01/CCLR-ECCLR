@echo off
REM Windows version of run_all.sh for the local layout:
REM   data\{plantvillage, plantvillage_cclr, plant_village_ecclr, plantdoc}
REM All folders already use the canonical class names, so the Kaggle class maps are switched off.
REM
REM Usage (from the repository root, with the package installed):
REM   scripts\run_all.bat                          all supervised runs, then all few-shot runs
REM   scripts\run_all.bat data.num_workers=4       extra overrides are passed to every run
REM --resume is on, so re-running skips seeds that already finished.
REM Any non-zero exit code stops the script (hard crashes return negative codes).

setlocal EnableDelayedExpansion
cd /d "%~dp0.."
REM Non-interactive matplotlib backend: avoids Tk crashes in long runs on Windows
set MPLBACKEND=Agg

set "DATA=%CD%/data"
set LOCAL="paths.data_root=%DATA%" datasets.plantvillage.class_map=null "datasets.plantdoc.roots=[%DATA%/plantdoc]" datasets.plantdoc.class_map=null
REM Few-shot evaluation uses 1000 episodes (the original notebooks used 100)
set FEWSHOT=fewshot.n_eval_tasks=1000

for %%F in (configs\experiments\supervised\exp*.yaml) do (
    echo ===== %%F
    plantdisease run "%%F" --resume --set %LOCAL% %*
    if !errorlevel! neq 0 goto :failed
)

for %%F in (configs\experiments\fewshot\cclr_*.yaml configs\experiments\fewshot\ecclr_*.yaml) do (
    echo ===== %%F
    plantdisease run "%%F" --resume --set %LOCAL% %FEWSHOT% %*
    if !errorlevel! neq 0 goto :failed
)

if not exist outputs\comparison mkdir outputs\comparison
plantdisease summarize outputs\exp1_efficientnet_autoaugment outputs\exp2_efficientnet_cutmix outputs\exp3_efficientnet_cclr outputs\exp4_efficientnet_ecclr outputs\exp5_vit_autoaugment outputs\exp6_vit_cutmix outputs\exp7_vit_cclr outputs\exp8_vit_ecclr --output outputs\comparison\supervised.md
plantdisease summarize outputs\fewshot_cclr_efficientnet_5shot outputs\fewshot_cclr_efficientnet_10shot outputs\fewshot_ecclr_efficientnet_5shot outputs\fewshot_ecclr_efficientnet_10shot outputs\fewshot_cclr_vit_5shot outputs\fewshot_cclr_vit_10shot outputs\fewshot_ecclr_vit_5shot outputs\fewshot_ecclr_vit_10shot --metrics test/accuracy test/accuracy_ci95 target/accuracy target/accuracy_ci95 target/macro_f1 --output outputs\comparison\fewshot.md
echo All done.
exit /b 0

:failed
echo A run failed (see the message above). Fix the error, then re-run: finished seeds are skipped.
exit /b 1
