param(
    [string]$Python = "python",
    [int]$Epochs = 30,
    [int]$BatchSize = 16,
    [string]$Output = ""
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not $Output) { $Output = Join-Path $PSScriptRoot ("runs\aam_masked_" + (Get-Date -Format "yyyyMMdd_HHmmss")) }
if (Test-Path -LiteralPath $Output) { throw "Choose a new output directory. Existing runs are preserved." }
& $Python -B -c "import torch,sys; print(torch.__version__); sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "This full-training launcher requires working CUDA." }
if (-not (Test-Path -LiteralPath ".\experiments\aam_annotated\dataset_manifest.json")) {
    throw "Prepare annotation data in a separate experiment directory before training."
}
& $Python -B -u -m ml.aam_annotated.train `
    --experiment ".\experiments\aam_annotated" --output $Output --mode full --seed 42 `
    --epochs $Epochs --batch-size $BatchSize --evaluation-batch-size $BatchSize --learning-rate 0.0003
if ($LASTEXITCODE -ne 0) { throw "Training/evaluation failed; inspect the new run directory." }
Write-Host "Complete: $Output. Historical experiment data and checkpoints are unchanged."
