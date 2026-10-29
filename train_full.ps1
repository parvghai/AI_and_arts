param(
    [string]$Python = "python",
    [int]$Epochs = 30,
    [int]$BatchSize = 16,
    [string]$Output = "",
    [switch]$SkipTraining,
    [string]$Checkpoint = ""
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not $Output) { $Output = Join-Path $PSScriptRoot ("runs\midi_masked_" + (Get-Date -Format "yyyyMMdd_HHmmss")) }
if (Test-Path -LiteralPath $Output) { throw "Choose a new output directory. Existing runs are preserved." }
& $Python -B -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Python/PyTorch is unavailable." }
if (-not $SkipTraining) {
    & $Python -B -m ml.train --data ".\data\processed" --output $Output `
        --epochs $Epochs --batch-size $BatchSize --sequence-length 256 `
        --hidden-size 768 --embedding-size 384 --layers 4 --dropout 0.30 `
        --learning-rate 0.0003 --weight-decay 0.01 --gradient-accumulation 2
    if ($LASTEXITCODE -ne 0) { throw "Training failed. Existing artifacts do not override a failed exit." }
    $Checkpoint = Join-Path $Output "models\best_model_4x768.pt"
} elseif (-not $Checkpoint -or -not (Test-Path -LiteralPath $Checkpoint)) {
    throw "With -SkipTraining, provide an existing -Checkpoint explicitly."
}
& $Python -B -m ml.evaluate --data ".\data\processed" --checkpoint $Checkpoint --output (Join-Path $Output "evaluation")
if ($LASTEXITCODE -ne 0) { throw "Masked evaluation failed." }
Write-Host "Complete: $Output. Original GUI checkpoints were not replaced."
