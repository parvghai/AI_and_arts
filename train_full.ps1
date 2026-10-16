param(
    [string]$Python = "python",
    [int]$Epochs = 30,
    [int]$BatchSize = 16,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "Checking PyTorch and CUDA with $Python"
& $Python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) { throw "Could not run Python/PyTorch. Pass the full python.exe path with -Python." }

$processed = Join-Path $PSScriptRoot "data\processed"
$jsonlCount = @(Get-ChildItem -LiteralPath $processed -Filter "*.jsonl" -File -ErrorAction SilentlyContinue).Count
if ($jsonlCount -lt 3) {
    Write-Host "Preprocessed dataset missing; processing local MIDI/ZIP files under data\raw first."
    & $Python -m ml.preprocessing --data ".\data\raw" --output ".\data\processed"
    if ($LASTEXITCODE -ne 0) { throw "MIDI preprocessing failed." }
    $jsonlCount = @(Get-ChildItem -LiteralPath $processed -Filter "*.jsonl" -File).Count
}

if (-not $SkipTraining) {
    Write-Host "Training the larger 4 x 768 LSTM on $jsonlCount MIDI sequences."
    & $Python -m ml.train --data ".\data\processed" --output "." `
        --epochs $Epochs --batch-size $BatchSize --sequence-length 256 `
        --hidden-size 768 --embedding-size 384 --layers 4 --dropout 0.30 `
        --learning-rate 0.0003 --weight-decay 0.01 --gradient-accumulation 2
    if ($LASTEXITCODE -ne 0) {
        $historyRows = if (Test-Path ".\logs\training_history.csv") { @(Import-Csv ".\logs\training_history.csv").Count } else { 0 }
        if ($historyRows -gt 0 -and (Test-Path ".\models\best_model_4x768.pt") -and (Test-Path ".\models\latest_model_4x768.pt")) {
            Write-Warning "Python returned a nonzero exit code after writing checkpoints and training history; continuing to checkpoint evaluation."
        } else {
            throw "LSTM training failed before producing usable checkpoint artifacts."
        }
    }
} elseif (-not (Test-Path ".\models\best_model_4x768.pt")) {
    throw "Cannot skip training: models\best_model_4x768.pt does not exist."
}

Write-Host "Evaluating the held-out test split and writing example MIDI files."
& $Python -m ml.evaluate --data ".\data\processed" --models ".\models" --checkpoint ".\models\best_model_4x768.pt" --output ".\test_outputs\large_lstm"
if ($LASTEXITCODE -ne 0) { throw "Model evaluation or MIDI export failed." }

Write-Host "Complete. The GUI defaults to models\best_model_3x512.pt; 4x768 remains experimental."
