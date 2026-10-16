param(
    [string]$Python = "python",
    [int]$Epochs = 30,
    [int]$BatchSize = 16
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "Checking annotation-aware training runtime: $Python"
& $Python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) { throw "Could not launch Python/PyTorch. Pass the Python executable that passed the smoke test." }
& $Python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA is required for this configured 4x768 experiment." }

$dataset = Join-Path $PSScriptRoot "experiments\aam_annotated\dataset_manifest.json"
if (-not (Test-Path -LiteralPath $dataset)) {
    throw "Annotation-aware dataset not found. Run: python -m ml.aam_annotated.preprocess"
}

Write-Host "Training annotation-aware AAM MIDI model (4 layers x 768 hidden)."
$runStartedAt = Get-Date
& $Python -u -m ml.aam_annotated.train `
    --experiment ".\experiments\aam_annotated" `
    --mode full `
    --seed 42 `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --evaluation-batch-size $BatchSize `
    --learning-rate 0.0003
$bestCheckpoint = Join-Path $PSScriptRoot "experiments\aam_annotated\models\aam_annotated_best.pt"
$evaluationReport = Join-Path $PSScriptRoot "experiments\aam_annotated\reports\evaluation_report.json"
if ($LASTEXITCODE -ne 0) {
    $reportStatus = if (Test-Path -LiteralPath $evaluationReport) {
        (Get-Content -LiteralPath $evaluationReport -Raw | ConvertFrom-Json).status
    } else { "missing" }
    $freshCheckpoint = (Test-Path -LiteralPath $bestCheckpoint) -and
        (Get-Item -LiteralPath $bestCheckpoint).LastWriteTime -ge $runStartedAt
    $freshReport = (Test-Path -LiteralPath $evaluationReport) -and
        (Get-Item -LiteralPath $evaluationReport).LastWriteTime -ge $runStartedAt
    if ($freshCheckpoint -and $freshReport -and $reportStatus -eq "completed") {
        Write-Warning "Python reported a nonzero exit after saving the best checkpoint and completed test report; both artifacts are verified."
    } else {
        throw "Annotation-aware training/evaluation failed before producing a complete checkpoint and test report. See experiments\aam_annotated\logs."
    }
}

Write-Host "Complete. Best checkpoint: experiments\aam_annotated\models\aam_annotated_best.pt"
Write-Host "Held-out report: experiments\aam_annotated\reports\evaluation_report.json"
