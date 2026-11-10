# MEME reader pilot (PREREG_MEME_READER.md). Each stage is resumable; rerun this file after an interruption.
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:HF_DATASETS_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = '.\.venv\Scripts\python.exe'
$rd = Join-Path $PSScriptRoot 'meme_reader.py'
& $py $rd answer                          *>> logs\meme_reader.log
& $py $rd judge                           *>> logs\meme_reader.log
& $py $rd answer --think --conds=oracle   *>> logs\meme_reader.log
& $py $rd judge                           *>> logs\meme_reader.log
& $py $rd answer --think --conds=rag      *>> logs\meme_reader.log
& $py $rd judge                           *>> logs\meme_reader.log
