param([Parameter(Mandatory)][string]$Docx)
$ErrorActionPreference='Stop'
$path=(Resolve-Path -LiteralPath $Docx).Path
$word=New-Object -ComObject Word.Application
$word.Visible=$false
$word.DisplayAlerts=0
try {
    $doc=$word.Documents.Open($path)
    $doc.Repaginate()
    [void]$doc.Fields.Update()
    $doc.Repaginate()
    [void]$doc.Fields.Update()
    $doc.Save()
    $doc.ExportAsFixedFormat([IO.Path]::ChangeExtension($path,'pdf'),17)
    [pscustomobject]@{Pages=$doc.ComputeStatistics(2); Tables=$doc.Tables.Count; Shapes=$doc.InlineShapes.Count}
    $doc.Close(0)
} finally { $word.Quit() }
