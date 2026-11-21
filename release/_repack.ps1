$ErrorActionPreference = 'Continue'
Add-Type -AssemblyName System.IO.Compression.FileSystem

function New-PkgZip($root, $out) {
  $name = (Get-Item $root).Name
  Write-Host "== $name"
  # stray junk: byte-code caches, traces, logs
  Get-ChildItem $root -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
      Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  Get-ChildItem $root -Recurse -File -Include '*.pyc','*.log','*.prof' -ErrorAction SilentlyContinue |
      ForEach-Object { Write-Host ("   stray: " + $_.FullName.Substring($root.Length + 1)); Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue }
  Remove-Item $out -Force -ErrorAction SilentlyContinue
  $zip = [IO.Compression.ZipFile]::Open($out, 'Create')
  $base = (Get-Item $root).FullName
  $files = 0
  Get-ChildItem $root -Recurse | Sort-Object FullName | ForEach-Object {
    $rel = $_.FullName.Substring($base.Length + 1).Replace('\', '/')
    if ($_.PSIsContainer) {
      $zip.CreateEntry("$name/$rel/") | Out-Null
    } else {
      [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $_.FullName, "$name/$rel", 'Optimal') | Out-Null
      $files++
    }
  }
  $zip.Dispose()
  $zi = Get-Item $out
  Write-Host ("   {0} bytes ({1:N1} MB), {2} files" -f $zi.Length, ($zi.Length / 1MB), $files)
  $h = (Get-FileHash $out -Algorithm SHA256).Hash
  Write-Host ("   sha256 {0}" -f $h)
  [IO.File]::WriteAllText("$out.sha256", "$h  $($zi.Name)`n")
}

New-PkgZip 'C:\tmp3\dist\Mantiq-Portable-CPU'    'C:\tmp3\dist\Mantiq-Portable-CPU.zip'
New-PkgZip 'C:\tmp3\dist\Mantiq-Portable-CUDA'   'C:\tmp3\dist\Mantiq-Portable-CUDA.zip'
