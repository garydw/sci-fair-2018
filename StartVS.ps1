# force the path to the bin dir to be first to avoid DLL conflicts.
$pathToBin = Resolve-path ".\bin"
$Env:path = "$pathToBin;$Env:path"

& "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\devenv.exe"