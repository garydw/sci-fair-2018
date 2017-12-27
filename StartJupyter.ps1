#
# start the jupyter notebook interactive python shell with the path fixed to see the openslide DLLs first (there's duplicates in the path)
#
# force the path to the bin dir to be first to avoid DLL conflicts.
$pathToBin = Resolve-path ".\bin"
$Env:path = "$pathToBin;$Env:path"


& jupyter notebook