param(
    [Parameter(Position=0,Mandatory=0)]
    [switch]$publish = $false
)

$base_dir = resolve-path .
$build_dir = "$base_dir\_build"
$tools_dir = "$base_dir\tools"

$package_files = @(Get-ChildItem src -include *packages.config -recurse)
foreach ($package in $package_files) {
	& $tools_dir\NuGet.exe install $package.FullName -OutputDirectory packages
}

$spec_files = @(Get-ChildItem $base_dir\src -include *.nuspec -recurse)
foreach ($spec in $spec_files) {
	& $tools_dir\NuGet.exe pack $spec.FullName -OutputDirectory $build_dir -Symbols -BasePath $base_dir
}

#$spec_files = @(Get-ChildItem $base_dir -include *.nupkg -recurse)
#foreach ($spec in $spec_files) {
#	& $tools_dir\NuGet.exe push $spec.FullName
#}