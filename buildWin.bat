@echo off
rem FOR /F "tokens=*" %%a in ('cd') do SET pwd=%%a
rem echo %pwd%

SET srcpath=%~dp0
set src_dir=%srcpath:~0,-1%
conan install %src_dir% -r conan-center --profile %src_dir%\conan_release_profile.txt
cmake -G "Visual Studio 16"  -DBUILD_TESTS=ON %src_dir%
cmake --build . --config Release -j -- /p:CharacterSet=Unicode