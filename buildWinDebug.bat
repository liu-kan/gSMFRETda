@echo off
rem FOR /F "tokens=*" %%a in ('cd') do SET pwd=%%a
rem echo %pwd%

SET srcpath=%~dp0
set src_dir=%srcpath:~0,-1%
conan install %src_dir% --profile %src_dir%\conan_debug_profile.txt --build=missing
cmake %src_dir% -G "Visual Studio 16"
cmake --build . --config Debug -- /p:CharacterSet=Unicode