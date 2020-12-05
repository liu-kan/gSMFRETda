@echo off
rem FOR /F "tokens=*" %%a in ('cd') do SET pwd=%%a
rem echo %pwd%

SET srcpath=%~dp0
set src_dir=%srcpath:~0,-1%
conan install %src_dir% --build=missing
cmake -G "Visual Studio 16" %src_dir%
cmake --build . --config Release -- /p:CharacterSet=Unicode