@echo off
rem FOR /F "tokens=*" %%a in ('cd') do SET pwd=%%a
rem echo %pwd%

SET srcpath=%~dp0
conan install %srcpath:~0,-1% --build=missing
SET srcpath=%~dp0
cmake %srcpath:~0,-1% -G "Visual Studio 16"
cmake --build . --config Release