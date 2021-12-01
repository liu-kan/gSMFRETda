@echo off
rem FOR /F "tokens=*" %%a in ('cd') do SET pwd=%%a
rem echo %pwd%

SET srcpath=%~dp0
set src_dir=%srcpath:~0,-1%
conan install %src_dir% --profile %src_dir%\conan_release_profile.txt --build=missing

set proto=OFF
set arg1=%1
:getarg
if "%1"=="" goto Continue
    if "%1"=="proto" set proto=ON
    shift
    goto getarg
:Continue
cmake -G "Visual Studio 16" -Dproto=%proto% -DBUILD_TESTS=ON -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake %src_dir%
cmake --build . --config Release -j -- /p:CharacterSet=Unicode