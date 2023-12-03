
@REM stable-diffusion.cpp这里有个bug 搜索"stable-diffusion.cpp"文件中 "void end()" 添加 work_output = NULL;
set SD_PATH=F:\sd.cpp\src\stable-diffusion.cpp-taesd-im\
set INCLUDE=%~dp0.\oatpp\src\;%SD_PATH%.\common\;%SD_PATH%.\ggml\src\;%SD_PATH%.\ggml\include\ggml\;%SD_PATH%.\ggml\include\;%SD_PATH%
set LIB=%SD_PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64;
call "D:\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
@REM call "J:\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64


mkdir build
cd build

del %~dp0.\build\*.obj 
del %~dp0.\build\*.exe 



cl.exe /EHsc /DGGML_USE_CUBLAS /DSD_USE_CUBLAS -c %~dp0.\main.cpp %SD_PATH%.\util.cpp %SD_PATH%.\model.cpp %SD_PATH%.\common\zip.c %SD_PATH%.\stable-diffusion.cpp %SD_PATH%.\ggml\src\ggml-alloc.c %SD_PATH%.\ggml\src\ggml-quants.c %SD_PATH%.\ggml\src\ggml-backend.c %SD_PATH%.\ggml\src\ggml.c
nvcc -DGGML_CUDA_FORCE_MMQ  %SD_PATH%.\ggml\src\ggml-cuda.cu -c -o %~dp0.\build\ggml-cuda-mmq.obj 
link.exe /OUT:%~dp0.\build\sd-cuda-mmq.exe cublas.lib cuda.lib cudart.lib cudart_static.lib %~dp0.\build\main.obj  %~dp0.\build\model.obj  %~dp0.\build\zip.obj %~dp0.\build\stable-diffusion.obj %~dp0.\build\ggml.obj %~dp0.\build\ggml-cuda-mmq.obj %~dp0.\build\util.obj %~dp0.\build\ggml-alloc.obj %~dp0.\build\ggml-quants.obj %~dp0.\build\ggml-backend.obj

copy %~dp0.\build\sd-cuda-mmq.exe %~dp0.\goapi\sd-cuda-mmq.exe
 

@REM del %~dp0.\build\*.obj 
@REM cl.exe /EHsc /arch:AVX2 /Ot /Ox /Gs /DSD_USE_FLASH_ATTENTION -c %~dp0.\main.cpp %SD_PATH%.\util.cpp %SD_PATH%.\stable-diffusion.cpp %SD_PATH%.\ggml\src\ggml-alloc.c %SD_PATH%.\ggml\src\ggml-quants.c %SD_PATH%.\ggml\src\ggml-backend.c %SD_PATH%.\ggml\src\ggml.c
@REM link.exe /OUT:%~dp0.\build\sd-falsh-attention.exe %~dp0.\build\main.obj %~dp0.\build\stable-diffusion.obj %~dp0.\build\ggml.obj %~dp0.\build\util.obj %~dp0.\build\ggml-alloc.obj %~dp0.\build\ggml-quants.obj %~dp0.\build\ggml-backend.obj
@REM  {"cfg_scale": "1", "width": "256", "height": "256", "sample_method": "LCM", "sample_steps": "5", "strength": "1", "seed": "-1", "output": "C:\\Users\\admin\\AppData\\Local\\Temp\\krita_diffusion_tmp2.png", "prompt": "<lora:lcm-lora-sdv1-5:1>1girl", "negative_prompt": "text", "input_path": "C:\\Users\\admin\\AppData\\Local\\Temp\\krita_diffusion_tmp1.png"}

pause