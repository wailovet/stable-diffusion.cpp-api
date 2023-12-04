
@REM stable-diffusion.cpp这里有个bug 搜索"stable-diffusion.cpp"文件中 "void end()" 添加 work_output = NULL; 
@REM 搜索auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool { 
@REM int n_dims              = tensor_storage.n_dims? tensor_storage.n_dims : 1;
set SD_PATH=F:\sd.cpp\src\stable-diffusion.cpp-taesd-im\
set INCLUDE=%~dp0.\oatpp\src\;%SD_PATH%.\common\;%SD_PATH%.\ggml\src\;%SD_PATH%.\ggml\include\ggml\;%SD_PATH%.\ggml\include\;%SD_PATH%
set LIB=%SD_PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64;
@REM call "D:\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
call "J:\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64


mkdir build
cd build

del %~dp0.\build\*.obj 
del %~dp0.\build\*.exe 



cl.exe /EHsc /DGGML_USE_CUBLAS /DSD_USE_CUBLAS -c %~dp0.\main.cpp %SD_PATH%.\*.cpp %SD_PATH%.\common\*.c  %SD_PATH%.\ggml\src\ggml-alloc.c %SD_PATH%.\ggml\src\ggml-quants.c %SD_PATH%.\ggml\src\ggml-backend.c %SD_PATH%.\ggml\src\ggml.c
nvcc -DGGML_CUDA_FORCE_MMQ  %SD_PATH%.\ggml\src\ggml-cuda.cu -c -o %~dp0.\build\ggml-cuda-mmq.obj 
link.exe /OUT:%~dp0.\build\sd-cuda-mmq.exe cublas.lib cuda.lib cudart.lib cudart_static.lib %~dp0.\build\*.obj

copy %~dp0.\build\sd-cuda-mmq.exe %~dp0.\goapi\sd-cuda-mmq.exe
 

del %~dp0.\build\*.obj 
cl.exe /EHsc /arch:AVX2 /Ot /Ox /Gs /DSD_USE_FLASH_ATTENTION -c %~dp0.\main.cpp %SD_PATH%.\*.cpp %SD_PATH%.\common\*.c  %SD_PATH%.\ggml\src\ggml-alloc.c %SD_PATH%.\ggml\src\ggml-quants.c %SD_PATH%.\ggml\src\ggml-backend.c %SD_PATH%.\ggml\src\ggml.c
link.exe /OUT:%~dp0.\build\sd-flash-attention.exe %~dp0.\build\*.obj
@REM  {"cfg_scale": "1", "width": "256", "height": "256", "sample_method": "LCM", "sample_steps": "5", "strength": "1", "seed": "-1", "output": "C:\\Users\\admin\\AppData\\Local\\Temp\\krita_diffusion_tmp2.png", "prompt": "<lora:lcm-lora-sdv1-5:1>1girl", "negative_prompt": "text", "input_path": "C:\\Users\\admin\\AppData\\Local\\Temp\\krita_diffusion_tmp1.png"}

copy %~dp0.\build\sd-flash-attention.exe %~dp0.\goapi\sd-flash-attention.exe

del %~dp0.\build\*.obj 
pause