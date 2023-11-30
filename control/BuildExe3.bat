set current_dir=%~dp0
echo %current_dir%  
set script_dir = "c:\Program Files\python36\Scripts"
REM cd c:\Program Files\python36\scripts
REM pyinstaller -F  --add-data opencv_ffmpeg410_64.dll;. script.py
REM "C:\Program Files\Python36\Scripts\pyinstaller" --onedir %current_dir%Main.py   -F --add-data opencv_videoio_ffmpeg412_64.dll;. -i %current_dir%ivi_icon.ico  

REM "C:\Program Files\Python36\Scripts\pyinstaller" --hidden-import="sklearn.utils._cython_blas" %current_dir%Main.py -i %current_dir%ivi_icon.ico

pyinstaller --hidden-import="sklearn.utils._cython_blas" %current_dir%Main.py -i %current_dir%ivi_icon.ico

pause