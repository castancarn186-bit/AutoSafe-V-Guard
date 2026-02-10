@echo off
echo [System] Starting V-Guard Windows Build Process...
cd ..
:: 安装打包工具
pip install pyinstaller
:: 执行打包命令
:: -F: 生成单文件  -w: 运行不显示黑窗口  -i: 指定图标  --add-data: 包含资源文件
pyinstaller --noconfirm --onefile --windowed --icon "assets/icon.png" --add-data "assets;assets" --add-data "core;core" --add-data "modules;modules" --add-data "ui;ui" --name "V-Guard-Pro" main.py
echo [System] Build Complete! Check the 'dist' folder.
pause