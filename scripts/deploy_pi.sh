#!/bin/bash
echo "[System] Deploying V-Guard on Raspberry Pi..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-tk
pip3 install -r requirements.txt
echo "[System] Setting up Autostart..."
# 这里可以写将 ctest_3.py 加入开机自启的逻辑
echo "[System] Deployment Finished."