#!/bin/bash
python3 gaussian-splatting/render.py -m model -s "$1" --render_path "$2"
