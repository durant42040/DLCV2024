#!/bin/bash
mkdir -p $2
python3 p2/inference.py $1 $2
