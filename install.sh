#!/bin/bash

python3 -m venv venv
source venv/bin/activate

pip install -r requirements4.5.txt --extra-index-url https://pypi.nvidia.com