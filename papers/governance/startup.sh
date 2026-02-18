#!/bin/bash

cd axc-backtesting
git pull
cd papers/governance
./splitfile.py source.md --delimiter "# %" --overwrite
myst build --html
serve --cors _build/html/ -l tcp://0.0.0.0:3000


