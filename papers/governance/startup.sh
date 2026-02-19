#!/bin/bash

if [ ! -d "axc-backtesting" ]; then
    git clone http://github.com/accelerate-protocol/axc-backtesting.git
fi
cd axc-backtesting
git pull origin ${GIT_BRANCH:-main}
git checkout ${GIT_BRANCH:-main}
cd papers/governance
./splitfile.py source.md --delimiter "# %" --overwrite
myst build --html
serve --cors _build/html/ -l tcp://0.0.0.0:3000


