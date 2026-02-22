#!/bin/bash
# Add a trap to kill background jobs when the process exits
trap "kill $(jobs -p) 2>/dev/null; exit" EXIT

if [ ! -d "axc-backtesting" ]; then
    git clone http://github.com/accelerate-protocol/axc-backtesting.git
fi
cd axc-backtesting
git config --global pull.rebase true
git pull origin ${GIT_BRANCH:-main}
git checkout ${GIT_BRANCH:-main}
cd papers/governance
./splitfile.py source.md --delimiter "# %" --overwrite

pushd en
myst build --html
npx serve --cors _build/html/ -l tcp://0.0.0.0:3000 &
popd
./splitfile.py source.md --delimiter "# %" --overwrite --lang zh
pushd zh
npx serve --cors _build/html/ -l tcp://0.0.0.0:3001 &
popd
sleep infinity
