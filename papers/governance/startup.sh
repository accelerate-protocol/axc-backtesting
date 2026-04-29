#!/bin/bash
# Add a trap to kill background jobs when the process exits
trap "kill $(jobs -p) 2>/dev/null; exit" EXIT

CURRENT_DIR=$(pwd)

if [[ ! "$CURRENT_DIR" == *axc-backtesting/papers/governance* ]]; then
if [ ! -d "axc-backtesting" ]; then
    git clone http://github.com/accelerate-protocol/axc-backtesting.git
fi
cd axc-backtesting
git config --global pull.rebase true
git pull origin ${GIT_BRANCH:-main}
git checkout ${GIT_BRANCH:-main}
cd papers/governance
else
    echo "--- Skipping setup script ---"
    echo "Current directory is already within axc-backtesting/papers/governance. No changes needed."
fi

./splitfile.py source.md --delimiter "# %" --overwrite

pushd en
cp ../*.png .
myst build --html
npx serve --cors _build/html/ -l tcp://0.0.0.0:3000 &
popd
./splitfile.py source.md --delimiter "# %" --overwrite --lang zh
pushd zh
cp ../*.png .
npx serve --cors _build/html/ -l tcp://0.0.0.0:3001 &
popd
sleep infinity
