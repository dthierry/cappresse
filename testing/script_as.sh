#!/usr/bin/env bash

ROUND=0
START_SESSION=0
END_SESSION=5

set -e

git clone https://github.com/dthierry/nmpc_mhe_q.git -b LarryTest
cd ./nmpc_mhe_q/
virtualenv running_framework
source ./running_framework/bin/activate
pip install -r requirements.txt
pip install -e .
cd ..

for ((i=$START_SESSION; i<$END_SESSION; i++))
        do
                mkdir ./run$i
                tmux new -s d$ROUND\_$i -d
                tmux send-keys -t d$ROUND\_$i "source ./nmpc_mhe_q/running_framework/bin/activate" C-m
                tmux send-keys -t d$ROUND\_$i "cp ./nmpc_mhe_q/testing/nmpc_as.py ./run$i/" C-m
                tmux send-keys -t d$ROUND\_$i "cp ./nmpc_mhe_q/testing/ref_ss.sol ./run$i/" C-m
                tmux send-keys -t d$ROUND\_$i "cd ./run$i" C-m
                tmux send-keys -t d$ROUND\_$i "python nmpc_as.py" C-m
        done
