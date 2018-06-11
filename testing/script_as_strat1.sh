#!/usr/bin/env bash

#/home/dmolinat/k_aug/bin

ROUND=2
START_SESSION=0
END_SESSION=4

set -e

git clone https://github.com/dthierry/nmpc_mhe_q.git -b Larrytestv2
cd ./nmpc_mhe_q/
virtualenv running_framework
source ./running_framework/bin/activate
pip install -r requirements.txt
pip install -e .
cd ..

for ((i=$START_SESSION; i<$END_SESSION; i++))
        do
                mkdir ./run$i
                cp ./nmpc_mhe_q/testing/nmpc_sens_strat1.py ./run$i/
                cd ./run$i/
                cd ..
                cp ./nmpc_mhe_q/testing/ref_ss.sol ./run$i/
                tmux new -s d$ROUND\_$i -d
                tmux send-keys -t d$ROUND\_$i "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dmolinat/k_aug/thirdparty/openblas/OpenBLAS" C-m
                tmux send-keys -t d$ROUND\_$i "export OMP_NUM_THREADS=1" C-m
                tmux send-keys -t d$ROUND\_$i "source ./nmpc_mhe_q/running_framework/bin/activate" C-m
                tmux send-keys -t d$ROUND\_$i "cd ./run$i" C-m
                tmux send-keys -t d$ROUND\_$i "python nmpc_sens_strat1.py" C-m
        done

