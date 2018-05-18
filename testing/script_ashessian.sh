#!/usr/bin/env bash

#/home/dmolinat/k_aug/bin

ROUND=1
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
                cp ./nmpc_mhe_q/testing/nmpc_sens.py ./run$i/
                cd ./run$i/
                sed -i.bak "s|/home/dav0/devzone/k_aug/cmake-build-k_aug/bin/k_aug|/home/dmolinat/k_aug/bin/k_aug|g" nmpc_sens.py
                sed -i.bak "s|/home/dav0/devzone/k_aug/src/k_aug/dot_driver/dot_driver|/home/dmolinat/k_aug/src/k_aug/dot_driver/dot_driver|g" nmpc_sens.py
                cd ..
                cp ./nmpc_mhe_q/testing/ref_ss.sol ./run$i/
                tmux new -s d$ROUND\_$i -d
                tmux send-keys -t d$ROUND\_$i "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dmolinat/k_aug/thirdparty/openblas/OpenBLAS" C-m
                tmux send-keys -t d$ROUND\_$i "source ./nmpc_mhe_q/running_framework/bin/activate" C-m
                tmux send-keys -t d$ROUND\_$i "cd ./run$i" C-m
                tmux send-keys -t d$ROUND\_$i "python nmpc_sens.py" C-m
        done

j