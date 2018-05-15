#!/bin/bash
ROUND=0
START_SESSION=0
END_SESSION=5

git clone https://github.com/dthierry/nmpc_mhe_q.git -b LarryTest
cd ./nmpc_mhe_q/
virtualenv running_framework
cd ..

for ((i=$START_SESSION; i<$END_SESSION; i++))
        do
                mkdir ./run$i
                tmux new -s d$ROUND\_$i
                tmux send-keys -t d$ROUND\_$i "source ./nmpc_mhe_q/running_framework/bin/activate" C-m
                tmux send-keys -t d$ROUND\_$i "cp ./nmpc_mhe_q/testing/nmpc_id.py ./run$i/" C-m
                tmux send-keys -t d$ROUND\_$i "cd ./run$i" C-m
                tmux send-keys -t d$ROUND\_$i "python nmpc_id.py" C-m
        done

