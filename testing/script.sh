#!/bin/bash
ROUND=6
START_SESSION=1
END_SESSION=3

for ((i=$START_SESSION; i<$END_SESSION; i++))
        do
                mkdir ./run$i
                cd ./run$i
                git clone https://github.com/dthierry/nmpc_mhe_q.git -b bfb
                cd ./nmpc_mhe_q/
                virtualenv d$ROUND\_$i
                source ./d$ROUND\_$i/bin/activate
                pip install -r requirements.txt
                pip install -e .
                deactivate
                cd ../..

        done

for ((i=$START_SESSION; i<$END_SESSION; i++))
        do
                tmux new -s d$ROUND\_$i -d
                tmux send-keys -t d$ROUND\_$i "cd ./run$i" C-m
                tmux send-keys -t d$ROUND\_$i "cd ./nmpc_mhe_q" C-m
                tmux send-keys -t d$ROUND\_$i "source ./d$ROUND\_$i/bin/activate" C-m
                tmux send-keys -t d$ROUND\_$i "cd ./testing/" C-m
                tmux send-keys -t d$ROUND\_$i "python tst_algv3_s1600as_n0.py && date" C-m
        done


