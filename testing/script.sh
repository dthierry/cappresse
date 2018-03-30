#!/bin/bash
ROUND=0
START_SESSION=0
END_SESSION=3

for ((i=$START_SESSION; i<$END_SESSION; i++))
        do
                mkdir ./run$i
                cd ./run$i
                git clone https://github.com/dthierry/nmpc_mhe_q.git -b fe0as_patch
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
                case "$i" in
                    0) tmux send-keys -t d$ROUND\_$i "python full_asas_v1.py && date" C-m
                    ;;
                    1) tmux send-keys -t d$ROUND\_$i "python full_asas_v1.py && date" C-m
                    ;;
                    2) tmux send-keys -t d$ROUND\_$i "python full_asas_v1.py && date" C-m
                    ;;
                    3) tmux send-keys -t d$ROUND\_$i "python full_asas_v1.py && date" C-m
                    ;;
                esac
#                if [ $i -eq 0 ]
#                then
#                    tmux send-keys -t d$ROUND\_$i "python tst_algv3_s1600as_n0_101.py && date" C-m
#                else
#                    tmux send-keys -t d$ROUND\_$i "python tst_algv3_s1600_n4_00.py && date" C-m
#                fi
        done


