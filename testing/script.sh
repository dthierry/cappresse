#!/bin/bash
START_SESSION=15
END_SESSION=19

for ((i=$START_SESSION; i<$END_SESSION; i++))
	do
		mkdir ./run$i
		cd ./run$i
		git clone https://github.com/dthierry/nmpc_mhe_q.git -b bfb
		cd ./nmpc_mhe_q/
		virtualenv dav_$i
		source ./dav_$i/bin/activate
		pip install -r requirements.txt
		pip install -e .
		deactivate
		cd ../..

	done

for ((i=$START_SESSION; i<$END_SESSION; i++))
	do
		tmux new -s dav_$i -d
		tmux send-keys -t dav_$i "cd ./run$i" C-m
		tmux send-keys -t dav_$i "cd ./nmpc_mhe_q" C-m
		tmux send-keys -t dav_$i "source ./dav_$i/bin/activate" C-m
		tmux send-keys -t dav_$i "cd ./testing/ && python tst_algv3_s1600_n3.py" C-m 

	done

