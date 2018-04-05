#!/bin/bash
ROUND=0
START_SESSION=0
END_SESSION=4


for ((i=$START_SESSION; i<$END_SESSION; i++))
        do
                case "$i" in
                1) echo "case 1" ;;
                2) echo "case zwei" ;;
                212) echo "case whatevs" ;;
                esac
        done