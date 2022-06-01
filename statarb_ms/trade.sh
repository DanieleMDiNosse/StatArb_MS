#!/bin/bash

python trade.py -g <<EOF
50
ScoreData50
gridsearch_ScoreData50
EOF

python trade.py -g <<EOF
60
ScoreData60
gridsearch_ScoreData60
EOF

python trade.py -g <<EOF
70
ScoreData70
gridsearch_ScoreData70
EOF

python trade.py -g <<EOF
80
ScoreData80
gridsearch_ScoreData80
EOF

python trade.py -g <<EOF
90
ScoreData90
gridsearch_ScoreData90
EOF

python trade.py -g <<EOF
100
ScoreData100
gridsearch_ScoreData100
EOF
