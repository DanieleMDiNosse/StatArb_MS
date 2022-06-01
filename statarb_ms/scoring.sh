#!/bin/bash

python scoring.py -s <<EOF
dis_res50
50
kappas50
EOF

python scoring.py -s <<EOF
dis_res60
60
kappas60
EOF

python scoring.py -s <<EOF
dis_res70
70
kappas70
EOF

python scoring.py -s <<EOF
dis_res90
90
kappas90
EOF

python scoring.py -s <<EOF
dis_res100
100
kappas100
EOF
