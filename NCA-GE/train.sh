#!/bin/bash

for RUN in 1 2 3 4 5 6 7 8 9 10; do
	python3 main.py  0       1         2    1e-3   5e-4    1   0     0      0      1024     128     1e-1      0.0       $RUN
	python3 main.py  2       1         2    1e-3   5e-4    1   0     0      0      1024     128     1e-1      0.0       $RUN
done