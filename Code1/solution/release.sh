#!/usr/bin/bash
set -e

pdfunite sol_q1q2q3.pdf sol_q4q5.pdf all_solution.pdf
zip -r code1_214146896_211515606.zip all_solution.pdf sol.py our_resource_files run_all.sh readme.txt
