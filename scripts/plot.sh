# source plot-env/bin/activate
file="offloaf_dynamic_out_avx_ON"
python plot.py ../output/$file

# python parse.py < aos.log | python analysis.py