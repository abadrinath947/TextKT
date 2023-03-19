cd $1
CUDA_VISIBLE_DEVICES=-1 python examples/run_dkt.py -f ../bBKT/data/database --epochs 20 --hidden_units 128
