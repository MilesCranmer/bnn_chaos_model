for seed in `seq 0 29`; do
    python find_minima.py --total_steps 300000 --swa_steps 50000 --version 51 --no_mmr --no_nan --no_eplusminus --seed $seed
    python run_swag.py --total_steps 300000 --swa_steps 50000 --version 51 --no_mmr --no_nan --no_eplusminus --seed $seed
done
