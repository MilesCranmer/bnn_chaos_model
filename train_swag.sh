for seed in `seq 0 29`; do
    python swag_part1.py --total_steps 300000 --swa_steps 50000 --version 51 --no_mmr --no_nan --no_eplusminus --seed $seed
    python swag_part2.py --total_steps 300000 --swa_steps 50000 --version 51 --no_mmr --no_nan --no_eplusminus --seed $seed
done
