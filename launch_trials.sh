for seed in `seq 0 29`; do 
    sbatch train_swag.sh --total_steps 300000 --swa_steps 50000 --version 51 --no_mmr --no_nan --no_eplusminus --seed $seed
done
