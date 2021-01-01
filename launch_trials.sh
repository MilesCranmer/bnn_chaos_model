for seed in `seq 0 29`; do 
#for power in --power_transform " "; do
#for megno in --megno " "; do
#for angles in --angles " "; do
#for mmr in --no_mmr " "; do
    ## Try a lot of combos
    #echo sbatch train_swag.sh --total_steps 100000 --swa_steps 30000 "$@" $angles $megno $power $mmr --seed $seed
    #if [[ "$seed" == "0" ]]; then sbatch plot_swag.sh --total_steps 100000 --swa_steps 30000 "$@" $angles $megno $power $mmr --seed -1; fi
    ## Longer:
    # sbatch train_swag.sh --total_steps 300000 --swa_steps 50000 "$@" --angles --no_mmr --no_nan --seed $seed
    #if [[ "$seed" == "0" ]]; then sbatch plot_swag.sh --total_steps 300000 --swa_steps 50000 "$@" --angles --no_mmr --no_nan --seed -1; fi
    ## Longer 2:
    #sbatch train_swag.sh --total_steps 300000 --swa_steps 50000 "$@" --angles --no_nan --seed $seed
    #if [[ "$seed" == "0" ]]; then sbatch plot_swag.sh --total_steps 300000 --swa_steps 50000 "$@" --angles --no_nan --seed -1; fi

    #This is v30:
    #sbatch train_swag.sh --total_steps 300000 --swa_steps 50000 "$@" --angles --no_mmr --no_nan --seed $seed

    #v50  (no e+/e-)
    #sbatch train_swag.sh --total_steps 300000 --swa_steps 50000 "$@" --angles --no_mmr --no_nan --no_eplusminus --seed $seed
    #v51 (no e+/e-/angles)
    sbatch train_swag.sh --total_steps 300000 --swa_steps 50000 "$@" --no_mmr --no_nan --no_eplusminus --seed $seed
#done
#done
#done
#done
done
