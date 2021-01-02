BASE=$HOME/ceph/MLstability/training_data
for batch in "full_resonant_p1*" "only_stable_full_resonant_p1*" "random" "resonant"; do
    for dataset in 'orbsummaryfeaturesxgbNorbits10000.0Nout1000window10' 'featuresNorbits10000.0Nout1000trio' 'additional_featuresNorbits10000.0Nout1000trio'; do
        for dir in $BASE/$batch/$dataset; do
            actualbatch=$(echo $dir | vims -s 'df/df/df/df/df/df/f/d$')
            echo mkdir -p ./data/summary_features/$actualbatch/
            echo cp -i $dir/* ./data/summary_features/$actualbatch/
        done
    done
done

