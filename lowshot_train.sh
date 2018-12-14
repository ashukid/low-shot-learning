for i in {1..1}
do
    for j in 1 2 5 10 20
        do
            python ./low_shot.py --lowshotmeta label_idx.json \
                --experimentpath experiment_cfgs/splitfile_{:d}.json \
                --experimentid  $i --lowshotn $j \
                --trainfile features/ResNet10_sgm/train.hdf5 \
                --testfile features/ResNet10_sgm/val.hdf5 \
                --outdir results \
                --lr 1 --wd 0.001 \
                --testsetup 1
        done
done
