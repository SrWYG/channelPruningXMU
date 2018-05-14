for i in `seq 7 16`
do
    let l=i*4
    python3 trainEDSR_Deconv_D.py --iterations=1010 --savedir="fine_ckpt/v2_$l" --reusedir="prune_ckpt/channel_pruning_v2_$l" --prunedlist_path="prune_ckpt/channel_pruning_v2_$l/prunedlist" --logdir="log/fine_tuning/v2_$l"
    wait   
done

#1
#python3 trainEDSR_Deconv.py --prune=True --iterations=500 --savedir="prune_ckpt/pruned/v1_8" --reusedir="prune_ckpt/futune/v1_4" --prunedlist_path="prune_ckpt/pruned/v1_4/prunedlist" --logdir="log/pruned/v1_8"
#python3 trainEDSR_Deconv.py --prune=False --iterations=2000 --savedir="prune_ckpt/futune/v1_8" --reusedir="prune_ckpt/pruned/v1_8" --prunedlist_path="prune_ckpt/pruned/v1_8/prunedlist" --logdir="log/futune/v1_8"

#2
#python3 trainEDSR_Deconv.py --prune=True --iterations=500 --savedir="prune_ckpt/pruned/v1_12" --reusedir="prune_ckpt/futune/v1_12" --prunedlist_path="prune_ckpt/pruned/v1_4/prunedlist" --logdir="log/pruned/v1_8"
#python3 trainEDSR_Deconv.py --prune=False --iterations=2000 --savedir="prune_ckpt/futune/v1_8" --reusedir="prune_ckpt/pruned/v1_8" --prunedlist_path="prune_ckpt/pruned/v1_8/prunedlist" --logdir="log/futune/v1_8"

