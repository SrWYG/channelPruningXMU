#--iterations=3
#--savedir
#--logdir
#--reusedir
#--prunedlist_path
#--prune

#0
python3 trainEDSR_Deconv_T.py --prune=True --iterations=500 --savedir="prune_ckpt/la_pruned_nodec/v1_4" --reusedir="ckpt" --prunedlist_path="aaaaa" --logdir="log/la_pruned_nodec/v1_4"
wait
python3 trainEDSR_Deconv_T.py --iterations=2000 --savedir="prune_ckpt/la_futune_nodec/v1_4" --reusedir="prune_ckpt/la_pruned_nodec/v1_4" --prunedlist_path="prune_ckpt/la_pruned_nodec/v1_4/prunedlist" --logdir="log/la_futune_nodec/v1_4"
wait
for i in `seq 2 31`  
do
    let l=i*4
    let m=l-4
    python3 trainEDSR_Deconv_T.py --prune=True --iterations=500 --savedir="prune_ckpt/la_pruned_nodec/v1_$l" --reusedir="prune_ckpt/la_futune_nodec/v1_$m" --prunedlist_path="prune_ckpt/la_pruned_nodec/v1_$m/prunedlist" --logdir="log/la_pruned_nodec/v1_$l"
    wait
    python3 trainEDSR_Deconv_T.py --iterations=2000 --savedir="prune_ckpt/la_futune_nodec/v1_$l" --reusedir="prune_ckpt/la_pruned_nodec/v1_$l" --prunedlist_path="prune_ckpt/la_pruned_nodec/v1_$l/prunedlist" --logdir="log/la_futune_nodec/v1_$l"
    wait    
done

#1
#python3 trainEDSR_Deconv.py --prune=True --iterations=500 --savedir="prune_ckpt/pruned/v1_8" --reusedir="prune_ckpt/futune/v1_4" --prunedlist_path="prune_ckpt/pruned/v1_4/prunedlist" --logdir="log/pruned/v1_8"
#python3 trainEDSR_Deconv.py --prune=False --iterations=2000 --savedir="prune_ckpt/futune/v1_8" --reusedir="prune_ckpt/pruned/v1_8" --prunedlist_path="prune_ckpt/pruned/v1_8/prunedlist" --logdir="log/futune/v1_8"

#2
#python3 trainEDSR_Deconv.py --prune=True --iterations=500 --savedir="prune_ckpt/pruned/v1_12" --reusedir="prune_ckpt/futune/v1_12" --prunedlist_path="prune_ckpt/pruned/v1_4/prunedlist" --logdir="log/pruned/v1_8"
#python3 trainEDSR_Deconv.py --prune=False --iterations=2000 --savedir="prune_ckpt/futune/v1_8" --reusedir="prune_ckpt/pruned/v1_8" --prunedlist_path="prune_ckpt/pruned/v1_8/prunedlist" --logdir="log/futune/v1_8"

