python3 predictEDSR_Deconv.py --prune=True --iterations=200 --reusedir="ckpt"
#python3 trainEDSR_Deconv_T.py --iterations=2000 --savedir="prune_ckpt/la_futune_nodec/v1_4" --reusedir="prune_ckpt/la_pruned_nodec/v1_4" --prunedlist_path="prune_ckpt/la_pruned_nodec/v1_4/prunedlist" --logdir="log/la_futune_nodec/v1_4"
#wait
#for i in `seq 2 31`  
#do
#    let l=i*4
#    let m=l-4
#    python3 trainEDSR_Deconv_T.py --prune=True --iterations=500 --savedir="prune_ckpt/la_pruned_nodec/v1_$l" --reusedir="prune_ckpt/la_futune_nodec/v1_$m" --prunedlist_path="prune_ckpt/la_pruned_nodec/v1_$m/prunedlist" --logdir="log/la_pruned_nodec/v1_$l"
#    wait
#    python3 trainEDSR_Deconv_T.py --iterations=2000 --savedir="prune_ckpt/la_futune_nodec/v1_$l" --reusedir="prune_ckpt/la_pruned_nodec/v1_$l" --prunedlist_path="prune_ckpt/la_pruned_nodec/v1_$l/prunedlist" --logdir="log/la_futune_nodec/v1_$l"
#    wait    
#done