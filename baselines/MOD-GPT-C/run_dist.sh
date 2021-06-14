cd  /apdcephfs/share_47076/alenfei/dstc10/new-MOD-GPT-C-D
python -m torch.distributed.launch --nproc_per_node=8 parallel_train.py
