cd  /apdcephfs/share_47076/alenfei/dstc10/MOD-GPT-E-D
python -m torch.distributed.launch --nproc_per_node=8 train_distributed.py
