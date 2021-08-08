cd  /apdcephfs/share_47076/yshzhu/shared/zhangzhexin/mod_dstc10/baselines/MOD-GPT-C
umask 0
python -u -m torch.distributed.launch --nproc_per_node=8 parallel_train.py > ../../logs/log.txt 2>&1
