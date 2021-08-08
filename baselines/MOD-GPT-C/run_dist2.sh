cd  /apdcephfs/share_47076/yshzhu/shared/zhangzhexin/mod_dstc10/baselines/MOD-GPT-C
umask 0
export PYTHONPATH=/apdcephfs/share_47076/yshzhu/shared/zhangzhexin/miniconda3/envs/py365/lib/python3.6/site-packages
python3 --version > ../../logs/log2.txt 2>&1
pip3 list >> ../../logs/log2.txt 2>&1
python3 -u -m torch.distributed.launch --nproc_per_node=8 parallel_train.py >> ../../logs/log2.txt 2>&1
