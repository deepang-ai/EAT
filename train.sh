export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
torchrun \
  --nproc_per_node 2 \
  --master_port 29400 \
  train.py