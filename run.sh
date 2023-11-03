set -uex
export CUDA_VISIBLE_DEVICES=2
export BATCH_SIZE=20

if [ -s "train.log" ]; then
  mv train.log train.log.$(date +%s)
fi

python3 /tmp/SwinLSTM/train.py --train_batch_size $BATCH_SIZE --valid_batch_size $BATCH_SIZE --test_batch_size $BATCH_SIZE 2>&1 | tee train.log
