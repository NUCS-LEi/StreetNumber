# StreetNumber1
1. Convert to TFRecords format
$ python convert_to_tfrecords.py --data_dir data
2. Train
$ python train.py --data_dir data --train_logdir logs\train
3. (optional)Retrain if you need
$ python train.py --data_dir data --train_logdir logs\train2 --restore_checkpoint logs\train\latest.ckpt
4. Evaluate
$ python eval.py --data_dir data --checkpoint_dir logs\train --eval_logdir logs\eval
5. Visualize
$ tensorboard --logdir logs
6. (Optional) Try to make an inference
$ python inference.py --image \path\to\image.jpg --restore_checkpoint logs\train\latest.ckpt
7. Clean
$ rm -rf logs
or
$ rm -rf logs\train2
or
$ rm -rf logs\eval
