#!/bin/bash

echo "[INFO] Starting..."

# train
python train_image_caption.py --data_format image_caption --arch resnet18_caption \
--batch-size 512 --workers 2 --classes 3 --epoch 20 --in-shape 3 224 224 \
--data-path ./dataset/image_caption/


echo "[INFO] Done."
