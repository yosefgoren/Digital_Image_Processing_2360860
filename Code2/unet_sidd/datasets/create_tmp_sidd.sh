#!/usr/bin/bash
set -e

SRC_PATH=/home/yogo/media/Datasets/SIDD_Small_sRGB_Only
TARGET_DIR=sidd_tmpfs
mount -t tmpfs . $TARGET_DIR
cp -r $SRC_PATH/Data $TARGET_DIR/
