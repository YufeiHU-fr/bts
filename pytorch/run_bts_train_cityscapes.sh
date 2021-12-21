#!/bin/sh
CUDA_VISIBLE_DEVICES=1 python bts_main.py  arguments_train_extra_cityscapes_p12.txt\
			> log_bts_resnet50_cityscapes.out
