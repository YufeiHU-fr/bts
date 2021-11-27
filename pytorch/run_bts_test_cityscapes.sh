#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python bts_test.py  arguments_test_cityscapes_deeplab.txt\
			> log_bts_resnet50_cityscapes_test.out
