#!/bin/sh
CUDA_VISIBLE_DEVICES=1 python bts_test.py  arguments_test_cityscapes.txt\
			> log_bts_resnet50_cityscapes_test.out
