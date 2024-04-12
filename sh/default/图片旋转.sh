#!/bin/bash
# -*- coding: utf-8 -*-

#要旋转的文件夹路径 ,分隔
rotate_pic_path_list='/Users/mabowen/Documents/www/coin_Identify/datasets/淳化元宝/origin/淳化元宝_小平_小字隔轮_草书,/Users/mabowen/Documents/www/coin_Identify/datasets/淳化元宝/origin/淳化元宝_小平_大宝_行书,/Users/mabowen/Documents/www/coin_Identify/datasets/淳化元宝/origin/淳化元宝_小平_大宝_草书'
#旋转角度
rotate_angle=5
python3 ../../image_process/rotate_pic.py  \
          --rotate_pic_path_list  ${rotate_pic_path_list}\
          --rotate_angle ${rotate_angle}