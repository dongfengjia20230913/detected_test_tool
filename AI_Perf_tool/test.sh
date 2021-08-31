#!/bin/bash
#模型测试


#生成测试集的结果detection_result
python AI_Perf_tool/Create_detection_ret_txt.py  -log val/phone_70000.txt  -o val
#生成测试集的groud_truth
python AI_Perf_tool/Create_ground_truth_txt.py  -test val
# #根据上述两步生成的groud_truth和detection_result，生成mAp，Recall,


#或者通过以下单行直接执行

python AI_Perf_tool/score_create.py -log val/phone_70000.txt -out val