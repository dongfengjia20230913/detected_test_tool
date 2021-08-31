 # -*- coding:UTF-8 -*-

import json
import os
import shutil
import operator
import sys
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import common.utils as utils

import cv2
import common.utils as utils
import matplotlib.pyplot as plt
import common.ap_tool as ap_tool
from common.ap_tool import print_log
import common.figure_tool as figure_tool
from common.create_detection_ret_txt import create_detected_txt
from common.create_ground_truth_txt import create_gt_txt



def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gt',  '--ground_truth', type=str, default='ground_truth', help="labeled groud truth txt file dir.")
    parser.add_argument('-dr',  '--detection_result', type=str, default='detection_result', help="detection result txt file dir.")
    parser.add_argument('-jpg', '--jpg_dir', default='JPEGImages', type=str, help="jpg images dir.")
    parser.add_argument('-val',   '--val', type=str, help="out result dir.")
    parser.add_argument('-log',  '--detector_log', type=str, help="model detected log file.")


    parser.add_argument('-na', '--no-animation', help="no animation is shown.", default="True", action="store_true")
    parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
    parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
    # argparse receiving list of classes to be ignored (e.g., python main.py --ignore person book)
    parser.add_argument('-i', '--ignore_class', nargs='+', type=str, help="ignore a list of classes.")
    # argparse receiving list of classes with specific IoU (e.g., python main.py --set-class-iou person 0.7)
    parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.,--set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] ")
    args = parser.parse_args()
    return args

def init_ap_tool(args):
    val_dir = args.val
    if not os.path.exists(val_dir):
        print('Error: val dir is not exists !')
        return False
    detection_result_dir = os.path.join(val_dir, args.detection_result)
    ground_truth_dir =  os.path.join(val_dir, args.ground_truth)
    test_result_out_dir = os.path.join(val_dir, ap_tool.ap_out_dir)
    import shutil

    if os.path.exists(detection_result_dir):
        shutil.rmtree(detection_result_dir)
    os.makedirs(detection_result_dir)

    if os.path.exists(ground_truth_dir):
        shutil.rmtree(ground_truth_dir)
    os.makedirs(ground_truth_dir)

    if os.path.exists(test_result_out_dir):
        shutil.rmtree(test_result_out_dir)
    os.makedirs(test_result_out_dir)

    ap_tool.set_log_dir(test_result_out_dir)


def get_ap_all_class(dr_list, gt_list, gt_class_names, gt_counter_per_class, split_num = 10):
     # print('get compare info for split_num[{}]'.format(ap_tool.SPLIT_NUM))
    class_compareInfos = ap_tool.get_tp_and_fp_info(dr_list, gt_list, gt_class_names, split_num)
    ap_infos = []
    for class_name in class_compareInfos:
        com_info = class_compareInfos[class_name]
        com_info.gt_count = gt_counter_per_class[class_name]

        recall_n, precision_n, fpr_n = ap_tool.get_recall_precision(com_info.tp10, com_info.fp10, gt_counter_per_class[class_name], split_num)
        ap = round(ap_tool.get_ap(recall_n[:], precision_n[:]), 3)

        ap_info = ap_tool.ApInfo(com_info, recall_n, precision_n, ap)
        ap_infos.append(ap_info)
    return ap_infos




if __name__ == '__main__':
    args = get_parse()
    if args.val is None or args.detector_log is None:
        print("Usage:")
        print("python AI_Perf_tool/score_create.py -log [log_file] -val [val_image_dir]")
        exit(0)
    init_ap_tool(args)

    val_image_dir = os.path.join(os.getcwd(), args.val)

    create_detected_txt(args.detector_log, val_image_dir)
    create_gt_txt(val_image_dir)

    gt_file_path = os.path.join(val_image_dir, args.ground_truth)
    dr_file_path = os.path.join(val_image_dir,  args.detection_result)
    img_file_path = os.path.join(val_image_dir, args.jpg_dir)
    # get dr and gt label info
    print('start get label info...')
    gt_counter_per_class, gt_list = ap_tool.get_label_file_info(gt_file_path, is_gt = True)
    print('finish get gt label info...')
    counter_per_class, dr_list = ap_tool.get_label_file_info(dr_file_path, is_gt = False)
    print('finish get dr label info...')

    gt_class_names = gt_counter_per_class.keys()
    print_log('--------------------------{}-------------------------------'.format('mAp'))
    print_log('class names:\n\t{}'.format(gt_class_names))
    print_log('class names:\n\t{}'.format(gt_class_names))
    print_log('gt class_num:\n\t{}'.format(gt_counter_per_class))
    print_log('dr class_num:\n\t{}'.format(counter_per_class))

    ap_infos = get_ap_all_class(dr_list, gt_list, gt_class_names, gt_counter_per_class, split_num = 10)

    sum_ap = 0
    for ap_info in ap_infos:
        com_info = ap_info.compareInfo
        print_log('---------------{}---------------'.format(com_info.class_name))
        print_log('tp:{}'.format(com_info.tp))
        print_log('fp:{}'.format(com_info.fp))
        print_log('tp10:{}'.format(com_info.tp10))
        print_log('fp10:{}'.format(com_info.fp10))
        print_log('gt_count:{}'.format(com_info.gt_count))
        print_log('fn(gt_count-tp):{}'.format(com_info.gt_count - com_info.tp))
        print_log('recall_n: {}'.format(ap_info.recall))
        print_log('precision_n: {}'.format(ap_info.prec))
        print_log('AP:{}'.format( ap_info.ap))
        sum_ap += ap_info.ap
    mAp = round(sum_ap/len(ap_infos), 3)
    print_log('\nmAp:{}'.format(mAp))


    figure_tool.draw_thresh_count(ap_infos, ap_tool.log_out_dir)
    figure_tool.draw_recall_fpr(ap_infos, ap_tool.log_out_dir)
    figure_tool.draw_ap(ap_infos, ap_tool.log_out_dir)
    figure_tool.draw_gt_tp_fp(ap_infos, ap_tool.log_out_dir)
    import shutil
    shutil.copy('AI_Perf_tool/image/test_result.html', ap_tool.log_out_dir)