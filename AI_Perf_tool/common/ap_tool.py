import glob
import math
import os
#config
MIN_IOU = 0.3 # default value (defined in the PASCAL VOC2012 challenge)
SPLIT_NUM = 100 #statistics score granularity.

ap_out_dir = 'output'

class LabelInfo:
    def __init__(self):
        self.size = []#[w, h]
        self.image_name = ''
        self.class_name=[]
        self.boxes = []
        self.scores = []
        self.is_used = []

    def __str__(self):
        print_info = []
        for i in range(len(self.class_name)):
            if len(self.scores)>0:
                print_info.append("{} {} {} {} ".format(self.class_name[i], self.scores[i] , self.boxes[i] , self.is_used[i]))
            else:
                print_info.append("{} {} {} ".format(self.class_name[i], self.boxes[i] , self.is_used[i]))
        return '[' + self.image_name + ':'+' '.join(print_info) +']'

class CompareInfo:
    def __init__(self, class_name, split_num=SPLIT_NUM):
        self.class_name = class_name
        self.tp = 0
        self.fp = 0
        self.tp10 = [0]*split_num
        self.fp10 = [0]*split_num
        self.gt_count = 0

class ApInfo:
    def __init__(self, class_compareInfo, recall, prec, ap):
        #CompareInfo object instance
        self.compareInfo = class_compareInfo
        self.recall = recall#recall every score
        self.prec = prec#prec every score
        self.ap = ap

def getFileLines(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def get_label_file_info(label_file_path, is_gt = True):
    '''
    Args:
        label_file_path: text label file,every line content is :
        is_gt==true,  [class_name, left,top,right,bottom]
        is_gt==false, [class_name, score,left,top,right,bottom]
    Return: 
        dicts, key is file name, value is a LabelInfo class instance
    '''
    label_file_list = glob.glob(label_file_path + '/*.txt')
    if len(label_file_list) == 0:
        utils.error("Error: No ground-truth files found!")
    label_file_list.sort()

    counter_per_class = {}

    label_list = {}
    for gt_file in label_file_list:
        #check gt file and dr file
        splits = gt_file.split('/')
        file_name = splits[len(splits)-1]

        gt_file_lines = getFileLines(gt_file)

        cur_img_label_info = LabelInfo()
        cur_img_label_info.image_name = file_name
        for line in gt_file_lines:
            if is_gt:
                class_name, left, top, right, bottom = line.split()
            else:
                class_name, score, left, top, right, bottom = line.split()
            # check if class is in the ignore list, if yes skip
            bbox = [int(left.strip()) , int(top.strip()) , int(right.strip()) , int(bottom.strip())]
            if not is_gt:
                cur_img_label_info.scores.append(float(score.strip()))
            cur_img_label_info.class_name.append(class_name)
            cur_img_label_info.boxes.append(bbox)
            cur_img_label_info.is_used.append(False)

            if class_name in counter_per_class:
                counter_per_class[class_name] += 1
            else:
                counter_per_class[class_name] = 1
        label_list[file_name] = cur_img_label_info
        # print(file_name, is_gt, cur_img_label_info)
    return counter_per_class, label_list


def get_tp_and_fp_info(dr_list, gt_list, gt_class_names, split_num = SPLIT_NUM):

    class_compareInfos = {}
    #iteration every detected file
    for key in dr_list.keys():
        dr_label_info = dr_list[key]
        gt_label_info = gt_list[key]
        #便利预测文件中的每一个预测结果
        for index in range(len(dr_label_info.class_name)):
            class_name = dr_label_info.class_name[index]
            if not  class_name in gt_class_names:
                continue
            box = dr_label_info.boxes[index]
            score = dr_label_info.scores[index]

            iou_max = -1
            iou_max_index = -1
            #create every class compare info
            if class_name in class_compareInfos:
                compareInfo = class_compareInfos[class_name]
            else:
                compareInfo = CompareInfo(class_name, split_num)
                class_compareInfos[class_name] = compareInfo

            #对比预测文件和gt文件中，是否存在符合IOU的相同分类
            for index2 in range(len(gt_label_info.class_name)):
                gt_class_name = gt_label_info.class_name[index2]
                box_gt = gt_label_info.boxes[index2]
                #找到gt中相同的类
                if gt_class_name == class_name:
                    #dr，gt两个框的交集部分
                    box_intercept = [max(box[0],box_gt[0]), max(box[1],box_gt[1]),\
                                     min(box[2],box_gt[2]), min(box[3],box_gt[3])]
                    w_intercept = box_intercept[2] - box_intercept[0] + 1
                    h_intercept = box_intercept[3] - box_intercept[1] + 1
                    #交集部分的面积
                    area_intercept = w_intercept*h_intercept
                    if w_intercept > 0 and h_intercept > 0:
                        ua = (box[2] - box[0] + 1) * (box[3] - box[1] + 1) + (box_gt[2] - box_gt[0]
                                        + 1) * (box_gt[3] - box_gt[1] + 1) - area_intercept
                        iou = area_intercept / ua
                        #计算最大IOU
                        if iou > iou_max:
                            iou_max = iou
                            iou_max_index = index2
            #如果当前预测类与实际预测类的IOU大于阈值，则tp+1
            if iou_max > MIN_IOU:
                gt_label_info.is_used[iou_max_index] = True
                compareInfo.tp +=  1
                tp10_index = math.floor(score*split_num)-1
                compareInfo.tp10[tp10_index]+=1
            else:
                compareInfo.fp +=  1
                fp10_index = math.floor(score*split_num)-1
                compareInfo.fp10[fp10_index]+=1
    return class_compareInfos


def get_recall_precision(tp10, fp10, n_gr, split_num = SPLIT_NUM):
    recall_10 = [0]*split_num
    precision_10 = [0]*split_num
    wrong_detect_10 = [0]*split_num
    mtp10 = tp10[:]
    for i in range(len(mtp10)-2, -1, -1):
        mtp10[i] = mtp10[i] + mtp10[i+1]
        recall_10[i] = round(mtp10[i]/float(n_gr),3)
    recall_10[len(mtp10)-1] = round(mtp10[len(mtp10)-1]/float(n_gr),3)


    mfp10 = fp10[:]
    for i in range(len(mfp10)-2, -1, -1):
        mfp10[i] = mfp10[i] + mfp10[i+1]
        if (mfp10[i] + mtp10[i]) != 0:
            precision_10[i] = round(float(mtp10[i])/(mfp10[i] + mtp10[i]),3)
        else:
            precision_10[i] = 0
        wrong_detect_10[i] = round(1-precision_10[i],3)

    last_i = len(mfp10)-1
    if (mfp10[last_i] + mtp10[last_i]) != 0:
        precision_10[last_i] = round(float(mtp10[last_i])/(mfp10[last_i] + mtp10[last_i]),3)
    else:
        precision_10[last_i] = 1.0
    wrong_detect_10[last_i] = round(1-precision_10[last_i],3)

    #draw_recall_fpr_10(recall_10, precision_10, wrong_detect_10)
    return recall_10, precision_10, wrong_detect_10

def get_ap(rec, prec):
    sorted_recall = sorted(rec)
    dealt_recall = [0]*len(sorted_recall)
    dealt_recall[0] = sorted_recall[0]
    for i in range(len(sorted_recall)):
        if i >=1:
            dealt_recall[i] = round(sorted_recall[i] - sorted_recall[i-1], 3)
    #[0, 0.576, 0.141, 0.058, 0.033, 0.035, 0.025, 0.024, 0.0, 0.0]
    # print(dealt_recall)
    ap = 0
    for i in range(len(sorted_recall)):
        ap += dealt_recall[i]*prec[i]

    return ap

log_out_dir='.'
def set_log_dir(test_result_out_dir):
    global log_out_dir 
    log_out_dir = test_result_out_dir

def print_log(content):
    print(content)
    f = open(os.path.join(log_out_dir, "test_result.txt"), "a")
    print(content, file=f)