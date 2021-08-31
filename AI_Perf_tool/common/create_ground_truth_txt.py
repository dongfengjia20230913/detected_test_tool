import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
from os.path import join
import shutil
import sys
import argparse


def create_gt_txt(test_images_dir):
    print('start create_gt_txt {}...'.format(test_images_dir))

    ground_truth_dir = test_images_dir+'/ground_truth'
    if os.path.exists(ground_truth_dir):
        shutil.rmtree(ground_truth_dir)
    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)

    annotations_dir = os.path.join(test_images_dir, 'Annotations')
    if not os.path.exists(annotations_dir):
        print('Error, ['+annotations_dir +'] not exists!')
        exit(0)

    xmls = os.listdir(annotations_dir)

    count = 0
    for xml in xmls:
        in_file = open(annotations_dir+'/%s'%(xml))
        out_gt_txt = ground_truth_dir+'/%s.txt'%(xml[:-4])
        out_file = open(out_gt_txt, 'w')
        #print('in_file:', in_file)
        #print('out_file:', out_file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            obj_name = obj.find('name').text

            if  int(difficult) == 1:
                continue
            xmlbox = obj.find('bndbox')
            left = xmlbox.find('xmin').text
            top = xmlbox.find('ymin').text
            right = xmlbox.find('xmax').text
            bottom = xmlbox.find('ymax').text
            #print(obj_name, left, top, right, bottom)
            out_file.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            # print(count,":",'create:'+xml[:-4]+'.txt')
            count = count + 1
    print('finish create_gt_txt %s...'% count)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', type=str, help="test images dir.")
    args = parser.parse_args()

    if args.test is None:
        print('Usage:' )
        print('\tpython AI_Perf_tool/Create_ground_truth_txt.py  -test val')
        exit(0)

    test_images_dir = os.path.join(os.getcwd(), args.test)
    print(test_images_dir)
    print('----Test images dir:', test_images_dir)
    create_gt_txt(test_images_dir)

