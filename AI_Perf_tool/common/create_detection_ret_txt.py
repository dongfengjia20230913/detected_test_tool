import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
from os.path import join
import shutil
import sys
import argparse
from common.ap_tool import print_log 

def create_detected_txt(log_file, output_dir):
    print('start create_detected_txt....')
    result_dir = os.path.join(output_dir, 'detection_result')

    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_file):
        print('Error, ['+log_file +'] not exists!\n')
        exit(0)
    filename=''
    index = 0
    with open(log_file, 'r') as file_to_read:
        for line in file_to_read.readlines():
           #print(line)
           if 'dectect out' in line or 'detect_out' in line or 'dectect out' in line or 'detect out' in line:
                 # print(line)
                 splits = line.replace('\n','').split(':')
                 if len(splits) < 4:
                    print('error:please check log print content!')
                    continue
                 className = splits[1].replace(" ", "")
                 thresh = splits[2].replace(" ", "")
                 box = splits[3]

                 imagesplit = splits[4].split('/')
                 lenthsplit = len(imagesplit)
                 imageFileName = imagesplit[lenthsplit-1].strip()
                 if imageFileName not in filename:
                    index = index+1
                    filename+=imageFileName
                 if imageFileName.endswith('.jpg'):
                    base_image_file_name = imageFileName.split('.jpg')[0]
                 elif imageFileName.endswith('.png'):
                    base_image_file_name = imageFileName.split('.png')[0]

                 out_file_name = os.path.join(result_dir, base_image_file_name+'.txt')
                 if os.path.exists(out_file_name):
                    out_file = open(out_file_name, 'a')
                 else:
                    out_file = open(out_file_name, 'w')
                 out_file.write("%s %s %s \n" % (className, thresh, box))
    print('finish create_detected_txt....')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-log',  '--detector_log', type=str, help="model detected log file.")
    parser.add_argument('-o',  '--output', type=str, help="detection result txt file dir.")
    args = parser.parse_args()


    if args.detector_log is None or args.output is None:
        error_msg = '\n -log [detector_out.txt] -o ./smoke '
        print('Error, missing arguments. Flag usage:' + error_msg)
        exit(0)

    log_file = os.path.join(os.getcwd(), args.detector_log)
    output_dir =  os.path.join(os.getcwd(), args.output)
    create_detected_txt(log_file, output_dir)

