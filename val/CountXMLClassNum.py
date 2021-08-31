# coding=UTF-8
import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
from os.path import join

class CountClassNumbers():

    def __init__(self):
        self.items = {}

    def add(self, k, v):    # 往表中添加元素
        self.items[k] = v
    def get(self, k):       # 线性方式查找元素
        if(k in self.items.keys()):
            return self.items[k]
        return 0

    def countXMLclassNum(self):
        xml_list = os.listdir('Annotations')
        print('Total data size(xml len):', len(xml_list))
        for xmlfile in xml_list:
            xml = open('Annotations/%s'%(xmlfile),encoding='utf-8')
            tree = ET.parse(xml)
            root = tree.getroot()
            for obj in root.iter('object'):
                cls = obj.find('name').text
                claNum = self.get(cls)
                self.add(cls, claNum + 1)

        for key in self.items.keys():
            print ((key, self.items[key]))

if __name__ == '__main__':
    demo = CountClassNumbers()
    demo.countXMLclassNum()