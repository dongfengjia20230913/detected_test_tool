import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd
from matplotlib.pyplot import MultipleLocator 

plot_w = 6

def show_value_for_barplot(barplot, h_v="v"):
    if h_v == "v":
        for p in barplot.patches:
            barplot.annotate(format(int(p.get_height()),'d'), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', color='gray',
             fontsize = 6)
    elif h_v == "h":
        for p in barplot.patches:
            barplot.annotate(format(p.get_width()), (p.get_width(), p.get_y()+ p.get_height() / 2.), ha = 'center', va = 'center', xytext = (30, 0), textcoords = 'offset points')

def draw_thresh_count(ap_infos, image_save_dir):
    '''
    Args:
        ApInfo[]
    '''
    #1. get draw info
    class_num = len(ap_infos)
    if class_num ==0 :
        return
    x = (np.arange(10) + 1)/10
    y_max = 0#limit y_value max value
    for ap_info in ap_infos:
        compareInfo = ap_info.compareInfo
        y_max=max(max(compareInfo.tp10),y_max)*1.2

    #2.draw sub plot
    sns.set_theme() 
    sns.axes_style("darkgrid")
    fig, axes=plt.subplots(class_num, 1, figsize=(plot_w, plot_w*class_num*0.5)) 
    ax_index = 0
    for ap_info in ap_infos:
        compareInfo = ap_info.compareInfo
        y = compareInfo.tp10
        ax=axes[ax_index]
        ax_index +=1
        splot=sns.barplot(x=x, y=y, ax=ax, palette="Dark2")
        show_value_for_barplot(splot)
        ax.set_title(compareInfo.class_name)
        ax.title.set_position([.1, 0.7])#设置标题位置
        ax.tick_params(labelsize=10)#设置刻度大小
    fig.tight_layout()
    plt.suptitle('thresh count', fontsize = 10 ,fontweight='bold')
    plt.savefig(os.path.join(image_save_dir,'cls_thresh.png'))
    # if show_fig:
    #     plt.show()
    plt.close()

def draw_recall_fpr(ap_infos:list, image_save_dir, max_cols = 1):
    '''
    Args:
        ApInfo[]
    '''
    #1. get draw info
    class_num = len(ap_infos)
    if class_num ==0 :
        return
    x = (np.arange(10) + 1)/10

    #根据传入的分类个数，决定rec-fpr图的行数和列数。列数，默认最大为2
    plot_cols = max_cols
    if class_num <=plot_cols:
        plot_cols = class_num
    plot_rows = math.ceil(class_num/plot_cols)

    #2.draw sub plot
    sns.set_theme() 
    sns.axes_style("darkgrid")
    fig = plt.figure(figsize=(plot_w*plot_cols , plot_w*plot_rows*0.5))
    # sns.set_theme(style="darkgrid")
    for row in range(plot_rows):
        for col in range(plot_cols):
            index = row*plot_cols+col
            if index < class_num:
                #获取要显示的信息
                ap_info = ap_infos[index]
                compareInfo = ap_info.compareInfo
                rec =  ap_info.recall
                fprs = [0]*len(ap_info.prec)
                fprs[:]=[round(1-prec, 3) for prec in ap_info.prec]
                datas_dict = {}
                datas_dict['rec'] = rec
                datas_dict['fpr'] = fprs
                rec_fpr_data = pd.DataFrame(datas_dict , index=x)
                #画图
                ax = fig.add_subplot(plot_rows, plot_cols, index+1)
                ax.xaxis.set_major_locator(MultipleLocator(0.1))#设置坐标间隔
                ax.yaxis.set_major_locator(MultipleLocator(0.2))
                ax.set(ylim=(0, 1.0))#设置y轴的显示值范围
                sns.lineplot(data=rec_fpr_data,linewidth=2, ax = ax)
                sns.scatterplot(x=x, y=rec, s=30, ax = ax)
                sns.scatterplot(x=x, y=fprs, s=30, ax = ax)

                ax.set_title(compareInfo.class_name)
                ax.title.set_position([.1, 0.7])#设置标题位置
                ax.tick_params(labelsize=10)#设置刻度字体大小

                for index,value in enumerate(x):#在点上显示数字
                    ax.text(value, rec[index], rec[index], fontsize=7)
                    ax.text(value, fprs[index], fprs[index], fontsize=7)
    fig.tight_layout()
    save_path = os.path.join(image_save_dir,'rec_fpr.png')
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path)
    # if show_fig:
    #     plt.show()
    plt.close()

def draw_ap(ap_infos, image_save_dir):
    '''
    Args:
        ApInfo[]
    '''
    #1. get draw info
    class_name = []
    ap = []
    for ap_info in ap_infos:
        compareInfo = ap_info.compareInfo
        class_name.append(compareInfo.class_name)
        ap.append(round(ap_info.ap, 2))
    map = round(sum(ap)/len(ap),4)

    #2.draw sub plot
    f, ax = plt.subplots(figsize=(plot_w, plot_w*0.8))
    #sns.set_theme(style="white")
    sns.set_theme()
    sns.axes_style("darkgrid")
    # sns.set_theme(style="darkgrid")
    sns.set_color_codes("muted")
    ax.set(xlim=(0, 1.1))
    splot = sns.barplot(x=ap, y=class_name, ax =ax, color='b')
    sns.despine(left=True, bottom=True)
    show_value_for_barplot(splot,h_v='h')
    ax.set_title('mAp = {}%'.format(map*100))
    save_path = os.path.join(image_save_dir,'mAp.png')
    plt.savefig(save_path)
    # if show_fig:
    #     plt.show()
    plt.close()

def draw_gt_tp_fp(ap_infos, image_save_dir):
    '''
    Args:
        ApInfo[]
    '''
    #1. get draw info
    class_name = []
    gt = []
    tp = []
    fp = []

    for ap_info in ap_infos:
        compareInfo = ap_info.compareInfo
        class_name.append(compareInfo.class_name)
        gt.append(compareInfo.gt_count)
        tp.append(compareInfo.tp)
        fp.append(compareInfo.fp)
    x_max = max(gt)*1.5
    f, ax = plt.subplots(figsize=(6, 5))
    #设置主题
    sns.set_theme()
    sns.axes_style("darkgrid")
    #ax.set(xlim=(0, 1.1))

    #画横向柱状图
    splot1 = sns.barplot(x=gt, y=class_name, label="gt", ax =ax, color="b")

    #柱状图显示数字
    index = 0
    for p in splot1.patches:
        x = p.get_width()
        y = p.get_y()+ p.get_height() / 2
        x_offset = 30
        splot1.annotate(format(fp[index]), (x, y), ha = 'center', va = 'center', color='r', xytext = (x_offset, 0), textcoords = 'offset points')
        x_offset +=30
        splot1.annotate(format(tp[index]), (x, y), ha = 'center', va = 'center', color='g',xytext = (x_offset, 0), textcoords = 'offset points')
        x_offset +=30
        splot1.annotate(format(gt[index]), (x, y), ha = 'center', va = 'center', color='b',xytext = (x_offset, 0), textcoords = 'offset points')
        index +=1

    #sns.set_color_codes("muted")
    splot2 = sns.barplot(x=tp, y=class_name, label="tp", ax =ax, color="g")
    splot3 = sns.barplot(x=fp, y=class_name, label="fp", ax =ax, color="r")
    ax.set(xlim=(0, x_max))

    ax.legend()#显示右下方的标注
    plt.legend(loc=4)#显示标注位置


    save_path = os.path.join(image_save_dir,'gt_tp_fp.png')
    plt.savefig(save_path)
    # if show_fig:
    #     plt.show()
    plt.close()