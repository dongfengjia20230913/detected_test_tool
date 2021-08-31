
## 1. 数据格式
本地测试的数据集是使用MRLabeler标注的检测数据，数据目录结构如下
```
└── val
    ├── Annotations
    ├── JPEGImages
    └── labels
    └── mrconfig.xml

```

Annotations: voc格式的标注文件
labels: yolo格式的标注文件,数据格式为[class_id, centx,ceny,w,h], 数据是相对图片宽和搞归一化后的数据
mrconfig.xml: 数据说明文件，包括分类名称的定义

## 2. 生成ground truth
```
python AI_Perf_tool/Create_ground_truth_txt.py  -test val

```
-test 表示数据标注文件所在的目录。 生成的gt文件是在-test指定的目录下的ground_truth目录。每个图片对应一个txt文件

每个txt的文件内容如下：
```
#va/ground_truth/*.txt
phone 99 66 152 122
hand 122 87 188 149
hand 283 95 310 134
hand 279 64 307 105
face 252 42 299 93
```

## 3. 生成预测文件

```
python AI_Perf_tool/Create_detection_ret_txt.py  -log val/phone_84000.txt  -o val

```

-log:表示模型日志测试日志，python文件主要通过解析日志中的检测结果日志生成预测结果。模型生成的日志格式要求如下,每个检测结果对应一行日志
```
#detect out: [class_name] : [score] : [box] :[image_file_path]

detect out: phone : 0.971 : 77 94 118 156 :../val/JPEGImages/01041816390000971_3394.jpg

```

-o: 表示预测日志文件生成预测结果文件目录，存在指定目录下的detection_result目录下


## 4. 根据预测文件和gt文件生成指标
### 4.1 召回率和精准率

|---- |  正例   | 反例  |
|---- |  ----  | ----  |
|正例 | TP      | FP |
|反例 | FN      | TN |

精度说明：
TP：正例预测正确的个数
FP：负例预测错误的个数
TN：负例预测正确的个数
FN：正例预测错误的个数

精确率：
Precision=TP/(TP+FP)
召回率：
Recall=TP/(TP+FN)

漏报率 = 1-Recall
误报率 = 1-Precision

在信息检索领域，精确率和召回率又被称为查准率和查全率，

查准率＝检索出的相关信息量 / 检索出的信息总量
查全率＝检索出的相关信息量 / 系统中的相关信息总量



两者只是分母不同。针对我们实际工业检测上报场景，因为系统中的相关信息总量位置，也就是没有对应的测试机集，因此查准率和查全率是相同的

我们需要通过对比预测框和真实标注框之间的IOU, 计算各个阈值的TP,FP,和FN

通过以上公式，我们可以求出给个分类的在不同阈值分布上的Recall和Precision

```
------class 1------
tp: 633
fp: 5
tp10: [0, 0, 13, 13, 29, 35, 59, 122, 362, 0]
fp10: [0, 0, 4, 0, 0, 0, 1, 0, 0, 0]
count class num(gt): 667
recall10    : [0.949, 0.949, 0.949, 0.93, 0.91, 0.867, 0.814, 0.726, 0.543, 0]
precision_10: [0.992, 0.992, 0.992, 0.998, 0.998, 0.998, 0.998, 1.0, 1.0, 0]

```

### 4.2 AP-Average Precision
以Recall为横轴，Precision为纵轴，就可以画出一条PR曲线，PR曲线下的面积就定义为AP，即：
![tp-fp](https://img-blog.csdnimg.cn/1da760feb12040a5ba4a9143b2fa59d6.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5Y-k6aOO5a2Q,size_20,color_FFFFFF,t_70,g_se,x_16)


有积分的定义可以知道，上述图像，单位横轴长度的面积计算公式为

Δr(i)×p(i)
Δr(i) = r(i)-r(i-1)



## 5. 命令整合
通过运行以下命令，可以直接获取gt，dr文件和对应的结果，map等结果会汇总显示在一个html中
```
python AI_Perf_tool/score_create.py -log val/phone_70000.txt -val val
```
[github](https://github.com/jdf-eng/detected_test_tool.git), 欢迎star

## 6. 最终的图片生成结果
### 6.1 mAp
![map](https://img-blog.csdnimg.cn/fdefca40649a49c985a9ec1439ebb7f9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5Y-k6aOO5a2Q,size_20,color_FFFFFF,t_70,g_se,x_16)

### 6.2 rec-fpr

![rec-fpr](https://img-blog.csdnimg.cn/46cd3b79a5d94cb995ef5a0584218b0c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5Y-k6aOO5a2Q,size_20,color_FFFFFF,t_70,g_se,x_16)

### 6.3 阈值分布
![thresh_count](https://img-blog.csdnimg.cn/3e294f86b58c4461be84095c5d476114.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5Y-k6aOO5a2Q,size_20,color_FFFFFF,t_70,g_se,x_16)

### 6.4 tp-fp
![tp-fp](https://img-blog.csdnimg.cn/49f7a087e97e4a16a19a1155effa119e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5Y-k6aOO5a2Q,size_20,color_FFFFFF,t_70,g_se,x_16)
