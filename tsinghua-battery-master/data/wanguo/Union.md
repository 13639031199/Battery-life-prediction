### 异常检测

* 异常检测部分代码位于data/wanguo文件夹下，文件树如下：

  ```
  .
  ├── anomoly_detect.py
  ├── data_process.py
  ├── group0_features.npy
  ├── group0_targets.npy
  ├── group1_features.npy
  ├── group1_targets.npy
  ├── group2_features.npy
  ├── group2_targets.npy
  ├── north
  │   ├── 202205北侧.xlsx
  │   ├── 202206北侧.xlsx
  │   ├── 202207北侧.xlsx
  │   ├── 202208北侧.xlsx
  │   ├── 202209北侧.xlsx
  │   ├── 202210北侧.xlsx
  │   ├── 202211北侧.xlsx
  │   ├── 202212北侧.xlsx
  │   ├── 202301北侧.xlsx
  │   ├── 202302北侧.xlsx
  │   ├── 202303北侧.xlsx
  │   └── 202304北侧.xlsx
  ├── south
  │   ├── 202207南侧.xlsx
  │   ├── 202208南侧.xlsx
  │   ├── 202209南侧.xlsx
  │   ├── 202210南侧.xlsx
  │   ├── 202211南侧.xlsx
  │   ├── 202212南侧.xlsx
  │   ├── 202301南侧.xlsx
  │   ├── 202302南侧.xlsx
  │   ├── 202303南侧.xlsx
  │   └── 202304南侧.xlsx
  ├── test
  │   └── soh.npy
  └── train
      └── soh.npy
  ```

* 原理：用时间序列分解算法，将电池单体的电压等数据中随机噪声与总体趋势分离。然后对大量正常的正常电池的趋势使用核密度估计，再对所有单体的当前时间点计算一个离群值，绘制出单体各自离群值曲线。同样对离群值曲线进行时间序列分解，获得离群值变化趋势，然后使用随机森林算法检测出异常曲线。

* 输入：首先使用data_process.py选择想要进行异常检测的特征（例如温度、电压等，仓库中默认的是电压），目前只支持同时对一个特征进行异常检测，若想要多个特征，可以使用不同的特征分别进行检测后，获得一个综合的评价结果），然后运行`python data_process.py`，会在目录下生成groupN_features.npy和groupN_targets.npy。由于数据中本身有三个电池簇，所以这里N的值为0、1、2。features.npy存储[m,1]的ndarray，第一维表示cycle输，第二维表示特征的值。

* 运行：

  ```
  python anomoly_detect.py -path [feature.npy的路径] -train [是否需要训练从而寻找合适的contamination值（随机森林算法使用，表示异常值占总样本的比例）] -epoch [训练轮数，配合train参数使用] -contamination [contamination预设值] -plot [是否绘图]
  ```

  准备好输入，可以先运行`python anomoly_detect.py -train True - epoch N`来获得一个较好的contamination值，然后在运行`python anomoly_detect.py -contamination x`来进行异常检测。

  全局变量：

  SEQ_LEN：计算离群值时，使用的曲线窗口大小。默认100cycle

  DEFAULT_OUTLIER_CURVE ：发生异常的曲线书目。默认1条曲线

* 输出：

  ```
  ERROR: [147]
  PREDICT: [147]
  precision: 1.0 recall: 1.0
  ```

  在代码中，会自动选取DEFAULT_OUTLIER_CURVE数量的曲线施加逐渐增大的噪声，使得值异常。输出结果中ERROR表示真正异常单体的编号，PREDICT表示算法检测出的单体编号。precision和recall分别表示算法的准确率和召回率。


### SOH预测

* 异常检测代码大多位于仓库根目录，文件树如下：

  ```
  .
  ├── __pycache__
  ├── data
  ├── data_aug.py
  ├── data_loader.py
  ├── dataset.py
  ├── figures
  ├── models.py
  ├── our_data_loader.py
  ├── reports
  ├── requirements.txt
  ├── rul_main.py
  ├── seq_sampler.py
  ├── soh_main.py
  ├── tool.py
  └── utils.py
  ```

* 原理：使用北侧的soh曲线，预测南侧的soh曲线。北侧的soh通过data_aug.py的插值采样进行数据增强后，获得若干条参考曲线。然后，按照每100个cycle为序列，将北侧和南侧的soh曲线分别切分为若干段，构成训练集和测试集。在训练时，通过当前序列的起始soh值寻找到参考曲线的对应参考点，并从参考点取同样长度的序列与原序列计算dtw，获得相似度。取topk个最相似的曲线，作为参考，将他们和原序列的相似度值经过MLP后计算一个权重值，并分别乘以它们在达到原序列后M周期的soh值所需的周期数，求和获取原序列后第M周期soh值所需的预测周期数，与M作差获得loss。测试时原理相同。

* 输入：测试集和训练集数据分别保存于根目录下data/wanguo/test和data/wanguo/train文件夹下的soh.npy中。其中存储的ndarray的形状为[m,1]，其中第一维m表示cycle数，第二维表示对应cycle的soh值。该数据可以通过data_process.py获得。

* 运行：

  ```
  usage: soh_main.py [-h] [-seq_len SEQ_LEN] [-N N] [-batch BATCH] [-valid_batch VALID_BATCH] [-num_worker NUM_WORKER] [-epoch EPOCH]
                     [-lr LR] [-top_k TOP_K] [-aug_path AUG_PATH] [-data_path DATA_PATH]
  
  options:
    -h, --help            show this help message and exit
    -seq_len SEQ_LEN      序列长度
    -N N                  预测当前序列末尾后N个cycle的soh值
    -batch BATCH          batch_size
    -valid_batch VALID_BATCH
                          batch_size
    -num_worker NUM_WORKER
                          number of worker
    -epoch EPOCH          num of epoch
    -lr LR                learning rate
    -top_k TOP_K          选择top_k个最相似的序列作为参考
    -aug_path AUG_PATH    数据增强使用的.npy文件的路径
    -data_path DATA_PATH  存有train和test文件夹的文件夹路径
  ```

  准备好数据后，可以直接运行`python soh_main.py`进行训练+测试。

* 输出：

  ```
  Epoch 7
  Start training
  train_average_loss 0.8806097408135732
  Start validating
  tensor(10.8354, grad_fn=<SumBackward0>) tensor(10.)
  tensor(10.0849, grad_fn=<SumBackward0>) tensor(10.)
  tensor(10.1431, grad_fn=<SumBackward0>) tensor(10.)
  tensor(9.9145, grad_fn=<SumBackward0>) tensor(10.)
  tensor(9.7786, grad_fn=<SumBackward0>) tensor(10.)
  test_average_loss 0.09580437342325847
  ```

这里的两个loss分别为该batch内训练集和测试集预测的周期数和实际值的平均差值。输出的tensor tensor则是测试集中实际的一些例子。第一个tensor为预测值，后一个为实际值。

**-------------------------------------data_process.py----------------------------------------**
import glob
import pandas as pd
import numpy as np
import re

def get_id(list, pattern):
    first_match_index = None
    last_match_index = None
    for i, string in enumerate(list):
        if len(re.findall(pattern, string)) > 0:
            if first_match_index is None:
                first_match_index = i
            last_match_index = i

    assert first_match_index is not None and last_match_index is not None
    return first_match_index, last_match_index

def s_k(x, avg, std):
    return (x - avg) ** 3 / (std ** 3)

def k_k(x, avg, std):
    return (x - avg) ** 4 / (std ** 4)

def get_items_feature(pattern, features, data):
    s, e = get_id(features, pattern)
    pos_temp = data[s: e]
    avg_pos_temp = pos_temp.mean()
    std_pos_temp = pos_temp.std()
    sk_pos_temp = pos_temp.apply(s_k, args=(avg_pos_temp, std_pos_temp)).mean()
    kk_pos_temp = pos_temp.apply(k_k, args=(avg_pos_temp, std_pos_temp)).mean()
    return [avg_pos_temp, std_pos_temp, sk_pos_temp, kk_pos_temp]

def main():

    cycles = 0
    cycle_list = []

    folder_path = './south'  # 替换为实际的文件夹路径

    # 使用 glob 模块匹配所有 .xlsx 文件
    xlsx_files = sorted(glob.glob(folder_path + '/*.xlsx'))

    # 使用 Pandas 打开每个 .xlsx 文件

    equip_features = [[], [], []]
    equip_targets = [[], [], []]


    for file in xlsx_files:
        df = pd.read_excel(file)
        print(f"文件名: {file}")

        # 有若干个电池组，分别算出哪些数据属于哪个电池组
        all_equipments = df['设备'].tolist()
        all_equipment_name = list(set(all_equipments))
        equip_index_range = []
        for equip in all_equipment_name:
            equip_index_range.append(len(all_equipments) - list(reversed(all_equipments)).index(equip)) # 不 -1，因为后面访问时是左闭右开
        equip_index_range.sort()
        print("segments:", equip_index_range)
       
        all_feature_name = df['单位'].tolist()
         
        for column, column_data in df.items():
            start_index = 0
            if re.findall(r'\d{4}-\d{1,2}-\d{1,2}', column): # 该列是某一天的数据
                for (i, end_index) in enumerate(equip_index_range):
                    features = []
                    targets = []

                    sub_feature_name = all_feature_name[start_index: end_index]
                    sub_col_data = column_data[start_index: end_index].reset_index(drop=True)

                    targets.append(sub_col_data[sub_feature_name.index('系统SOH')])
                    # features.append(sub_col_data[sub_feature_name.index('系统SOH')])

                    # features.append(sub_col_data[sub_feature_name.index('系统总电压_V')])
                    # features.append(sub_col_data[sub_feature_name.index('系统平均电压')])
                    # features.append(sub_col_data[sub_feature_name.index('系统平均温度')])

                    # 1簇feature
                    # features.append(sub_col_data[sub_feature_name.index('1_簇并机总电压')])
                    for j in range(1, 160):
                        features.append(sub_col_data[sub_feature_name.index(f'1_簇单体电压{j}')])
                    # features += get_items_feature(r'1_模组\d{1,2}正极柱温度', sub_feature_name, sub_col_data)
                    # features += get_items_feature(r'1_模组\d{1,2}负极柱温度', sub_feature_name, sub_col_data)
                    # features += get_items_feature(r'1_簇单体电压\d{1,2}', sub_feature_name, sub_col_data)
                    # features += get_items_feature(r'1_簇单体温度\d{1,2}', sub_feature_name, sub_col_data)

                    # # 2簇feature
                    # features.append(sub_col_data[sub_feature_name.index('2_簇并机总电压')])
                    # features.append(sub_col_data[sub_feature_name.index('2_簇电池总电压')])
                    # features += get_items_feature(r'2_模组\d{1,2}正极柱温度', sub_feature_name, sub_col_data)
                    # features += get_items_feature(r'2_模组\d{1,2}负极柱温度', sub_feature_name, sub_col_data)
                    # features += get_items_feature(r'2_簇单体电压\d{1,2}', sub_feature_name, sub_col_data)
                    # features += get_items_feature(r'2_簇单体温度\d{1,2}', sub_feature_name, sub_col_data)

                    # if i != 0:
                    #     # 3簇feature
                    #     features.append(sub_col_data[sub_feature_name.index('3_簇并机总电压')])
                    #     features.append(sub_col_data[sub_feature_name.index('3_簇电池总电压')])
                    #     features += get_items_feature(r'3_模组\d{1,2}正极柱温度', sub_feature_name, sub_col_data)
                    #     features += get_items_feature(r'3_模组\d{1,2}负极柱温度', sub_feature_name, sub_col_data)
                    #     features += get_items_feature(r'3_簇单体电压\d{1,2}', sub_feature_name, sub_col_data)
                    #     features += get_items_feature(r'3_簇单体温度\d{1,2}', sub_feature_name, sub_col_data)

                    equip_features[i].append(features)
                    equip_targets[i].append(targets)
                    
                    start_index = end_index

                cycles += 1

    for (i, features) in enumerate(equip_features):
        arr = np.array(features)
        print(arr.shape)
        np.save(f'group{i}_features.npy', arr)

    for (i, targets) in enumerate(equip_targets):
        targets = np.array(targets)
        targets = targets.flatten()
        targets = targets[targets != 0]
        arr = targets.reshape(-1, 1)
        print(arr.shape)
        np.save(f'group{i}_targets.npy', arr)
    

if __name__ == "__main__":
    main()

**-------------------------------------anomoly_detect.py-------------------------------------**
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
import random
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import argparse

SEQ_LEN = 100
DEFAULT_OUTLIER_CURVE = 1

def ts_decompose(data):
    time_index = pd.date_range(start='2022-05-01', periods=len(data))
    time_series = pd.Series(data, index=time_index)
    # 对时间序列进行分解
    result = seasonal_decompose(time_series, model='additive')
    return result

# 定义计算离群值的函数
def calculate_outlier_values(data):
    print(data.shape)
    outlier_values = []
    for i in range(data.shape[0]):
        sequence = data[:i+1]
        distances = []
        for j in range(data.shape[1]):
            if j == i:
                continue
            other_sequence = data[:i+1, j]
            distance, path = fastdtw(sequence, other_sequence, dist=euclidean)
            distances.append(distance)
        outlier_values.append(np.mean(distances))
    return np.array(outlier_values)

def main(contamination, path):
    # 读取.npy文件
    data = np.load(path)

    random_indices = random.sample(range(0, data.shape[1]-1), DEFAULT_OUTLIER_CURVE) 
    random_indices.sort()
    print("ERROR:", random_indices)

    # 生成扰动
    for random_index in random_indices:
        start = int((0.1 + 0.4 * np.random.rand()) * len(data[:, random_index]))
        disturbance_length = len(data[:, random_index]) - start
        disturbance = np.array([(1.00001 + 0.00009 * np.random.rand()) ** i for i in range(disturbance_length)])
        data[start:, random_index] *= disturbance

    decomposed = [seasonal_decompose(data[:, i], model='additive', period=1) for i in range(data.shape[1])]
    trends = [decomp.trend for decomp in decomposed]
    trends = np.array(trends)

    mask = np.isin(range(data.shape[-1]), random_indices)
    normal_trends = trends[~mask]
    abnorm_trends = trends[mask]

    from sklearn.neighbors import KernelDensity
    # train_trends = np.transpose(normal_trends)
    probs_list = []
    for t in range(SEQ_LEN, trends.shape[1]):
        partial_trends = normal_trends[:, t - SEQ_LEN:t]
        # print(normal_trends.shape, abnorm_trends.shape)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
        kde.fit(partial_trends)
        log_probs = kde.score_samples(trends[:,t - SEQ_LEN:t])
        scaled_log_probs = (log_probs - np.min(log_probs)) / (np.max(log_probs) - np.min(log_probs))
        # threshold = np.percentile(log_probs, int(len(random_indices) / len(trends) * 100))
        # outlier_indices = np.where(log_probs < threshold)[0]
        probs_list.append(scaled_log_probs)

    probs_list = np.array(probs_list).transpose()
    decomposed_probs = [seasonal_decompose(probs_list[i], model='additive', period=1) for i in range(probs_list.shape[0])]
    prob_trends = [decomp.trend for decomp in decomposed_probs]

    clf = IsolationForest(contamination=contamination).fit(probs_list)

    # 预测异常的KDE曲线
    pred = clf.predict(probs_list)

    # 找出被认为是异常的曲线的索引
    outlier_indices = np.where(pred == -1)[0]
    print("PREDICT:", outlier_indices)
    # plt.legend()
    true_pos = set(outlier_indices) & set(random_indices)
    precision = len(list(true_pos)) / len(outlier_indices)
    recall = len(list(true_pos)) / len(random_indices)
    print("precision:", precision, "recall:", recall)
    # plot
    if args.plot:
        for i, prob_trend in enumerate(prob_trends):
            # print(prob_curve)
            # plt.plot([i for i in range(len(prob_curve.trend))], prob_curve.trend, label=f'{i}')
            plt.plot(prob_trend)
        plt.xlabel("cycle")
        plt.ylabel("prob")
        plt.show()

    return precision, recall

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-contamination", default=0.005, type=float)
    arg_parser.add_argument("-path", default="./group0_features.npy", type=str)
    arg_parser.add_argument("-epoch", default=20, type=int)
    arg_parser.add_argument("-train", help="get best_contamination", default=False, type=bool)
    arg_parser.add_argument("-plot", help="get plot", default=False, type=bool)

    args = arg_parser.parse_args()

    if args.train:
        state_dict = {}
        for epoch in range(args.epoch):
            max_precision, max_recall = 0, 0
            best_contamination = 0
            print("Epoch 1")
            for i,  contamination in enumerate([0.07 + 0.01 * j for j in range(10)]):
                # print("Test:", i)
                precision, recall = main(contamination, args.path)
                if precision > max_precision and recall > max_recall:
                    max_precision = precision
                    max_recall = recall
                    best_contamination = contamination

            print("Best:", max_precision, max_recall, best_contamination)
            if best_contamination in  state_dict.keys():
                state_dict[best_contamination] += 1
            else:
                state_dict[best_contamination] = 1
        
        print("Best contamination", best_contamination)
    else:
        main(args.contamination, args.path)

**------------------------------------------data_aug.py--------------------------------------**
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib.pyplot as plt

def interpolate(data, scale_rate=1):

    # 原始数据的长度
    original_length = len(data)    

    # 你希望的新数据的长度（降采样后）
    new_length = int(original_length * scale_rate)

    # 创建原始数据的x坐标
    x = np.linspace(0, 1, original_length)  

    # 创建插值函数
    f = interp1d(x, data.squeeze(1))  

    spline = UnivariateSpline(x, data)

    # 创建新的x坐标
    new_x = np.linspace(0, 1, new_length)

    # 使用插值函数计算新的y坐标（数据）
    new_data = f(new_x) 

    new_spline = spline(new_x)

    # print("Original data:", data)
    # print("New data:", new_data)
    return new_data, new_spline, new_length, f, spline, original_length

def data_aug(path):
    train_data = np.load('/'.join([path, 'soh.npy']))
    scale_rates = [0.8, 0.9, 0.93, 0.96, 0.98, 1.02, 1.05, 1.08, 1.1, 1.2, 1.3]
    curve_list, s_curve_list, curve_func_list, s_curve_func_list, new_length_list, origin_length_list = [], [], [], [], [], []
    for scale_rate in scale_rates:
        curve, smooth_curve, new_length, curve_f, smooth_curve_f, original_length = interpolate(train_data, scale_rate)
        curve_list.append(curve)
        s_curve_list.append(smooth_curve)
        curve_func_list.append(curve_f)
        s_curve_func_list.append(smooth_curve_f)
        new_length_list.append(new_length)
        origin_length_list.append(original_length)
    #     plt.plot([i for i in range(len(curve))], curve)
    #     plt.plot([i for i in range(len(curve))], smooth_curve)
    # plt.show()
    return s_curve_list, s_curve_func_list, new_length_list


if __name__ == "__main__":
    data_aug()
        
**----------------------------------------soh_main.py-----------------------------------------**
from data_aug import data_aug, interpolate
import numpy as np
import argparse
from scipy.optimize import root
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SeqDataset
from dtw import *
from models import MLP
import matplotlib.pyplot as plt
import os
from utils import score_weight_loss

class NegativeMSELoss(nn.Module):
    def __init__(self):
        super(NegativeMSELoss, self).__init__()

    def forward(self, input, target):
        return -torch.mean((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

# def custom_loss(y_pred, x):
#     mean_centered_pred = y_pred - torch.mean(y_pred)
#     mean_centered_input = x - torch.mean(x)
#     corr = torch.sum(mean_centered_pred * mean_centered_input) / (torch.sqrt(torch.sum(mean_centered_pred ** 2)) * torch.sqrt(torch.sum(mean_centered_input ** 2)))
#     return corr



def build_retrieval_set(curve_funcs, curve_lens, soh, seq_len):
    retrieval_set = []
    for j, curve_func in enumerate(curve_funcs):
        point = root(get_root, x0=0, args=(curve_func, soh))
        curve_len = curve_lens[j]
        start_cycle = round(point.x.item() * curve_len)
        retrieval_seq = []
        for k in range(seq_len):
            retrieval_seq.append(curve_func((start_cycle - seq_len + k) / curve_len))
        retrieval_set.append(np.array(retrieval_seq))
    return retrieval_set

def build_dataset(raw, seq_len, N):
    seqs = []
    targets = []
    for i in range(raw.shape[0] - seq_len - N):
        seq = raw[i: i + seq_len]
        target = raw[i + seq_len + N - 1]
        seqs.append(seq)
        targets.append([N, *target])
    return np.array(seqs), np.array(targets)

def get_root(x, spline, y_target):
    return spline(x) - y_target 

def run(seq_len, N, curve_lens, curve_funcs, args):
    train_data = np.load(f'{args.data_path}train/soh.npy')
    test_data = np.load(f'{args.data_path}test/soh.npy')
    _, smooth_train_data, _, _, _, _ = interpolate(train_data)
    _, smooth_test_data, _, _, _, _ = interpolate(test_data)
    smooth_train_data = smooth_train_data.reshape(-1, 1)
    smooth_test_data = smooth_test_data.reshape(-1, 1)
    # for i, curve in enumerate(curves):
        # plt.plot([i for i in range(len(curve))], curve, label="i")
    # plt.plot([i for i in range(len(smooth_test_data))], smooth_test_data, label='test')
    # plt.legend()
    # plt.show()
    train_seqs, train_targets = build_dataset(smooth_train_data, seq_len, N)
    test_seqs, test_targets = build_dataset(smooth_test_data, seq_len, N)
    print("Train Dataset:", train_seqs.shape, train_targets.shape, "Test Dataset:", test_seqs.shape, test_targets.shape)

    train_set = SeqDataset(train_seqs, train_targets)
    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.num_worker, shuffle=False)

    test_set = SeqDataset(test_seqs, test_targets)
    test_loader = DataLoader(test_set, batch_size=args.valid_batch, num_workers=args.num_worker, shuffle=False)

    criterion = nn.MSELoss()
    score_model = MLP(args.top_k, args.top_k)
    optimizer = torch.optim.Adam(score_model.parameters(), lr=args.lr)

    score_model.train()
    min_valid_loss = float('inf')
    min_valid_epoch = 0
    for epoch in range(args.epoch):
        print("Epoch", epoch)
        # Train
        print("Start training")
        sum_loss, batch_num = 0, 0
        self_curve_len = len(smooth_train_data)
        for _, (seq, target) in enumerate(train_loader):
            loss = 0
            start_soh = seq[:, -1, :]
            for i in range(len(start_soh)):
                retrieval_set = build_retrieval_set(curve_funcs, curve_lens, start_soh[i].item(), seq_len)
                x = seq[i].numpy().reshape(-1)
                scores = []
                for retrieval_seq in retrieval_set:
                    alignment = dtw(x, retrieval_seq, keep_internals=True) # DTW算法
                    scores.append(alignment.distance)
                scores = np.array(scores)
                chosen_scores, indices = torch.topk(torch.Tensor(-scores), args.top_k) # 乘 -1 找最小的
                chosen_scores = -chosen_scores
                # print(chosen_scores)
                max_ = torch.max(chosen_scores)
                min_ = torch.min(chosen_scores)
                chosen_scores = (chosen_scores - min_) / (max_ - min_)
                output = score_model(chosen_scores)
                # weight = 1 / (1 + output)
                # print(chosen_scores, output, indices)
                error, diff = score_weight_loss(output, chosen_scores)
                loss_0 = score_model.loss_weights[0] * error + score_model.loss_weights[1] * diff

                target_cycle = target[i][0].float()
                target_soh = target[i][1]
                # TODO：这里的权重到最后权重的映射不是线性的
                output_clone = output.clone()
                for n in range(len(indices)):
                    curve_id = indices[n]
                    start_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], start_soh[i].data.item()))
                    predict_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], target_soh.item()))
                    output_clone[n] *= (predict_cycle.x.item() - start_cycle.x.item()) * curve_lens[curve_id]
                output = output_clone
                # loss += criterion(output.sum() / self_curve_len, target_cycle / self_curve_len)
                loss_1 = criterion(output.sum(), target_cycle) 
                loss += loss_0 + loss_1

            loss /= args.batch
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sum_loss += loss.item()
            batch_num += 1
        print("train_average_loss", sum_loss / batch_num)
        # Eval
        print("Start validating")
        score_model.eval()
        loss = 0
        batch_num = 0
        predict_list, target_list = [], []
        self_curve_len = len(smooth_test_data)
        for step, (seq, target) in enumerate(test_loader):
            if step == 0:
                for i in range(len(seq[0])):
                    predict_list.append(i)
                    target_list.append((i, seq[0][i].item()))
            start_soh = seq[:, -1, :]
            retrieval_set =  build_retrieval_set(curve_funcs, curve_lens, start_soh.item(), seq_len)
            x = seq.numpy().reshape(-1)
            scores = []
            for retrieval_seq in retrieval_set:
                alignment = dtw(x, retrieval_seq, keep_internals=True) # DTW算法
                scores.append(alignment.distance)
            scores = np.array(scores)
            chosen_scores, indices = torch.topk(torch.Tensor(-scores), args.top_k) # 乘 -1 找最小的
            chosen_scores = -chosen_scores
            max_ = torch.max(chosen_scores)
            min_ = torch.min(chosen_scores)
            chosen_scores = (chosen_scores - min_) / (max_ - min_)
            output = score_model(chosen_scores)
            # print(chosen_scores, output, indices)
            target = target.flatten()
            target_cycle = target[0].float()
            target_soh = target[1]
            #TODO：这里的权重到最后权重的映射不是线性的
            output_clone = output.clone()
            for n in range(len(indices)):
                curve_id = indices[n]
                start_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], start_soh.item()))
                predict_cycle = root(get_root, x0=0, args=(curve_funcs[curve_id], target_soh.item()))
                output_clone[n] *= (predict_cycle.x.item() - start_cycle.x.item()) * curve_lens[curve_id]
            if step % 40 == 0:
                print(output_clone.sum(), target_cycle)
            output = output_clone
            # print(output_clone.sum().detach().item() * self_curve_len, target_cycle.detach().item())
            target_list.append((target_cycle.detach().numpy(), target_soh.detach().numpy()))
            predict_list.append(step + seq_len + output_clone.sum().detach().item())
            loss += (output_clone.sum().detach().item() - target_cycle.detach().item())
            batch_num += 1
        print("test_average_loss", loss / batch_num)
        # for i in range(len(target_list)):
        #     print(target_list[i], predict_list[i])
        # assert 0
        if abs(loss / batch_num) < min_valid_loss:
            min_valid_loss = abs(loss / batch_num)
            min_valid_epoch = epoch
            plt.plot([i for i in range(len(smooth_test_data))], smooth_test_data)
            plt.plot([i for i in predict_list], [soh for cycle, soh in target_list])
            save_path = f"./figures/seq_len={seq_len}&N={N}"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig('/'.join([save_path, f'epoch{epoch}_{min_valid_loss}.png']))
            plt.close()
    with open(f"./reports/seq_len={seq_len}&N={N}.txt", 'w') as f:
        f.write(f"Summary: min_valid_loss: {min_valid_loss} in epoch: {min_valid_epoch}, relative_min_loss: {min_valid_loss / N * 100}%")
        f.close()

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-seq_len", help="The sequence length", default=100, type=int)
    arg_parser.add_argument("-N", help="Predict the next N cycle's SoH", default=10, type=int)
    arg_parser.add_argument("-batch", help="batch_size", default=16, type=int)
    arg_parser.add_argument("-valid_batch", help="batch_size", default=1, type=int)
    arg_parser.add_argument("-num_worker", help="number of worker", default=0, type=int)
    arg_parser.add_argument("-epoch", help="num of epoch", default=10, type=int)
    arg_parser.add_argument("-lr", help="learning rate", default=1e-3, type=float)
    arg_parser.add_argument("-top_k", help="choose top_k to retrieval", default=3, type=int)
    arg_parser.add_argument("-aug_path", help="the path of train set's soh.npy for data augmentation", default='./data/wanguo/train', type=str)
    arg_parser.add_argument("-data_path", help="the path of train set and test set", default='./data/wanguo/', type=str)


    args = arg_parser.parse_args()

    curves, curve_funcs, curve_lens = data_aug(args.aug_path)
    # print(curve_lens)
    # assert 0
    print("Curves number after augmentation:", len(curves))
    
    run(args.seq_len, args.N, curve_lens, curve_funcs, args)

            

if __name__ == "__main__":
    main()

**---------------------------------------data_loader.py-----------------------------------**
import numpy as np
from scipy import interpolate

def interp(x, y, num, ruls, rul_factor):
    ynew = []
    for i in range(y.shape[1]):
        f = interpolate.interp1d(x, y[:, i], kind='linear')
        x_new = np.linspace(x[0], x[-1], num)
        ytmp = f(x_new)
        ynew.append(ytmp)
    ynew = np.vstack(ynew)
    ynew = ynew.T
    newruls = [i for i in range(1, ynew.shape[0] + 1)]
    newruls.reverse()
    newruls = np.array(newruls).astype(float)
    # remove rul_factor
    # newruls /= rul_factor
    new_right_end_value = ruls[-1] * (num / len(x))
    for i in range(len(newruls)):
        newruls[i] += new_right_end_value
    return ynew, newruls


def data_aug(feas, ruls, scale_ratios, rul_factor):
    augmented_feas, augmented_ruls = [], []
    for scaleratio in scale_ratios:
        if int(scaleratio * feas.shape[0]) <= 100:
            continue
        augmented, rul = interp([i for i in range(feas.shape[0])], feas,
                                int(scaleratio * feas.shape[0]), ruls,
                                rul_factor)
        augmented_feas.append(augmented)
        augmented_ruls.append(rul)
    return augmented_feas, augmented_ruls

def split_seq(fullseq, rul_labels, seqlen, seqnum):
    if isinstance(fullseq, list):
        all_fea, all_lbls = [], []
        for seqidx in range(len(fullseq)):
            tmp_all_fea = np.lib.stride_tricks.sliding_window_view(
                fullseq[seqidx], (seqlen, fullseq[seqidx].shape[1]))

            tmp_all_fea = tmp_all_fea.squeeze()
            tmp_lbls = rul_labels[seqidx][seqlen - 1:]
            tmp_fullseqlen = rul_labels[seqidx][0]
            fullseqlens = np.array(
                [tmp_fullseqlen for _ in range(tmp_all_fea.shape[0])])
            # print(tmp_lbls.shape, fullseqlens.shape, fullseq[seqidx].shape, rul_labels[seqidx].shape)
            lbls = np.vstack((tmp_lbls, fullseqlens)).T
            if seqnum <= tmp_all_fea.shape[0]:
                all_fea.append(tmp_all_fea[:seqnum])
                all_lbls.append(lbls[:seqnum])
            else:
                all_fea.append(tmp_all_fea)
                all_lbls.append(lbls)
        all_fea = np.vstack(all_fea)
        all_lbls = np.vstack(all_lbls)
        all_lbls = all_lbls.astype(int)
        return all_fea, all_lbls
    else:
        all_fea = np.lib.stride_tricks.sliding_window_view(
            fullseq, (seqlen, fullseq.shape[1]))
        all_fea = all_fea.squeeze()
        # ruls = rul_labels[seqlen-1:]
        fullseqlen = rul_labels[0]
        lbls = rul_labels[seqlen - 1:]
        fullseqlens = np.array([fullseqlen for _ in range(all_fea.shape[0])])
        lbls = np.vstack((lbls, fullseqlens)).T
        lbls = lbls.astype(int)
        if seqnum <= all_fea.shape[0]:
            return all_fea[:seqnum], lbls[:seqnum]
        else:
            return all_fea, lbls

def get_train_test_val(series_len=100,
                       rul_factor=3000,
                       dataset_name='train',
                       seqnum=500,
                       data_aug_scale_ratios=None):

    metadata = np.load('ne_data/meta_data.npy', allow_pickle=True)
    if dataset_name == 'train':
        set = metadata[0]
    elif dataset_name == 'valid':
        set = metadata[1]
    elif dataset_name == 'trainvalid':
        set = metadata[0] + metadata[1]
    else:
        set = metadata[2]

    allseqs, allruls, batteryids = [], [], []
    batteryid = 0
    for batteryname in set:
        seqname = 'ne_data/' + batteryname + '.npy'
        lblname = 'ne_data/' + batteryname + '_rul.npy'
        seq = np.load(seqname, allow_pickle=True)
        lbls = np.load(lblname, allow_pickle=True)
        origin_dq = np.array([seq[0][0] for i in range(seq.shape[0])]).reshape(
            (-1, 1))
        seq = np.hstack((seq, origin_dq))

        if data_aug_scale_ratios is not None:
            seqs, ruls = data_aug(seq, lbls, data_aug_scale_ratios, rul_factor)
            feas, ruls = split_seq(seqs, ruls, series_len, seqnum)
        else:
            feas, ruls = split_seq(seq, lbls, series_len, seqnum)

        allseqs.append(feas)
        allruls.append(ruls)
        batteryids += [batteryid for _ in range(feas.shape[0])]
        batteryid += 1
    batteryids = np.array(batteryids).reshape((-1, 1))
    allruls = np.vstack(allruls)
    allruls = np.hstack((allruls, batteryids))
    allseqs = np.vstack(allseqs)
    print("origin data:", allruls.shape, allseqs.shape)
    return allseqs, allruls

**-----------------------------------------dataset.py---------------------------------------**
from torch.utils.data import Dataset, DataLoader

class ContrastiveDataset(Dataset):
    def __init__(self, features, labels, batchsize):
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels
        self.batch = batchsize

    def __getitem__(self, index):
        start = index
        end = index + self.batch
        next_start = index + 1
        next_end = index + 1 + self.batch
        if next_end > len(self.features):  # 超过tensor边界
            return None
        current_features = self.features[start:end]
        next_features = self.features[next_start:next_end]
        current_labels = self.labels[start:end]
        next_labels = self.labels[next_start:next_end]
        return (current_features, current_labels), (next_features, next_labels)

    def __len__(self):
        return len(self.features) - self.batch - 1


class SeqDataset(Dataset):
    def __init__(self, seqs, targets) -> None:
        super().__init__()
        self.seqs = seqs
        self.targets = targets

    def __getitem__(self, index):
        return self.seqs[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


**----------------------------------------models.py----------------------------------------**
import torch
from torch import nn
import math
from einops import rearrange, repeat

# from einops.layers.torch import Rearrange
import torch.nn.functional as F

# helpers

# 定义MLP网络
class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channel, 64),  # 输入层，假设有100个神经元
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 64),  # 隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(64, out_channel), # 输出层，假设有1个神经元
            nn.Softmax(dim=0)
        )
        self.loss_weights = nn.Parameter(torch.ones(2))

    def forward(self, x):
        output = self.layers(x)
        # output = self.std_layer(output)
        # output = nn.ReLU()(output)
        return output


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head=64, mlp_dim=64, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class lstm_encoder(nn.Module):
    def __init__(self, indim, hiddendim, fcdim, outdim, n_layers, dropout=0.4):
        super(lstm_encoder, self).__init__()
        self.lstm1 = torch.nn.LSTM(
            input_size=indim,
            hidden_size=hiddendim,
            batch_first=True,
            bidirectional=False,
            num_layers=n_layers,
        )
        # self.lstm2 = torch.nn.LSTM(input_size=hiddendim, hidden_size=hiddendim, batch_first=True, bidirectional=False, num_layers=n_layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hiddendim * n_layers, fcdim)
        self.bn1 = torch.nn.LayerNorm(normalized_shape=fcdim)
        self.fc2 = torch.nn.Linear(fcdim, outdim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        out, (h, c) = self.lstm1(x)
        # out, (h, c) = self.lstm2(h)
        h = h.reshape(x.size(0), -1)
        h = self.dropout(h)
        # h = h.squeeze()
        x = self.fc1(h)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.sigmoid(x)
        return x
    
**-----------------------------------our_data_loader.py-------------------------------------**
import os
import time
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy import interpolate
from datetime import datetime
import pandas as pd
from sklearn.metrics import roc_auc_score,mean_squared_error,mean_absolute_error, r2_score
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import get_xy

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.sampler import RandomSampler
    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')

def our_data_loader_generate():

    i_low = -2199
    i_upp = 5498
    v_low = 3.36
    v_upp = 3.60
    q_low = 610
    q_upp = 1190
    rul_factor = 3000
    cap_factor = 1190
    pkl_dir = './data/our_data/'
    series_lens = [100]

    new_valid = ['4-3', '5-7', '3-3', '2-3', '9-3', '10-5', '3-2', '3-7']
    new_train = ['9-1', '2-2', '4-7','9-7', '1-8','4-6','2-7','8-4', '7-2','10-3', '2-4', '7-4', '3-4',
            '5-4', '8-7','7-7', '4-4','1-3', '7-1','5-2', '6-4', '9-8','9-5','6-3','10-8','1-6','3-5',
             '2-6', '3-8', '3-6', '4-8', '7-8','5-1', '2-8', '8-2','1-5','7-3', '10-2','5-5', '9-2','5-6', '1-7', 
             '8-3', '4-1','4-2','1-4','6-5', ]
    new_test  = ['9-6','4-5','1-2', '10-7','1-1', '6-1','6-6', '9-4','10-4','8-5', '5-3','10-6',
            '2-5','6-2','3-1','8-8', '8-1','8-6','7-6','6-8','7-5','10-1']

    train_fea, train_lbl = [], []
    for name in new_train + new_valid:
        print(f"loading {name}")
        '''
        label: [rul, full_seq]
        '''
        tmp_fea, tmp_lbl = get_xy(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp,
                                                   rul_factor, cap_factor, pkl_dir, raw_features=False)
        # tmp_fea, tmp_lbl = get_xy(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor)
        train_fea.append(tmp_fea)
        train_lbl.append(tmp_lbl)
    
    test_fea, test_lbl = [], []
    for name in new_test:
        print(f"loading {name}")
        '''
        label: [rul, full_seq]
        '''
        tmp_fea, tmp_lbl = get_xy(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp,
                                                   rul_factor, cap_factor, pkl_dir, raw_features=False)
        # tmp_fea, tmp_lbl = get_xy(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor)
        test_fea.append(tmp_fea)
        test_lbl.append(tmp_lbl)

    
    train_fea = np.vstack(train_fea)
    train_lbl = np.vstack(train_lbl)
    test_fea = np.vstack(test_fea)
    test_lbl = np.vstack(test_lbl)

    train_soh_index = np.argsort(train_lbl[:, -1])
    sorted_train_fea = train_fea[train_soh_index]
    sorted_train_lbl = train_lbl[train_soh_index]
    # print(train_fea.shape, train_lbl.shape, test_fea.shape, test_lbl.shape)
    return sorted_train_fea, sorted_train_lbl, test_fea, test_lbl

if __name__ == "__main__":
    our_data_loader_generate()

**----------------------------------rul_main.py-------------------------------------**
from our_data_loader import our_data_loader_generate
from torch.utils.data import DataLoader, TensorDataset
import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from models import lstm_encoder, MLP
import math
from dataset import ContrastiveDataset
from dtw import dtw
import numpy as np
from utils import score_weight_loss
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def contrastive_loss(source, pos_sample, tao):

    assert source.shape[0] == pos_sample.shape[0]
    N = source.shape[0]

    def sim(tensor1, tensor2):
        if tensor1.shape != tensor2.shape:
            tensor1 = tensor1.reshape(1, -1)
            return torch.cosine_similarity(tensor1, tensor2)
        else:
            return torch.cosine_similarity(tensor1, tensor2, dim=0)

    def _l(i, type):
        denominator = 0
        if type == "src":
            denominator += torch.sum(torch.exp(sim(source[i], source) / tao))
            denominator += torch.sum(torch.exp(sim(source[i], pos_sample) / tao))
        else:
            denominator += torch.sum(torch.exp(sim(pos_sample[i], pos_sample) / tao))
            denominator += torch.sum(torch.exp(sim(pos_sample[i], source) / tao))
        denominator -= math.exp(1 / tao)
        numerator = torch.exp(sim(pos_sample[i], source[i]) / tao)
        return -torch.log(numerator / denominator).item()

    L = 0
    for i in range(N):
        L += _l(i, "src") + _l(i, "pos")
    # print((e-s).microseconds / 10**6)
    return L / (2 * N)

def run(train_loader, test_loader, args, fea_num):
    encoder = lstm_encoder(indim=fea_num, hiddendim=args.lstm_hidden, fcdim=args.fc_hidden, outdim=args.fc_out, n_layers=args.lstm_layer, dropout=args.dropout)
    encoder_optimizer = Adam(encoder.parameters(), lr=args.lr)
    encoder_lr_scheduler = StepLR(encoder_optimizer, step_size=1, gamma=args.gamma)
    relation_model = MLP(in_channel=args.top_k, out_channel=args.top_k)
    relation_model_optimizer = Adam(
            relation_model.parameters(),
            lr=args.lr,
        )
    relation_lr_scheduler = StepLR(
            relation_model_optimizer, step_size=100, gamma=args.gamma
        )
    


    train_loss = []
    train_loss = []
    for e in range(args.epoch):
        all_retrieval_feas = []
        all_retrieval_lbls = []   


        encoder_lr_scheduler.step(e)
        relation_lr_scheduler.step(e)

            
        print(
            "training epoch:",
            e,
            "learning rate:",
            encoder_optimizer.param_groups[0]["lr"],
        )


        encoder.train().cuda()
        relation_model.train().cuda()


        for step, ((features, labels), (nei_features, _)) in enumerate(train_loader):
            features = features.squeeze(0).cuda()
            nei_features = nei_features.squeeze(0).cuda()
            labels = labels.squeeze(0).cuda()
            encoded_source = encoder(features)
            all_retrieval_feas.append(encoded_source.cpu())
            all_retrieval_lbls.append(labels.cpu())
            encoded_neigh = encoder(nei_features)
            assert encoded_source.shape == encoded_neigh.shape
            loss = 0
            # contrastive_l = contrastive_loss(encoded_source, encoded_neigh, args.tao)
            for i in range(len(features)):
                '''
                    generate retrieval set
                '''
                retreival_lbls = torch.cat((labels[:i], labels[i + 1 :]), dim=0)
                retreival_ruls = retreival_lbls[:, -1]
                target_rul = labels[i, -1]

                encoded_retrieval_feas = torch.cat((encoded_source[:i], encoded_source[i + 1:]), dim=0)

                relation_scores = []
                        

                # for retrieval_tensor in encoded_retrieval_feas:
                #     s = F.cosine_similarity(encoded_source[i], retrieval_tensor, dim=0).cpu().detach().numpy()
                #     relation_scores.append(s)
                # relation_scores = np.array(relation_scores)

                        
                # import pdb;pdb.set_trace()
                relation_scores = F.cosine_similarity(encoded_source[i].unsqueeze(0), encoded_retrieval_feas,dim=1)
                chosen_scores, indices = torch.topk(torch.Tensor(relation_scores), args.top_k,dim=0) # 乘 -1 找最小的


                # relation_scores = F.cosine_similarity(encoded_source, encoded_retrieval_feas)
                # chosen_scores, indices = torch.topk(relation_scores, args.top_k)
                        
                # relation_scores = F.cosine_similarity(encoded_source[:-1], encoded_retrieval_feas, dim=1)
                # relation_scores = relation_scores.t() # 转置操作，使得relation_scores的形状与原来代码中的一致
                # chosen_scores, indices = torch.topk(relation_scores, args.top_k)


                # chosen_scores = -chosen_scores
                max_ = torch.max(chosen_scores)
                min_ = torch.min(chosen_scores)
                chosen_scores = (chosen_scores - min_) / (max_ - min_)
                chosen_scores=chosen_scores.cuda()

                output = relation_model(chosen_scores)
                
                error, diff = score_weight_loss(output, chosen_scores)
                # TODO: Here meet a bug. the loss_0 will become negative
                # loss_0 = relation_model.loss_weights[0] * error + relation_model.loss_weights[1] * diff
                loss_0 = error + diff

                predict_rul = torch.sum(chosen_scores * retreival_ruls[indices])
                loss_1 = nn.MSELoss()(target_rul, predict_rul)
                loss += loss_0 + loss_1

            loss /= len(features)
            # loss += contrastive_l * args.alpha
            encoder_optimizer.zero_grad()
            relation_model_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            relation_model_optimizer.step()
            train_loss.append(loss.cpu().detach().numpy())

            if step % 40 == 0:
                print(
                    "step:",
                    step,
                    "train loss:",
                    train_loss[-1],
                    "avg train loss:",
                    np.average(train_loss),
                )
            
        print("Start validating")
        encoder.eval()
        relation_model.eval()
        all_retrieval_feas = torch.vstack(all_retrieval_feas)
        all_retrieval_lbls = torch.vstack(all_retrieval_lbls).cuda()
        print(all_retrieval_feas.shape, all_retrieval_lbls.shape)
        loss = 0
        percent_rul=0
        # percent = 0
        batch_num = 0
        with torch.no_grad():
            for step, (seq, target) in enumerate(test_loader):
                target_rul = target[:, -1].cuda()
                x = encoder(seq.cuda())
                #scores = []
                #import pdb;pdb.set_trace()
                scores = F.cosine_similarity(x, all_retrieval_feas.cuda(),dim=1)
                # for retrieval_seq in all_retrieval_feas:
                #     alignment = dtw(x.detach().cpu().numpy(), retrieval_seq.detach().cpu().numpy(), keep_internals=True) # DTW算法
                #     scores.append(alignment.distance)
                # scores = np.array(scores)
                chosen_scores, indices = torch.topk(torch.Tensor(scores), args.top_k,dim=0) # 乘 -1 找最小的
                # chosen_scores = -chosen_scores
                max_ = torch.max(chosen_scores)
                min_ = torch.min(chosen_scores)
                chosen_scores = (chosen_scores - min_) / (max_ - min_)
                output = relation_model(chosen_scores)

                predict_rul = torch.sum(chosen_scores * all_retrieval_lbls[indices, -1])

                if step % 40 == 0:
                    print("step:",
                            step,
                            "predict_rul",
                            predict_rul * 3000, 
                            "target_rul",
                            target_rul * 3000)
                # print(output_clone.sum().detach().item() * self_curve_len, target_cycle.detach().item())
               
                loss_rul = nn.MSELoss()(predict_rul, target_rul)
                percent_rul+=torch.abs(torch.sqrt(loss_rul)/target_rul)
                
                # percent_rul = loss_rul/target_rul
                loss = loss+loss_rul
                # percent = percent+percent_rul
                batch_num += 1
            print("test_average_loss", 
                  loss / batch_num,
                   "test_loss_percent",
                   percent_rul / batch_num,
                  )
 

            

def main(args):
    train_fea, train_lbl, test_fea, test_lbl = our_data_loader_generate()
    print(train_fea.shape, train_lbl.shape, test_fea.shape, test_lbl.shape)
    train_set = ContrastiveDataset(torch.Tensor(train_fea), torch.Tensor(train_lbl), args.batch)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    test_set = TensorDataset(torch.Tensor(test_fea), torch.Tensor(test_lbl))
    test_loader = DataLoader(test_set, batch_size=args.valid_batch, shuffle=True)

    # ==== Train =====
    run(train_loader=train_loader, test_loader=test_loader, args=args, fea_num=train_fea.shape[2])

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-batch", type=int, default=32)
    argparser.add_argument("-valid_batch", type=int, default=1)
    argparser.add_argument("-epoch", type=int, default=100)#100
    argparser.add_argument("-lr", help="initial learning rate", type=float, default=1e-3)
    argparser.add_argument("-gamma", help="learning rate decay rate", type=float, default=0.9)
    argparser.add_argument("--lstm-hidden", type=int, help="lstm hidden layer number", default=128)  # 128
    argparser.add_argument("--fc-hidden", type=int, help="fully connect layer hidden dimension", default=98)  # 128
    argparser.add_argument("--fc-out", type=int, help="embedded sequence dimmension", default=64)  # 128
    argparser.add_argument("--dropout", type=float, default=0.3)  # 0.1
    argparser.add_argument("--lstm-layer", type=int, default=1)  # 0.1
    argparser.add_argument("-top_k", help="use top k curves to retrieve", type=int, default=5)
    argparser.add_argument("-tao", help="tao in contrastive loss calculation ", type=float, default=0.5)
    argparser.add_argument("-alpha", help="zoom factor of contrastive loss", type=float, default=0.1)
       

    args = argparser.parse_args()
    main(args)

**---------------------------------seq_sampler.py--------------------------------------**
from torch.utils.data import Sampler, TensorDataset

class SeqSampler(Sampler):
    def __init__(self, data_source: TensorDataset, type) -> None:
        super().__init__(data_source)
        self.data = data_source
        self.type = type

    def __iter__(self):
        """tensors:[features,label]
        features: seq,seq_len,feature_num
        label: seq, feas [rul,len,num]
        """
        indices_map = {}
        features = self.data.tensors[0]
        labels = self.data.tensors[1]
        for i in range(features.shape[0]):
            tail_dq = features[i][-1][-1]
            origin_dq = features[i][-1][0]
            # rul = labels[i][0]  # tail rul
            # tot_seq_len = labels[i][1]
            # pos = int((tot_seq_len - rul).item())
            # pos = "%.4f" % (tail_dq / origin_dq)
            pos = labels[i][-1]
            if pos in indices_map.keys():
                indices_map[pos].append(i)
            else:
                indices_map[pos] = [i]
        indices = []
        keys = list(indices_map.keys())
        if self.type == "train":
            nei_keys = [keys[i + 1] for i in range(len(keys) - 1)]
            nei_keys.append(keys[-2])
            assert len(keys) == len(nei_keys)
            for i in range(len(keys)):
                indices += indices_map[keys[i]]
                indices += indices_map[nei_keys[i]]
        else:
            for i in range(len(keys)):
                indices += indices_map[keys[i]]
        return iter(indices)

    def __len__(self):
        if self.type == "train":
            return self.data.tensors[0].shape[0] * 2
        else:
            return self.data.tensors[0].shape[0]

**---------------------------------------tool.py---------------------------------------------**
import os
import torch
from copy import deepcopy
import numpy as np
import pandas as pd
import torch.nn as nn
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, check_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, check_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, check_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, check_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), check_name)
        self.val_loss_min = val_loss

**-------------------------------------utils.py------------------------------------------**
import os
import time
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime
import pandas as pd
from tool import EarlyStopping
from sklearn.metrics import roc_auc_score,mean_squared_error
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.sampler import RandomSampler
    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')


# save dict    
def save_obj(obj,name):
    with open(name + '.pkl','wb') as f:
        pickle.dump(obj,f)
                  
#load dict        
def load_obj(name):
    with open(name +'.pkl','rb') as f:
        return pickle.load(f)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def interp(v, q, num):
    f = interpolate.interp1d(v,q,kind='linear')
    v_new = np.linspace(v[0],v[-1],num)
    q_new = f(v_new)
    vq_new = np.concatenate((v_new.reshape(-1,1),q_new.reshape(-1,1)),axis=1)
    return q_new

def get_xy(name, series_lens, i_low, i_upp, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor, pkl_dir,
             raw_features=True, fill_with_zero=True, seriesnum=1500):
    """
    Args:
        n_cyc (int): The previous cycles number for model input
        in_stride (int): The interval in the previous cycles number
        fea_num (int): The number of interpolation
        v_low (float): Voltage minimum for normalization
        v_upp (float): Voltage maximum for normalization
        q_low (float): Capacity minimum for normalization
        q_upp (float): Capacity maximum for normalization
        rul_factor (float): The RUL factor for normalization
        cap_factor (float): The capacity factor for normalization
        seriesnum: The number of series sliced from this degradation curve
    """
    def preprocess(data):
        datamean = np.mean(data)
        datastdvar = math.sqrt(np.var(data))
        return [datamean, datastdvar]
    # print("loading", name)
    if not os.path.exists(pkl_dir + name + '_fea.npy'):
        A = load_obj(pkl_dir + name)[name]
        A_rul = A['rul']
        A_dq = A['dq']
        A_df = A['data']
        all_fea = []
        all_idx = list(A_dq.keys())[9:]
        ruls = []
        for cyc in all_idx:
            feature = [A_dq[cyc] / cap_factor]
            time, v, i, q, dv, di, dq, dtime = [], [], [], [], [], [], [], []
            for timeidx in range(len(A_df[cyc]['Status'])):
                if 'discharge' in A_df[cyc]['Status'][timeidx]:
                    time.append(A_df[cyc]['Time (s)'][timeidx])
                    v.append((A_df[cyc]['Voltage (V)'][timeidx] - v_low) / (v_upp - v_low))
                    i.append((A_df[cyc]['Current (mA)'][timeidx] - i_low) / (i_upp - i_low))
                    q.append((A_df[cyc]['Capacity (mAh)'][timeidx] - q_low) / (q_upp - q_low))
                    if timeidx < len(A_df[all_idx[0]]['Voltage (V)']):
                        # import pdb;pdb.set_trace()
                        dv.append((A_df[cyc]['Voltage (V)'][timeidx] - A_df[all_idx[0]]['Voltage (V)'][timeidx]) / (v_upp - v_low))
                        di.append((A_df[cyc]['Current (mA)'][timeidx] - A_df[all_idx[0]]['Current (mA)'][timeidx]) / (i_upp - i_low))
                        dq.append((A_df[cyc]['Capacity (mAh)'][timeidx] - A_df[all_idx[0]]['Capacity (mAh)'][timeidx]) / (q_upp - q_low))
                        dtime.append(A_df[cyc]['Time (s)'][timeidx])
            feature += preprocess(v)
            feature += preprocess(i)
            feature += preprocess(q)
            feature += preprocess(dv)
            feature += preprocess(di)
            feature += preprocess(dq)
            all_fea.append(feature)
            ruls.append(A_rul[cyc])

        np.save(pkl_dir + name + '_fea.npy', all_fea)
        np.save(pkl_dir + name + '_rul.npy', ruls)
    else:
        all_fea = np.load(pkl_dir + name + '_fea.npy', allow_pickle=True)
        A_rul = np.load(pkl_dir + name + '_rul.npy', allow_pickle=True)
        # import pdb;pdb.set_trace()
    if raw_features:
        return np.array(all_fea), A_rul
    feature_num = len(all_fea[0])
    all_series, all_ruls = np.empty((0, np.max(series_lens), feature_num)), np.empty((0, 3))
    for ratio in range(4):
        tmpfea=copy.deepcopy(all_fea)
        tmprul=copy.deepcopy(A_rul)
        tmpfea=tmpfea[0::ratio+1]
        tmprul=tmprul[0::ratio+1]
        for series_len in series_lens:
            # series_num = len(all_fea) // series_len
            # series = np.lib.stride_tricks.as_strided(np.array(all_fea), (series_num, series_len, feature_num))
            series = np.lib.stride_tricks.sliding_window_view(tmpfea, (series_len, feature_num))
            series = series.squeeze()
            full_series = []
            if series_len < np.max(series_lens) and fill_with_zero:
                zeros = np.zeros((np.max(series_lens) - series_len, feature_num))
                for seriesidx in range(series.shape[0]):
                    # import pdb;pdb.set_trace()
                    full_series.append(np.concatenate((series[seriesidx], zeros)))
            elif series_len == np.max(series_lens):
                full_series = series
            # ruls = np.array(A_rul[series_len - 1:]) / rul_factor
            # series.tolist()
            full_series = np.array(full_series)

            full_seq_len = len(tmprul)

            if isinstance(A_rul, dict):
                tmp = []
                for k, v in A_rul.items():
                    if k >= series_len:
                        tmp.append([v / rul_factor, full_seq_len / rul_factor, v / full_seq_len])
                ruls = tmp
            else:
                ruls = tmprul[series_len/(ratio+1) - 1:].tolist()
                for i in range(len(ruls)):
                    ruls[i] = [ruls[i] / rul_factor, full_seq_len / rul_factor, ruls[i] / full_seq_len]
            # import pdb;pdb.set_trace()
            # print(all_series.shape, all_ruls.shape)
            all_series = np.append(all_series, full_series, axis=0)
            ruls = np.array(ruls).astype(float)
            all_ruls = np.append(all_ruls, ruls, axis=0)
    if seriesnum is not None:
        all_series = all_series[:seriesnum]
        all_ruls = all_ruls[:seriesnum]
    return all_series, all_ruls


class Trainer():
    
    def __init__(self, lr, n_epochs,device, patience, lamda, alpha, model_name):
        """
        Args:
            lr (float): Learning rate
            n_epochs (int): The number of training epoch
            device: 'cuda' or 'cpu'
            patience (int): How long to wait after last time validation loss improved.
            lamda (float): The weight of RUL loss
            alpha (List: [float]): The weights of Capacity loss
            model_name (str): The model save path
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device
        self.patience = patience
        self.model_name = model_name
        self.lamda = lamda
        self.alpha = alpha

    def train(self, train_loader, valid_loader, model, load_model):
        model = model.to(self.device)
        device = self.device
        optimizer = optim.Adam(model.parameters(), lr=self.lr,)
        model_name = self.model_name
        lamda = self.lamda
        alpha = self.alpha
        
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStopping(self.patience, verbose=True)
        loss_fn.to(self.device)

        # Training
        train_loss = []
        valid_loss = []
        total_loss = []
        
        for epoch in range(self.n_epochs):
            model.train()
            y_true, y_pred = [], []
            losses = []
            for step, (x,y) in enumerate(train_loader):  
                optimizer.zero_grad()
                
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)
                
                loss = lamda * loss_fn(y_.squeeze(), y[:,0])
                
                for i in range(y.shape[1] - 1):
                    loss += loss_fn(soh_[:,i], y[:,i+1]) * alpha[i]
                    
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().detach().numpy())

                y_pred.append(y_)
                y_true.append(y[:,0])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            train_loss.append(epoch_loss)
            
            losses = np.mean(losses)
            total_loss.append(losses)
            
            # validate
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for step, (x,y) in enumerate(valid_loader):
                    x = x.to(device)
                    y = y.to(device)
                    y_, soh_ = model(x)

                    y_pred.append(y_)
                    y_true.append(y[:,0])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            valid_loss.append(epoch_loss)
            
            if self.n_epochs > 100:
                if (epoch % 100 == 0 and epoch !=0):
                    print('Epoch number : ', epoch)
                    print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}',f'-- "total" loss {losses:.4}')
            else :
                print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}',f'-- "total" loss {losses:.4}')

            early_stopping(epoch_loss, model, f'{model_name}_best.pt')
            if early_stopping.early_stop:
                break
                
        if load_model:
            model.load_state_dict(torch.load(f'{model_name}_best.pt'))
        else:
            torch.save(model.state_dict(), f'{model_name}_end.pt')

        return model, train_loss, valid_loss, total_loss

    def test(self, test_loader, model):
        model = model.to(self.device)
        device = self.device

        y_true, y_pred, soh_true, soh_pred = [], [], [], []
        model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)

                y_pred.append(y_)
                y_true.append(y[:,0])
                soh_pred.append(soh_)
                soh_true.append(y[:,1:])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            soh_true = torch.cat(soh_true, axis=0)
            soh_pred = torch.cat(soh_pred, axis=0)
            mse_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        return y_true, y_pred, mse_loss, soh_true, soh_pred
    

class FineTrainer():
    
    def __init__(self, lr, n_epochs,device, patience, lamda, train_alpha, valid_alpha, model_name):
        """
        Args:
            lr (float): Learning rate
            n_epochs (int): The number of training epoch
            device: 'cuda' or 'cpu'
            patience (int): How long to wait after last time validation loss improved.
            lamda (float): The weight of RUL loss. In fine-tuning part, set 0.
            train_alpha (List: [float]): The weights of Capacity loss in model training
            valid_alpha (List: [float]): The weights of Capacity loss in model validation
            model_name (str): The model save path
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device
        self.patience = patience
        self.model_name = model_name
        self.lamda = lamda
        self.train_alpha = train_alpha
        self.valid_alpha = valid_alpha

    def train(self, train_loader, valid_loader, model, load_model):
        model = model.to(self.device)
        device = self.device
        optimizer = optim.Adam(model.parameters(), lr=self.lr,)
        model_name = self.model_name
        lamda = self.lamda
        train_alpha = self.train_alpha
        valid_alpha = self.valid_alpha
        
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStopping(self.patience, verbose=True)
        loss_fn.to(self.device)

        # Training
        train_loss = []
        valid_loss = []
        total_loss = []
        added_loss = []
        
        for epoch in range(self.n_epochs):
            model.train()
            y_true, y_pred = [], []
            losses = []
            for step, (x,y) in enumerate(train_loader):  
                optimizer.zero_grad()
                
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)
                soh_ = soh_.view(y_.shape[0], -1)
                
                loss = lamda * loss_fn(y_.squeeze(), y[:,0])
                
                for i in range(y.shape[1] - 1):
                    loss += loss_fn(soh_[:,i], y[:,i+1]) * train_alpha[i]
                    
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().detach().numpy())

                y_pred.append(y_)
                y_true.append(y[:,0])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            train_loss.append(epoch_loss)
            
            losses = np.mean(losses)
            total_loss.append(losses)

            # validate
            model.eval()
            y_true, y_pred, all_true, all_pred = [], [], [], []
            with torch.no_grad():
                for step, (x,y) in enumerate(valid_loader):
                    x = x.to(device)
                    y = y.to(device)
                    y_, soh_ = model(x)
                    soh_ = soh_.view(y_.shape[0], -1)

                    y_pred.append(y_)
                    y_true.append(y[:,0])
                    all_true.append(y[:,1:])
                    all_pred.append(soh_)

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            all_true = torch.cat(all_true, axis=0)
            all_pred = torch.cat(all_pred, axis=0)
            epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            valid_loss.append(epoch_loss)
            
            temp = 0
            for i in range(all_true.shape[1]):
                temp += mean_squared_error(all_true[0:1,i].cpu().detach().numpy(), 
                                           all_pred[0:1,i].cpu().detach().numpy()) * valid_alpha[i]
            added_loss.append(temp)
            
            if self.n_epochs > 10:
                if (epoch % 200 == 0 and epoch !=0):
                    print('Epoch number : ', epoch)
                    print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}',
                          f'-- "total" loss {losses:.4}',f'-- "added" loss {temp:.4}')
            else :
                print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}',
                      f'-- "total" loss {losses:.4}',f'-- "added" loss {temp:.4}')

            early_stopping(temp, model, f'{model_name}_fine_best.pt')
            if early_stopping.early_stop:
                break
                
        if load_model:
            model.load_state_dict(torch.load(f'{model_name}_fine_best.pt'))
        else:
            torch.save(model.state_dict(), f'{model_name}_fine_end.pt')

        return model, train_loss, valid_loss, total_loss, added_loss

    def test(self, test_loader, model):
        model = model.to(self.device)
        device = self.device

        y_true, y_pred, soh_true, soh_pred = [], [], [], []
        model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)
                soh_ = soh_.view(y_.shape[0], -1)

                y_pred.append(y_)
                y_true.append(y[:,0])
                soh_pred.append(soh_)
                soh_true.append(y[:,1:])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            soh_true = torch.cat(soh_true, axis=0)
            soh_pred = torch.cat(soh_pred, axis=0)
            mse_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        return y_true, y_pred, mse_loss, soh_true, soh_pred


def score_weight_loss(input, output):
    # 反转输入以使得较小的输入对应较大的输出
    inverted_input = 1.0 - input

    # 计算输入和输出的误差
    error = (inverted_input - output).abs()

    # 计算输入和输出间的差异
    diff_input = torch.abs(input[:-1] - input[1:])
    diff_output = torch.abs(output[:-1] - output[1:])

    # 计算差异的误差
    diff_error = (diff_input - diff_output).abs()

    # 将两个误差组合起来得到总的损失
    # loss = error.sum() + diff_error.sum()

    return error.sum(), diff_error.sum()
