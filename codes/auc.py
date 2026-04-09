import numpy as np
def AUC(y_scores,y_true):
    # 按照预测分数升序排序
    indexs=sorted(range(len(y_scores)),key=lambda i:y_scores[i])
    # 按照排序后的索引提取真实标签
    sorted_labels=[y_true[i] for i in indexs]
    # 统计正负样本数量
    pos_cnt=np.sum(y_true)
    neg_cnt=len(y_true)-pos_cnt
    # 计算所有正样本的秩次和
    rank_sum=0
    for i in range(len(sorted_labels)):
        if sorted_labels[i]==1:
            rank_sum += i
    auc = (rank_sum-pos_cnt*(pos_cnt-1)/2)/(pos_cnt*neg_cnt)
    return auc

def GAUC(y_scores, y_true, group_ids):
    """
    计算 Group AUC
    
    参数:
        y_scores: 预测分数
        y_true: 真实标签
        group_ids: 分组ID（如用户ID）
    """
    # 按组聚合数据
    groups = {}
    for i in range(len(y_scores)):
        gid = group_ids[i]
        if gid not in groups:
            groups[gid] = {'scores': [], 'labels': []}
        groups[gid]['scores'].append(y_scores[i])
        groups[gid]['labels'].append(y_true[i])
    
    # 计算每组的 AUC 和权重
    total_auc = 0
    total_weight = 0
    
    for gid, data in groups.items():
        scores = np.array(data['scores'])
        labels = np.array(data['labels'])
        
        # 跳过只有一个类别的组
        if len(np.unique(labels)) < 2:
            continue
            
        # 计算该组的 AUC
        group_auc = AUC(scores, labels)
        
        # 权重通常用正负样本数的乘积
        pos_cnt = np.sum(labels)
        neg_cnt = len(labels) - pos_cnt
        weight = pos_cnt * neg_cnt
        
        total_auc += group_auc * weight
        total_weight += weight
    
    return total_auc / total_weight if total_weight > 0 else 0


def AUC_with_ties(y_scores, y_true):
    # 创建(score, label, original_index)的元组
    data = [(y_scores[i], y_true[i], i) for i in range(len(y_scores))]
    # 按分数排序，分数相同时保持稳定
    data.sort(key=lambda x: x[0])
    
    # 处理相同分数的平均秩次
    n = len(data)
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        # 找到所有相同分数的样本
        while j < n and data[j][0] == data[i][0]:
            j += 1
        # 计算平均秩次
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    
    # 计算正样本的秩次和
    rank_sum = 0
    pos_cnt = 0
    for k in range(n):
        if data[k][1] == 1:  # 正样本
            rank_sum += ranks[k]
            pos_cnt += 1
    
    neg_cnt = n - pos_cnt
    auc = (rank_sum - pos_cnt * (pos_cnt - 1) / 2) / (pos_cnt * neg_cnt)
    return auc
