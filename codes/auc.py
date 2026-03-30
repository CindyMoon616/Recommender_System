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