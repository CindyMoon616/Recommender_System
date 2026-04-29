import numpy as np
def auc(y_scores,y_true):
    indices=sorted(range(len(y_scores)),key=lambda i:y_scores[i])
    rank_sum=0
    for i,index in enumerate(indices):
        rank_sum=rank_sum+i if y_true[index]==1 else rank_sum
    num_pos=np.sum(y_true)
    num_neg=len(y_true)-num_pos
    auc=(rank_sum-(num_pos-1)*num_pos/2)/(num_pos*num_neg)
    return auc

def group_auc(y_scores,y_true,group_id):
    y_true_groups=dict()
    y_scores_groups=dict()
    n=len(y_scores)
    for i in range(n):
        if group_id[i] not in y_true_groups:
            y_true_groups[group_id[i]]=[]
            y_scores_groups[group_id[i]]=[]
        y_true_groups[group_id[i]].append(y_true[i])
        y_scores_groups[group_id[i]].append(y_scores[i])
    auc_groups=[]
    for group in y_true_groups.keys():
        auc_groups.append(auc(y_scores_groups[group],y_true_groups[group]))
    return np.nanmea(auc_groups)


y_true=np.random.randint(0,2,10)
y_scores=y_true+np.random.randn(10)*0.1
auc=auc(y_scores,y_true)
print(auc)