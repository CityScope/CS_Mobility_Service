#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:59:30 2019

@author: doorleyr
"""
import pandas as pd
import numpy as np
import editdistance
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

def create_motif_var_len(row, time_interval_minutes, default_seq_len):
    """ Converts the full sequence of activities and times into 
    a sequence suitable for clustering using EditDistance
    Each activity is repeated by the number of hours it spans
    """
    default_seq_len=int((24*60)/time_interval_minutes)
    if len(row['activities'])==1:
        return 'H'*default_seq_len
    motif=''
    activity_sequence=row['activities'].split('_')
    start_times=[0]+[
            int(st) for st in row['start_times'].split('_')
            ] +[24*60*60]
    for i in range(len(activity_sequence)):
        act_len=start_times[i+1]-start_times[i]
        act_intervals=int(act_len/(time_interval_minutes*60))
        motif+=activity_sequence[i]*(act_intervals+1)
    return motif


def get_seq_matrix_df(people, time_interval_minutes, seq_len):
    """ Converts the full sequence of activities and times into 
    a matrix suitable for tramineR
    """
    seq_mat=[]
    for ind, row in people.iterrows():
        if len(row['activities'])==1:
            p_sequence=['H']*seq_len
        else:
            end_last=0
            activities=row['activities'].split('_')
            end_times=[int(st) for st in row['start_times'].split('_')
                ] +[24*60*60]        
            end_last=0
            p_sequence=[None]*seq_len
            for i in range(len(activities)):
                end_interval=int(end_times[i]/(60*time_interval_minutes))
                for h in range(end_last,end_interval):
                    p_sequence[h]=activities[i]
                end_last=end_interval
        seq_mat.append({'P'+str(i).zfill(3): p_sequence[i] for i in range(seq_len)})
    seq_df=pd.DataFrame(seq_mat)
    seq_df['id']=range(1, len(seq_df)+1)
    return seq_df

def create_motif_list_fixed_len(seq_matrix_df):
    motif_list=[]
    for i in range(len(seq_matrix_df)):
        motif_list.append(list(seq_matrix_df.iloc[i])[:-1])
    return motif_list

def get_motif_edit_dist_mat(motif_list):
    dist_mat=np.zeros([len(motif_list), len(motif_list)])
    for i in range(len(motif_list)):
        print(i)
        for j in range(i, len(motif_list)):
            if i==j:
                dist_mat[i,j]=0
            else:
                dist=editdistance.eval(motif_list[i], motif_list[j])
                dist_mat[i,j]=dist
                dist_mat[j, i]=dist
    return dist_mat

def cluster_motifs(dist_mat, n_clusters, linkage):
    clustering = AgglomerativeClustering(
            n_clusters, affinity='precomputed', linkage=linkage # “complete”, “average”, “single”
            ).fit(dist_mat)
#    clustering = SpectralClustering(
#            n_clusters, affinity='precomputed'
#            ).fit(dist_mat)
    return clustering.labels_

city='Detroit'
PERSON_SCHED_TABLE_PATH='./scripts/cities/'+city+'/clean/person_sched_weekday.csv'

# =============================================================================
# Motif Parameters
# =============================================================================
time_interval_minutes=60
seq_len=int((24*60)/time_interval_minutes)
SEQ_MATRIX_DF_PATH='./scripts/cities/'+city+'/clean/seq_matrix_{}.csv'.format(seq_len)

# =============================================================================
# Compute the motifs
# =============================================================================
sched_df=pd.read_csv(PERSON_SCHED_TABLE_PATH)
sched_df['motif_var_len']=sched_df.apply(lambda row: 
    create_motif_var_len(row, time_interval_minutes, seq_len), axis=1)
seq_matrix_df=get_seq_matrix_df(sched_df, time_interval_minutes, seq_len) 
seq_matrix_df.to_csv(SEQ_MATRIX_DF_PATH, index=False)

# =============================================================================
# Compute distance matrix for motif pairs
# =============================================================================
motif_list_fixed=create_motif_list_fixed_len(seq_matrix_df)
edit_dist_mat=get_motif_edit_dist_mat(motif_list_fixed)

# =============================================================================
# Clustering Parameters
# =============================================================================
linkage="complete"
for n_clusters in range(5,9):    
    CLUSTERED_SEQ_PATH='./scripts/cities/'+city+'/clean/clust_mat_{}_{}_{}.csv'.format(
            time_interval_minutes, n_clusters, linkage)  
    # =============================================================================
    # Cluster motifs and save results
    # =============================================================================
    seq_matrix_df['clust']=cluster_motifs(edit_dist_mat, n_clusters, linkage)
    seq_matrix_df.to_csv(CLUSTERED_SEQ_PATH, index=False)

