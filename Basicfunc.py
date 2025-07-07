import pandas as pd
import numpy as np
import random
import itertools
from scipy import stats
import math
import cv2
import sys
from tqdm import tqdm
import time
import multiprocessing as mp
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import matplotlib.pyplot as plt 
import seaborn as sns
import arviz as az
import copy 
from itertools import combinations
import math
from scipy.stats import hypergeom
from scipy.stats import chi2

from scipy.stats import beta, nbinom, cauchy
from scipy.stats import truncnorm
from arsenal.maths.rvs import TruncatedDistribution

from pybedtools import BedTool

import warnings
warnings.filterwarnings("ignore")

from input import *


DebugName = f"{settings['outdir']}/debug.txt"
with open(DebugName, 'w') as f:
    f.write('debug\n')

with open(f"{settings['outdir']}/permutationTestPost.txt", 'w') as f:
    f.write('\t'.join(['filename', 'keepvalue']) + '\n')

# pairnum = int(knum*(knum-1)/2)

# print('----------------------------------- data cut Functions --------------------------------------------')

# # print('----------------------------------- Data Loading Functions --------------------------------------------')
def expandReads(bedpe,settings):
    if settings['expandFlag'] == False:
        return bedpe
    else:
        startL1= min(bedpe['chromStart1'])
        startL2= min(bedpe['chromStart2'])
        endL1 = max(bedpe['chromEnd2'])
        endL2 = max(bedpe['chromEnd1'])

        bedpe_expand = bedpe.copy()
        bedpe_expand['chromStart1'] = bedpe_expand.pipe(lambda x: np.minimum(np.maximum(startL1,x['midPoint1'] - int(settings['minreadLen']/2)), x['chromStart1']))
        bedpe_expand['chromEnd1']   = bedpe_expand.pipe(lambda x: np.maximum(np.minimum(endL1,x['midPoint1'] + int(settings['minreadLen']/2)), x['chromEnd1']))
        bedpe_expand['chromStart2'] = bedpe_expand.pipe(lambda x: np.minimum(np.maximum(startL2,x['midPoint2'] - int(settings['minreadLen']/2)), x['chromStart2']))
        bedpe_expand['chromEnd2']   = bedpe_expand.pipe(lambda x: np.maximum(np.minimum(endL2,x['midPoint2'] + int(settings['minreadLen']/2)), x['chromEnd2']))
        return bedpe_expand
    
def PCRDuplicate(bedpe, settings):
    # 创建列的布尔掩码，避免重复计算
    diff_midPoint1 = bedpe["midPoint1"].diff().abs() <= settings['mergeDis']
    diff_midPoint2 = bedpe["midPoint2"].diff().abs() <= settings['mergeDis']
    
    # 筛选满足条件的重复行
    Dup_similar_links = bedpe[diff_midPoint1 & diff_midPoint2]
    print('Dup_similar_links: ', len(Dup_similar_links))

    with open(settings['outdir'] + '/worklog.txt', 'a') as f:
        f.write('PCRDuplicate: ' + str(len(Dup_similar_links)) + '\n')

    bedpe = bedpe[~(diff_midPoint1 & diff_midPoint2)].reset_index(drop=True)
    # 检查剩余的完全重复项
    duplicate_count = len(bedpe[bedpe.duplicated(subset=['midPoint1', 'midPoint2'], keep=False)])
    print('PCRDuplicate done: ', duplicate_count)
    
    return bedpe

def changeRep(bedpe,settings):

    lenmid1Dup = len(bedpe[bedpe.duplicated(subset=['midPoint1'], keep='first')])
    while len(bedpe[bedpe.duplicated(subset=['midPoint1'], keep='first')]) > 0:
        one_end_rep = bedpe[bedpe.duplicated(subset=['midPoint1'], keep='first')]
        for index, row in one_end_rep.iterrows():
            bedpe.loc[index, 'midPoint1'] = bedpe.loc[index, 'midPoint1'] + 1
            
    lenmid2Dup = len(bedpe[bedpe.duplicated(subset=['midPoint2'], keep='first')])
    while len(bedpe[bedpe.duplicated(subset=['midPoint2'], keep='first')]) > 0:
        one_end_rep = bedpe[bedpe.duplicated(subset=['midPoint2'], keep='first')]
        for index, row in one_end_rep.iterrows():
            bedpe.loc[index, 'midPoint2'] = bedpe.loc[index, 'midPoint2'] + 1

    with open(settings['outdir']+'/worklog.txt', 'a') as f:
        f.write('One End Duplicate: ' + str(lenmid1Dup) + ' ' + str(lenmid2Dup) + '\n')
                
    return bedpe

def Getdata(file_path,settings):

    bedpe_name = os.path.splitext(os.path.basename(file_path))[0]

    c = os.path.splitext(os.path.basename(file_path))[0].split('_')[1]

    data_bedpe = pd.read_csv(file_path, sep='\t', header=None)
    header = ['chrom1', 'chromStart1', 'chromEnd1', 'chrom2', 'chromStart2', 'chromEnd2', 'name','score','strand1','strand2','str1','str2']
    data_bedpe.columns = header[:len(data_bedpe.columns)]


    # data_bedpe_self = pd.read_csv(settings['outdir']+'/ChromSep.part'+'/'+settings['prefix']+'_'+c+'_self.bedpe', sep='\t', header=None)
    data_bedpe_self = pd.read_csv('./input/test_multiself.bedpe', sep='\t', header=None)
    data_bedpe_self.columns = header[:len(data_bedpe_self.columns)]

    data_bedpe['midPoint1'] = (data_bedpe['chromStart1']+(data_bedpe['chromEnd1']-data_bedpe['chromStart1'])/2).astype("int")
    data_bedpe['midPoint2'] = (data_bedpe['chromStart2']+(data_bedpe['chromEnd2']-data_bedpe['chromStart2'])/2).astype("int")
    data_bedpe = data_bedpe.sort_values(by=['midPoint1','midPoint2'])
    data_bedpe = data_bedpe.reset_index(drop=True)

    data_bedpe['distance'] = data_bedpe['midPoint2']-data_bedpe['midPoint1']
    # change duplicate 
    data_bedpe = PCRDuplicate(data_bedpe,settings)
    data_bedpe_original = data_bedpe.copy()

    # data_bedpe = changeRep(data_bedpe,settings)
    # cutoff selflinks
    # data_bedpe = data_bedpe[data_bedpe['distance']>settings['selfthreshold']].reset_index(drop=True)
    # if settings['keeptempFlag'] == True:
    #     data_bedpe.to_csv(settings['outdir'] + '/tempfiles/' + bedpe_name +'_cutoff%s.bedpe'%settings['selfthreshold'],index=False,header=False,sep='\t')
    # expand reads
    data_bedpe = expandReads(data_bedpe,settings)

    header = ['chrom', 'chromStart', 'chromEnd','name','strand','midPoint','distance']
    data_bed1 = data_bedpe[['chrom1', 'chromStart1', 'chromEnd1','name','strand1','midPoint1','distance']]
    data_bed2 = data_bedpe[['chrom2', 'chromStart2', 'chromEnd2','name','strand2','midPoint2','distance']]
    data_bed1_original = data_bedpe_original[['chrom1', 'chromStart1', 'chromEnd1','name','strand1','midPoint1','distance']]
    data_bed2_original = data_bedpe_original[['chrom2', 'chromStart2', 'chromEnd2','name','strand2','midPoint2','distance']]
    
    data_bed1.columns = header[:len(data_bed1.columns)]; data_bed1['category'] = 1; data_bed1 = data_bed1.reset_index(drop=True)
    data_bed2.columns = header[:len(data_bed2.columns)]; data_bed2['category'] = 2; data_bed2 = data_bed2.reset_index(drop=True)
    data_bed1_original.columns = header[:len(data_bed1_original.columns)]; data_bed1_original['category'] = 1; data_bed1_original = data_bed1_original.reset_index(drop=True)
    data_bed2_original.columns = header[:len(data_bed2_original.columns)]; data_bed2_original['category'] = 2; data_bed2_original = data_bed2_original.reset_index(drop=True)
    
    data_bed = pd.concat([data_bed1,data_bed2],axis=0)
    data_bed = data_bed.sort_values(by="midPoint")
    data_bed = data_bed.reset_index(drop=True)

    data_bed_original = pd.concat([data_bed1_original,data_bed2_original],axis=0)
    data_bed_original = data_bed_original.sort_values(by="midPoint")
    data_bed_original = data_bed_original.reset_index(drop=True)

    data_bed.to_csv(settings['outdir']+'/tempfiles/'+bedpe_name+'_reads_expand_a.bed',index=False,header=False,sep='\t')
    data_bed1.to_csv(settings['outdir']+'/tempfiles/'+bedpe_name+'_reads_expand_l.bed',index=False,header=False,sep='\t')
    data_bed2.to_csv(settings['outdir']+'/tempfiles/'+bedpe_name+'_reads_expand_r.bed',index=False,header=False,sep='\t')

    data_bed_directory  = os.path.join(settings['outdir'],'tempfiles',bedpe_name+'_reads_expand_a.bed')
    data_bed1_directory = os.path.join(settings['outdir'],'tempfiles',bedpe_name+'_reads_expand_l.bed')
    data_bed2_directory = os.path.join(settings['outdir'],'tempfiles',bedpe_name+'_reads_expand_r.bed')

    os.system(f"bedtools coverage -a {data_bed_directory}  -b {data_bed_directory}  > {settings['outdir']}/tempfiles/{bedpe_name}_reads_expand_depth_a.bed")
    os.system(f"bedtools coverage -a {data_bed1_directory} -b {data_bed1_directory} > {settings['outdir']}/tempfiles/{bedpe_name}_reads_expand_depth_l.bed")
    os.system(f"bedtools coverage -a {data_bed2_directory} -b {data_bed2_directory} > {settings['outdir']}/tempfiles/{bedpe_name}_reads_expand_depth_r.bed")

    depth_dir_a = os.path.join(settings['outdir'],'tempfiles',bedpe_name+'_reads_expand_depth_a.bed')
    depth_dir_l = os.path.join(settings['outdir'],'tempfiles',bedpe_name+'_reads_expand_depth_l.bed')
    depth_dir_r = os.path.join(settings['outdir'],'tempfiles',bedpe_name+'_reads_expand_depth_r.bed')

    header = ['chrom', 'chromStart', 'chromEnd', 'name','strand','midPoint','distance','category','number','length','cover','rate']
    # load depth all
    data_with_depth_a = pd.read_csv(depth_dir_a, sep='\t', header=None)
    data_with_depth_a.columns = header[:len(data_with_depth_a.columns)]
    data_with_depth_a = data_with_depth_a.sort_values(by="midPoint"); data_with_depth_a = data_with_depth_a.reset_index(drop=True)
    # load depth first
    data_with_depth_l = pd.read_csv(depth_dir_l, sep='\t', header=None)
    data_with_depth_l.columns = header[:len(data_with_depth_l.columns)]
    data_with_depth_l = data_with_depth_l.sort_values(by="midPoint"); data_with_depth_l = data_with_depth_l.reset_index(drop=True)
    # load depth second 
    data_with_depth_r = pd.read_csv(depth_dir_r, sep='\t', header=None)
    data_with_depth_r.columns = header[:len(data_with_depth_r.columns)]
    data_with_depth_r = data_with_depth_r.sort_values(by="midPoint"); data_with_depth_r = data_with_depth_r.reset_index(drop=True)

    # get relative cover
    depth1 = np.array(data_with_depth_l['number']); depth2 = np.array(data_with_depth_r['number']); depth = np.array(data_with_depth_a['number'])
    data_bed['new_id'] = data_bed.groupby(data_bed['midPoint'].tolist(), sort=False).ngroup(); data_bed.index = data_bed['new_id'].tolist()

    data_bed1= data_bed[data_bed['category']==1].reset_index(drop=True); data_bed1['rela_cover'] = (depth1-0.9).astype(int)
    data_bed2= data_bed[data_bed['category']==2].reset_index(drop=True); data_bed2['rela_cover'] = (depth2-0.9).astype(int)
    data_bed['rela_cover'] = (depth-0.9).astype(int)

    data_bedpe['depth1'] = depth1-0.9; data_bedpe = data_bedpe.sort_values(by="midPoint2")
    data_bedpe['depth2'] = depth2-0.9; data_bedpe = data_bedpe.sort_values(by=['midPoint1','midPoint2'])

    with open(settings['outdir']+'/worklog.txt', 'a') as f:
        f.write('Number of read with depth>1 after expand read: ' + str(settings['expandFlag']) + ' ' + str(np.sum(depth1 > 1)) + ' ' + str(np.sum(depth2 > 1)))

    data_bed_unique = data_bed.drop_duplicates(subset=['midPoint'], keep='first')
    data_bed_unique1 = data_bed1.drop_duplicates(subset=['midPoint'], keep='first').reset_index(drop=True)
    data_bed_unique2 = data_bed2.drop_duplicates(subset=['midPoint'], keep='first').reset_index(drop=True)

    data_bed_unique.to_csv(settings['outdir']+'/tempfiles/'+bedpe_name+'_reads_expand_a.bed',index=False,header=False,sep='\t')
    data_bed_unique1.to_csv(settings['outdir']+'/tempfiles/'+bedpe_name+'_reads_expand_l.bed',index=False,header=False,sep='\t')
    data_bed_unique2.to_csv(settings['outdir']+'/tempfiles/'+bedpe_name+'_reads_expand_r.bed',index=False,header=False,sep='\t')

    if settings['keeptempFlag'] == False:
        os.remove(data_bed_directory); os.remove(data_bed1_directory); os.remove(data_bed2_directory)
        # os.remove(depth_dir_a); os.remove(depth_dir_l); os.remove(depth_dir_r)
    print('DATA GENERATION DONE!')

    return data_bedpe,data_bed1,data_bed2,data_bed,data_bed_unique1,data_bed_unique2,data_bed_unique,data_bedpe_self,data_bed_original

# print('--------------------------------------- Basic Functions -----------------------------------------------')

def FindML(locs):
    m1 = int((locs[0]+locs[1])/2); l1 = locs[1]-locs[0]+1
    m2 = int((locs[2]+locs[3])/2); l2 = locs[3]-locs[2]+1

    return m1,l1,m2,l2

def FindML_index(i,j,block_group):
    l1 = block_group[i][1]-block_group[i][0] +1
    l2 = block_group[j][1]-block_group[j][0] +1
    m1 = int((block_group[i][0]+block_group[i][1])/2)
    m2 = int((block_group[j][0]+block_group[j][1])/2)
    return m1,l1,m2,l2

def FindReadSE(readm1,w1,readm2,w2):
    reads1 = readm1 - math.floor((w1-1)/2); reade1 = readm1 + math.ceil((w1-1)/2)
    reads2 = readm2 - math.floor((w2-1)/2); reade2 = readm2 + math.ceil((w2-1)/2)

    return reads1,reade1,reads2,reade2

def FindReadSE2(datas,m1,l1,m2,l2):

    locs = FindLocation(m1,l1,m2,l2)

    reads1list = datas['data_bed_unique1'][(datas['data_bed_unique1'].midPoint>=locs[0]) & (datas['data_bed_unique1'].midPoint<=locs[1])].index.tolist()
    reade2list = datas['data_bed_unique2'][(datas['data_bed_unique2'].midPoint>=locs[2]) & (datas['data_bed_unique2'].midPoint<=locs[3])].index.tolist()

    if len(reads1list) == 0 or len(reade2list) == 0:
        # print('no reads for FindReadSE2')
        return 0,0,0,0,0,0
    else: 
        w1 = len(reads1list); w2 = len(reade2list)
        reads1 = min(reads1list); reade1 = max(reads1list)
        reads2 = min(reade2list); reade2 = max(reade2list)

        return reads1,reade1,reads2,reade2,w1,w2
    
def GetW(datas,m1,l1,m2,l2):
    locs = FindLocation(m1,l1,m2,l2)
    w1 = len(datas['data_bed_unique1'][(datas['data_bed_unique1']['midPoint']>=locs[0])&(datas['data_bed_unique1']['midPoint']<=locs[1])])
    w2 = len(datas['data_bed_unique2'][(datas['data_bed_unique2']['midPoint']>=locs[2])&(datas['data_bed_unique2']['midPoint']<=locs[3])])
    return w1,w2

def GetReadsNum_simple(datas,settings,block_group,expandFlag):
    if expandFlag:
        for i in range(len(block_group)-1):
            for j in range(i+1,len(block_group)):
                locs = [block_group[i][0],block_group[i][1],block_group[j][0],block_group[j][1]]
                m1,l1,m2,l2 = FindML(locs)
                l1 = max(l1,settings['CACBminLen']); l2 = max(l2,settings['CACBminLen'])
                locs = FindLocation(m1,l1,m2,l2)
                block_group[i][0],block_group[i][1],block_group[j][0],block_group[j][1] = locs[0],locs[1],locs[2],locs[3]         
    readsNums = np.zeros(len(block_group))
    for i in range(len(block_group)):
        start = block_group[i][0]; end = block_group[i][1]
        readsNums[i] = len(datas['data_bed_original'][(datas['data_bed_original']['midPoint']>=start)&(datas['data_bed_original']['midPoint']<=end)])
    return readsNums

def GetReadsNum(datas,settings,m1,l1,m2,l2,shrinkFlag,expandFlag):
    if expandFlag:
        l1 = max(l1,settings['CACBminLen']); l2 = max(l2,settings['CACBminLen'])
        locs = FindLocation(m1,l1,m2,l2)
    if shrinkFlag:
        mid1 = datas['data_bedpe']['midPoint1']; mid2 = datas['data_bedpe']['midPoint2']
        shrinkchoose  = datas['data_bedpe'].loc[(mid1>=locs[0])&(mid1<=locs[1])&(mid2>=locs[2])&(mid2<=locs[3])]
        # if len(shrinkchoose) >= settings['minLinks'] :
        if len(shrinkchoose) >= 2:
            locs = np.zeros(4)
            locs[0] = datas['data_bed_unique1'].loc[datas['data_bed_unique1']['midPoint'] == shrinkchoose['midPoint1'].min(), 'midPoint']
            locs[1] = datas['data_bed_unique1'].loc[datas['data_bed_unique1']['midPoint'] == shrinkchoose['midPoint1'].max(), 'midPoint']
            locs[2] = datas['data_bed_unique2'].loc[datas['data_bed_unique2']['midPoint'] == shrinkchoose['midPoint2'].min(), 'midPoint']
            locs[3] = datas['data_bed_unique2'].loc[datas['data_bed_unique2']['midPoint'] == shrinkchoose['midPoint2'].max(), 'midPoint']
            m1,l1,m2,l2 = FindML(locs)
            if expandFlag:
                l1 = max(l1,settings['CACBminLen']); l2 = max(l2,settings['CACBminLen'])
                locs = FindLocation(m1,l1,m2,l2)  

    w1 = len(datas['data_bed_original'][(datas['data_bed_original']['midPoint']>=locs[0])&(datas['data_bed_original']['midPoint']<=locs[1])])
    w2 = len(datas['data_bed_original'][(datas['data_bed_original']['midPoint']>=locs[2])&(datas['data_bed_original']['midPoint']<=locs[3])])
    return w1,w2

def FindLocation(m1,l1,m2,l2):
    locs = np.zeros(4)
    locs[0] = m1 - math.floor((l1-1)/2); locs[1] = m1 + math.ceil((l1-1)/2)
    locs[2] = m2 - math.floor((l2-1)/2); locs[3] = m2 + math.ceil((l2-1)/2)

    return locs

def indicator(datas,locs):

    names1 = np.sort(datas['data_bed_unique1']['midPoint'])
    names2 = np.sort(datas['data_bed_unique2']['midPoint'])

    T1 = pd.DataFrame()
    T1['block1'] = np.zeros(datas['N1']); T1['midPoint'] = names1; T1['location'] = list(datas['data_bed_unique1']['new_id'])
    T2 = pd.DataFrame()
    T2['block2'] = np.zeros(datas['N2']); T2['midPoint'] = names2; T2['location'] = list(datas['data_bed_unique2']['new_id'])
    T1.loc[(T1['midPoint'] >= locs[0])&(T1['midPoint']<= locs[1]),'block1'] = 1
    T2.loc[(T2['midPoint'] >= locs[2])&(T2['midPoint']<= locs[3]),'block2'] = 1

    return T1,T2

def countnn(datas,m1,l1,m2,l2):
    nn = [int((l1+10)*(l2+10)),int(datas['L']*(datas['L']-1)/2-(l1+10)*(l2+10))]
    return nn

def countX(datas,m1,l1,m2,l2):
    X = [0,0]
    locs = FindLocation(m1,l1,m2,l2)
    X[0] = len(datas['data_bedpe'][(datas['data_bedpe']['midPoint1']>=locs[0])&(datas['data_bedpe']['midPoint1']<=locs[1])&(datas['data_bedpe']['midPoint2']>=locs[2])&(datas['data_bedpe']['midPoint2']<=locs[3])])
    X[1] = len(datas['data_bedpe'])-X[0]
    return X

def choose_links(datas,sample1,sample2):
    links = datas['data_bedpe'][(datas['data_bedpe']['midPoint1']>=sample1[0])&(datas['data_bedpe']['midPoint1']<=sample1[1])&(datas['data_bedpe']['midPoint2']>=sample2[0])&(datas['data_bedpe']['midPoint2']<=sample2[1])]
    return links

def countnn_list(datas,knum,block_group):
    pairnum = int(knum*(knum-1)/2)
    nn = np.zeros(pairnum+1)
    pth = 0
    for i in range(knum-1):
        for j in range(i+1,knum):
            l1 = block_group[i][1]-block_group[i][0]
            l2 = block_group[j][1]-block_group[j][0]
            nn[pth] = int((l1+10)*(l2+10))
            pth = pth +1
    nn[pairnum] = int(datas['L']*(datas['L']-1)/2-np.sum(nn))
    return nn

def countX_list(datas,knum,block_group):
    pairnum = int(knum*(knum-1)/2)
    X = np.zeros(pairnum+1)
    pth = 0
    for i in range(knum-1):
        for j in range(i+1,knum):
            locs = [block_group[i][0],block_group[i][1],block_group[j][0],block_group[j][1]]
            X[pth] = len(datas['data_bedpe'][(datas['data_bedpe']['midPoint1']>=locs[0])&(datas['data_bedpe']['midPoint1']<=locs[1])&(datas['data_bedpe']['midPoint2']>=locs[2])&(datas['data_bedpe']['midPoint2']<=locs[3])])
            pth = pth +1
    X[pairnum] = len(datas['data_bedpe'])-np.sum(X)
    return X

def Generate_P_all(knum,X,nn,settings): # contain P1,P234
    pairnum = int(knum*(knum-1)/2)

    np.random.seed(random.choice(list(range(500))))

    alpha_list = np.zeros(pairnum+1)
    beta_list  = np.zeros(pairnum+1)
    update_p_list = np.zeros(pairnum+1)

    alpha_list = [x + settings['a1'] for x in X[:-1]] + [X[-1]+settings['a234']]
    nn_X  = [x - y for x, y in zip(nn, X)]
    beta_list  = [x + settings['b1'] for x in nn_X[:-1]] + [nn_X[-1]+settings['b234']]

    cauchymean_list = [x/(x+y) for x, y in zip(alpha_list[:-1], beta_list[:-1])]
    cauchyscale_list = [x/10 for x in cauchymean_list]
    
    for pth in range(pairnum):
        D = cauchy(cauchymean_list[pth], cauchyscale_list[pth]);  T = TruncatedDistribution(D, settings['minp1'], 1)
        samples = T.rvs(1); update_p_list[pth] = samples[0]+ 10**(-20)

    D = beta(alpha_list[pairnum],beta_list[pairnum]); T = TruncatedDistribution(D,0,min(update_p_list[:-1]))
    samples = T.rvs(1); update_p_list[pairnum] = samples[0]+ 10**(-20)

    return update_p_list

def comb(n,m):
    return int(math.factorial(n)/(math.factorial(n-m)*math.factorial(m)))

def countCC_list(datas,knum,block_group):
    pairnum = int(knum*(knum-1)/2)
    CC = np.zeros((pairnum+1,2)); C = np.zeros(len(block_group)+1)
    pth = 0
    for i in range(knum-1):
        for j in range(i+1,knum):
            locs = [block_group[i][0],block_group[i][1],block_group[j][0],block_group[j][1]]
            C_i = len(datas['data_bed_original'][(datas['data_bed_original']['midPoint']>=locs[0])&(datas['data_bed_original']['midPoint']<=locs[1])])
            C_j = len(datas['data_bed_original'][(datas['data_bed_original']['midPoint']>=locs[2])&(datas['data_bed_original']['midPoint']<=locs[3])])  
            CC[pth] = [C_i,C_j]     
            pth = pth +1
            C[i] = C_i; C[j] = C_j
    C[-1] = len(datas['data_bedpe'])-np.sum(C)
    CC[-1] = [C[-1],C[-1]]
    return CC,C

def calculate_W(N,CA,CB,CAB,alpha=1000):
    pvalue = getPvalue_hypergeom(N,CA,CB,CAB)
    pvalue = float(pvalue)
    w = 1 / (1 + np.exp(-(alpha) * ((1-pvalue)-0.995)))
    if w == 0:
        w = 1e-100

    return w

def logposterior(datas,settings,knum,block_group,weightFlag=True):

    X = countX_list(datas,knum,block_group)
    nn = countnn_list(datas,knum,block_group)

    p_list = Generate_P_all(knum,X,nn,settings)
    if weightFlag == False:
        ll_p = [int(x)*(math.log(y)-math.log(1-y))+int(z)*math.log(1-y) for x,y,z in zip(X,p_list,nn)]
        ll = np.sum(ll_p)
    else:
        CC,C = countCC_list(datas,knum,block_group)
        W  = [calculate_W(N=np.sum(X),CA=CA,CB=CB,CAB=CAB,alpha=1000) for CA,CB,CAB in zip([x[0] for x in CC],[x[1] for x in CC],X)]
        W[-1] = 1
        # W[-1] = 这里的W[-1]的计算有点问题，不过本来就是background，不影响 , bg的pvalue近似1
        # print('W:',W)
        # ll_p_noW = [int(x)*(math.log(y)-math.log(1-y))+int(z)*math.log(1-y) for x,y,z in zip(X,p_list,nn)]
        # ll_p = [int(x)*(math.log(w*y)-math.log(1-y))+int(z)*math.log(1-y) for x,w,y,z in zip(X,W,p_list,nn)]
        try:
            ll_p = [int(x)*(math.log(w*y)-math.log(1-y))+int(z)*math.log(1-y) for x,w,y,z in zip(X,W,p_list,nn)]
        except ValueError as e:
            if 'math domain error' in str(e):
                print("X:", X)
                print("W:", W)
                print("p_list:", p_list)
                print("nn:", nn)
                raise
        # ll_noW = np.sum(ll_p_noW)
        ll = np.sum(ll_p)
        # print('delta: ', ll - ll_noW )

    M1prior = 0; M2prior = 0; IABprior = 0; blockLenPrior = 0; P4prior = 0
    pth = 0
    for i in range(knum-1):
        for j in range(i+1,knum):

            m1,l1,m2,l2 = FindML_index(i,j,block_group)

            dep1 = datas['data_bed_unique1'].loc[(datas['data_bed_unique1']['midPoint']-m1).abs().argsort()[0],'rela_cover']
            dep2 = datas['data_bed_unique2'].loc[(datas['data_bed_unique2']['midPoint']-m2).abs().argsort()[0],'rela_cover']
            M1prior = M1prior + math.log((dep1+1)/((datas['data_bed_unique1']['rela_cover']+1).sum()))
            M2prior = M2prior + math.log((dep2+1)/((datas['data_bed_unique2']['rela_cover']+1).sum()))

            if X[pth]>=2:
                # IABprior = IABprior + math.log(1-10**(-200))
                IABprior = IABprior
            else:
                # IABprior = IABprior + math.log(10**(-200))
                IABprior = IABprior -10000


            blockLenPenalty = np.log(np.abs(l2-l1)+1)*np.maximum(0,np.log(max(l1,l2)/(1500+min(l1,l2))))
            blockLenPrior   = blockLenPrior + -0.5*((blockLenPenalty**2)/4 + np.log(2*np.pi*2))

            locs = FindLocation(m1,l1,m2,l2)
            self_l = datas['data_bedpe_self'][(datas['data_bedpe_self']['chromStart1']>=locs[0]-1000)&(datas['data_bedpe_self']['chromEnd1']<=locs[1]+1000)&(datas['data_bedpe_self']['chromStart2']>=locs[0]-1000)&(datas['data_bedpe_self']['chromEnd2']<=locs[1]+1000)]
            self_r = datas['data_bedpe_self'][(datas['data_bedpe_self']['chromStart1']>=locs[2]-1000)&(datas['data_bedpe_self']['chromEnd1']<=locs[3]+1000)&(datas['data_bedpe_self']['chromStart2']>=locs[2]-1000)&(datas['data_bedpe_self']['chromEnd2']<=locs[3]+1000)]
            p4_l = len(self_l)/((l1+2000)*(l1+2000-1)/2)+1e-5 # 之前是20, 结果有点差异
            p4_r = len(self_r)/((l2+2000)*(l2+2000-1)/2)+1e-5
            P4prior = P4prior + min(math.log(p4_l),math.log(p4_r))
            pth = pth + 1

    # W12prior = math.log(comb(w1+settings['r']-1,w1)) + w1*math.log(1-settings['pp']) + settings['r']*math.log(settings['pp']) + math.log(comb(w2+settings['r']-1,w2)) + w2*math.log(1-settings['pp']) + settings['r']*math.log(settings['pp'])
    # lp = ll + W12prior + M1prior + M2prior
    lp = ll + blockLenPrior + M1prior + M2prior + IABprior + P4prior 
    return lp, p_list

def logposterior_noP(datas,settings,knum,block_group,p_list,weightFlag=True):

    X = countX_list(datas,knum,block_group)
    nn = countnn_list(datas,knum,block_group)

    if weightFlag == False:
        ll_p = [int(x)*(math.log(y)-math.log(1-y))+int(z)*math.log(1-y) for x,y,z in zip(X,p_list,nn)]
        ll = np.sum(ll_p)
    else:
        CC,C = countCC_list(datas,knum,block_group)
        W  = [calculate_W(N=np.sum(X),CA=CA,CB=CB,CAB=CAB,alpha=1000) for CA,CB,CAB in zip([x[0] for x in CC],[x[1] for x in CC],X)]
        W[-1] = 1
        # print('W:',W)
        ll_p_noW = [int(x)*(math.log(y)-math.log(1-y))+int(z)*math.log(1-y) for x,y,z in zip(X,p_list,nn)]
        ll_p = [int(x)*(math.log(w*y)-math.log(1-y))+int(z)*math.log(1-y) for x,w,y,z in zip(X,W,p_list,nn)]
        ll_noW = np.sum(ll_p_noW)
        ll = np.sum(ll_p)
        # print('delta: ', ll - ll_noW )

    M1prior = 0; M2prior = 0; IABprior = 0; blockLenPrior = 0; P4prior = 0
    pth = 0
    for i in range(knum-1):
        for j in range(i+1,knum):

            m1,l1,m2,l2 = FindML_index(i,j,block_group)

            dep1 = datas['data_bed_unique1'].loc[(datas['data_bed_unique1']['midPoint']-m1).abs().argsort()[0],'rela_cover']
            dep2 = datas['data_bed_unique2'].loc[(datas['data_bed_unique2']['midPoint']-m2).abs().argsort()[0],'rela_cover']
            M1prior = M1prior + math.log((dep1+1)/((datas['data_bed_unique1']['rela_cover']+1).sum()))
            M2prior = M2prior + math.log((dep2+1)/((datas['data_bed_unique2']['rela_cover']+1).sum()))

            if X[pth]>=2:
                # IABprior = IABprior + math.log(1-10**(-200))
                IABprior = IABprior
            else:
                # IABprior = IABprior + math.log(10**(-200))
                IABprior = IABprior -10000

            blockLenPenalty = np.log(np.abs(l2-l1)+1)*np.maximum(0,np.log(max(l1,l2)/(1500+min(l1,l2))))
            blockLenPrior   = blockLenPrior + -0.5*((blockLenPenalty**2)/4 + np.log(2*np.pi*2))

            locs = FindLocation(m1,l1,m2,l2)
            self_l = datas['data_bedpe_self'][(datas['data_bedpe_self']['chromStart1']>=locs[0]-1000)&(datas['data_bedpe_self']['chromEnd1']<=locs[1]+1000)&(datas['data_bedpe_self']['chromStart2']>=locs[0]-1000)&(datas['data_bedpe_self']['chromEnd2']<=locs[1]+1000)]
            self_r = datas['data_bedpe_self'][(datas['data_bedpe_self']['chromStart1']>=locs[2]-1000)&(datas['data_bedpe_self']['chromEnd1']<=locs[3]+1000)&(datas['data_bedpe_self']['chromStart2']>=locs[2]-1000)&(datas['data_bedpe_self']['chromEnd2']<=locs[3]+1000)]
            p4_l = len(self_l)/((l1+2000)*(l1+2000-1)/2)+1e-5 # 之前是20, 结果有点差异
            p4_r = len(self_r)/((l2+2000)*(l2+2000-1)/2)+1e-5
            P4prior = P4prior + min(math.log(p4_l),math.log(p4_r))
            pth = pth + 1

    # W12prior = math.log(comb(w1+settings['r']-1,w1)) + w1*math.log(1-settings['pp']) + settings['r']*math.log(settings['pp']) + math.log(comb(w2+settings['r']-1,w2)) + w2*math.log(1-settings['pp']) + settings['r']*math.log(settings['pp'])
    # lp = ll + W12prior + M1prior + M2prior
    lp = ll + blockLenPrior + M1prior + M2prior + IABprior + P4prior 
    return lp

def all_equal(lst):
    return all(x == lst[0] for x in lst)

def all_similar(lst,err):
    for i in range(len(lst)-1):
        for j in range(i+1, len(lst)):
            if abs(lst[i] - lst[j]) > err:
                return False
    return True

def is_overlapped(sample1, sample2):
    if sample1[0] > sample2[1] or sample1[1] < sample2[0]:
        return False
    return True

def overlap(x1, x2, x3, x4):
    return x2 >= x3 and x4 >= x1

def overlap_list(x1_list, x2_list, x3_list, x4_list):
    overlapFlag = False
    for kth in range(len(x1_list)):
        if overlap(x1_list[kth], x2_list[kth], x3_list[kth], x4_list[kth]) == True:
            overlapFlag = True
            return overlapFlag
    return overlapFlag

def check_interval_overlap(interval1, interval2):
    for i in range(len(interval1)):
        for j in range(i+1, len(interval1)):
            if (interval1[i][0] <= interval1[j][1] and interval1[j][0] <= interval1[i][1]) and (interval2[i][0] <= interval2[j][1] and interval2[j][0] <= interval2[i][1]):
                print('overlap:', i, j, interval1[i], interval1[j], interval2[i], interval2[j])
                return True
    return False

def check_overlap(m_list, l_list):


    interval = []

    for i in range(len(m_list)):
        for j in range(i+1, len(m_list)):
            interval.append([m_list[i]-math.floor((l_list[i]-1)/2),m_list[i]+math.ceil((l_list[i]-1)/2),m_list[j]-math.floor((l_list[j]-1)/2),m_list[j]+math.ceil((l_list[j]-1)/2)])

    sorted_interval = sorted(interval, key=lambda x: x[0])
    
    for i in range(len(sorted_interval) - 1):
        for j in range(i+1, len(sorted_interval)):
            if (sorted_interval[i][1] > sorted_interval[j][0])&(sorted_interval[i][3] >= sorted_interval[j][2])&(sorted_interval[j][3] >= sorted_interval[i][2]):
                # print(m_list[i],l_list[i],m2_list[i],l2_list[i],m_list[j],l_list[j],m2_list[j],l2_list[j])
                return True 
    
    return False 

def check_result_overlap(groups_result,extendFlag):
    # allow knum -1 overlap

    # extendFlag = True 的时候，extend每一个block的范围，然后再检查overlap
    expand_groups_result = copy.deepcopy(groups_result)
    if extendFlag == True:
        for i in range(len(expand_groups_result)):
            for index,interval in enumerate(expand_groups_result[i]):
                interval = list(interval)
                interval[0] = interval[0] - 10000
                interval[1] = interval[1] + 10000
                expand_groups_result[i][index] = interval

    for i in range(len(expand_groups_result)):
        for j in range(i+1,len(expand_groups_result)):
            overlap_count_1 = [0] * len(expand_groups_result[i])
            overlap_count_2 = [0] * len(expand_groups_result[j])
            for idx1, interval1 in enumerate(expand_groups_result[i]):
                for idx2, interval2 in enumerate(expand_groups_result[j]):
                    if interval1[0] <= interval2[1] and interval1[1] >= interval2[0]:
                        overlap_count_1[idx1] += 1
                        overlap_count_2[idx2] += 1
                        # print(interval1,interval2,overlap_count_1,overlap_count_2)
                        if overlap_count_1.count(0) == 0 and overlap_count_2.count(0) == 0:
                            # print('check_result_overlap: ',expand_groups_result[i])
                            # print('check_result_overlap: ',expand_groups_result[j])
                            return True
    return False

def check_group_distance(settings, block_group):
    # 遍历 block_group 中的每一对相邻的 blocks，检查它们之间的距离
    for i in range(len(block_group) - 1):
        # 获取当前 block 和下一个 block 的右端点和左端点
        current_block_end = block_group[i][1]
        next_block_start = block_group[i + 1][0]
        # 如果两者之间的距离小于 minBlockDis，返回 False
        if next_block_start - current_block_end < settings['minBlockDis']:
            return False 
    # 如果所有的 block 之间的距离都大于 minBlockDis，返回 True
    return True

def UpdateNewClass(new_block_group,blocks_class_):
    blocks_class_ = [[list(pair) for pair in sublist] for sublist in blocks_class_]
    new_block_group = [list(pair) for pair in new_block_group]
    findFlag = False
    expanded_new_blocks = [[block[0] - 1000, block[1] + 1000] for block in new_block_group]

    merge_group_idx = []

    # 遍历 blocks_class_ 中的所有区间
    for idx_1, intervals in enumerate(blocks_class_):
        intervals_update = copy.deepcopy(intervals)
        # 检查新区间是否与当前区间有重叠
        for idx_2,new_block in enumerate(expanded_new_blocks):
            for exi_block in intervals:
                # 判断是否重叠
                if new_block[0] <= exi_block[1] and new_block[1] >= exi_block[0]:
                    findFlag = True
                    merge_group_idx.append(idx_1)
                    break  # 找到重叠，跳出内部循环
    merge_group_idx = list(set(merge_group_idx))               
    if findFlag:
        intervals_update = [item for i in merge_group_idx for item in blocks_class_[i]] + expanded_new_blocks

        # 合并区间
        intervals_update.sort(key=lambda x: x[0])
        merged_intervals = []
        for interval in intervals_update:
            if not merged_intervals or merged_intervals[-1][1] < interval[0]:
                merged_intervals.append(interval)
        else:
            # 当前区间与最后一个合并区间重叠，合并它们
            merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])

        # 现在更新 blocks_class_ 中每个区间，找到每个区间应该对应的合并区间
        for idx1,intervals in enumerate(blocks_class_):
            updated_intervals = []
            for idx2,exi_block in enumerate(intervals):
                # 查找每个区间对应的合并区间
                for merged_interval in merged_intervals:
                    if exi_block[0] <= merged_interval[1] and exi_block[1] >= merged_interval[0]:
                        # 如果重叠，更新它
                        blocks_class_[idx1][idx2] = merged_interval
        
        # 现在更新 expanded_new_blocks 中每个区间，找到每个区间应该对应的合并区间
        merged_new_blocks = []
        for new_block in expanded_new_blocks:
            for merged_interval in merged_intervals:
                if new_block[0] <= merged_interval[1] and new_block[1] >= merged_interval[0]:
                    merged_new_blocks.append(merged_interval)
                    break
        newLen = len(merged_new_blocks)

        # 使用 itertools.combinations 获取所有两两组合
        combinations_2_all = list(itertools.combinations(merged_intervals, 2))
        # 确保每对组合中的区间按顺序排列，较小的在前
        combinations_2_all = [list(pair) for pair in combinations_2_all]

        combinations_2_merged_all = list(itertools.combinations(merged_new_blocks, 2))
        combinations_2_merged_all = [list(pair) for pair in combinations_2_merged_all]
        for i in merge_group_idx:
            combinations_2_class = list(itertools.combinations(blocks_class_[i], 2))
            combinations_2_class = [list(pair) for pair in combinations_2_class]
            combinations_2_merged_all = combinations_2_merged_all + combinations_2_class
        combinations_2_merged_all = list(map(lambda x: [list(i) for i in x], set(map(lambda x: tuple(map(tuple, x)), combinations_2_merged_all))))
        combinations_2_merged_all.sort(key=lambda x: x[0][0])
        
        candidate_loops_knum2 = [pair for pair in combinations_2_all if pair not in combinations_2_merged_all]

        combinations_3_all = list(itertools.combinations(merged_intervals, 3))
        combinations_3_all = [list(pair) for pair in combinations_3_all]
        if newLen >= 3:
            combinations_3_merged_all = list(itertools.combinations(merged_new_blocks, 3))
            combinations_3_merged_all = [list(pair) for pair in combinations_3_merged_all]
        else:
            combinations_3_merged_all = []
        for i in merge_group_idx:
            if len(blocks_class_[i]) >=3:
                combinations_3_class = list(itertools.combinations(blocks_class_[i], 3))
                combinations_3_class = [list(pair) for pair in combinations_3_class]
                combinations_3_merged_all = combinations_3_merged_all + combinations_3_class
            else:
                pass
        combinations_3_merged_all = list(map(lambda x: [list(i) for i in x], set(map(lambda x: tuple(map(tuple, x)), combinations_3_merged_all))))
        combinations_3_merged_all.sort(key=lambda x: x[0][0])
        candidate_loops_knum3 = [pair for pair in combinations_3_all if pair not in combinations_3_merged_all]

        blocks_class_ = [block for idx, block in enumerate(blocks_class_) if idx not in merge_group_idx]
        blocks_class_.append(merged_intervals)
        blocks_class_.sort(key=lambda x: x[0][0])
        return blocks_class_, candidate_loops_knum2,candidate_loops_knum3
    else:
        # 如果没有找到重叠，直接将 new_block_group 添加为新的一组
        blocks_class_.append(new_block_group)
        blocks_class_.sort(key=lambda x: x[0][0])
        return blocks_class_,[],[]   

def check_elements(X,flag):
    if flag == 'all':
        for element in X[:-1]:
            if element <= settings['minLinks']:
                return False
        return True
    if flag == 'exist':
        for element in X[:-1]:
            if element > settings['minLinks']:
                return True
        return False

def FindminL(datas,settings, name, mid):

    closest_values = datas[name]['midPoint'].sort_values(key=lambda x: abs(x-mid)).iloc[:settings['minw']]
    s1 = min(closest_values)
    e1 = max(closest_values)
    minL = max(2*(mid-s1)+2,2*(e1-mid)+1)

    return minL

def list_to_group(m_list, l_list):
    block_group = []
    for i in range(len(m_list)):
        block_group.append([m_list[i]-math.floor((l_list[i]-1)/2),m_list[i]+math.ceil((l_list[i]-1)/2)])
    return block_group

def group_to_list(block_group):
    m_list = []
    l_list = []
    for i in range(len(block_group)):
        m_list.append(int((block_group[i][0]+block_group[i][1])/2))
        l_list.append(block_group[i][1]-block_group[i][0]+1)
    return m_list, l_list

def getPvalue_hypergeom(N,CA,CB,CAB):
    pvalue = hypergeom.sf(max(-1,CAB-2), 2*N, CA, CB)
    # formatted_pvalue = f"{pvalue:.2e}"
    return pvalue

def harmonic_mean_p(p_values_list):
    """ 使用 Harmonic Mean P-Value (HMP) 计算整体 p 值 """
    k = len(p_values_list)
    hmp = k / np.sum(1 / np.array(p_values_list))  # 计算调和平均 p 值
    p_combined = chi2.sf(-2 * np.log(hmp), 2 * k)  # 计算整体 p 值
    return p_combined

# print('-------------------------------------- Sampling Functions ----------------------------------------------')



def deletebedpe(datas,settings,m1,l1,m2,l2):
    indexList_list = datas['indexList_list']
    locs = FindLocation(m1,l1,m2,l2)
    block1flag = (datas['data_bedpe']['midPoint1'] >= locs[0]-settings['delWid']) & (datas['data_bedpe']['midPoint1'] <= locs[1]+settings['delWid'])
    block2flag = (datas['data_bedpe']['midPoint2'] >= locs[2]-settings['delWid']) & (datas['data_bedpe']['midPoint2'] <= locs[3]+settings['delWid'])
    indexList_list = datas['data_bedpe'][(~(block1flag&block2flag))&(datas['data_bedpe'].index.isin(indexList_list))&(datas['data_bedpe'].index.isin(datas['indexList']))].index.tolist()
    return indexList_list

def choose_links(datas,sample1,sample2):
    links = datas['data_bedpe'][(datas['data_bedpe']['midPoint1']>=sample1[0])&(datas['data_bedpe']['midPoint1']<=sample1[1])&(datas['data_bedpe']['midPoint2']>=sample2[0])&(datas['data_bedpe']['midPoint2']<=sample2[1])]
    return links

def calculate_total_length(valid_scopes):
    total_length = 0
    for interval in valid_scopes:
        length = interval[1] - interval[0]
        total_length += length
    return total_length

def calculate_scope_score(datas,block_group,valid_scopes):
    scope_scores = []
    all_scope_reads  = []
    # if len(block_group) == 0:
    #     scope_scores.append(1)
    #     inside_reads = pd.DataFrame(columns=['chrom', 'chromStart', 'chromEnd'])
    #     inside_reads = pd.concat([inside_reads,datas['data_bedpe'][['chrom1','chromStart1','chromEnd1']].rename(columns={'chrom1':'chrom','chromStart1':'chromStart','chromEnd1':'chromEnd'})])
    #     inside_reads = pd.concat([inside_reads,datas['data_bedpe'][['chrom2','chromStart2','chromEnd2']].rename(columns={'chrom2':'chrom','chromStart2':'chromStart','chromEnd2':'chromEnd'})])
    #     all_scope_reads.append(inside_reads.sort_values(by=['chromStart']).reset_index(drop=True))
    #     # all_scope_reads = [datas['data_bedpe']]
    if len(block_group) == 0:
        scope_scores.append(1)
        # 合并 chrom1 和 chrom2 数据，并排序
        inside_reads = pd.concat([
            datas['data_bedpe'][['chrom1', 'chromStart1', 'chromEnd1']].rename(columns={'chrom1': 'chrom', 'chromStart1': 'chromStart', 'chromEnd1': 'chromEnd'}),
            datas['data_bedpe'][['chrom2', 'chromStart2', 'chromEnd2']].rename(columns={'chrom2': 'chrom', 'chromStart2': 'chromStart', 'chromEnd2': 'chromEnd'})
        ], ignore_index=True).sort_values(by=['chromStart']).reset_index(drop=True)
        all_scope_reads.append(inside_reads)
    else: #比如在确定两个blocks，想要找第三个block的时候，可以先看valid_scopes（可选范围内）和这两个block有links的地方的reads的分布情，提高搜索效率与收敛速度
        for scope in valid_scopes: 
            inside_score = 0
            inside_reads = pd.DataFrame(columns=['chrom', 'chromStart', 'chromEnd'])
            for sample in block_group:
                if scope[1] < sample[0]:
                    inside_bedpe = choose_links(datas,scope,sample)
                    inside_reads = pd.concat([inside_reads,inside_bedpe[['chrom1','chromStart1','chromEnd1']].rename(columns={'chrom1':'chrom','chromStart1':'chromStart','chromEnd1':'chromEnd'})])
                elif sample[1] < scope[0]:
                    inside_bedpe = choose_links(datas,sample,scope)
                    inside_reads = pd.concat([inside_reads,inside_bedpe[['chrom2','chromStart2','chromEnd2']].rename(columns={'chrom2':'chrom','chromStart2':'chromStart','chromEnd2':'chromEnd'})])
                else:
                    print('Error: overlap between scope and sample')
                    sys.exit()
                inside_score = inside_score+len(inside_bedpe)
            all_scope_reads.append(inside_reads.sort_values(by=['chromStart']).reset_index(drop=True))
            scope_scores.append(inside_score)
    return scope_scores,all_scope_reads

def calculate_reads_score(scope_reads):

    scope_reads.sort_values(by=['chromStart'],inplace=True)
    bed = BedTool(scope_reads.to_string(index=False, header=False), from_string=True)
    bed_with_coverage = bed.coverage(bed).to_dataframe()
    scope_reads['coverage'] = bed_with_coverage.iloc[:, [scope_reads.shape[1]]]
    
    return scope_reads

def valid_scopes_for_nextblock(datas,settings,block_group,extendFlag):
    valid_scopes = [(datas['staL1'], datas['endL2'])]
    extend_block_group = copy.deepcopy(block_group)
    if extendFlag == True:
        for i in range(len(extend_block_group)):
            extend_block_group[i][0] = extend_block_group[i][0] - settings['minBlockDis']
            extend_block_group[i][1] = extend_block_group[i][1] + settings['minBlockDis']

    for sample in extend_block_group:
        for scope in  valid_scopes:
            if  is_overlapped(scope, sample):
                valid_scopes.remove(scope)
                if scope[0] <= sample[0] and scope[1] >= sample[1]:
                    if (sample[0] - 1) - scope[0] >= settings['minBlockLen']:
                        valid_scopes.append((scope[0], sample[0] - 1))
                    if scope[1] - (sample[1] + 1) >= settings['minBlockLen']:
                        valid_scopes.append((sample[1] + 1, scope[1]))
                elif scope[0] < sample[0] and scope[1] < sample[1]:
                    if (sample[0] - 1) - scope[0] >= settings['minBlockLen']:
                        valid_scopes.append((scope[0], sample[0] - 1))
                elif scope[0] >= sample[0] and scope[1] >= sample[1]:
                    if scope[1] - (sample[1] + 1) >= settings['minBlockLen']:
                        valid_scopes.append((sample[1] + 1, scope[1]))
                break
    return valid_scopes

def remove_from_valid_scopes(settings,valid_scopes, sample, extendFlag): 

    expand_sample = copy.deepcopy(sample)
    expand_sample = list(expand_sample)
    if extendFlag == True:
        expand_sample[0] = expand_sample[0] - settings['minBlockDis']
        expand_sample[1] = expand_sample[1] + settings['minBlockDis']

    updated_valid_scopes = []
    for scope in valid_scopes:
        if is_overlapped(scope, expand_sample):
            if scope[0] <= expand_sample[0] and scope[1] >= expand_sample[1]:
                if (expand_sample[0] - 1) - scope[0] >= settings['minBlockLen']:
                    updated_valid_scopes.append((scope[0], expand_sample[0] - 1))
                if scope[1] - (expand_sample[1] + 1) >= settings['minBlockLen']:
                    updated_valid_scopes.append((expand_sample[1] + 1, scope[1]))
            elif scope[0] < expand_sample[0] and scope[1] < expand_sample[1]:
                if (expand_sample[0] - 1) - scope[0] >= settings['minBlockLen']:
                    updated_valid_scopes.append((scope[0], sample[0] - 1))
            elif scope[0] >= expand_sample[0] and scope[1] >= expand_sample[1]:
                if scope[1] - (expand_sample[1] + 1) >= settings['minBlockLen']:
                    updated_valid_scopes.append((expand_sample[1] + 1, scope[1]))
            # else:
            #     print("Error: unexpected scope")
        else:
            updated_valid_scopes.append(scope)
    return updated_valid_scopes

#待提速
def calculate_valid_scopes(datas,settings,knum,block_group_part,groups_result,extendFlag):
    valid_scopes = [(datas['staL1'], datas['endL2'])]

    expand_block_group_part = copy.deepcopy(block_group_part)
    expand_groups_result = copy.deepcopy(groups_result)
    if extendFlag == True:
        for i in range(len(expand_block_group_part)):
            expand_block_group_part[i][0] = expand_block_group_part[i][0] - settings['minBlockDis']
            expand_block_group_part[i][1] = expand_block_group_part[i][1] + settings['minBlockDis']
        for existing_group in expand_groups_result:
            for index, existing_sample in enumerate(existing_group):
                existing_sample = list(existing_sample)
                existing_sample[0] = existing_sample[0] - settings['minBlockDis']
                existing_sample[1] = existing_sample[1] + settings['minBlockDis']
                existing_group[index] = existing_sample

    for existing_group in expand_groups_result:
        has_overlap = np.zeros(knum, dtype=bool)
        for new_sample in expand_block_group_part:
            for existing_sample in existing_group:
                if is_overlapped(existing_sample, new_sample):
                    has_overlap[expand_block_group_part.index(new_sample)] = True
                    break
            if sum(has_overlap) >= knum - 1:
                # for existing_sample in existing_group:
                #     valid_scopes = remove_from_valid_scopes(settings,valid_scopes, existing_sample) #还没有加extendFlag

                 # 使用 valid_scopes_for_nextblock 替代 remove_from_valid_scopes 2025.2.13
                valid_scopes = valid_scopes_for_nextblock(datas, settings, expand_block_group_part, extendFlag=False)
                break

    return valid_scopes

def calculate_reads_posterior(datas,settings,knum,m_list,l_list,p_list,kth):

    candidate_reads  = []
    candidate_reads_all = pd.DataFrame()
    half_window_len = 2000
    halfmaxLen = 3000
    block_group = list_to_group(m_list, l_list)
    block_group_part = copy.deepcopy(block_group)
    block_group_part.pop(kth)
    valid_scopes = valid_scopes_for_nextblock(datas,settings,block_group_part,extendFlag=True)

    for k in range(knum):
        each_candidate_reads = pd.DataFrame()
        if k == kth:
            candidate_reads.append(each_candidate_reads)
            continue
        else:
            each_candidate_left  = datas['data_bedpe'][(datas['data_bedpe']['midPoint2']>=m_list[k]-halfmaxLen)&(datas['data_bedpe']['midPoint2']<=m_list[k]+halfmaxLen)]
            each_candidate_right = datas['data_bedpe'][(datas['data_bedpe']['midPoint1']>=m_list[k]-halfmaxLen)&(datas['data_bedpe']['midPoint1']<=m_list[k]+halfmaxLen)]
            # each_candidate_left  = datas['data_bedpe'][(datas['data_bedpe']['midPoint2']>=m_list[k]-int(l_list[k]))&(datas['data_bedpe']['midPoint2']<=m_list[k]+int(l_list[k]))]
            # each_candidate_right = datas['data_bedpe'][(datas['data_bedpe']['midPoint1']>=m_list[k]-int(l_list[k]))&(datas['data_bedpe']['midPoint1']<=m_list[k]+int(l_list[k]))]
            each_candidate_reads = pd.concat([each_candidate_reads, each_candidate_left[['chrom1','chromStart1','chromEnd1']].rename(columns={'chrom1':'chrom','chromStart1':'chromStart','chromEnd1':'chromEnd'})])
            each_candidate_reads = pd.concat([each_candidate_reads,each_candidate_right[['chrom2','chromStart2','chromEnd2']].rename(columns={'chrom2':'chrom','chromStart2':'chromStart','chromEnd2':'chromEnd'})])
            each_candidate_reads['midpoint'] = ((each_candidate_reads['chromStart'] + each_candidate_reads['chromEnd'])/2).astype(int)
            each_candidate_reads = each_candidate_reads[each_candidate_reads['midpoint'].apply(lambda x: any(a <= x <= b for a, b in valid_scopes))]    
            if len(each_candidate_reads) > 0:      
                each_candidate_reads = each_candidate_reads.sort_values(by=['chromStart']).reset_index(drop=True)
            # each_candidate_reads.to_csv('each_candidate_reads.bed', sep='\t', header=None, index=None)

            candidate_reads.append(each_candidate_reads)
            candidate_reads_all = pd.concat([candidate_reads_all,each_candidate_reads])

    candidate_reads_all['chromStart'] = candidate_reads_all['chromStart'] - half_window_len
    candidate_reads_all['chromEnd'] = candidate_reads_all['chromEnd'] + half_window_len
    candidate_reads_all = candidate_reads_all.sort_values(by=['chromStart']).reset_index(drop=True)

    #calculate each_candidate_reads coverage
    col_num = candidate_reads_all.shape[1]
    # candidate_reads_all.to_csv('candidate_reads_all.bed', sep='\t', header=None, index=None)
    refWindow = BedTool(candidate_reads_all.to_string(index=False, header=False), from_string=True)
    for k in range(knum):  
        if k!=kth:
            
            each_candidate_reads = candidate_reads[k]

            if len(each_candidate_reads)==0:
                candidate_reads_all['coverage'+str(k)] = 0
            else:
                block_bed = BedTool(each_candidate_reads.to_string(index=False, header=False), from_string=True)
                refWindow_with_coverage = refWindow.coverage(block_bed).to_dataframe()
                candidate_reads_all['coverage'+str(k)] = refWindow_with_coverage.iloc[:, [col_num]]

    # 是否需要向中间卷积？

    #calculate each_candidate_reads log posterior:
    p_matrix = np.zeros((knum,knum))
    pair = 0
    for i in range(knum):
        for j in range(i+1,knum):
            p_matrix[i,j]  = p_list[pair]
            pair = pair + 1

    p4 = p_list[-1]
    candidate_reads_all['m1_posterior'] = 1
    for k in range(knum):
        each_candidate_reads = candidate_reads[k]
        if k < kth:
            p1 = p_matrix[k,kth]
            candidate_reads_all['indicater'+str(k)] = [0 if x>1 else 1 for x in candidate_reads_all['coverage'+str(k)]]
            each_ll_m1 = np.exp((math.log(p1/(1-p1))-math.log(p4/(1-p4)))*candidate_reads_all['coverage'+str(k)]+math.log(1e-200)*candidate_reads_all['indicater'+str(k)])
        elif k > kth:
            p1 = p_matrix[kth,k]
            candidate_reads_all['indicater'+str(k)] = [0 if x>1 else 1 for x in candidate_reads_all['coverage'+str(k)]]
            each_ll_m1 = np.exp((math.log(p1/(1-p1))-math.log(p4/(1-p4)))*candidate_reads_all['coverage'+str(k)]+math.log(1e-200)*candidate_reads_all['indicater'+str(k)])
        else:
            each_ll_m1 = 1
        candidate_reads_all['m1_posterior'] = candidate_reads_all['m1_posterior'] * each_ll_m1

    candidate_reads_all['m1_post_prob'] = candidate_reads_all['m1_posterior']/candidate_reads_all['m1_posterior'].sum()

    return candidate_reads_all

#choose initial or update all
def generate_random_sample(datas,settings,knum,groups_result,extendFlag):
    # allow knum -1 overlap

    valid_scopes = [(datas['staL1'], datas['endL2'])]
    block_group = []
    scope_try_count = 0
    stopFlag = False

    while True:

        len_try_count = 0

        if calculate_total_length(valid_scopes)<(knum-len(block_group))*settings['minBlockLen']:
            # print(valid_scopes)
            print('no enough space')
            stopFlag = True
            break
        if scope_try_count > 200:
            print('scope_try_count > 20')
            stopFlag = True
            break
        # random a scope
        scope_scores,all_scope_reads = calculate_scope_score(datas,block_group,valid_scopes)
        # random select mid point: random a scope, then random within the scope
        if sum(scope_scores) == 0:
            valid_scopes = [(datas['staL1'], datas['endL2'])]
            block_group = []
            scope_try_count = scope_try_count+1
            # print('scope_try_count: ',scope_try_count)
            continue
        scope_idx = random.choices(range(len(valid_scopes)), weights=scope_scores)[0]
        scope = valid_scopes[scope_idx]
        scope_reads = calculate_reads_score(all_scope_reads[scope_idx]) # with coverage
        # print('minreads:',scope_reads['chromStart'].min(),'maxreads:',scope_reads['chromEnd'].max())
        while True:

            # mid_point = random.randint(scope[0], scope[1])
            mid_point_index = random.choices(range(len(scope_reads)), weights=scope_reads['coverage'])[0]
            # print('mid_point_index: ',mid_point_index)
            mid_point = int((scope_reads['chromEnd'][mid_point_index]+scope_reads['chromStart'][mid_point_index])/2)
            # print('mid_point: ',mid_point)        
            # resolve max half length, should be the shortest distance to the scope boundary
            max_len = min(mid_point - scope[0], scope[1] - mid_point)*2
            max_len = min(max_len, int(settings['maxBlockLen']))

            if max_len > int(settings['minBlockLen']):
                break
            elif len_try_count > 20:
                break
            else:
                len_try_count = len_try_count+1
                continue
            
        if len_try_count > 20:
            stopFlag = True
            break
        # generate length
        update_len = TruncatedDistribution(nbinom(settings['r'],settings['pp']),settings['minBlockLen'],max_len).rvs(1)[0]
        half_len = int(update_len/2)
        # generate sample
        new_sample = (mid_point - half_len, mid_point + half_len)


        # Check if the new sample is too close to any existing blocks
        if any(abs(new_sample[0] - other_sample[1]) < settings['minBlockDis'] or abs(new_sample[1] - other_sample[0]) < settings['minBlockDis'] for other_sample in block_group):
            print('----too close : ----',block_group,new_sample,valid_scopes)
            continue  # Skip if the new sample is too close to an existing block
        
        block_group.append(new_sample)
        # print('new_len: ',update_len)
        # update valid scopes
        # 1. the selected scope is consumed
        left_new_scope = (scope[0], new_sample[0] - 1-settings['minBlockDis'])
        right_new_scope = (new_sample[1] + 1 + settings['minBlockDis'], scope[1])
        valid_scopes.remove(scope)
        if  (left_new_scope[1]-left_new_scope[0])>settings['minBlockLen']:
            valid_scopes.insert(scope_idx, left_new_scope)
        if (right_new_scope[1]-right_new_scope[0])>settings['minBlockLen']:
            valid_scopes.insert(scope_idx + 1, right_new_scope)
        # 2. if the sample is overlapped with existing samples from other groups, all other samples from the other groups are removed
        
        expand_groups_result = copy.deepcopy(groups_result)
        if extendFlag == True:
            for existing_group in expand_groups_result:
                for index, existing_sample in enumerate(existing_group):
                    existing_sample = list(existing_sample)
                    existing_sample[0] = existing_sample[0] - settings['minBlockDis']
                    existing_sample[1] = existing_sample[1] + settings['minBlockDis']
                    existing_group[index] = existing_sample

        for existing_group in expand_groups_result:
            has_overlap = np.zeros(knum, dtype=bool)
            for new_sample in block_group:
                for existing_sample in existing_group:
                    if is_overlapped(existing_sample, new_sample):
                        has_overlap[block_group.index(new_sample)] = True
                        break
                if sum(has_overlap) >= knum - 1:
                    for existing_sample in existing_group:
                        valid_scopes = remove_from_valid_scopes(settings,valid_scopes, existing_sample, extendFlag=False)
                # print('valid_scopes: ',len(valid_scopes),valid_scopes)
        # if three samples are generated, return
        if len(block_group) == knum:
            break
    block_group_sorted = sorted(block_group, key=lambda x: x[0])
    if stopFlag == False:
        check_flag = check_result_overlap(groups_result + [block_group_sorted], True)
        if check_flag:
            stopFlag = True
    return stopFlag,block_group_sorted

def update_group_each(datas,settings,knum,m_list,l_list,p_list,posterior,groups_result,wethercheck):

    count = 0
    while True:
        stopFlag , update_group = generate_random_sample(datas,settings,knum,groups_result,extendFlag=True)
        if stopFlag == False:
            break
        elif count > 20:
            return m_list,l_list,p_list,posterior
        else:
            count = count+1
            continue
    # print(stopFlag , update_group)
    update_m_list = [int((x[0]+x[1])/2 )for x in update_group]
    update_l_list = [x[1]-x[0]+1 for x in update_group]
    # update_posterior,update_p_list = logposterior(datas,settings,knum,update_group,weightFlag=True)

    seed = random.choice([1,2,2,2])

    expand_l_list = update_l_list.copy()
    for i in range(knum):
        expand_len_candidates = list(range(2000,int(min(20000,settings['maxBlockLen']*2-update_l_list[i]))))
        if len(expand_len_candidates) == 0 or seed == 1:
            expand_len = 0
        else:
            expand_len = random.choice(expand_len_candidates)
        expand_l_list[i] = expand_l_list[i] + expand_len
    expand_update_group = list_to_group(update_m_list, expand_l_list)

    shrinkflag, update_group = shrinkInterval_list(datas,settings,knum,expand_update_group)
    for k in range(len(update_group)):
        if update_group[k][1] - update_group[k][0] > settings['maxBlockLen']:
            update_group = list_to_group(update_m_list, update_l_list)
            break
    update_m_list, update_l_list = group_to_list(update_group)

    if check_group_distance(settings, update_group):
        if wethercheck:
            test_groups_result = copy.deepcopy(groups_result)
            test_groups_result.append(update_group)
            check_flag = check_result_overlap(test_groups_result,True)
        else:
            check_flag = False

        if not check_flag:
            update_posterior,update_p_list = logposterior(datas,settings,knum,update_group,weightFlag=True)
            if update_posterior >= posterior:
                acceptance = 1
            else:
                acceptance = math.exp(update_posterior-posterior)
            u = np.random.uniform(0,1)
            if (acceptance >= u):
        
                # with open(DebugName, 'a') as f:
                #     f.write(f'accepted!!! group_each, update_group:{update_group},update_m_list: {update_m_list}, update_l_list: {update_l_list}, update_p_list: {update_p_list}, update_posterior: {update_posterior},last_posterior:{posterior}\n')
                return update_m_list,update_l_list,update_p_list,update_posterior
            
    # with open(DebugName, 'a') as f:
    #     f.write(f'rejected!!! group_each, update_group:{update_group},m_list: {m_list}, l_list: {l_list}, p_list: {p_list}, update_posterior: {update_posterior},last_posterior:{posterior}\n')
    return m_list,l_list,p_list,posterior

def update_L_pair(datas,settings,knum,m_list,l_list,p_list,posterior,groups_result,wethercheck,pth):

    sigma = random.choice([4, settings['sigma']])
    maxBlockLen = settings['maxBlockLen']
    minBlockLen = settings['minBlockLen']

    update_l_list = l_list.copy()
    
    pair = 0
    for i in range(knum-1):
        for j in range(i+1,knum):  
            if pair == pth:
                i_temp = i; j_temp = j
                break
            pair = pair+1 

    m1 = m_list[i_temp]; l1 = l_list[i_temp]; m2 = m_list[j_temp]; l2 = l_list[j_temp]

    update_l1 = truncnorm.rvs((minBlockLen - l1) / sigma, (maxBlockLen - l1+0.1) / sigma, loc=l1, scale=sigma, size=1).astype(int)[0]
    update_l2 = truncnorm.rvs((minBlockLen - l2) / sigma, (maxBlockLen - l2+0.1) / sigma, loc=l2, scale=sigma, size=1).astype(int)[0]

    update_l_list[i_temp] = update_l1
    update_l_list[j_temp] = update_l2
    
    # locs = FindLocation(m1,update_l1,m2,update_l2)

    update_group = list_to_group(m_list, update_l_list)
    # 检查 block 之间的距离是否满足条件
    if check_group_distance(settings, update_group):
        if wethercheck:
            test_groups_result = copy.deepcopy(groups_result)
            test_groups_result.append(update_group)
            # 检查是否有重叠
            check_flag = check_result_overlap(test_groups_result, True)
        else:
            check_flag = False
        if not check_flag:
            update_posterior, update_p_list = logposterior(datas,settings,knum,update_group,weightFlag=True)
            # 计算接受概率
            if update_posterior >= posterior:
                acceptance = 1
            else:
                acceptance = math.exp(update_posterior-posterior)
            # 生成随机数并判断是否接受
            if acceptance >= np.random.uniform(0, 1):
        
                # with open(DebugName, 'a') as f:
                #     f.write(f'accepted!!! pair, update_group:{update_group},m_list: {m_list}, update_l_list: {update_l_list}, update_p_list: {update_p_list}, update_posterior: {update_posterior},last_posterior:{posterior}\n')
                return m_list, update_l_list, update_p_list, update_posterior
        # 如果不接受更新，或者存在重叠，返回原始值

        # with open(DebugName, 'a') as f:
        #     f.write(f'rejected!!! pair, update_group:{update_group},m_list: {m_list}, l_list: {l_list}, p_list: {p_list}, update_posterior: {update_posterior},last_posterior:{posterior}\n')
        return m_list, l_list, p_list, posterior

    # 如果 block 之间的距离不满足条件，直接返回原始值
    return m_list, l_list, p_list, posterior
    
def update_L_all(datas,settings,knum,m_list,l_list,p_list,posterior,groups_result,wethercheck):

    sigma = random.choice([4, settings['sigma']])
    maxBlockLen = settings['maxBlockLen']
    minBlockLen = settings['minBlockLen']

    update_l_list = l_list.copy()

    for i in range(knum):
        length = l_list[i]
        midpoint = m_list[i]
        update_length = truncnorm.rvs((minBlockLen - length) / sigma, (maxBlockLen - length+0.1) / sigma, loc=length, scale=sigma, size=1).astype(int)[0]
        update_l_list[i] = update_length

    update_group = list_to_group(m_list, update_l_list)
    if check_group_distance(settings, update_group):
        if wethercheck:
            test_groups_result = copy.deepcopy(groups_result)
            test_groups_result.append(update_group)
            # 检查是否有重叠
            check_flag = check_result_overlap(test_groups_result, True)
        else:
            check_flag = False

        if not check_flag:
            update_posterior,update_p_list = logposterior(datas,settings,knum,update_group,weightFlag=True)
            if update_posterior >= posterior:
                acceptance = 1
            else:
                acceptance = math.exp(update_posterior-posterior)
            if (acceptance >= np.random.uniform(0,1)):
        
                # with open(DebugName, 'a') as f:
                #     f.write(f'accepted!!! all, update_group:{update_group},m_list: {m_list}, update_l_list: {update_l_list}, update_p_list: {update_p_list}, update_posterior: {update_posterior},last_posterior:{posterior}\n')
                return m_list,update_l_list,update_p_list,update_posterior
    # with open(DebugName, 'a') as f:
    #     f.write(f'rejected!!! all, update_group:{update_group},m_list: {m_list}, l_list: {l_list}, p_list: {p_list}, update_posterior: {update_posterior},last_posterior:{posterior}\n')
    return m_list,l_list,p_list,posterior
    
def update_L(datas,settings,knum,m_list,l_list,p_list,posterior,groups_result,wethercheck,pth):
    seed = random.choice([1,2])
    if seed == 1:
        return update_L_pair(datas,settings,knum,m_list,l_list,p_list,posterior,groups_result,wethercheck,pth)
    else:
        return update_L_all(datas,settings,knum,m_list,l_list,p_list,posterior,groups_result,wethercheck)
    
# 在shift里面会用到
def shrinkInterval(datas,settings,m1,l1,m2,l2): # only decide whether it can be shrink

    pre_loc = FindLocation(m1,l1,m2,l2)
    mid1 = datas['data_bedpe']['midPoint1']; mid2 = datas['data_bedpe']['midPoint2']
    shrinkchoose  = datas['data_bedpe'].loc[(mid1>=pre_loc[0])&(mid1<=pre_loc[1])&(mid2>=pre_loc[2])&(mid2<=pre_loc[3])]
    # if len(shrinkchoose) >= settings['minLinks'] :
    if len(shrinkchoose) >= 2:
        shrinkflag = True
        locs = np.zeros(4)
        locs[0] = datas['data_bed_unique1'].loc[datas['data_bed_unique1']['midPoint'] == shrinkchoose['midPoint1'].min(), 'midPoint']
        locs[1] = datas['data_bed_unique1'].loc[datas['data_bed_unique1']['midPoint'] == shrinkchoose['midPoint1'].max(), 'midPoint']
        locs[2] = datas['data_bed_unique2'].loc[datas['data_bed_unique2']['midPoint'] == shrinkchoose['midPoint2'].min(), 'midPoint']
        locs[3] = datas['data_bed_unique2'].loc[datas['data_bed_unique2']['midPoint'] == shrinkchoose['midPoint2'].max(), 'midPoint']
        # print(s1,e1,s2,e2)
        sk_m1,sk_l1,sk_m2,sk_l2 = FindML(locs)
        if sk_l1 < settings['minBlockLen']:
            sk_l1 = settings['minBlockLen']
        if sk_l2 < settings['minBlockLen']:
            sk_l2 = settings['minBlockLen']
        return shrinkflag, sk_m1,sk_l1,sk_m2,sk_l2
    else:
        shrinkflag = False
        return shrinkflag, m1,l1,m2,l2

def shrinkInterval_list(datas,settings,knum,block_group): # only decide whether it can be shrink

    L = datas['L']
    shrinkflag = False
    shrink_block_group = copy.deepcopy(block_group)

    for kth in range(knum):
        shrinkchoose = pd.DataFrame()
        shrinkflag1 = False; shrinkflag2 = False
        b1_l = datas['staL1']+L; b1_r = 0
        b2_l = datas['staL1']+L; b2_r = 0

        for k in range(kth):
            m1,l1,m2,l2 = FindML_index(k,kth,shrink_block_group)
            pre_loc = FindLocation(m1,l1,m2,l2)
            mid1 = datas['data_bedpe']['midPoint1']; mid2 = datas['data_bedpe']['midPoint2']
            shrinkchoose = shrinkchoose.append(datas['data_bedpe'].loc[(mid1>=pre_loc[0])&(mid1<=pre_loc[1])&(mid2>=pre_loc[2])&(mid2<=pre_loc[3])])
            # pd.concat([df1, df2], axis=1)
        if len(shrinkchoose) >= 2:
            shrinkflag2 = True
            b2_l = datas['data_bed_unique2'].loc[datas['data_bed_unique2']['midPoint'] == shrinkchoose['midPoint2'].min(), 'midPoint'].values.item()
            b2_r = datas['data_bed_unique2'].loc[datas['data_bed_unique2']['midPoint'] == shrinkchoose['midPoint2'].max(), 'midPoint'].values.item()

        shrinkchoose = pd.DataFrame()
        for k in range(kth+1,knum):
            m1,l1,m2,l2 = FindML_index(kth,k,shrink_block_group)
            pre_loc = FindLocation(m1,l1,m2,l2)
            mid1 = datas['data_bedpe']['midPoint1']; mid2 = datas['data_bedpe']['midPoint2']
            shrinkchoose = shrinkchoose.append(datas['data_bedpe'].loc[(mid1>=pre_loc[0])&(mid1<=pre_loc[1])&(mid2>=pre_loc[2])&(mid2<=pre_loc[3])])
        if len(shrinkchoose) >= 2:
            shrinkflag1 = True
            b1_l = datas['data_bed_unique1'].loc[datas['data_bed_unique1']['midPoint'] == shrinkchoose['midPoint1'].min(), 'midPoint'].values.item()
            b1_r = datas['data_bed_unique1'].loc[datas['data_bed_unique1']['midPoint'] == shrinkchoose['midPoint1'].max(), 'midPoint'].values.item()
    
        if (shrinkflag1 == True) | (shrinkflag2 == True):
            kth_l = min(b1_l,b2_l); kth_r = max(b1_r,b2_r)
            update_m_kth = int((kth_l+kth_r)/2); update_l_kth  = kth_r-kth_l+1
            if update_l_kth < settings['minBlockLen']:
                update_l_kth = settings['minBlockLen']
            shrink_block_group[kth] = (update_m_kth-math.floor((update_l_kth-1)/2),update_m_kth+math.ceil((update_l_kth-1)/2))
            shrinkflag = True
            
    return shrinkflag, shrink_block_group    

def update_shrink_list(datas,settings,knum,m_list,l_list,p_list,posterior,groups_result):

    block_group = list_to_group(m_list, l_list)

    shrinkflag, update_group = shrinkInterval_list(datas,settings,knum,block_group)
    if shrinkflag > 0:

        update_posterior,update_p_list = logposterior(datas,settings,knum,update_group,weightFlag=True)

        if update_posterior >= posterior:
            acceptance = 1
        else:
            acceptance = math.exp(update_posterior-posterior)
        u = np.random.uniform(0,1)
        if (acceptance >= u):
            update_m_list,update_l_list = group_to_list(update_group)
            return update_m_list,update_l_list,update_p_list,update_posterior
        else:
            return m_list,l_list,p_list,posterior
    else:    
        return m_list,l_list,p_list,posterior
    
# 发现之前的没有expand
def update_expand_shrink_list(datas,settings,knum,m_list,l_list,p_list,posterior,groups_result,wethercheck):

    seed = random.choice([1,2,2,2])
    if seed == 1: #shrink directly
        block_group = list_to_group(m_list, l_list)
    else:         #expand first
        expand_l_list = l_list.copy()
        for i in range(knum):
            expand_len_candidates = list(range(2000,int(min(20000,settings['maxBlockLen']*2-l_list[i]))))
            if len(expand_len_candidates) == 0:
                # print(m_list,l_list,posterior)
                expand_len = 0
            else:
                expand_len = random.choice(expand_len_candidates)
            expand_l_list[i] = expand_l_list[i] + expand_len
        block_group = list_to_group(m_list, expand_l_list)

    shrinkflag, update_group = shrinkInterval_list(datas,settings,knum,block_group)
    for k in range(len(update_group)):
        if update_group[k][1] - update_group[k][0] > settings['maxBlockLen']:
            update_group = list_to_group(m_list, l_list)
            break


    if check_group_distance(settings, update_group):

        if wethercheck:
            test_groups_result = copy.deepcopy(groups_result)
            test_groups_result.append(update_group)
            # 检查是否有重叠
            check_flag = check_result_overlap(test_groups_result, True)
        else:
            check_flag = False

        if shrinkflag > 0 and check_flag == False:
            update_posterior,update_p_list = logposterior(datas,settings,knum,update_group,weightFlag=True)

            if update_posterior >= posterior:
                acceptance = 1
            else:
                acceptance = math.exp(update_posterior-posterior)
            if (acceptance >= np.random.uniform(0,1)):
                update_m_list, update_l_list = group_to_list(update_group)
        
                # with open(DebugName, 'a') as f:
                #     f.write(f'accepted!!! expand_shrink, update_group:{update_group},update_m_list: {update_m_list}, update_l_list: {update_l_list}, update_p_list: {update_p_list}, update_posterior: {update_posterior},last_posterior:{posterior}\n')          
                return update_m_list,update_l_list,update_p_list,update_posterior
    # with open(DebugName, 'a') as f:
    #     f.write(f'rejected!!! expand_shrink, update_group:{update_group},m_list: {m_list}, l_list: {l_list}, p_list: {p_list},last_posterior:{posterior}\n')
    return m_list,l_list,p_list,posterior

# use this one to shift
def update_kth_block(datas,settings,knum,m_list,l_list,p_list,posterior,groups_result,wethercheck,kth):
    block_group = list_to_group(m_list, l_list)
    update_group = copy.deepcopy(block_group)
    update_m_list = m_list.copy()
    update_l_list = l_list.copy()

    block_group_part = copy.deepcopy(block_group)
    block_group_part.pop(kth)
    valid_scopes = calculate_valid_scopes(datas,settings,knum,block_group_part,groups_result,extendFlag=True)
    candidate_reads_all = calculate_reads_posterior(datas,settings,knum,m_list, l_list,p_list,kth)
    # candidate_reads_all['midpoint'] = ((candidate_reads_all['chromStart'] + candidate_reads_all['chromEnd'])/2).astype(int)
    candidate_reads_valid = candidate_reads_all[candidate_reads_all['midpoint'].apply(lambda x: any(a <= x <= b for a, b in valid_scopes))].reset_index(drop=True)
    
    if len(candidate_reads_valid) == 0:
        print('no candidate reads')
        print(kth,block_group,block_group_part,p_list,posterior,groups_result)        
    
    update_m = random.choices(candidate_reads_valid['midpoint'], weights=candidate_reads_valid['m1_post_prob'], k=1)[0]
    if update_m is None:
        print('update_m is None')
        return m_list,l_list,p_list,posterior
    else:
        update_l = TruncatedDistribution(nbinom(3,3/(l_list[kth]*2)),2*l_list[kth]/3,l_list[kth]*3).rvs(1)[0]
        if math.isnan(update_l):
            update_l = l_list[kth].copy()
        # print(l_list[kth],update_l)
        update_group[kth] = (update_m-math.floor((update_l-1)/2),update_m+math.ceil((update_l-1)/2))

        update_group = sorted(update_group, key=lambda x: x[0])

        shrinkflag, shrink_group = shrinkInterval_list(datas,settings,knum,update_group)
        if shrinkflag == True:
            update_group = shrink_group.copy()
            
        update_m_list, update_l_list = group_to_list(update_group)

        if wethercheck:
            test_groups_result = copy.deepcopy(groups_result)
            test_groups_result.append(update_group)
            # 检查是否有重叠
            check_flag = check_result_overlap(test_groups_result, True)
        else:
            check_flag = False
            
        if not check_flag:
            update_posterior,update_p_list = logposterior(datas,settings,knum,update_group,weightFlag=True)
            # print(update_m_list, update_l_list,update_posterior)
            if update_posterior >= posterior:
                acceptance = 1
            else:
                acceptance = math.exp(update_posterior-posterior)
            u = np.random.uniform(0,1)
            if (acceptance >= u):
        
                # with open(DebugName, 'a') as f:
                #     f.write(f'accepted!!!kth_block, kth: {kth}, update_group:{update_group},update_m_list: {update_m_list}, update_l_list: {update_l_list}, update_p_list: {update_p_list}, update_posterior: {update_posterior}\n')
                return update_m_list,update_l_list,update_p_list,update_posterior
            else:
        
                # with open(DebugName, 'a') as f:
                #     f.write(f'rejected!!!kth_block, kth: {kth}, update_group:{update_group},block_group: {block_group}, m_list: {m_list}, l_list: {l_list}, p_list: {p_list}, posterior: {posterior}\n')
                return m_list,l_list,p_list,posterior
        else:
    
            # with open(DebugName, 'a') as f:
            #     f.write(f'rejected!!!kth_block, kth: {kth}, update_group:{update_group},block_group: {block_group}, m_list: {m_list}, l_list: {l_list}, p_list: {p_list}, posterior: {posterior}\n')
            return m_list,l_list,p_list,posterior


def update_P_list(datas,settings,knum,m_list,l_list,p_list,posterior,groups_result): # contain P1,P234

    block_group = list_to_group(m_list, l_list)

    X = countX_list(datas,knum,block_group); 
    nn = countnn_list(datas,knum,block_group)

    update_p_list = Generate_P_all(knum,X,nn,settings)

    update_posterior = logposterior_noP(datas,settings,knum,block_group,update_p_list,weightFlag=True)

    if update_posterior >= posterior:
        acceptance = 1
    else:
        acceptance = math.exp(update_posterior-posterior)
    u = np.random.uniform(0,1)
    if acceptance >= u:

        # with open(DebugName, 'a') as f:
        #     f.write(f'accepted!!!P_list,m_list: {m_list}, m_list: {m_list}, update_p_list: {update_p_list}, update_posterior: {update_posterior}\n')
        return m_list,l_list,update_p_list,update_posterior
    else:

        # with open(DebugName, 'a') as f:
        #     f.write(f'rejected!!!P_list, m_list: {m_list}, l_list: {l_list}, p_list: {p_list}, posterior: {posterior}\n')
        return m_list,l_list,p_list,posterior
# print('--------------------------------------- Gibbs Functions -----------------------------------------------')

def inittxt(outputdir, knum):
    filenames = [
        'm1', 'l1', 'm2', 'l2', 
        'p1', 'p234', 
        'posterior',
        'max_m1', 'max_l1', 'max_m2', 'max_l2', 'max_p1', 'max_p234', 
        'max_post',
        'foundLinksNum', 'max_m', 'max_l', 'max_p'
    ]
    for name in filenames:
        open(f"{outputdir}/{name}_{knum}.txt", 'w').close()

    loop_colnames = ['chrom1', 'chromStart1', 'chromEnd1', 'chrom2', 'chromStart2', 'chromEnd2','IAB','CA','CB','p-value','CA(s)','CB(s)','p-value(s)','post','group','keep','ChromSep','time']
    filename = f"{outputdir}/Final_Loops_{knum}.bedpe"
    with open(filename, 'w') as f:
        f.write('\t'.join(loop_colnames))
        f.write('\n')
        
    filename = f"{outputdir}/Final_Loops_ambiguous_{knum}.bedpe"
    with open(filename, 'w') as f:
        f.write('\t'.join(loop_colnames))
        f.write('\n')

    filename = f"{outputdir}/Final_Loops_test_{knum}.bedpe"
    with open(filename, 'w') as f:
        f.write('\t'.join(loop_colnames))
        f.write('\n')

def savetxt(knum,mcmcparas,outputdir,p,m_list,l_list,p_list,posterior,max_m_list,max_l_list,max_p_list,max_post,foundLinksNum):

    pair = 0
    for i in range(knum-1):
        for j in range(i+1,knum):
            m1 = m_list[:,:,i,:]; l1 = l_list[:,:,i,:]; m2 = m_list[:,:,j,:]; l2 = l_list[:,:,j,:]

        
            with open(outputdir + '/m1.txt','a') as outfile:
                for slice_2d in m1[p]:
                    outfile.write(','.join('{:.0f}'.format(float(num)) for num in slice_2d))
                    outfile.write('\n')

            with open(outputdir + '/l1.txt','a') as outfile:
                for slice_2d in l1[p]:
                    outfile.write(','.join('{:.0f}'.format(float(num)) for num in slice_2d))
                    outfile.write('\n')

            with open(outputdir + '/m2.txt','a') as outfile:
                for slice_2d in m2[p]:
                    outfile.write(','.join('{:.0f}'.format(float(num)) for num in slice_2d))
                    outfile.write('\n')

            with open(outputdir + '/l2.txt','a') as outfile:
                for slice_2d in l2[p]:
                    outfile.write(','.join('{:.0f}'.format(float(num)) for num in slice_2d))
                    outfile.write('\n')
        
            p1 = p_list[:,:,pair,:]
            with open(outputdir + '/p1.txt','a') as outfile:
                for slice_2d in p1[p]:
                    outfile.write(','.join('{:.20e}'.format(float(num)) for num in slice_2d))
                    outfile.write('\n')
            pair = pair+1
        
    p234 = p_list[:,:,-1,:]
    with open(outputdir + '/p234.txt','a') as outfile:
        for slice_2d in p234[p]:
            outfile.write(','.join('{:.20e}'.format(float(num)) for num in slice_2d))
            outfile.write('\n')
    with open(outputdir + '/posterior.txt','a') as outfile:
        for slice_2d in posterior[p]:
            outfile.write(','.join('{:.10e}'.format(float(num)) for num in slice_2d))
            outfile.write('\n')

    #2025.2.14
    for t in range(mcmcparas['rep']):
        max_m = max_m_list[:,t,:]; max_l = max_l_list[:,t,:]; max_p = max_p_list[:,t,:]
        # print(t,max_m,max_l)
        with open(outputdir + '/max_m.txt', 'a') as outfile:
            outfile.write(','.join('{:.0f}'.format(float(num)) for num in max_m[p]))
            outfile.write('\n')
        with open(outputdir + '/max_l.txt', 'a') as outfile:
            outfile.write(','.join('{:.0f}'.format(float(num)) for num in max_l[p]))
            outfile.write('\n')   
        with open(outputdir + '/max_p.txt', 'a') as outfile:
            outfile.write(','.join('{:.20e}'.format(float(num)) for num in max_p[p]))
            outfile.write('\n')

    pair = 0
    for i in range(knum-1):
        for j in range(i+1,knum):
            max_m1 = max_m_list[:,:,i]; max_l1 = max_l_list[:,:,i]; max_m2 = max_m_list[:,:,j]; max_l2 = max_l_list[:,:,j]
            # print(i,j,max_m1,max_l1,max_m2,max_l2)

            with open(outputdir + '/max_m1.txt', 'a') as outfile:
                outfile.write(','.join('{:.0f}'.format(float(num)) for num in max_m1[p]))
                outfile.write('\n')

            with open(outputdir + '/max_l1.txt', 'a') as outfile:
                outfile.write(','.join('{:.0f}'.format(float(num)) for num in max_l1[p]))
                outfile.write('\n')

            with open(outputdir + '/max_m2.txt', 'a') as outfile:
                outfile.write(','.join('{:.0f}'.format(float(num)) for num in max_m2[p]))
                outfile.write('\n')

            with open(outputdir + '/max_l2.txt', 'a') as outfile:
                outfile.write(','.join('{:.0f}'.format(float(num)) for num in max_l2[p]))
                outfile.write('\n')

            max_p1 = max_p_list[:,:,pair]
            with open(outputdir + '/max_p1.txt', 'a') as outfile:
                outfile.write(','.join('{:.20e}'.format(float(num)) for num in max_p1[p]))
                outfile.write('\n')
            pair = pair+1

    max_p234 = max_p_list[:,:,-1]
    with open(outputdir + '/max_p234.txt', 'a') as outfile:
        outfile.write(','.join('{:.20e}'.format(float(num)) for num in max_p234[p]))
        outfile.write('\n')

    with open(outputdir + '/max_post.txt', 'a') as outfile:
        outfile.write(','.join('{:.10e}'.format(float(num)) for num in max_post[p]))
        outfile.write('\n')

    with open(outputdir + '/foundLinksNum.txt', 'a') as outfile:
        for slice_2d in foundLinksNum[p]:
            outfile.write(','.join('{:.0f}'.format(float(num)) for num in slice_2d))
            outfile.write('\n')

def savetxt(knum, mcmcparas, outputdir, p, m_list, l_list, p_list, posterior, max_m_list, max_l_list, max_p_list, max_post, foundLinksNum):

    pair = 0
    for i in range(knum - 1):
        for j in range(i + 1, knum):
            m1 = m_list[:, :, i, :]; l1 = l_list[:, :, i, :]; m2 = m_list[:, :, j, :]; l2 = l_list[:, :, j, :]

            with open(f"{outputdir}/m1_{knum}.txt", 'a') as outfile:
                for slice_2d in m1[p]:
                    outfile.write(','.join('{:.0f}'.format(float(num)) for num in slice_2d))
                    outfile.write('\n')

            with open(f"{outputdir}/l1_{knum}.txt", 'a') as outfile:
                for slice_2d in l1[p]:
                    outfile.write(','.join('{:.0f}'.format(float(num)) for num in slice_2d))
                    outfile.write('\n')

            with open(f"{outputdir}/m2_{knum}.txt", 'a') as outfile:
                for slice_2d in m2[p]:
                    outfile.write(','.join('{:.0f}'.format(float(num)) for num in slice_2d))
                    outfile.write('\n')

            with open(f"{outputdir}/l2_{knum}.txt", 'a') as outfile:
                for slice_2d in l2[p]:
                    outfile.write(','.join('{:.0f}'.format(float(num)) for num in slice_2d))
                    outfile.write('\n')

            p1 = p_list[:, :, pair, :]
            with open(f"{outputdir}/p1_{knum}.txt", 'a') as outfile:
                for slice_2d in p1[p]:
                    outfile.write(','.join('{:.20e}'.format(float(num)) for num in slice_2d))
                    outfile.write('\n')
            pair += 1

    p234 = p_list[:, :, -1, :]
    with open(f"{outputdir}/p234_{knum}.txt", 'a') as outfile:
        for slice_2d in p234[p]:
            outfile.write(','.join('{:.20e}'.format(float(num)) for num in slice_2d))
            outfile.write('\n')

    with open(f"{outputdir}/posterior_{knum}.txt", 'a') as outfile:
        for slice_2d in posterior[p]:
            outfile.write(','.join('{:.10e}'.format(float(num)) for num in slice_2d))
            outfile.write('\n')

    # 2025.2.14
    for t in range(mcmcparas['rep']):
        max_m = max_m_list[:, t, :]; max_l = max_l_list[:, t, :]; max_p = max_p_list[:, t, :]

        with open(f"{outputdir}/max_m_{knum}.txt", 'a') as outfile:
            outfile.write(','.join('{:.0f}'.format(float(num)) for num in max_m[p]))
            outfile.write('\n')

        with open(f"{outputdir}/max_l_{knum}.txt", 'a') as outfile:
            outfile.write(','.join('{:.0f}'.format(float(num)) for num in max_l[p]))
            outfile.write('\n')   

        with open(f"{outputdir}/max_p_{knum}.txt", 'a') as outfile:
            outfile.write(','.join('{:.20e}'.format(float(num)) for num in max_p[p]))
            outfile.write('\n')

    pair = 0
    for i in range(knum - 1):
        for j in range(i + 1, knum):
            max_m1 = max_m_list[:, :, i]; max_l1 = max_l_list[:, :, i]; max_m2 = max_m_list[:, :, j]; max_l2 = max_l_list[:, :, j]

            with open(f"{outputdir}/max_m1_{knum}.txt", 'a') as outfile:
                outfile.write(','.join('{:.0f}'.format(float(num)) for num in max_m1[p]))
                outfile.write('\n')

            with open(f"{outputdir}/max_l1_{knum}.txt", 'a') as outfile:
                outfile.write(','.join('{:.0f}'.format(float(num)) for num in max_l1[p]))
                outfile.write('\n')

            with open(f"{outputdir}/max_m2_{knum}.txt", 'a') as outfile:
                outfile.write(','.join('{:.0f}'.format(float(num)) for num in max_m2[p]))
                outfile.write('\n')

            with open(f"{outputdir}/max_l2_{knum}.txt", 'a') as outfile:
                outfile.write(','.join('{:.0f}'.format(float(num)) for num in max_l2[p]))
                outfile.write('\n')

            max_p1 = max_p_list[:, :, pair]
            with open(f"{outputdir}/max_p1_{knum}.txt", 'a') as outfile:
                outfile.write(','.join('{:.20e}'.format(float(num)) for num in max_p1[p]))
                outfile.write('\n')
            pair += 1

    max_p234 = max_p_list[:, :, -1]
    with open(f"{outputdir}/max_p234_{knum}.txt", 'a') as outfile:
        outfile.write(','.join('{:.20e}'.format(float(num)) for num in max_p234[p]))
        outfile.write('\n')

    with open(f"{outputdir}/max_post_{knum}.txt", 'a') as outfile:
        outfile.write(','.join('{:.10e}'.format(float(num)) for num in max_post[p]))
        outfile.write('\n')

    with open(f"{outputdir}/foundLinksNum_{knum}.txt", 'a') as outfile:
        for slice_2d in foundLinksNum[p]:
            outfile.write(','.join('{:.0f}'.format(float(num)) for num in slice_2d))
            outfile.write('\n')


def ConstructDatas(file_path, settings):
    datas = {}
    name_seq = ['data_bedpe','data_bed1','data_bed2','data_bed','data_bed_unique1','data_bed_unique2','data_bed_unique','data_bedpe_self','data_bed_original']
    data_seq = Getdata(file_path, settings)
    datas = dict(zip(name_seq, data_seq))

    datas['discard_chains'] = []

    datas['N'] = len(datas['data_bed_unique'])
    datas['N1'] = len(datas['data_bed_unique1'])
    datas['N2'] = len(datas['data_bed_unique2'])
    datas['L'] = datas['data_bed_unique']['midPoint'].max()-datas['data_bed_unique']['midPoint'].min()+1
    datas['minp1'] = settings['minp1']


    datas['staL1'] = datas['data_bed_unique1']['midPoint'].min()
    datas['staL2'] = settings['minBlockDis']+datas['data_bed_unique1'].midPoint[settings['minLinks']-1]
    datas['endL1'] = datas['data_bed_unique2'].midPoint[datas['N2']-settings['minLinks']-1]-settings['minBlockDis']
    datas['endL2'] = datas['data_bed_unique2']['midPoint'].max()
    

    chooseM1M2 = ((datas['data_bedpe']['depth1']!=0)|(datas['data_bedpe']['depth2']!=0))&(datas['data_bedpe']['distance']>=settings['minBlockDis'])
    chooseM1M2 = chooseM1M2&(datas['data_bedpe']['midPoint2']<datas['data_bedpe']['midPoint2'].nlargest(math.ceil((settings['minw']-1)/2)).iloc[-1])
    chooseM1M2 = chooseM1M2&(datas['data_bedpe']['midPoint1']>datas['data_bedpe']['midPoint1'].nsmallest(math.ceil((settings['minw']-1)/2)).iloc[-1])
    datas['indexList'] = datas['data_bedpe'][chooseM1M2].index.tolist()
    datas['indexList_list'] = datas['indexList'].copy()

    datas['permuTestPost'] = []

    with open(settings['outdir']+'/worklog.txt', 'a') as f:
        f.write("N = %s, N1 = %s, N2 = %s, L = %s, minp1 = %s" % (datas['N'], datas['N1'], datas['N2'], datas['L'], datas['minp1']) + '\n')
    # print('N = %s'%datas['N'],'N1 = %s'%datas['N1'],'N2 = %s'%datas['N2'],'L = %s'%datas['L'],'minp1 = %s'%datas['minp1'])

    datas['filename'] = file_path.split('/')[-1].split('.')[0]

    return datas

def Initial2Chains(datas,settings,knum,mcmcparas, groups_result):
    pairnum = int(knum*(knum-1)/2)

    rep, rep_in, iters, subiters = mcmcparas['rep'], mcmcparas['rep_in'], mcmcparas['iters'], mcmcparas['subiters']

    chains = {}
    chains['m_list'] = np.nan * np.zeros((rep * rep_in, knum, subiters * iters + 1))
    chains['l_list'] = np.nan * np.zeros((rep * rep_in, knum, subiters * iters + 1))
    chains['p_list'] = np.nan * np.zeros((rep * rep_in, pairnum + 1, subiters * iters + 1))
    chains['post'] = np.full((rep * rep_in, subiters * iters + 1), np.NINF)
    chains['startsite'] = np.zeros(rep * rep_in)
    chains['min_index'] = np.zeros(rep)

    stopFlag_all = np.zeros(rep * rep_in, dtype=bool)
    
    for t in range(rep * rep_in):
        start = time.time()
        stopFlag_all[t], initial_group = generate_random_sample(datas,settings,knum,groups_result, extendFlag=True)
        end = time.time()
        time_cost = end - start
        
        # If stopFlag_all[t] is True, break the loop
        if stopFlag_all[t] == True:
            print('cannot generate', knum, 'blocks for one group')
            stopFlag = True
            return stopFlag, chains
        
        init_post, init_p_list = logposterior(datas,settings,knum,initial_group,weightFlag=True)
        init_m_list = [int((x[0] + x[1]) / 2) for x in initial_group]
        init_l_list = [x[1] - x[0] + 1 for x in initial_group]

        chains['m_list'][t, :, 0] = init_m_list
        chains['l_list'][t, :, 0] = init_l_list
        chains['p_list'][t, :, 0] = init_p_list
        chains['post'][t][0] = init_post

        init_X = countX_list(datas,knum,initial_group) 
        print('initial for t = ', t, ' : m1=', init_m_list, ',l1=', init_l_list, ',post=', '%.3f' % init_post, ', Case1 = ', ['%.f' % x for x in init_X[:-1]], ', time = ', time_cost, 's', ', iters=', iters)

    stopFlag = False
    return stopFlag, chains

def ConstructOutput(knum,mcmcparas):
    pairnum = int(knum*(knum-1)/2)

    groupNum,rep,iters,subiters = mcmcparas['groupNum'],mcmcparas['rep'],mcmcparas['iters'],mcmcparas['subiters']

    posterior = np.full((groupNum,rep,subiters*iters+1), np.NINF)
    m_list = np.nan*np.zeros((groupNum,rep,knum,subiters*iters+1)); l_list   = np.nan*np.zeros((groupNum,rep,knum,subiters*iters+1))
    p_list = np.nan*np.zeros((groupNum,rep,pairnum+1,subiters*iters+1))

    max_post = np.full((groupNum,rep), np.NINF)
    max_m_list = np.nan*np.zeros((groupNum,rep,knum)); max_l_list = np.nan*np.zeros((groupNum,rep,knum))
    max_p_list = np.nan*np.zeros((groupNum,rep,pairnum+1))

    foundLinksNum = np.full((groupNum,rep+1,pairnum),np.inf)
    return posterior,m_list,l_list,max_post,max_m_list,max_l_list,max_p_list,p_list,foundLinksNum

def R_hat(para,keepnum):
    # lastvalue = len([x for x in para[0] if x != -float('inf')])
    lastvalue = len([x for x in para[0] if not np.isnan(x)])
    # print('lastvalue = ',lastvalue)
    chains_array = np.array(para[:,max(0,lastvalue-keepnum):lastvalue])
    if np.var(chains_array) == 0:
        rhat = 1
    else:
        para_data = az.from_dict({'para': chains_array})
        rhat = az.summary(para_data).r_hat.values[0]

    return rhat

def R_hat_part_flag(para, keepnum):
    lastvalue = len([x for x in para[0] if not np.isnan(x)])
    chains_array = np.array(para[:, max(0, lastvalue - keepnum):lastvalue])
    
    num_chains = chains_array.shape[0]
    
    # 遍历所有两两组合，计算 R_hat
    for i, j in itertools.combinations(range(num_chains), 2):
        sub_chains = np.array([chains_array[i], chains_array[j]])
        if np.var(sub_chains) == 0:
            rhat_part = 1
        else:
            para_data = az.from_dict({'para': sub_chains})
            rhat_part = az.summary(para_data).r_hat.values[0]
        if rhat_part < 1.2:
            return True
    return False

def write2Final_Loops(datas,settings,knum,filename,p,t,s_list,e_list,max_post_rep,foundLinksNum,timeuse):
    pair_temp = 0
    for i in range(knum-1):
        for j in range(i+1,knum):
            s1,e1,s2,e2 = s_list[i],e_list[i],s_list[j],e_list[j]
            m1,l1,m2,l2 = FindML([s1,e1,s2,e2])
            # w1,w2 = GetW(datas,m1,l1,m2,l2)
            readsnum1_shrink,readsnum2_shrink = GetReadsNum(datas,settings,m1,l1,m2,l2,shrinkFlag=True,expandFlag=True)
            readsnum1,readsnum2 = GetReadsNum(datas,settings,m1,l1,m2,l2,shrinkFlag=False,expandFlag=True)

            # write to Final_Loops.bedpe
            chrom1 = os.path.splitext(os.path.basename(datas['filename']))[0].split('_')[1]; chrom2 = chrom1
            IAB = foundLinksNum[p,t,pair_temp]
            pvalue_shrink = getPvalue_hypergeom(len(datas['data_bedpe']),readsnum1_shrink,readsnum2_shrink,IAB)
            pvalue = getPvalue_hypergeom(len(datas['data_bedpe']),readsnum1,readsnum2,IAB)
            with open(settings['outdir'] + f'/{filename}_{knum}.bedpe','a') as outfile:
                outfile.write(f"{chrom1}\t{int(s1)}\t{int(e1)}\t{chrom2}\t{int(s2)}\t{int(e2)}\t{int(IAB)}\t{int(readsnum1)}\t{int(readsnum2)}\t{pvalue:.2e}\t{int(readsnum1_shrink)}\t{int(readsnum2_shrink)}\t{pvalue_shrink:.2e}\t{max_post_rep}\t{p}\t{pvalue<0.05}\t{datas['filename']}\t{timeuse}")
                outfile.write('\n')
            pair_temp = pair_temp+1

def resultprocess(datas,settings,knum,file_path,p,t,s_list,e_list,max_post_rep,foundLinksNum,groups_result,blocks_class,timeuse,writeflag):

    new_block_group = []
    for i in range(len(s_list)):
        new_block_group.append((s_list[i],e_list[i]))
    
    print(p,new_block_group, blocks_class)
    if writeflag:
        blocks_class,candidate_loops_knum2,candidate_loops_knum3 = UpdateNewClass(new_block_group, blocks_class)
    else:
        candidate_loops_knum2 = []
        candidate_loops_knum3 = []

    groups_result.append(new_block_group)

    if knum==3:
        groups_result = groups_result+candidate_loops_knum3
    elif knum==2:
        groups_result = groups_result+candidate_loops_knum2
    else:
        pass

    with open(settings['outdir'] + f'/Groups_{p}_{knum}.txt','a') as outfile:
        for group in groups_result:
            outfile.write(str(group)+'\n')
        outfile.write('\n')
    with open(settings['outdir'] + f'/Class_{p}_{knum}.txt','a') as outfile:
        for group in blocks_class:
            outfile.write(str(group)+'\n')
        outfile.write('\n')
            
    pair = 0
    for i in range(knum-1):
        for j in range(i+1,knum):
            s1,e1,s2,e2 = s_list[i],e_list[i],s_list[j],e_list[j]
            m1,l1,m2,l2 = FindML([s1,e1,s2,e2])
            # w1,w2 = GetW(datas,m1,l1,m2,l2)
            readsnum1_shrink,readsnum2_shrink = GetReadsNum(datas,settings,m1,l1,m2,l2,shrinkFlag=True,expandFlag=True)
            readsnum1,readsnum2 = GetReadsNum(datas,settings,m1,l1,m2,l2,shrinkFlag=False,expandFlag=True)

            foundLinksNum[p,-1,pair] = t
            # write to Final_Loops.bedpe
            chrom1 = os.path.splitext(os.path.basename(file_path))[0].split('_')[1]; chrom2 = chrom1
            IAB = foundLinksNum[p,t,pair]
            pvalue_shrink = getPvalue_hypergeom(len(datas['data_bedpe']),readsnum1_shrink,readsnum2_shrink,IAB)
            pvalue = getPvalue_hypergeom(len(datas['data_bedpe']),readsnum1,readsnum2,IAB)
            with open(settings['outdir'] + f'/Final_Loops_{knum}.bedpe','a') as outfile:
                outfile.write(f"{chrom1}\t{int(s1)}\t{int(e1)}\t{chrom2}\t{int(s2)}\t{int(e2)}\t{int(IAB)}\t{int(readsnum1)}\t{int(readsnum2)}\t{pvalue:.2e}\t{int(readsnum1_shrink)}\t{int(readsnum2_shrink)}\t{pvalue_shrink:.2e}\t{max_post_rep}\t{p}\t{pvalue<0.05}\t{datas['filename']}\t{timeuse}s")
                outfile.write('\n')

            # delete found links
            if settings['delFlag']:

                print('settings[delFlag] = ',settings['delFlag'])

                # loc_delete = FindLocation(m1,l1,m2,l2)
                # choose_delete = (data_bedpe['midPoint1'] >= loc_delete[0]-settings['delWid']) & (data_bedpe['midPoint1'] <= loc_delete[1]+settings['delWid']) & (data_bedpe['midPoint2'] >= loc_delete[2]-settings['delWid']) & (data_bedpe['midPoint2'] <= loc_delete[3]+settings['delWid'])
                # choose_found = (data_bedpe['midPoint1'] >= loc_delete[0]) & (data_bedpe['midPoint1'] <= loc_delete[1]) & (data_bedpe['midPoint2'] >= loc_delete[2]) & (data_bedpe['midPoint2'] <= loc_delete[3])
                # previousLinksNum = len(data_bedpe)
                # data_bedpe = data_bedpe.drop(data_bedpe[choose_delete].index).reset_index(drop=True)
                # # data_bedpe.to_csv(methodname + '/%s.bedpe'%p,index=False,header=False,sep='\t')
                # currentLinksNum = len(data_bedpe)
                # if currentLinksNum > 0:
                    
                #     print('Found ',foundLinksNum[p,t,:],' links, delete ',previousLinksNum-currentLinksNum,' links, there are ',currentLinksNum,' links left')
                # else:
                #     print('All of links are found')
                #     sys.exit(1)

            else:
                print('-----Found ',foundLinksNum[p],' links, keepindex is ',t,' -----')

            pair = pair+1

    return groups_result,blocks_class,candidate_loops_knum2,foundLinksNum

def generate_knum_combinations(groups_result_knum, knum, knum_next):
    if knum_next < knum:
        groups_result_knum_next = []  # 存储最终结果

        # 遍历 groups_result_knum 里的所有 group
        for group in groups_result_knum:
            # 生成所有可能的 knum_next 组合
            for comb in combinations(range(knum), knum_next):
                # 正确索引方式：用 `list comprehension` 选择组合后的元素
                new_group = [group[i] for i in comb]  
                groups_result_knum_next.append(new_group)

        return groups_result_knum_next
    else:
        print("knum_next should be smaller than knum")
        return None

def merge_overlapping_groups(groups_result_knum_next):
    merged_groups = []  # 存储合并后的 group
    merged_flags = [False] * len(groups_result_knum_next)  # 记录是否已合并

    for i in range(len(groups_result_knum_next)):
        if merged_flags[i]:  # 如果已被合并，跳过
            continue

        current_group = groups_result_knum_next[i]  # 选取当前 group
        for j in range(i + 1, len(groups_result_knum_next)):
            if merged_flags[j]:  # 如果 j 已合并，跳过
                continue

            overlap_count_1 = [0] * len(current_group)
            overlap_count_2 = [0] * len(groups_result_knum_next[j])
            mergeFlag = False

            for idx1, interval1 in enumerate(current_group):
                for idx2, interval2 in enumerate(groups_result_knum_next[j]):
                    if interval1[0] <= interval2[1] and interval1[1] >= interval2[0]:  # 判断区间是否重叠
                        overlap_count_1[idx1] += 1
                        overlap_count_2[idx2] += 1
                        if all(overlap_count_1) and all(overlap_count_2):  # 所有区间都有重叠
                            mergeFlag = True

            if mergeFlag:
                # 生成新的合并 group
                new_group = [
                    [min(current_group[idx][0], groups_result_knum_next[j][idx][0]),
                     max(current_group[idx][1], groups_result_knum_next[j][idx][1])]
                    for idx in range(len(current_group))
                ]
                current_group = new_group  # 更新 current_group
                merged_flags[j] = True  # 标记 j 已合并

        merged_groups.append(current_group)  # 存入合并结果

    return merged_groups  # 返回合并后的 group 列表


def PermutationTest_all(testlist,listLen = 3,expandNum=5,permuNum=10000):
    TestStopflag = False #初始值
    for i in range(len(testlist)-(2*listLen-1)):
        list1 = testlist[i:i+listLen]
        list2 = testlist[i+listLen:i+2*listLen]
        list1 = list1*int(expandNum)
        list2 = list2*int(expandNum)
        observed_diff = np.mean(list1) - np.mean(list2)
        permuted_diffs = []
        combined_data = np.concatenate([list1, list2])

        for _ in range(permuNum):
            # 随机打乱所有数据
            np.random.shuffle(combined_data)
            # 将打乱后的数据重新分成两组
            new_group1 = combined_data[:len(list1)]
            new_group2 = combined_data[len(list2):]
            # 计算新的统计量
            permuted_diff = np.mean(new_group1) - np.mean(new_group2)
            permuted_diffs.append(permuted_diff)
        # p-value：置换后统计量大于等于原始统计量的比例
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        p_value = p_value.round(3)
        # print(f"Permutation p-value: {p_value}, posts= {list1} vs {list2}")
        if p_value > 0.05 and TestStopflag:
            keep_value = max(list1+list2)+1
            TestStopflag = True
            print(f"Permutation p-value: {p_value}, posts= {list1} vs {list2} is not significant, so we can stop here. Keep value is {keep_value}")
            break
    else:
        # print('all p_value is significant, so we keep all the value.')
        keep_value = min(testlist)
    return TestStopflag,keep_value

def PermutationTest_last(testlist,listLen = 3,expandNum=5,permuNum=10000):
    TestStopflag = False 
    keep_value = min(testlist) #初始值

    if len(testlist) < 2*listLen:
        return TestStopflag,keep_value
    else:
        list1 = testlist[-2*listLen:-listLen]
        list2 = testlist[-listLen:]
        list1 = list1*int(expandNum)
        list2 = list2*int(expandNum)
        observed_diff = np.mean(list1) - np.mean(list2)
        permuted_diffs = []
        combined_data = np.concatenate([list1, list2])

        for _ in range(permuNum):
            # 随机打乱所有数据
            np.random.shuffle(combined_data)
            # 将打乱后的数据重新分成两组
            new_group1 = combined_data[:len(list1)]
            new_group2 = combined_data[len(list2):]
            # 计算新的统计量
            permuted_diff = np.mean(new_group1) - np.mean(new_group2)
            permuted_diffs.append(permuted_diff)
        # p-value：置换后统计量大于等于原始统计量的比例
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        p_value = p_value.round(3)
        # print(f"Permutation p-value: {p_value}, posts= {list1} vs {list2}")
        if p_value > 0.05 and TestStopflag:
            keep_value = max(list1+list2)+1
            TestStopflag = True
            print(f"Permutation p-value: {p_value}, posts= {list1} vs {list2} is not significant, so we can stop here. Keep value is {keep_value}")
        return TestStopflag,keep_value

# print('--------------------------------------- Analyze Functions -----------------------------------------------')

def FindIndex(datas,loops_true):

    num = len(loops_true)
    m1_true = np.zeros(num);l1_true = np.zeros(num)
    m2_true = np.zeros(num);l2_true = np.zeros(num)
    s1_true = np.zeros(num);e1_true = np.zeros(num)
    s2_true = np.zeros(num);e2_true = np.zeros(num)
    locs_true = np.zeros((num,4))

    for p in range(num):
        loc1 = loops_true.iloc[p,:].chromStart1
        loc2 = loops_true.iloc[p,:].chromEnd1
        loc3 = loops_true.iloc[p,:].chromStart2
        loc4 = loops_true.iloc[p,:].chromEnd2
        # print(p,loc1,loc2,loc3,loc4)

        case1 = datas['data_bed_unique1'][(datas['data_bed_unique1']['midPoint']>=loc1)&(datas['data_bed_unique1']['midPoint']<=loc2)]
        case2 = datas['data_bed_unique2'][(datas['data_bed_unique2']['midPoint']>=loc3)&(datas['data_bed_unique2']['midPoint']<=loc4)]

        if (len(case1)>1) and (len(case2)>1):

            s1_true[p] = min(case1.midPoint); e1_true[p] = max(case1.midPoint)
            s2_true[p] = min(case2.midPoint); e2_true[p] = max(case2.midPoint)

            l1_true[p] = e1_true[p] - s1_true[p] + 1
            l2_true[p] = e2_true[p] - s2_true[p] + 1

            m1_true[p] = (s1_true[p] + e1_true[p]) / 2
            m2_true[p] = (s2_true[p] + e2_true[p]) / 2


            locs_true[p,:] = s1_true[p],e1_true[p],s2_true[p],e2_true[p]
        else:
            # print(p,'None')
            continue 
    
    m1_true = m1_true[m1_true!=0];l1_true = l1_true[l1_true!=0]
    m2_true = m2_true[m2_true!=0];l2_true = l2_true[l2_true!=0]
    s1_true = s1_true[s1_true!=0];e1_true = e1_true[e1_true!=0]
    s2_true = s2_true[s2_true!=0];e2_true = e2_true[e2_true!=0]
    locs_true = locs_true[locs_true[:,0]!=0,:]
    validnum = len(m1_true); print('validnum:',validnum)

    return m1_true,l1_true,m2_true,l2_true,s1_true,e1_true,s2_true,e2_true,locs_true

def selectColor(num,k):

  def hex_to_rgb(hex):
    rgb = []
    for i in (1, 3, 5):
      decimal = int(hex[i:i+2], 16)
      rgb.append(decimal)
  
    return tuple(rgb)

  palette = sns.color_palette("hls", num)
  colorList = palette.as_hex()
  hex = colorList[k]
  rgb = hex_to_rgb(hex)

  return rgb

def selectColor2(which_pair,rep,which_rep):

  def hex_to_rgb(hex):
    rgb = []
    for i in (1, 3, 5):
      decimal = int(hex[i:i+2], 16)
      rgb.append(decimal)
  
    return tuple(rgb)

  colorList = []
  colormaps = ['Blues','Greens','Reds','Purples','Oranges','Greys','YlOrBr','YlOrRd','OrRd','PuRd','RdPu','BuPu','GnBu','PuBu','YlGnBu','PuBuGn','BuGn','YlGn']
  for p in range(len(colormaps)):
      palette = sns.color_palette(colormaps[p],rep)
      colorList = np.append(colorList,np.array(palette.as_hex()))
  colorList = np.reshape(colorList,(len(colormaps),rep))

  
  hex = colorList[which_pair][which_rep]
  rgbwrong = hex_to_rgb(hex)
  rgb = (rgbwrong[2],rgbwrong[1],rgbwrong[0])

  return rgb

def selectColorList(rep):

  colorList = []
  colormaps = ['Blues','Greens','Reds','Purples','Oranges','Greys','YlOrBr','YlOrRd','OrRd','PuRd','RdPu','BuPu','GnBu','PuBu','YlGnBu','PuBuGn','BuGn','YlGn']
  for c in range(len(colormaps)):
      palette = sns.color_palette(colormaps[c],rep)
      colorList = np.append(colorList,np.array(palette.as_hex()))
  colorList = np.reshape(colorList,(len(colormaps),rep))

  return colorList

