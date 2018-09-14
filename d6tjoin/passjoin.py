import pandas as pd
import numpy as np
from collections import defaultdict

class PIndex:
    def __init__(self, stPos, Lo, partLen, indexed_len):
        self.stPos=stPos
        self.Lo = Lo
        self.partLen = partLen
        self.indexed_len = indexed_len 
        
def DJB_hash(string,pLen):
    #print(string[:pLen])
    return hash(string[:pLen])

def edit_distance(s1,s2,threshold,xpos=0,ypos=0,xlen=-1,ylen=-1):
    if xlen==-1:
        xlen=len(s1)-xpos
    if ylen==-1:
        ylen=len(s2)-ypos
    if (xlen>ylen+threshold) or (ylen>xlen+threshold):
        return threshold + 1
    if xlen==0:
        return ylen
    
    matrix=np.zeros((xlen+1, 2*threshold+1),dtype=int)
    for k in range(threshold+1):
        matrix[0][threshold+k]=k
        
    right=(threshold+(ylen-xlen))//2
    left=(threshold-(ylen-xlen))//2
    
    for i in range(1,xlen+1):
        valid=False
        if i<=left:
            matrix[i][threshold-i]=i
            valid=True
        for j in range(i-left if (i-left>=1) else 1, i+right+1 if (i+right<=ylen) else ylen + 1):
            if (s1[xpos+i-1]==s2[ypos+j-1]):
                matrix[i][j-i+threshold]=matrix[i-1][j-i+threshold]
            else:
                matrix[i][j-i+threshold]=min(matrix[i-1][j-i+threshold],
                                             matrix[i][j-i+threshold-1] if (j-1>=i-left) else threshold,
                                             matrix[i-1][j-i+threshold+1] if (j+1<=i+right) else threshold)+1
            if abs(xlen-ylen-i+j)+matrix[i][j-i+threshold]<=threshold:
                valid=True
        if valid==False:
            return threshold+1
    return matrix[xlen][ylen-xlen+threshold]
    
    
def pass_join(strings1,strings2,threshold):
    N1=len(strings1)  # number of strings1
    N2=len(strings2)  # number of strings2
    D=threshold       # threshold tau
    PN=D+1            # part number
    max_len=len(max(strings1,key=len)) # max length of strings1
    min_len=len(min(strings1,key=len)) # min length of strings1
    
    # Initialize the matrix for partPos, partLen, partIndex, invLists and dist
    partPos=np.zeros((PN+1,max_len+1),dtype=int)
    partLen=np.zeros((PN,max_len+1),dtype=int)
    dist=np.full(max_len+2,N1)

    # partPos, partLen
    for i in range(min_len,max_len+1):
        partPos[0][i]=0
        partLen[0][i]=i//PN
        partPos[PN][i]=i

    for i in range(1,PN):
        for j in range(min_len,max_len+1):
            partPos[i][j]=partPos[i-1][j]+partLen[i-1][j]
            if i==(PN-j%PN):
                partLen[i][j]=partLen[i-1][j]+1
            else:
                partLen[i][j]=partLen[i-1][j]

    # partIndex
    partIndex = np.empty((PN, max_len + 1 ), dtype=object)
    for row in partIndex:        
        for cell in range(len(row)):
            row[cell] = []

    # invLists
    invLists = np.empty((PN, max_len + 1), dtype=object)
    for row in invLists:
        for cell in range(len(row)):
            row[cell] = defaultdict(list)
            
    # Prepare dist
    clen=0 # clen of S1
    for i in range(N1): # N1
        if clen!=len(strings1[i]):
            for j in range(clen+1,len(strings1[i])+1):
                dist[j]=i
            clen=len(strings1[i])
    
    # Prepare partIndex
    clen=0 # clen of S2
    for each in strings2: # consider each string
        if clen==len(each):
            continue
        clen=len(each)
        for pid in range(PN): # pid is i-1 (1<=i<=tau+1);pid=0,1,2,3
            # indexed_len is l (|s|-tau<=l<=|s|+tau)
            for indexed_len in range(max(clen-D,min_len),min(clen+D,max_len)+1): #modified
                if dist[indexed_len]==dist[indexed_len+1]:
                    continue
                min_start=max(0, partPos[pid][indexed_len] - pid, partPos[pid][indexed_len] + (clen - indexed_len) - (D - pid))
                max_start=min(clen - partLen[pid][indexed_len], partPos[pid][indexed_len] + pid,partPos[pid][indexed_len] + (clen - indexed_len) + (D - pid))
                for stPos in range(min_start,max_start+1):
                    partIndex[pid][clen].append(PIndex(stPos, partPos[pid][indexed_len], partLen[pid][indexed_len], indexed_len))

    # Perform join                
    # Prepare invLists
    for _id in range(N1):# consider each string in S1
        clen=len(strings1[_id]) # clen of S1
        for partId in range(PN):# partId=0,1,2,3
            pLen=partLen[partId][clen]
            stPos = partPos[partId][clen]
            hash_value=DJB_hash(strings1[_id][stPos:],pLen)
            invLists[partId][clen][hash_value].append(_id)
            
    # Start join
    candNum=0
    realNum=0
    cand_id=[]
    real_id=[]
    real_str=[]
    for _id in range(N2):# consider each string in S2
        checked_ids=[]
        clen=len(strings2[_id])  # clen of S2
        for partId in range(PN): # partId=0,1,2,3
            for lp in range(len(partIndex[partId][clen])):
                stPos = partIndex[partId][clen][lp].stPos
                Lo = partIndex[partId][clen][lp].Lo
                pLen = partIndex[partId][clen][lp].partLen
                indexed_len = partIndex[partId][clen][lp].indexed_len

                hash_value = DJB_hash(strings2[_id][stPos:],pLen)
                if hash_value not in invLists[partId][indexed_len].keys():
                    continue
                for cand in invLists[partId][indexed_len][hash_value]:
                    if cand not in checked_ids:
                        candNum+=1
                        cand_id.append([cand,_id]) # s1,s2
                        #print('S1:',cand,'S2:',_id)
                        if partId==D:
                            checked_ids.append(cand)
                        if partId==0 or edit_distance(strings1[cand],strings2[_id],partId,0,0,Lo,stPos)<=partId:
                            if partId==0:
                                checked_ids.append(cand)    

                            if partId==D or edit_distance(strings1[cand],strings2[_id],D-partId,Lo+pLen,stPos+pLen)<=D-partId:
                                if edit_distance(strings1[cand],strings2[_id],D)<=D:
                                    checked_ids.append(cand)
                                    #print('find a real: S1:',cand,'S2:', _id)
                                    #print('ie:',strings1[cand],',', strings2[_id])
                                    #print('')
                                    realNum+=1   
                                    real_id.append([cand,_id]) # s1,s2
                                    real_str.append([strings1[cand],strings2[_id]])
    return real_str#,cand_id,real_id,candNum,realNum
