#-*- coding: utf-8 -*-

import numpy as np

#Before training,the mean must be substract
def train(trainingset,label):
    # the total num of image
    # m=len(label)
    # the dim of features
    n=trainingset.shape[1]
    # filter the complicate label,for count the total people num
    classes,labels=np.unique(label,return_inverse=True)
    # the total people num
    nc=len(classes)
    Sw=np.zeros([n,n])
    # save each people items
    cur={}
    withinCount=0
    # record the count of each people
    numberBuff=np.zeros(1000)
    for i in range(nc):
        # get the item of i
        cur[i]=trainingset[labels==i]
        if cur[i].shape[0]>1:
            withinCount=withinCount+cur[i].shape[0]
        if numberBuff[cur[i].shape[0]] == 0 :
            numberBuff[cur[i].shape[0]] = 1
    u=np.zeros([n,nc])
    ep=np.zeros([n,withinCount])
    nowp=0
    for i in range(nc):
        # the mean of cur[i]
        u[:,i]=np.mean(cur[i],0)
        b=u[:,i].reshape(n,1)
        if cur[i].shape[0]>1:
            ep[:,nowp:nowp+cur[i].shape[0]]=cur[i].T-b
            nowp=nowp+cur[i].shape[0]
    Su=np.cov(u.T,rowvar=0)
    Sw=np.cov(ep.T,rowvar=0)
    oldSw=Sw
    SuFG={}
    SwG={}
    for l in range(500):
        F=np.linalg.pinv(Sw)
        u=np.zeros([n,nc])
        ep=np.zeros([n,withinCount])
        nowp=0
        for g in range(1000):
            if numberBuff[g]==1:
                #G = −(mS μ + S ε )−1*Su*Sw−1
                G=-np.dot(np.dot(np.linalg.pinv(g*Su+Sw),Su),F)
                #Su*(F+g*G) for u
                SuFG[g]=np.dot(Su,(F+g*G))
                #Sw*G for e
                SwG[g]=np.dot(Sw,G)
        for i in range(nc):
            nnc=cur[i].shape[0]
            #formula 7 in suppl_760
            u[:,i]=np.sum(np.dot(SuFG[nnc],cur[i].T),1)
            #formula 8 in suppl_760
            ep[:,nowp:nowp+cur[i].shape[0]]=cur[i].T+np.sum(np.dot(SwG[nnc],cur[i].T),1).reshape(n,1)
            nowp=nowp+nnc
        Su=np.cov(u.T,rowvar=0)
        Sw=np.cov(ep.T,rowvar=0)
        print l,np.linalg.norm(Sw-oldSw)/np.linalg.norm(Sw)
        if np.linalg.norm(Sw-oldSw)/np.linalg.norm(Sw)<1e-6:
            break;
        oldSw=Sw
    F=np.linalg.pinv(Sw)
    G=-np.dot(np.dot(np.linalg.pinv(2*Su+Sw),Su),F)
    A=np.linalg.pinv(Su+Sw)-(F+G)
    return A,G

#ratio of similar,the threshold we always choose in (-1,-2)
def verify(A,G,x1,x2):
    # x1.shape=(-1,1)
    # x2.shape=(-1,1)
    ratio=np.dot(np.dot(np.transpose(x1),A),x1)+np.dot(np.dot(np.transpose(x2),A),x2)-2*np.dot(np.dot(np.transpose(x1),G),x2)
    return ratio
