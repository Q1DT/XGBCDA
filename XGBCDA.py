from feature_extract import *
from sklearn.decomposition import PCA
import random
import time
import math
from sklearn.model_selection import train_test_split  
from sklearn import metrics  
from xgboost.sklearn import XGBClassifier  
import pandas as pd
import numpy as np  
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import pylab as plt

pca=PCA(n_components=10)

#**********Read the file and output the list in [exoRbase id,circR2disease id] format ************
circRNA_id = np.loadtxt(r'.\data\exoRBase-circR2disease id conversion.txt',dtype=bytes).astype(str)

a=np.array(circRNA_id)
exoRBase_id_all=a[:,0]
circBase_id_all=a[:,1]

all_id=[]
for i in range(1000):
    exo_id=exoRBase_id_all[i]
    circ_id=circBase_id_all[i]
    all_id.append([exo_id,circ_id])


# **************Delete useless columns, only keep those used for calculation **************
exoRBase_all= np.loadtxt(r'.\data\Normal_circRNA_RPM.txt',dtype=bytes).astype(str)
exoRBase_for_calculate=np.delete(exoRBase_all,[0,1],1)
exoRBase_id=exoRBase_all[:,0] 

exoR_id_in_Normal_circRNA_RPM=[]
exoR_id_to_circRNA_id=[]
for i in exoRBase_id:
    for j in range(1000):
        if i==all_id[j][0]:
            exoR_id_in_Normal_circRNA_RPM.append(all_id[j])
new_all_id=[]
a=0
for i in all_id:
    if all_id[a][1]!='NA':
        new_all_id.append(all_id[a])
    a=a+1
# print(new_all_id)
    
#*********************Screen out the circRNA that exists in the circRNA number ID.txt file 
circRNA_number_ID= np.loadtxt(r'.\data\circRNA number ID.txt',dtype=bytes).astype(str)
circRNA_number_ID=np.delete(circRNA_number_ID,0,1)
# print(circRNA_number_ID)
circ_location=[]
new_all_id_justCircID=np.delete(new_all_id,0,1)
new_all_id_in_circRNA_number_ID=[]
for i in range(604):
    for j in range(824):
        if circRNA_number_ID[i][0]==new_all_id[j][1]:
            circ_location.append(i)
            new_all_id_in_circRNA_number_ID.append(new_all_id[j])



exoR_id_in_circRNA_number_ID=[] 
for i in exoRBase_id:
    for j in range(49):
        if i==new_all_id_in_circRNA_number_ID[j][0]:
            exoR_id_in_circRNA_number_ID.append(i)


new_exoR_id_in_circRNA_number_ID=[]
for i in exoR_id_in_circRNA_number_ID:
    for j in range(40645):
        if i==exoRBase_all[j][0]:
            new_exoR_id_in_circRNA_number_ID.append(exoRBase_all[j])
new_exoR_id_in_circRNA_number_ID=np.delete(new_exoR_id_in_circRNA_number_ID,[0,1],1)

FS_integration_1=np.zeros((604,604))
for i in range(49):
    for j in range(49): 
        x=list(map(float,new_exoR_id_in_circRNA_number_ID[i]))
        y=list(map(float,new_exoR_id_in_circRNA_number_ID[j]))
        p=pearsonr(x, y)
        a=circ_location[i]
        b=circ_location[j]
        FS_integration_1[a,b] = p[1]

for i in range(604):
    for j in range(604):
        if  FS_integration_1[i,j] >=0.4:
            FS_integration_1[i,j] = FS_integration_1[i,j]
        else:
            FS_integration_1[i,j] = 0

def mergeToOne(X,X2):  
    X3=[]  
    for i in range(X.shape[0]):  
        tmp=np.array([list(X[i]),list(X2[i])])
        X3.append(list(np.hstack(tmp)))  
    X3=np.array(X3)  
    return X3  

na = 604 
nd = 88 
na = 659 
r = 0.5 
nn = 604*88-659 

circRNAnumbercode = np.loadtxt(r'.\data\circRNA number ID.txt',dtype=bytes).astype(str)
diseasenumbercode = np.genfromtxt(r'.\data\disease number ID.txt',dtype=str,delimiter='\t')


def Getgauss_circRNA(adjacentmatrix,nc):
       KC = np.zeros((nc,nc))
       gamaa=1
       sumnormm=0
       for i in range(nc):
           normm = np.linalg.norm(adjacentmatrix[i])**2
           sumnormm = sumnormm + normm  
       gamam = gamaa/(sumnormm/nc)
       for i in range(nc):
              for j in range(nc):
                      KC[i,j]= math.exp (-gamam*(np.linalg.norm(adjacentmatrix[i]-adjacentmatrix[j])**2))
       return KC
       
def Getgauss_disease(adjacentmatrix,nd):
       KD = np.zeros((nd,nd))
       gamaa=1
       sumnormd=0
       for i in range(nd):
              normd = np.linalg.norm(adjacentmatrix[:,i])**2
              sumnormd = sumnormd + normd
       gamad=gamaa/(sumnormd/nd)
       for i in range(nd):
           for j in range(nd):
               KD[i,j]= math.exp(-(gamad*(np.linalg.norm(adjacentmatrix[:,i]-adjacentmatrix[:,j])**2)))
       return KD


A = np.zeros((nc,nd),dtype=float)
ConnectDate = np.loadtxt(r'.\data\known disease-circRNA association number ID.txt',dtype=int)-1 
for i in range(na):
    A[ConnectDate[i,0], ConnectDate[i,1]] = 1 

dataset_n = np.argwhere(A == 0)
Trainset_p = np.argwhere(A == 1)
disease_sm = np.loadtxt(r'.\data\disease_sm.txt',dtype=int)
FS_integration = Getgauss_circRNA(A,nc)
FS_integration = 0.5*FS_integration+0.5*FS_integration_1
DS_integration = 0.5*Getgauss_disease(A,nd)+0.5*disease_sm 


circRNAFeature,DiseaseFeature,numberOfDiseaseNeighborAssociations,\
numberOfcircRNANeighborAssociations = threetypes_features(nc,nd,A,FS_integration,DS_integration)
predict_0 =np.zeros((dataset_n.shape[0]+Trainset_p.shape[0]))
Trainset_n = dataset_n[random.sample(list(range(nn)),na)]
Trainset= np.vstack((Trainset_n,Trainset_p))   
     
    
TraincircRNAFeature = circRNAFeature[Trainset[:,0]]
TrainDiseaseFeature = DiseaseFeature[Trainset[:,1]]

circRNANumberNeighborTrain = numberOfcircRNANeighborAssociations[Trainset[:,0],Trainset[:,1]]
DiseaseNumberNeighborTrain = numberOfDiseaseNeighborAssociations[Trainset[:,0],Trainset[:,1]]
    
TraincircRNAFeatureOfPair = np.hstack((TraincircRNAFeature, DiseaseNumberNeighborTrain.reshape(DiseaseNumberNeighborTrain.shape[0],1)))
PCA_TraincircRNAFeatureOfPair = pca.fit_transform(TraincircRNAFeatureOfPair)
TrainDiseaseFeatureOfPair = np.hstack((TrainDiseaseFeature, circRNANumberNeighborTrain.reshape(circRNANumberNeighborTrain.shape[0],1)))
PCA_TrainDiseaseFeatureOfPair = pca.transform(TrainDiseaseFeatureOfPair)

X_train = np.hstack((PCA_TraincircRNAFeatureOfPair,PCA_TrainDiseaseFeatureOfPair))
Y_value=[]
for i in range(Trainset_n.shape[0]):
    Y_value.append(0.0)
for i in range(Trainset_n.shape[0],Trainset.shape[0]):
    Y_value.append(1.0)

X1_train, X1_test, y1_train, y1_test = train_test_split(X_train, Y_value, test_size=0.3, random_state=0)

clf = XGBClassifier(
    learning_rate =0.2, 
    n_estimators=200,   
    max_depth=8,  
    min_child_weight=10,  
    gamma=0.5,  
    subsample=0.75,  
    colsample_bytree=0.75,  
    objective= 'binary:logistic', 
    nthread=8,   
    scale_pos_weight=1,  
    reg_alpha=1e-05,  
    reg_lambda=10, 
    seed=1024)  
  
clf.fit(X1_train, y1_train)  


new_feature= clf.apply(X1_train)
new_feature2=clf.apply(X1_test)
new_feature_all=clf.apply(X_train)
X_train_new=mergeToOne(X1_train,new_feature)
X_train_new2=mergeToOne(X1_test,new_feature2)
X_train_all=mergeToOne(X_train,new_feature_all)
model = XGBClassifier(  
    learning_rate =0.3,   
    n_estimators=200,   
    max_depth=5,  
    min_child_weight=1,  
    gamma=0.5,  
    subsample=0.8,  
    colsample_bytree=0.8,  
    objective= 'binary:logistic', 
    nthread=8,   
    scale_pos_weight=1,  
    reg_alpha=1e-05,  
    reg_lambda=1,  
    seed=1024)  
model.fit(X_train_new, y1_train)

predict_0 = model.predict_proba(X_test_new)[:,1]
predict_0scoreranknumber =np.argsort(-predict_0)
predict_0scorerank = predict_0[predict_0scoreranknumber]
diseaserankname_pos = Trainset[predict_0scoreranknumber,1]
diseaserankname = diseasenumbercode[diseaserankname_pos,1]
circRNArankname_pos = Trainset[predict_0scoreranknumber,0]
circRNArankname = circRNAnumbercode[circRNArankname_pos,1]
predict_0scorerank_pd=pd.Series(predict_0scorerank)
diseaserankname_pd=pd.Series(diseaserankname)
circRNArankname_pd=pd.Series(circRNArankname)
prediction_0_out = pd.concat([diseaserankname_pd,circRNArankname_pd,predict_0scorerank_pd],axis=1)
prediction_0_out.columns=['Disease','circRNA','Score']
prediction_0_out.to_excel(r'prediction results for all unknown samples.xlsx', sheet_name='Sheet1',index=False)


    






     
        

    
        
    
    
    
    


