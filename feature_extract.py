#!user/bin/python
#coding=utf-8
from sklearn.decomposition import NMF
import math
#from math import e
#from nmf import *
import numpy as np
import time
import networkx as nx



  ##############################
  ## Type 1 feature of circRNAs ##
  ##############################


def threetypes_features(nm,nd,A,FS_integration,DS_integration):
    noOfObervationsOfcircRNA=np.zeros((nm,1)) # number of observations in each row of MDA
    aveOfSimilaritiesOfcircRNA=np.zeros((nm,1))# average of all similarity scores for each circRNA
    # histogram feature: cut [0, 1] into five bins and count the proportion of similarity scores that fall into each bin
    hist1circRNA=np.zeros((nm,1))
    hist2circRNA=np.zeros((nm,1))
    hist3circRNA=np.zeros((nm,1))
    hist4circRNA=np.zeros((nm,1))
    hist5circRNA=np.zeros((nm,1))

    for i in range(nm):
        noOfObervationsOfcircRNA[i,0]=np.sum(A[i, ])
        aveOfSimilaritiesOfcircRNA[i,0]=np.mean(FS_integration[i, ])
        #print (aveOfSimilaritiesOfcircRNA[i,0])
        hist1Count = 0.0
        hist2Count = 0.0
        hist3Count = 0.0
        hist4Count = 0.0
        hist5Count = 0.0
        for j in range(nm):
            if(FS_integration[i, j] < 0.2):
                hist1Count = hist1Count + 1.0
            elif(FS_integration[i, j] < 0.4):
                hist2Count = hist2Count + 1.0
            elif(FS_integration[i, j] < 0.6):
                hist3Count = hist3Count + 1.0
            elif(FS_integration[i, j] < 0.8):
                hist4Count = hist4Count + 1.0
            elif(FS_integration[i, j] <= 1):
                hist5Count = hist5Count + 1.0
            
            
        hist1circRNA[i,0]=hist1Count /nm
        hist2circRNA[i,0]=hist2Count /nm
        hist3circRNA[i,0]=hist3Count /nm
        hist4circRNA[i,0]=hist4Count /nm
        hist5circRNA[i,0]=hist5Count /nm
                
   
    #print (hist1circRNA,hist2circRNA,hist3circRNA,hist4circRNA,hist5circRNA)
    feature1OfcircRNA=np.hstack((noOfObervationsOfcircRNA, aveOfSimilaritiesOfcircRNA, hist1circRNA,hist2circRNA, hist3circRNA, hist4circRNA, hist5circRNA))
    #print ('feature1OfcircRNA',feature1OfcircRNA[0])
      ################################
      ## Type 1 feature of diseases ##
      ################################


    noOfObervationsOfdisease=np.zeros((nd,1))# number of observations in each column of MDA
    aveOfsimilaritiesOfDisease=np.zeros((nd,1))# average of all similarity scores for each disease
    hist1disease=np.zeros((nd,1))# histogram feature: cut [0, 1] into five bins and count the proportion of similarity scores that fall into each bin
    hist2disease=np.zeros((nd,1))
    hist3disease=np.zeros((nd,1))
    hist4disease=np.zeros((nd,1))
    hist5disease=np.zeros((nd,1))
    for i in range(nd):
        noOfObervationsOfdisease[i,0]=np.sum(A[:, i])
        aveOfsimilaritiesOfDisease[i]=np.mean(DS_integration[i])
        hist1Count = 0.0
        hist2Count = 0.0
        hist3Count = 0.0
        hist4Count = 0.0
        hist5Count = 0.0
        for j in range(nd):
            if(DS_integration[i, j] < 0.2):
                hist1Count = hist1Count + 1.0
            elif(DS_integration[i, j] < 0.4):
                hist2Count = hist2Count + 1.0
            elif(DS_integration[i, j] < 0.6):
                hist3Count = hist3Count + 1.0
            elif(DS_integration[i, j] < 0.8):
                hist4Count = hist4Count + 1.0
            elif(DS_integration[i, j] <= 1):
                hist5Count = hist5Count + 1.0

        hist1disease[i,0]=hist1Count /nd
        hist2disease[i,0]=hist2Count /nd
        hist3disease[i,0]=hist3Count /nd
        hist4disease[i,0]=hist4Count /nd
        hist5disease[i,0]=hist5Count /nd

    feature1OfDisease=np.hstack((noOfObervationsOfdisease, aveOfsimilaritiesOfDisease, hist1disease,hist2disease, hist3disease, hist4disease, hist5disease))
    #print ('feature1OfDisease',feature1OfDisease[0])
      #############################
      # Type 2 feature of circRNAs ##
      #############################

    #number of neighbors of circRNAs and similarity values for 10 nearest neighbors
    numberOfNeighborscircRNA=np.zeros((nm,1))
    similarities10KnncircRNA=np.zeros((nm,10))
    averageOfFeature1circRNA=np.zeros((nm,7))
    weightedAverageOfFeature1circRNA=np.zeros((nm,7))
    similarityGraphcircRNA=np.zeros((nm,nm))
    meanSimilaritycircRNA=np.mean(FS_integration)
    for i in range(nm):
        neighborCount = 0 - 1 # similarity between an circRNA and itself is not counted
        for j in range(nm):
            if(FS_integration[i, j] >= meanSimilaritycircRNA):
                neighborCount = neighborCount + 1
                similarityGraphcircRNA[i, j] = 1
        numberOfNeighborscircRNA[i,0]=neighborCount

        similarities10KnncircRNA[i, ]=sorted(FS_integration[i, ], reverse= True )[1:11]
        indices=np.argsort(-FS_integration[i, ])[1:11]

        averageOfFeature1circRNA[i, ]=np.mean(feature1OfcircRNA[indices, ],0)
        weightedAverageOfFeature1circRNA[i, ]=np.dot(similarities10KnncircRNA[i, ],feature1OfcircRNA[indices, ])/10
        # build circRNA similarity graph
    mSGraph = nx.from_numpy_matrix(similarityGraphcircRNA)
    betweennessCentralitycircRNA=np.array(list(nx.betweenness_centrality(mSGraph).values())).T
    #print ("numberOfNeighborscircRNA",numberOfNeighborscircRNA[0,0],'similarities10KnncircRNA',sicirclarities10KnncircRNA[0])#betweennessCentralitycircRNA.shape
    #print (betweennessCentralitycircRNA)
    #print (np.array(betweennessCentralitycircRNA.values()))
    #closeness_centrality
    closenessCentralitycircRNA=np.array(list(nx.closeness_centrality(mSGraph).values())).T
    #print (closenessCentralitycircRNA.shape)
    #pagerank
    pageRankcircRNA=np.array(list(nx.pagerank(mSGraph).values())).T
    #print (pageRankcircRNA.shape)
    #eigenvector_centrality
    # eigenvector_centrality=nx.eigenvector_centrality(mSGraph)
    eigenVectorCentralitycircRNA=np.array(list(nx.eigenvector_centrality(mSGraph).values())).T
    #print (eigenVectorCentralitycircRNA.shape)
    combination=np.array([betweennessCentralitycircRNA,closenessCentralitycircRNA,pageRankcircRNA,eigenVectorCentralitycircRNA])
    #print (combination)
    #print (combination.shape)
      # # concatenation
    feature2OfcircRNA=np.hstack((numberOfNeighborscircRNA, similarities10KnncircRNA, averageOfFeature1circRNA, weightedAverageOfFeature1circRNA,combination.T))#betweennessCentralitycircRNA, closenessCentralitycircRNA, eigenVectorCentralitycircRNA, pageRankcircRNA))
    #print ('feature2OfcircRNA',feature2OfcircRNA[0])
      ###############################
      # Type 2 feature of diseases ##
      ###############################

      # number of neighbors of diseases and similarity values for 10 nearest neighbors
    numberOfNeighborsDisease=np.zeros((nd,1))
    similarities10KnnDisease=np.zeros((nd,10))
    averageOfFeature1Disease=np.zeros((nd,7))
    weightedAverageOfFeature1Disease=np.zeros((nd,7))
    similarityGraphDisease=np.zeros((nd,nd))
    meanSimilarityDisease=np.mean(DS_integration)
    for i in range(nd):
        neighborCount = 0 - 1 
        for j in range(nd):
            if(DS_integration[i, j] >= meanSimilarityDisease):
                neighborCount = neighborCount + 1
                similarityGraphDisease[i, j] = 1

        numberOfNeighborsDisease[i,0]=neighborCount

        similarities10KnnDisease[i, ]=sorted(DS_integration[i, ], reverse= True)[1:11]
        indices=np.argsort(-DS_integration[i, ])[1:11]


        averageOfFeature1Disease[i, ]=np.mean(feature1OfDisease[indices, ],0)
        weightedAverageOfFeature1Disease[i, ]=np.dot(similarities10KnnDisease[i, ],feature1OfDisease[indices, ])/10

    # build disease similarity graph
    dSGraph = nx.from_numpy_matrix(similarityGraphDisease)
    betweennessCentralityDisease=np.array(list(nx.betweenness_centrality(dSGraph).values())).T
    #print (betweenness_centrality)
    #closeness_centrality
    closenessCentralityDisease=np.array(list(nx.closeness_centrality(dSGraph).values())).T
    #print (closeness_centrality)
    #pagerank
    pageRankDisease=np.array(list(nx.pagerank(dSGraph).values())).T
    #print (pagerank)
    #eigenvector_centrality
    eigenVectorCentralityDisease=np.array(list(nx.eigenvector_centrality(dSGraph).values())).T
    #print (eigenvector_centrality)
    combination=np.array([betweennessCentralityDisease,closenessCentralityDisease,pageRankDisease,eigenVectorCentralityDisease])
    #print (combination)
    #print (combination.shape)

      # concatenation
    feature2OfDisease=np.hstack((numberOfNeighborsDisease, similarities10KnnDisease, averageOfFeature1Disease, weightedAverageOfFeature1Disease,combination.T))#betweennessCentralityDisease, closenessCentralityDisease, eigenVectorCentralityDisease, pageRankDisease))
    #print ('feature2OfDisease',feature2OfDisease[0])
      ###########################################
      ## Type 3 feature of circRNA-disease pairs ##
      ###########################################

      # matrix factorization
    # number of associations between an circRNA and a disease's neighbors
    nmf_model = NMF(n_components=20)
    latentVectorscircRNA = nmf_model.fit_transform(A)
    latentVectorsDisease = nmf_model.components_
    numberOfDiseaseNeighborAssociations=np.zeros((nm,nd))
    numberOfcircRNANeighborAssociations=np.zeros((nm,nd))
    MDAGraph=nx.Graph() 
    MDAGraph.add_nodes_from(list(range(nm+nd)))
    for i in range(nm):
        for j in range(nd):
            if A[i,j]==1:
                MDAGraph.add_edge(i, j+604)# build MDA graph
            for k in range(nd):
                if DS_integration[j,k]>= meanSimilarityDisease and A[i,k]==1 :
                    numberOfDiseaseNeighborAssociations[i,j]= numberOfDiseaseNeighborAssociations[i,j] + 1
                    
            for l in range (nm):
                if FS_integration[i,l]>= meanSimilaritycircRNA and A[l,j]==1 :
                    numberOfcircRNANeighborAssociations[i,j]= numberOfcircRNANeighborAssociations[i,j] + 1

    #betweennessCentralityMDA=nx.betweenness_centrality(MDAGraph)
    betweennessCentralityMDA=np.array(list(nx.betweenness_centrality(MDAGraph).values())).T
    betweennessCentralitycircRNAInMDA=betweennessCentralityMDA[0:604]
    betweennessCentralityDiseaseInMDA=betweennessCentralityMDA[604:692]
    #print (betweenness_centrality)
    closenessCentralityMDA=np.array(list(nx.closeness_centrality(MDAGraph).values())).T
    closenessCentralitycircRNAInMDA=closenessCentralityMDA[0:604]
    closenessCentralityDiseaseInMDA=closenessCentralityMDA[604:692]
    eigenVectorCentralityMDA=np.array(list(nx.eigenvector_centrality_numpy(MDAGraph).values())).T#nx.eigenvector_centrality(MDAGraph)
    eigenVectorCentralitycircRNAInMDA=eigenVectorCentralityMDA[0:604]
    eigenVectorCentralityDiseaseInMDA=eigenVectorCentralityMDA[604:692]
    pageRankMDA=np.array(list(nx.pagerank(MDAGraph).values())).T#nx.pagerank(MDAGraph)
    pageRankcircRNAInMDA=pageRankMDA[0:604]
    pageRankDiseaseInMDA=pageRankMDA[604:692]

    Diseasecombination=np.array([betweennessCentralityDiseaseInMDA,closenessCentralityDiseaseInMDA,eigenVectorCentralityDiseaseInMDA,pageRankDiseaseInMDA])
    feature3OfDisease=np.hstack((latentVectorsDisease.T,Diseasecombination.T))
    circRNAcombination=np.array([betweennessCentralitycircRNAInMDA,closenessCentralitycircRNAInMDA,eigenVectorCentralitycircRNAInMDA,pageRankcircRNAInMDA])
    feature3OfcircRNA=np.hstack((latentVectorscircRNA,circRNAcombination.T))
    #print ('feature3OfcircRNA',feature3OfcircRNA[0])
    #print ('feature3OfDisease',feature3OfDisease[0])
    Feature_circRNA = np.hstack ((feature1OfcircRNA,feature2OfcircRNA,feature3OfcircRNA))
    Feature_disease = np.hstack((feature1OfDisease,feature2OfDisease,feature3OfDisease))
    return Feature_circRNA,Feature_disease,numberOfDiseaseNeighborAssociations,numberOfcircRNANeighborAssociations





