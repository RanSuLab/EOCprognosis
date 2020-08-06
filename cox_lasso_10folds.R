rm(list=ls())
library(survival)
library(caret)
library(glmnet)
library(survminer)
library(survivalROC)
library(Hmisc)

path = 'D:\\Dling\\MyProject\\Ovarian Cancer\\Cox\\image\\features\\10folds'
first_path_list = dir(path)
for(firstn in 1:length(first_path_list)){
  
  first_path = paste0(path,'\\',first_path_list[firstn],'\\')
  second_path_list = dir(first_path)
  alltrainCI = 0
  alltestCI = 0
  
  for(secondn in 1:length(second_path_list)){
    second_path = paste0(first_path,'\\',second_path_list[secondn],'\\')
    third_path_list = dir(second_path)
    
    test_path = paste0(second_path,'\\',third_path_list[1])
    train_path = paste0(second_path,'\\',third_path_list[2])
    
    train = read.csv(train_path, header = T,check.names = F,row.names = 1)      #��ȡ�����ļ�
    train[,"time"]=train[,"time"]/365                                               #����ʱ�䵥λ��Ϊ��
    test = read.csv(test_path, header = T,check.names = F,row.names = 1)      #��ȡ�����ļ�
    test[,"time"]=test[,"time"]/365                                               #����ʱ�䵥λ��Ϊ��
    
    
    # #############������COX����#############
    # pFilter=0.01
    # sigGenes=c("time","status")
    # for(i in colnames(train[,3:ncol(train)])){
    #   cox <- coxph(Surv(time, status) ~ train[,i], data = train)
    #   coxSummary = summary(cox)
    #   coxP=coxSummary$coefficients[,"Pr(>|z|)"]
    #   if(coxP<pFilter){
    #     sigGenes=c(sigGenes,i)
    #   }
    # }
    # train=train[,sigGenes]
    # test=test[,sigGenes]
    
    #############lasso�ع�#############
    trainLasso=train
    
    trainLasso$time[trainLasso$time<=0]=0.003
    x=as.matrix(trainLasso[,c(3:ncol(trainLasso))])
    y=data.matrix(Surv(trainLasso$time,trainLasso$status))
    fit <- glmnet(x, y, family = "cox", maxit = 1000)
    cvfit <- cv.glmnet(x, y, family="cox", maxit = 1000)
    coef <- coef(fit, s = cvfit$lambda.min)
    index <- which(coef != 0)
    actCoef <- coef[index]
    lassoGene=row.names(coef)[index]
    lassoGene=c("time","status",lassoGene)
    # print(lassoGene)
    
    if(length(lassoGene)==2){
      next
    }	
    train=train[,lassoGene]
    test=test[,lassoGene]
    lassoSigExp=train
    lassoSigExp=cbind(id=row.names(lassoSigExp),lassoSigExp)
    
    
    #############����COXģ��#############
    multiCox <- coxph(Surv(time, status) ~ ., data = train)
    multiCoxSum=summary(multiCox)
    
  
    #���train������ļ�
    riskScore=predict(multiCox,type="risk",newdata=train)           #����train�õ�ģ��Ԥ��train��Ʒ����
    trainCI=1-rcorr.cens(riskScore,Surv(train$time,train$status)) [[1]]#����һ����ָ��
    
    
    #���test������ļ�
    riskScoreTest=predict(multiCox,type="risk",newdata=test)      #����train�õ�ģ��Ԥ��test��Ʒ����
    testCI=1-rcorr.cens(riskScoreTest,Surv(test$time,test$status))[[1]]#����һ����ָ��
    
    alltrainCI = alltrainCI+trainCI
    alltestCI = alltestCI+testCI
    
  }
  
  cat(first_path_list[firstn],'    ','trainCI:',(alltrainCI/10),'testCI:','testCI:',(alltestCI/10))
  cat('\n')
}