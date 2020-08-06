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
    
    train = read.csv(train_path, header = T,check.names = F,row.names = 1)      #读取输入文件
    train[,"time"]=train[,"time"]/365                                               #生存时间单位改为年
    test = read.csv(test_path, header = T,check.names = F,row.names = 1)      #读取输入文件
    test[,"time"]=test[,"time"]/365                                               #生存时间单位改为年
    
    
    # #############单因素COX分析#############
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
    
    #############lasso回归#############
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
    
    
    #############构建COX模型#############
    multiCox <- coxph(Surv(time, status) ~ ., data = train)
    multiCoxSum=summary(multiCox)
    
  
    #输出train组风险文件
    riskScore=predict(multiCox,type="risk",newdata=train)           #利用train得到模型预测train样品风险
    trainCI=1-rcorr.cens(riskScore,Surv(train$time,train$status)) [[1]]#计算一致性指数
    
    
    #输出test组风险文件
    riskScoreTest=predict(multiCox,type="risk",newdata=test)      #利用train得到模型预测test样品风险
    testCI=1-rcorr.cens(riskScoreTest,Surv(test$time,test$status))[[1]]#计算一致性指数
    
    alltrainCI = alltrainCI+trainCI
    alltestCI = alltestCI+testCI
    
  }
  
  cat(first_path_list[firstn],'    ','trainCI:',(alltrainCI/10),'testCI:','testCI:',(alltestCI/10))
  cat('\n')
}