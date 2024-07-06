install.packages("oncoPredict")

#BiocManager::install("GenomicFeatures")
#BiocManager::install("TxDb.Hsapiens.UCSC.hg19.knownGene")

library(oncoPredict)


trainingExprData=readRDS(file='./DataFiles/Training Data/GDSC2_Expr (RMA Normalized and Log Transformed).rds')
dim(trainingExprData) #
trainingExprData[1:4,1:4]

#IMPORTANT note: here I do e^IC50 since the IC50s are actual ln values/log transformed already, and the calcPhenotype function Paul #has will do a power transformation (I assumed it would be better to not have both transformations)
trainingPtype<-exp(trainingPtype)
trainingPtype[1:4,1:4]

load("PDAC.RData")


PDAC <- PDAC %>%
  column_to_rownames("gene") %>%
  as.matrix()

testExprData = PDAC

batchCorrect<-"eb"
powerTransformPhenotype<-TRUE
removeLowVaryingGenes<-0.2
removeLowVaringGenesFrom<-"homogenizeData"
minNumSamples=10
selection<- 1
printOutput=TRUE
pcr=FALSE
report_pc=FALSEcc=FALSE
rsq=FALSE
percent=80

calcPhenotype(trainingExprData=trainingExprData,
              trainingPtype=trainingPtype,
              testExprData=testExprData,
              batchCorrect=batchCorrect,
              powerTransformPhenotype=powerTransformPhenotype,
              removeLowVaryingGenes=removeLowVaryingGenes,
              minNumSamples=minNumSamples,
              selection=selection,
              printOutput=printOutput,
              pcr=pcr,
              removeLowVaringGenesFrom=removeLowVaringGenesFrom,
              report_pc=report_pc,
              cc=cc,
              percent=percent,
              rsq=rsq)
