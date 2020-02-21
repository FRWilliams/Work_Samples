data <- read.csv(file.choose(), header = TRUE, sep = ",")

set.seed(1029)
## Remove rows that do not have target variable values
drops = "Retain 01"
X <- data[,!(names(data) %in% drops)]

data$Retain.01 <- factor(data$Retain.01)

library(caTools)

split <- sample.split(data$Retain.01, SplitRatio = 0.60)

dresstrain <- subset(data, split == TRUE)
dresstest <- subset(data, split == FALSE)


## Let's check the count of unique value in the target variable
as.data.frame(table(dresstrain$Retain.01))
## Loading DMwr to balance the unbalanced class
install.packages('DMwR')
library(DMwR)

## Smote : Synthetic Minority Oversampling Technique To Handle Class Imbalancy In Binary Classification
balanced.data <- SMOTE(Retain.01 ~., dresstrain, perc.over = 4800, k = 5, perc.under = 1000)

as.data.frame(table(balanced.data$Retain.01))
library(caret)  

model <- glm (Retain.01~., data=balanced.data, family = binomial)
summary(model)

## Predict the Values
predict <- predict(model, dresstest, type = 'response')

## Create Confusion Matrix
table(dresstest$Retain.01, predict > 0.5)
#ROCR Curve
library(ROCR)
ROCRpred <- prediction(predict, dresstest$Retain.01)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
ROCRpred
