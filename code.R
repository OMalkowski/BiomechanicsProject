# Installing packages
r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)
install.packages("formatR",repos = "http://cran.us.r-project.org")
install.packages("tidyverse",repos = "http://cran.us.r-project.org")
install.packages("caret", repos = "http://cran.us.r-project.org")
install.packages("data.table", repos = "http://cran.us.r-project.org")
install.packages("lubridate", repos = "http://cran.us.r-project.org")
install.packages("stringr", repos = "http://cran.us.r-project.org")
install.packages("ggplot2", repos = "http://cran.us.r-project.org")
install.packages("rpart", repos = "http://cran.us.r-project.org")
install.packages("randomForest", repos = "http://cran.us.r-project.org")
install.packages("dplyr")

# Loading libraries
library(tidyverse)
library(lubridate)
library(stringr)
library(ggplot2)
install.packages("caret")
library(caret)
install.packages("rpart")
library(rpart)
install.packages("readr")
library(readr)

# Download file part 1
# https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients/data
filename <- "column_3C_weka.csv"
library(readr)
dat3C<-read_csv(filename)

# Summarise dataset
table(dat3C$class)
summary(dat3C)

# Head dataset
head(dat3C)

# Tidy data
dat3C%>%as_tibble

# Check missing data
any(is.na(dat3C))
sum(is.na(dat3C))

# Transform dataset into dataframe for classification purposes
class(dat3C)
dat3C$class<-as.factor(dat3C$class)
dat3C<-as.data.frame(dat3C)
class(dat3C)

# Pelvic incidence
options(scipen=999)
dat3C%>%group_by(class)%>%ggplot(aes(class,pelvic_incidence))+geom_boxplot(aes(class, pelvic_incidence,col=class))+ggtitle("Pelvic Incidence per Class with Individual Data")+xlab("Class")+ylab("Pelvic Incidence per Individual")+geom_point(alpha=0.1)

# Pelvic tilt
options(scipen=999)
dat3C%>%group_by(class)%>%ggplot(aes(class,pelvic_tilt))+geom_boxplot(aes(class, pelvic_tilt,col=class))+ggtitle("Pelvic Tilt per Class with Individual Data")+xlab("Class")+ylab("Pelvic Tilt per Individual")+geom_point(alpha=0.1)

# Lumbar lordosis angle
options(scipen=999)
dat3C%>%group_by(class)%>%ggplot(aes(class,lumbar_lordosis_angle))+geom_boxplot(aes(class, lumbar_lordosis_angle,col=class))+ggtitle("Lumbar Lordosis Angle per Class with Individual Data")+xlab("Class")+ylab("Lumbar Lordosis Angle per Individual")+geom_point(alpha=0.1)

# Sacral slope
options(scipen=999)
dat3C%>%group_by(class)%>%ggplot(aes(class,sacral_slope))+geom_boxplot(aes(class, sacral_slope,col=class))+ggtitle("Sacral Slope per Class with Individual Data")+xlab("Class")+ylab("Sacral Slope per Individual")+geom_point(alpha=0.1)

# Pelvic radius
options(scipen=999)
dat3C%>%group_by(class)%>%ggplot(aes(class,pelvic_radius))+geom_boxplot(aes(class, pelvic_radius,col=class))+ggtitle("Pelvic Radius per Class with Individual Data")+xlab("Class")+ylab("Pelvic Radius per Individual")+geom_point(alpha=0.1)

# Degree of spondylolisthesis
options(scipen=999)
dat3C%>%group_by(class)%>%ggplot(aes(class,degree_spondylolisthesis))+geom_boxplot(aes(class, degree_spondylolisthesis,col=class))+ggtitle("Degree of Spondylolisthesis per Class with Individual Data")+xlab("Class")+ylab("Degree of Spondylolisthesis per Individual")+geom_point(alpha=0.1)

# Create train and test sets
set.seed(1,sample.kind="Rounding")
test_index1 <- createDataPartition(y = dat3C$class, times = 1, p = 0.2, list = FALSE) 
train_set1 <- dat3C[-test_index1,] # Create train set
test_set1 <- dat3C[test_index1,] # Create test set

# k-nearest neighbors - train and test sets
fit_knn <- train(class ~ ., method = "knn",data = train_set1)
fit_knn
ggplot(fit_knn,highlight=TRUE)
fit_knn$bestTune
fit_knn$finalModel
y_hat_knn <- predict(fit_knn, test_set1, type = "raw")
confusionMatrix(y_hat_knn, test_set1$class)$overall[["Accuracy"]]

# k-nearest neighbors - whole dataset
fit_knn1 <- train(class ~ ., method = "knn",data = dat3C)
fit_knn1
ggplot(fit_knn1,highlight=TRUE)
fit_knn1$bestTune
fit_knn1$finalModel

# rpart - whole dataset
install.packages("rpart")
library(rpart)
fit_2 <- rpart(class ~ ., data=dat3C)
fit_2
plot(fit_2,margin=0.1)
text(fit_2,cex=0.75)
install.packages("rpart.plot")
library(rpart.plot)
install.packages("RColorBrewer")
library(RColorBrewer)
rpart.plot(fit_2)
printcp(fit_2)
train_rpart <- train(class ~., method = "rpart", data = dat3C)
train_rpart
ggplot(train_rpart)
plot(train_rpart$finalModel,margin=0.1)
text(train_rpart$finalModel,cex=0.75)
pruned_fit <- prune(fit_2, cp = 0.01)
pruned_fit
ind <- !(train_rpart$finalModel$frame$var == "<leaf>") 
tree_terms <-
  train_rpart$finalModel$frame$var[ind] %>% unique() %>%
  as.character()
tree_terms

# rpart - train and test sets
train_rpart1 <- train(class ~ pelvic_incidence + pelvic_tilt + lumbar_lordosis_angle + sacral_slope + pelvic_radius + degree_spondylolisthesis, method = "rpart", data = train_set1)
train_rpart1
confusionMatrix(predict(train_rpart1, test_set1), test_set1$class)$overall["Accuracy"]
ind <- !(train_rpart1$finalModel$frame$var == "<leaf>") 
tree_terms <-
  train_rpart1$finalModel$frame$var[ind] %>% unique() %>%
  as.character()
tree_terms

# randomForest - whole dataset
library(randomForest)
fit_3 <- randomForest(class ~ ., data=dat3C)
fit_3
plot(fit_3)

# randomForest - train and test sets
nodesize <- seq(1, 50, 10)
acc <- sapply(nodesize, function(ns){
  train(class ~ ., method = "rf", data = train_set1, tuneGrid = data.frame(mtry = 2),
        nodesize = ns)$results$Accuracy 
})
qplot(nodesize, acc)
train_rf <- randomForest(class ~ ., data=train_set1, ns = ns[which.max(acc)])
train_rf
confusionMatrix(predict(train_rf, test_set1), test_set1$class)$overall["Accuracy"]

# Multiclass support vector machine
install.packages("e1071", repos="https://cran.r-project.org/package=e1071")
library(e1071)
svm1 <- svm(class~., data=train_set1, 
            method="C-classification", kernal="radial", 
            gamma=0.1, cost=10)
summary(svm1)
svm1$SV
predict<-predict(svm1,test_set1)
xtab<-table(test_set1$class,predict)
xtab
(8+15+29)/nrow(test_set1)

# Download file part 2
library(readr)
filename <- "column_2C_weka.csv"
dat<-read_csv(filename)

# Summarise dataset
table(dat$class)
summary(dat)

# Head dataset
head(dat)

# Tidy data
dat%>%as_tibble

# Check missing data
any(is.na(dat))
sum(is.na(dat))

# Transform dataset into dataframe for classification purposes
class(dat)
dat$class<-as.factor(dat$class)
dat<-as.data.frame(dat)
class(dat)

# Change column name for simplicity
colnames(dat)[colnames(dat)=="pelvic_tilt numeric"] <- "pelvic_tilt_numeric"

# Pelvic incidence
options(scipen=999)
dat%>%group_by(class)%>%ggplot(aes(class,pelvic_incidence))+geom_boxplot(aes(class, pelvic_incidence,col=class))+ggtitle("Pelvic Incidence per Class with Individual Data")+xlab("Class")+ylab("Pelvic Incidence per Individual")+geom_point(alpha=0.1)

# Pelvic tilt
options(scipen=999)
dat%>%group_by(class)%>%ggplot(aes(class,pelvic_tilt_numeric))+geom_boxplot(aes(class, pelvic_tilt_numeric,col=class))+ggtitle("Pelvic Tilt per Class with Individual Data")+xlab("Class")+ylab("Pelvic Tilt per Individual")+geom_point(alpha=0.1)

# Lumbar lordosis angle
options(scipen=999)
dat%>%group_by(class)%>%ggplot(aes(class,lumbar_lordosis_angle))+geom_boxplot(aes(class, lumbar_lordosis_angle,col=class))+ggtitle("Lumbar Lordosis Angle per Class with Individual Data")+xlab("Class")+ylab("Lumbar Lordosis Angle per Individual")+geom_point(alpha=0.1)

# Sacral slope
options(scipen=999)
dat%>%group_by(class)%>%ggplot(aes(class,sacral_slope))+geom_boxplot(aes(class, sacral_slope,col=class))+ggtitle("Sacral Slope per Class with Individual Data")+xlab("Class")+ylab("Sacral Slope per Individual")+geom_point(alpha=0.1)

# Pelvic radius
options(scipen=999)
dat%>%group_by(class)%>%ggplot(aes(class,pelvic_radius))+geom_boxplot(aes(class, pelvic_radius,col=class))+ggtitle("Pelvic Radius per Class with Individual Data")+xlab("Class")+ylab("Pelvic Radius per Individual")+geom_point(alpha=0.1)

# Degree of spondylolisthesis
options(scipen=999)
dat%>%group_by(class)%>%ggplot(aes(class,degree_spondylolisthesis))+geom_boxplot(aes(class, degree_spondylolisthesis,col=class))+ggtitle("Degree of Spondylolisthesis per Class with Individual Data")+xlab("Class")+ylab("Degree of Spondylolisthesis per Individual")+geom_point(alpha=0.1)

# Create train and test sets
set.seed(1,sample.kind="Rounding")
test_index <- createDataPartition(y = dat$class, times = 1, p = 0.2, list = FALSE) 
train_set <- dat[-test_index,] # Create train set
test_set <- dat[test_index,] # Create test set

# k-nearest neighbors - train and test sets
fit_knn2 <- train(class ~ ., method = "knn",data = train_set)
fit_knn2
ggplot(fit_knn2,highlight=TRUE)
fit_knn2$bestTune
fit_knn2$finalModel
y_hat_knn1 <- predict(fit_knn2, test_set, type = "raw")
confusionMatrix(y_hat_knn1, test_set$class)$overall[["Accuracy"]]

# k-nearest neighbors - whole dataset
fit_knn3 <- train(class ~ ., method = "knn",data = dat)
fit_knn3
ggplot(fit_knn3,highlight=TRUE)
fit_knn3$bestTune
fit_knn3$finalModel

# rpart - whole dataset
install.packages("rpart")
library(rpart)
fit_4 <- rpart(class ~ ., data=dat)
fit_4
plot(fit_4,margin=0.1)
text(fit_4,cex=0.75)
install.packages("rpart.plot")
library(rpart.plot)
install.packages("RColorBrewer")
library(RColorBrewer)
rpart.plot(fit_4)
printcp(fit_4)
train_rpart2 <- train(class ~., method = "rpart", data = dat)
train_rpart2
ggplot(train_rpart2)
plot(train_rpart2$finalModel,margin=0.1)
text(train_rpart2$finalModel,cex=0.75)
pruned_fit <- prune(fit_4, cp = 0.01)
pruned_fit
ind <- !(train_rpart2$finalModel$frame$var == "<leaf>") 
tree_terms <-
  train_rpart2$finalModel$frame$var[ind] %>% unique() %>%
  as.character()
tree_terms

# rpart - train and test sets
train_rpart3 <- train(class ~., method = "rpart", data = train_set)
train_rpart3
plot(train_rpart3)
confusionMatrix(predict(train_rpart3, test_set), test_set$class)$overall["Accuracy"]
ind <- !(train_rpart3$finalModel$frame$var == "<leaf>") 
tree_terms <-
  train_rpart3$finalModel$frame$var[ind] %>% unique() %>%
  as.character()
tree_terms

# randomForest - whole dataset
library(randomForest)
fit_5 <- randomForest(class ~ ., data=dat)
fit_5
plot(fit_5)

# randomForest - train and test sets
nodesize1 <- seq(1, 50, 10)
acc1 <- sapply(nodesize, function(ns){
  train(class ~ ., method = "rf", data = train_set, tuneGrid = data.frame(mtry = 2),
        nodesize1 = ns)$results$Accuracy 
})
qplot(nodesize1, acc1)
train_rf1 <- randomForest(class ~ ., data=train_set, ns = ns[which.max(acc)])
train_rf1
confusionMatrix(predict(train_rf1, test_set), test_set$class)$overall["Accuracy"]

# Multiclass support vector machine
install.packages("e1071")
library(e1071)
svm2 <- svm(class~., data=train_set, 
            method="C-classification", kernal="radial", 
            gamma=0.1, cost=10)
summary(svm2)
svm2$SV
predict<-predict(svm2,test_set)
xtab<-table(test_set$class,predict)
xtab
(39+15)/nrow(test_set)
