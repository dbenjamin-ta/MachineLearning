# Predicting Qualitative Dumbell Lifting Performance with Random Forests
Darryl Benjamin  
1 April 2018  

#Executive Summary


This analysis predicts one of five Dumbell Lifting classifications based on data collected from accelometers on participants' belt, forearm, arm and dumbell.

The prediction uses Random Forests with 5-fold cross-validation. Due to the high level of accuracy achieved with this approach, other approaches were not explored.

This document explains the data preparation, analysis used, validation results, as well as the prediction for the test dataset.

#Data Preparation
###Set Directories, Define Libraries and Read in the Data



###Clean fields with Multiple NA and Blank Values

In the both the training and the test set, there are multiple fields that have a large number of NA or blank values.  These values coincide with observations where the variable new_window has the value "yes", of which there were 406 cases in the training data, and none in the testing data.

These values were tentatively removed.  Subsequent analysis results showed that the prediction was successful without these fields for all values of the new_window variable, and therefore no attempt was made to integrate these variables into the analysis.


```r
training.short <- training[,colSums(!is.na(training))>406]
training.short <- training.short[,colSums(training.short!="")>406]

testing.short <- testing[,colSums((!is.na(testing)))>0]
testing.short <- testing.short[,colSums(testing.short!="")>0]
```

###Split the training data into a "Pure" Training and a Validation Set.

Though the cross-validation in the Random Forest procedure provides some comfort of accuracy, additional validation was performed on hold-out data in a validation set.  A validation set was created.

```r
set.seed(1234)
trainIndex = createDataPartition(y=training.short$classe,p=0.75,list=FALSE)
training.short <- training.short[trainIndex,]
validation.short <- training.short[-trainIndex,]
```

###Alignment of Factor Variables between Training and Testing Datasets
The training data and testing data have factors whose levels are not defined consistently.  For the first, new_window, the factor levels are aligned.  For the second, cvtd_timestamp, the factor is converted to date format.

```r
new_window_values <- unique(as.character(training$new_window) )
training.short$new_window <- factor(training.short$new_window, new_window_values)
validation.short$new_window <- factor(validation.short$new_window, new_window_values)
testing.short$new_window <- factor(testing.short$new_window, new_window_values)

training.short$cvtd_timestamp <-as.Date(training.short$cvtd_timestamp, "%d/%m/%Y %H:%M")
validation.short$cvtd_timestamp <- as.Date(validation.short$cvtd_timestamp, "%d/%m/%Y %H:%M")
```

#Random Forest Model Run

###Define Model Varaibles
The modelled variable y is defined as the classe, the 60th variable in the dataframe.

The first variable is a row numbering.  As the variable is classe is sorted, if not removed the machine learning algorithm will use this to predict the variables.  In addition, the modelled variable is removed from the set of predictors (x).

```r
x <- training.short[,-c(1,60)]
y <- training.short[,60]
```

###Random Forest Model Run
The model is run using parallel processing to speed up processing time.
Rather than using bootstrap simulations, the Random Forest uses 5-fold cross-validation.


```r
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

##Configure trainControl object
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

modFit.rf <- train(x,y, method = "rf",trControl = fitControl)

##De-register parallel processing cluster
stopCluster(cluster)
registerDoSEQ()
```

###Model Output

```r
modFit.rf
```

```
## Random Forest 
## 
## 14718 samples
##    58 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11773, 11775, 11774, 11776, 11774 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9957872  0.9946709
##   30    0.9989129  0.9986250
##   58    0.9985053  0.9981094
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 30.
```

```r
modFit.rf$resample
```

```
##    Accuracy     Kappa Resample
## 1 0.9983022 0.9978526    Fold1
## 2 0.9993207 0.9991407    Fold3
## 3 0.9996602 0.9995702    Fold2
## 4 0.9989810 0.9987111    Fold5
## 5 0.9983005 0.9978502    Fold4
```

```r
confusionMatrix.train(modFit.rf)
```

```
## Cross-Validated (5 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.0  0.0  0.0  0.0
##          B  0.0 19.3  0.0  0.0  0.0
##          C  0.0  0.0 17.4  0.0  0.0
##          D  0.0  0.0  0.0 16.4  0.0
##          E  0.0  0.0  0.0  0.0 18.4
##                             
##  Accuracy (average) : 0.9989
```

```r
plot(modFit.rf, main="Accuracy by Predictors")
```

![](Machine_Learning_Week_4_Assignment_-_DB_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

```r
modFit.rf$finalModel$importance
```

```
##                      MeanDecreaseGini
## user_name                6.940768e+01
## raw_timestamp_part_1     2.738169e+03
## raw_timestamp_part_2     6.019398e+00
## cvtd_timestamp           3.903881e+01
## new_window               1.374047e-02
## num_window               1.470788e+03
## roll_belt                1.223152e+03
## pitch_belt               4.471137e+02
## yaw_belt                 5.231164e+02
## total_accel_belt         4.828672e+01
## gyros_belt_x             1.748176e+01
## gyros_belt_y             2.805661e+01
## gyros_belt_z             8.322072e+01
## accel_belt_x             1.909147e+01
## accel_belt_y             3.110930e+01
## accel_belt_z             1.715360e+02
## magnet_belt_x            7.669640e+01
## magnet_belt_y            1.769529e+02
## magnet_belt_z            1.328176e+02
## roll_arm                 7.607438e+01
## pitch_arm                4.473234e+01
## yaw_arm                  5.170509e+01
## total_accel_arm          1.678495e+01
## gyros_arm_x              2.177190e+01
## gyros_arm_y              2.397836e+01
## gyros_arm_z              7.423615e+00
## accel_arm_x              4.757989e+01
## accel_arm_y              2.205529e+01
## accel_arm_z              1.855673e+01
## magnet_arm_x             6.022474e+01
## magnet_arm_y             4.927211e+01
## magnet_arm_z             2.507192e+01
## roll_dumbbell            1.955823e+02
## pitch_dumbbell           3.953985e+01
## yaw_dumbbell             9.933857e+01
## total_accel_dumbbell     1.447806e+02
## gyros_dumbbell_x         1.892797e+01
## gyros_dumbbell_y         5.157359e+01
## gyros_dumbbell_z         1.089757e+01
## accel_dumbbell_x         8.360417e+01
## accel_dumbbell_y         2.399534e+02
## accel_dumbbell_z         1.389119e+02
## magnet_dumbbell_x        1.893459e+02
## magnet_dumbbell_y        4.660276e+02
## magnet_dumbbell_z        5.563638e+02
## roll_forearm             3.586525e+02
## pitch_forearm            7.414954e+02
## yaw_forearm              4.232616e+01
## total_accel_forearm      1.687338e+01
## gyros_forearm_x          1.015641e+01
## gyros_forearm_y          2.679211e+01
## gyros_forearm_z          1.264280e+01
## accel_forearm_x          1.724213e+02
## accel_forearm_y          2.424948e+01
## accel_forearm_z          6.438346e+01
## magnet_forearm_x         4.815085e+01
## magnet_forearm_y         4.559233e+01
## magnet_forearm_z         7.026264e+01
```
This output shows extemely good accuracy across folds, and highlights the factors that are most predictive.  Three variables stand out - raw_timestamp_part_1, num_window, and roll_belt, with another five variables being less predictive, but still standing out from the pack.

###Model Validation
Calculate the confusion matrix and accuracy on the validation set.

```r
pred.rf.valid <-predict(modFit.rf$finalModel, validation.short[,-c(1,60)])
confus.matrix <- table(validation.short$classe, pred.rf.valid)
valid.accuracy <- (confus.matrix[1,1] + confus.matrix[2,2] + confus.matrix[3,3] + confus.matrix[4,4] + confus.matrix[5,5])/length(pred.rf.valid)
confus.matrix
```

```
##    pred.rf.valid
##        A    B    C    D    E
##   A 1046    0    0    0    0
##   B    0  711    0    0    0
##   C    0    0  642    0    0
##   D    0    0    0  615    0
##   E    0    0    0    0  693
```

```r
valid.accuracy
```

```
## [1] 1
```

With accuracy over 99%, this is satisfactory for our purposes.  I only wish real life was like this.

#Predicting the Test Set
The test set predictions are obtained by applying the fitted model to predict on the testing data.

These are the values used on the quiz.


```r
testing.short$cvtd_timestamp <-as.Date(testing.short$cvtd_timestamp, "%d/%m/%Y %H:%M")
testing.short <- testing.short[,-1]

pred.rf <- predict(modFit.rf$finalModel, testing.short)
pred.rf
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
