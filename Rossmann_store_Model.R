library(ggplot2)
library(sqldf)
library(zoo)
library(reshape2)
library(car)
library(plyr)
library(readr)
library(dplyr)
library(caret)
library(repr)



train_data <- read.csv("data/train.csv")
str(train_data)
summary(train_data)
train_data <- train_data %>% mutate(StateHoliday = as.factor(StateHoliday))
train_data <- train_data %>% mutate(StateHoliday = as.numeric(StateHoliday))

store_data <- read.csv("data/store.csv")
str (store_data)
summary(store_data)
unique(store_data$Promo2)


#impute NAs
store_data$CompetitionDistance[is.na(store_data$CompetitionDistance)] <- median(store_data$CompetitionDistance, na.rm=TRUE)

store_data[is.na(store_data)] <- 0


train_store <- merge(train_data, store_data, by = "Store")
summary(train_store)
str(train_store)

unique(is.na(train_store))

train_store <- train_store %>% mutate(StoreType = as.factor(StoreType))
train_store <- train_store %>% mutate(StoreType = as.numeric(StoreType))

train_store <- train_store %>% mutate(Assortment = as.factor(Assortment))
train_store <- train_store %>% mutate(Assortment = as.numeric(Assortment))

train_store <- train_store %>% mutate(PromoInterval = as.factor(PromoInterval))
train_store <- train_store %>% mutate(PromoInterval = as.numeric(PromoInterval))

train_store <- subset(train_store,select=-c(Date))



# sets the random seed for reproducibility of results

set.seed(100)



#We will build our model on the training set and evaluate its performance on the test set.
#This is called the holdout-validation approach for evaluating model performance.

trainingRowIndex <- sample(1:nrow(train_store), 0.8*nrow(train_store)) 
trainingData <- train_store[trainingRowIndex, ] 
testData  <- train_store[-trainingRowIndex, ]
str(trainingData)
dim(trainingData)
dim(testData)



#Numeric Features Scaling
#To avoid problems with the modeling process, the numerical features must be rescaled.

#1-creates a list that contains the names of independent numeric variables.
cols = c( 'Store', 'Open', 'DayOfWeek', 'SchoolHoliday', 'StateHoliday', 'Promo','Customers',
         'StoreType','Assortment','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear',
         'Promo2','Promo2SinceWeek','Promo2SinceYear','PromoInterval')

#2-uses the preProcess function from the caret package to complete the scaling task.
pre_proc_val <- preProcess(trainingData[,cols], method = c("center", "scale"))

#The pre-processing object is fit only to the training data, 
#while the scaling is applied on both the train and test sets.

trainingData[,cols] = predict(pre_proc_val, trainingData[,cols])
testData[,cols] = predict(pre_proc_val, testData[,cols])

summary(trainingData)


K_train <- read.csv("data/K1 - Train.csv")
str(K_train)
summary(K_train)

K_test <- read.csv("data/K1 - Test.csv")
str(K_test)
summary(K_test)

K_test <- K_test %>% mutate(PromoInterval = as.factor(PromoInterval))
K_test <- K_test %>% mutate(PromoInterval = as.numeric(PromoInterval))

K_test <- K_test %>% mutate(StateHoliday = as.factor(StateHoliday))
K_test <- K_test %>% mutate(StateHoliday = as.numeric(StateHoliday))


#impute NAs
K_test$Assortment[is.na(K_test$Assortment)] <- median(K_test$Assortment, na.rm=TRUE)

K_test[is.na(K_test)] <- 0

unique(is.na(K_test))
K_test <- subset(K_test,select=-c(Date))

K_train <- K_train %>% mutate(PromoInterval = as.factor(PromoInterval))
K_train <- K_train %>% mutate(PromoInterval = as.numeric(PromoInterval))

K_train <- K_train %>% mutate(StateHoliday = as.factor(StateHoliday))
K_train <- K_train %>% mutate(StateHoliday = as.numeric(StateHoliday))


#impute NAs
K_train$Assortment[is.na(K_train$Assortment)] <- median(K_train$Assortment, na.rm=TRUE)

K_train[is.na(K_train)] <- 0

unique(is.na(K_train))
K_train <- subset(K_train,select=-c(Date))



lmMod <- lm(Sales ~ Store + Open +Customers+StoreType+Assortment+CompetitionDistance+CompetitionOpenSinceMonth+ CompetitionOpenSinceYear+Promo2+Promo2SinceWeek+Promo2SinceYear+PromoInterval+ DayOfWeek + SchoolHoliday + StateHoliday + Promo , data=K_train) 
Pred <- predict(lmMod, K_test)
summary (lmMod)

# build Linear Regression model

lmMod <- lm(Sales ~ Store + Open +Customers+StoreType+Assortment+CompetitionDistance+CompetitionOpenSinceMonth+ CompetitionOpenSinceYear+Promo2+Promo2SinceWeek+Promo2SinceYear+PromoInterval+ DayOfWeek + SchoolHoliday + StateHoliday + Promo , data=trainingData) 
Pred <- predict(lmMod, testData)
summary (lmMod)



#Calculating Prediction Accuracy
actuals_preds <- data.frame(cbind(actuals=testData$Sales, predicteds=Pred))
correlation_accuracy <- cor(actuals_preds)
correlation_accuracy


#Model Evaluation Metrics
#The first step is to create a function for calculating the evaluation metrics
#R-squared and RMSE. The second step is to predict and evaluate the model on train data,
#while the third step is to predict and evaluate the model on test data.


#Step 1 - create the evaluation metrics function

eval_metrics = function(model, df, predictions, target){
  resids = df[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  r2 = as.character(round(summary(model)$r.squared, 2))
  adj_r2 = as.character(round(summary(model)$adj.r.squared, 2))
  print(adj_r2) #Adjusted R-squared
  print(as.character(round(sqrt(sum(resids2)/N), 2))) #RMSE
}

# Step 2 - predicting and evaluating the model on train data
predictions = predict(lmMod, newdata = trainingData)
eval_metrics(lmMod, trainingData, predictions, target = 'Sales')

# Step 3 - predicting and evaluating the model on test data
predictions = predict(lmMod, newdata = testData)
eval_metrics(lmMod, testData, predictions, target = 'Sales')




#Regularization
#Linear regression algorithm works by selecting coefficients for each independent
#variable that minimizes a loss function. However, if the coefficients are large, 
#they can lead to over-fitting on the training dataset, 
#and such a model will not generalize well on the unseen test data.
#To overcome this shortcoming, we'll do regularization, which penalizes large coefficients.




cols_reg = c('Sales', 'Store', 'Open', 'DayOfWeek', 'SchoolHoliday', 'StateHoliday', 'Promo','Customers',
             'StoreType','Assortment','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear',
             'Promo2','Promo2SinceWeek','Promo2SinceYear','PromoInterval')

dummies <- dummyVars(Sales ~ ., data = train_store[,cols_reg])

train_dummies = predict(dummies, newdata = trainingData[,cols_reg])

test_dummies = predict(dummies, newdata = testData[,cols_reg])

print(dim(train_dummies)); print(dim(test_dummies))


# build Ridge Regression model
#Ridge regression(also referred to as l2 regularization)is an extension of linear regression where the loss function is 
#modified to minimize the complexity of the model. This modification is done by adding
#a penalty parameter that is equivalent to the square of the magnitude of the coefficients.
library(glmnet)

# Need matrices for glmnet() function. Automatically conducts conversions as well 
# for factor variables into dummy variables

x = as.matrix(train_dummies)
y_train = trainingData$Sales

x_test = as.matrix(test_dummies)
y_test = testData$Sales

lambdas <- 10^seq(2, -3, by = -.1)
ridge_reg = glmnet(x, y_train, nlambda = 25, alpha = 0, family = 'gaussian', lambda = lambdas)

summary(ridge_reg)

cv_ridge <- cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda

# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

# Prediction and evaluation on train data
predictions_train <- predict(ridge_reg, s = optimal_lambda, newx = x)
eval_results(y_train, predictions_train, trainingData)

# Prediction and evaluation on test data
predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
eval_results(y_test, predictions_test, testData)



# build lasso Regression model
#Lasso regression, or the Least Absolute Shrinkage and Selection Operator, 
#is also a modification of linear regression. In lasso, the loss function is modified to minimize 
#the complexity of the model by limiting the sum of the absolute values of 
#the model coefficients (also called the l1-norm).

lambdas <- 10^seq(2, -3, by = -.1)

# Setting alpha = 1 implements lasso regression
lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)

# Best 
lambda_best <- lasso_reg$lambda.min 
lambda_best

lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)

predictions_train <- predict(lasso_model, s = lambda_best, newx = x)
eval_results(y_train, predictions_train, trainingData)

predictions_test <- predict(lasso_model, s = lambda_best, newx = x_test)
eval_results(y_test, predictions_test, testData)

# K-fold cross-validation K equal to 5

# defining training control
# as cross-validation and
# value of K equal to 5
train_control <- trainControl(method = "cv",
                              number = 5)

# training the model by assigning sales column
# as target variable and rest other column
# as independent variable
model <- train(Sales ~., data = train_store,
               method = "lm",
               trControl = train_control)

# printing model performance metrics
# along with other details
print(model)


# K-fold cross-validation K equal to 10

# defining training control
# as cross-validation and
# value of K equal to 10
train_control <- trainControl(method = "cv",
                              number = 10)

# training the model by assigning sales column
# as target variable and rest other column
# as independent variable
model <- train(Sales ~., data = train_store,
               method = "lm",
               trControl = train_control)

# printing model performance metrics
# along with other details
print(model)





# repeated K-fold cross-validation

# defining training control as
# repeated cross-validation and
# value of K is 10 and repetition is 3 times
train_control <- trainControl(method = "repeatedcv",
                              number = 10, repeats = 3)

# training the model by assigning sales column
# as target variable and rest other column
# as independent variable
model <- train(Sales ~., data = train_store,
               method = "lm",
               trControl = train_control)

# printing model performance metrics
# along with other details
print(model)








