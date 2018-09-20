# quartic
#Data Scientist Challenge

1.CONCEPTUAL APPROACH:
  At first, the target variable is dichotomous and there is no description about the variables and also there are more NaN values in the data set.So ,for identifing the correlation between the variables and filling the NaN's with sutiable values , I had choosen the Logistic Regression model.Scine the model is unbiased, and also with the help of sensitivity ,specificity, ROC curve and Confusion matrix ,we can validate and do futher process for imporving reliability.
  
TRADE-OFF:
  sensitivity-Specificity trade off:
    Even though the accuracy of the model is important,the reliability is more important than that.If we train the model and predict the validation set with normal threshold(0.5) the sensitivity was 1 and the specificity was 0. That is they are inversely propotional.so that, I find the optimal thershold value and I fix with nice mix of sensitivity and specificity values.
    
2.MODEL PERFORMANCE:
   Model performance of this model is good on comparing with other models like Xgboost and LightGBM.
   Model evaluation was done using Receiver Operating Characteristic(ROC) curve,confusion matrix and classifiction report(sensitivity and specificity). Model Performance is according to thid data, because the distribution of data for each class(0,1) is not equal. If the description of the data is there ,then we will decide to go with either sensitivity or specificity.But there is no description. Inspite of the accuracy is not high,there is  a nice mix of sensitivity and specificity values.
   For the data having almost equal distribution of data in each class,this model will definitely leads to good accuracy.
  
COMPLEXITY:
    The data is not that much big, so there is no memory problem.
    The insufficient size of data in each class,increase the complexity of the problem.(you can see the Visualisation.png ). There is only 3.6% of data for class 1 and 96.4% of data for class 2 in the training test. Even for the machine,it is some what complex to understand the characteristics of the classes to fit the accurate curve for the data.
    
BOTTLE NECK:
    Usually ,Heatmap will guide us to select the features , but in this data(see heatmap.png) ,it is the major bottle neck for me.None of the variables are highly correlated to the target. So, It is very hard to go further for feature selection.
     
3.IMPROVISATIONS:
    (i) I will study about the relation between each variable and target . so, that we can do feature selection and drop the variables with less Coefficient in the decision function and it will reduce the memory space for the data.
    (ii)I will find the variables that are contributing more towards the class 1 and by giving more weight to these to increase  Reliability and Accuary together.
