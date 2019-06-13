readme file:
These modules contain techniques to obtain effective predictors for binary classification problem.
The output will be in the form of .csv for tabular output and .pdf for visulaization of decision trees

Usage:
These molues are written in python.Codes can be downloaded and save them in path of the required directory
PIP install version is on the process

Folder Structure

.current path
IntelligentDrivers
	modules
		DecisionTreeVisualization.py
		WoEGeneration.py 
         	init.py 

calling the function: 
from IntelligentDrivers.modules.DecisionTreeVisualization import DecisionTreeVisualization

Objective:

1. Decision Tree visualization.py 
To identify the top variables whihch are having high gini index
To get idea on creation of interaction variables. In some cases, individual variables may not come significant and DT doesn't the effect of one independent variable 
on the other, but the combination of least significant variables may have significant effect of dependent variable
So, we can indentify the interation of these variables and use them in model. This is one of the oppraich that we easily the get the idea on interation just by analysing the decision tree

Inputs will be the training data, Dependent variable, list of independent variables and Decision Tree classifier object with appropriate parameters
Output will the Decision Tree graph and can be saved as PDF format
Note: Parameters to format the Tree are inside the module. Even if we change the parameters of the trees the importance / top variables obtained in the tree graph won't change

2. Weight of Evidence and Infomation Value
Weight of evidence (WOE) is a measure of how much the evidence supports or undermines a hypothesis
Variables having high Information value have significant effect on dependent variable.
We can use WoE to transform a continuous independent variable into a set of groups or bins based on similarity of dependent variable distribution
WoE = ln(probability of event/probability of non-event)
IV = sum[(probability of event-probability of non-event)*WoE]

For categorical variables, it will give WoE and information value for each category whereas for continuous variables it will calculate values for each decile
The WoE recoding of predictors is particularly well suited for subsequent modeling using Logistic Regression.
Inputs will be the training data, Dependent variable, list of independent variables 
Output will be the .csv file with WoE and IV for selected independent variables
Note: Categorical Values should be in form of string type
