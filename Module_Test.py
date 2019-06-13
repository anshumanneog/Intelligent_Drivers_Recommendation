import sys
sys.path.append('D:\Learnings')

import pandas as pd
from IntelligentDrivers.modules.DecisionTreeVisualization import DecisionTreeVisualization
from IntelligentDrivers.modules.WoEGeneration import WoEGeneration
from sklearn import tree
from IntelligentDrivers.modules import  config


## Decision Tree Object
classifier = tree.DecisionTreeClassifier(criterion=config.DECISIONTREE_CONFIG['criterion'],
              splitter=config.DECISIONTREE_CONFIG['splitter'], 
              max_depth=config.DECISIONTREE_CONFIG['max_depth'], 
              min_samples_split=config.DECISIONTREE_CONFIG['min_samples_split'],
              min_samples_leaf=config.DECISIONTREE_CONFIG['min_samples_leaf'],
              min_weight_fraction_leaf=config.DECISIONTREE_CONFIG['min_weight_fraction_leaf'],
              max_features=config.DECISIONTREE_CONFIG['max_features'], 
              random_state=config.DECISIONTREE_CONFIG['random_state'],
              max_leaf_nodes=config.DECISIONTREE_CONFIG['max_leaf_nodes'],
              min_impurity_decrease=config.DECISIONTREE_CONFIG['min_impurity_decrease'],
              min_impurity_split=config.DECISIONTREE_CONFIG['min_impurity_split'],
              class_weight=config.DECISIONTREE_CONFIG['class_weight'],
              presort=config.DECISIONTREE_CONFIG['presort'])

########## Test 1################### 
df=pd.read_csv('D://Learnings//CODING//Shop_Phase_Project_AD//Shop_Phase_AD.csv')
all_vars = list(df.columns)

## Dependent Variable
DV = 'booked'
# IV
list_cols = ['wkend_flag', 'absence', 'distinct_ratio', 'search_ratio',
       'users_rk', 'search_rk', 'Search_Window', 'sw/bw'] 

## Dependent Variable
data = df
data = data[~data['sw/bw'].isnull()]
data.reset_index(drop = True, inplace= True)

y = data[DV]
check3 = WoEGeneration(data,list_cols,y,10,0.7).Create_WoE() 
df_all = check3[0]

df_all.to_csv('D:/Learnings/CODING/Shop_Phase_Project_AD/EXpedia_woe1.csv' , index = False)

## Visulaize the Decision Tree
xx = DecisionTreeVisualization(data,list_cols,DV,classifier).DecisionTree()
graph = DecisionTreeVisualization(data,list_cols,DV,classifier).TreeGraph()
graph.write_pdf('D:/Learnings/CODING/Shop_Phase_Project_AD/Significant_Variables_EXpedia1.pdf')

 DecisionTreeVisualization(data,list_cols,DV,classifier).tree_to_code()
            
########## Test 2###################
## Read Data
data = pd.read_csv("D:\\BlueDart\\Data_v6.csv")

## Select columns on which Decision Tree has to be trained
list_cols  = ['Capacity',
'Crit',
'Distance',
'Intra/Inter_Inter',
'O_D_city_type',
'O_D',
'Pcs',
'Run Type_SPEED',
'Sch/NSch/Spl',
'Time_Interval']

## Dependent Variable
DV ='Delay_Flag'

y =  data[DV]
check3 = WoEGeneration(data,list_cols,y,10,0.7).Create_WoE() 
df_all = check3[0]
df_all.to_csv('D:/Learnings/CODING/BlueDart_Woe1.csv' , index = False)



## Visulaize the Decision Tree
list_cols2   = ['Capacity',
'Crit_<40%',
'Crit_>70%',
'Distance',
'Imp_CVEHTYPE_M17',
'Imp_Routes_BBL',
'Imp_Routes_Others',
'Imp_Vendors_ABC10',
'Imp_Vendors_ABC60',
'Imp_Vendors_Others',
'Intra/Inter_Inter',
'O_D_city_type_Metro/Metro',
'O_D_city_type_Non-Metro/Non-Metro',
'O_D_SOUTH2-SOUTH2',
'O_D_WEST1-SOUTH2',
'Pcs',
'Run Type_SPEED',
'Sch/NSch/Spl_Sch Run',
'Sch/NSch/Spl_Spl Run',
'Time_Interval_[20, 24)']
graph = DecisionTreeVisualization(data,list_cols2,DV,classifier).TreeGraph()
graph.write_pdf('D:/Learnings/CODING/Significant_Variables_BlueDart1.pdf')

########## Test 3###################
import re  
df = pd.read_csv('D:/Learnings/CODING/HP/SEALS_AD_v9.csv')
all_vars = list(df.columns)
dv_list = list(filter(lambda x: re.findall('_err', x), all_vars))

df_dv = df[dv_list]
df_dv.sum()
# DV
DV = 'Scan_axis_err' ## It has more number of 1's
# IV
list_cols = list(set(all_vars[4:])-set(dv_list[1:]))

## Dependent Variable
data = df
y = data[DV]
#Parameters
#data
#list_cols is list of independent variables
# y - dependent variable
# 10 - number of bins
#  rho - absolute cutoff correlation value for spearman (to select optimum number of bins)
check3 = WoEGeneration(data,list_cols[1:10],y,10,0.8).Create_WoE() 
df_all = check3[0]

df_all.to_csv('D:/Learnings/CODING/HP/HP_woe1.csv' , index = False)

## Visulaize the Decision Tree
graph = DecisionTreeVisualization(data,list_cols,DV,classifier).DecisionTree()

graph.write_pdf('D:/Learnings/CODING/HP/Significant_Variables_check1.pdf')
DecisionTreeVisualization(data,list_cols,DV,classifier).tree_to_code()



