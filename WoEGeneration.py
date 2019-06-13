# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 19:20:36 2018

@author: Rayudu Pujitha
"""



import numpy as np
import pandas as pd
import traceback
import re
import pandas.core.algorithms as algos
from  scipy.stats import spearmanr

class WoEGeneration: 
    """
    This WoEGeneration contains the function to create Weight of Evidence and 
    Information Value for a given data.
    """
    def __init__(self,data,selected_var,y,max_bin,rho):
        '''Initialization
        :param data:  Training Data
        :param selected_var: List of independent variables
        :param y: Dependent variable as list
        :return: None
        '''
        self.data = data
        self.selected_var = selected_var
        self.y = y
        self.max_bin = max_bin
        self.rho = rho
        self.force_bins = 3
        
   
   
    
    def woe_single_x(self,X, Y,rho):
        '''
        :For numerical variables function **woe_single_x** calculates the WoE and IV value for each of the bin and stores them in iv_list
        :param x: 1-D numpy stands for single feature
        :param y: 1-D numpy array target variable
        :param event: value of binary stands for the event to predict
        :return: dictionary contains woe values for categories of this feature
                 information value of this feature
        '''
        df1 = pd.DataFrame({"X": X, "Y": Y})
        justmiss = df1[['X','Y']][df1.X.isnull()]
        notmiss = df1[['X','Y']][df1.X.notnull()]
        n=self.max_bin
        r = 0
        while np.abs(r) < rho and n >0:
            x_labels,bins = pd.qcut(X.drop_duplicates(),n,labels = False,retbins=True)
            bins2 = [-np.inf]+list(bins[1:-1])+ [np.inf]
            
            names = list(range(len(bins[0:-1])))


            d1 = pd.DataFrame({'X': X,'Y':Y})
            d1['Bucket'] = pd.cut(d1['X'], bins2, labels=names)

            d2 = d1.groupby('Bucket', as_index=True)
            r, p = spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
            print(n)
                
        force_bin = self.force_bins
        if len(d2) == 1:
            n = force_bin         
            bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
            if len(np.unique(bins)) == 2:
                bins = np.insert(bins, 0, 1)
                bins[1] = bins[1]-(bins[1]/2)
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
            d2 = d1.groupby('Bucket', as_index=True)
        
        d3 = pd.DataFrame({},index=[])
        d3["MIN_VALUE"] = d2.min().X
        d3["MAX_VALUE"] = d2.max().X
        d3["COUNT"] = d2.count().Y
        d3["EVENT"] = d2.sum().Y
        d3["NONEVENT"] = d2.count().Y - d2.sum().Y
        d3=d3.reset_index(drop=True)
        
        if len(justmiss.index) > 0:
            d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
            d4["MAX_VALUE"] = np.nan
            d4["COUNT"] = justmiss.count().Y
            d4["EVENT"] = justmiss.sum().Y
            d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
            d3 = d3.append(d4,ignore_index=True)
        
        d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
        d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
        d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
        d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
        d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
        d3["IV_BUCKET"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
        d3["VAR_NAME"] = "VAR"
        d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV_BUCKET']]       
        d3 = d3.replace([np.inf, -np.inf], 0)
        
        return(d3)
    
    
    def woe_cat_x(self,X, Y):
        '''
        Function for Categorical variables  **woe_cat_x** which calculates the IV value for each type of the categorical variable. **woe_dict** stores the WOE values and **iv_dict** stores the iv values for each type of the categorical variable
        :param x: 1-D numpy stands for single feature
        :param y: 1-D numpy array target variable
        :param event: value of binary stands for the event to predict
        :return: dictionary contains woe values for categories of this feature
                 information value of this feature
        '''
        df1 = pd.DataFrame({"X": X, "Y": Y})
        justmiss = df1[['X','Y']][df1.X.isnull()]
        notmiss = df1[['X','Y']][df1.X.notnull()]    
        df2 = notmiss.groupby('X',as_index=True)
        
        d3 = pd.DataFrame({},index=[])
        d3["COUNT"] = df2.count().Y
        d3["MIN_VALUE"] = df2.sum().Y.index
        d3["MAX_VALUE"] = d3["MIN_VALUE"]
        d3["EVENT"] = df2.sum().Y
        d3["NONEVENT"] = df2.count().Y - df2.sum().Y
        
        if len(justmiss.index) > 0:
            d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
            d4["MAX_VALUE"] = np.nan
            d4["COUNT"] = justmiss.count().Y
            d4["EVENT"] = justmiss.sum().Y
            d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
            d3 = d3.append(d4,ignore_index=True)
        
        d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
        d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
        d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
        d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
        d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
        d3["IV_BUCKET"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
        d3["VAR_NAME"] = "VAR"
        d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV_BUCKET']]      
        d3 = d3.replace([np.inf, -np.inf], 0)
        d3 = d3.reset_index(drop=True)
        
        return(d3)
    
        
    # The **Adult Dataset** data file is read as pandas dataframe and depending on type of variable i.e. Categrical or numeric, the values are appended and stored into final **IV result** csv file
    
     
    def Create_WoE(self):
        data = self.data 
        x = self.selected_var 
        y = self.y
        rho = self.rho
        stack = traceback.extract_stack()
        filename, lineno, function_name, code = stack[-2]
        vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
        final = (re.findall(r"[\w']+", vars_name))[-1]
        count = -1
        
        for i in x:
            if i.upper() not in (final.upper()):
                if np.issubdtype(data[i], np.number) and len(pd.Series.unique(data[i])) > 2:
                    conv = self.woe_single_x(data[i],y,rho)
                    conv["VAR_NAME"] = i
                    count = count + 1
                else:
                    conv = self.woe_cat_x(data[i],y)
                    conv["VAR_NAME"] = i            
                    count = count + 1
                    
                if count == 0:
                    iv_df = conv
                else:
                    iv_df = iv_df.append(conv,ignore_index=True)
        
        iv_df['IV'] = iv_df.groupby('VAR_NAME')['IV_BUCKET'].transform('sum') 
        iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
        iv = iv.reset_index()

        return(iv_df,iv)
        


