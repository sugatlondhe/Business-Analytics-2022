#!/usr/bin/env python
# coding: utf-8

# # Name : Sugat Londhe - 220185538

# # EDA PROJECT

# # Introduction
# In the context of college data, exploratory analysis involves examining various aspects of the data such as the distribution of enrollment numbers,application numbers, student demographics, or academic performance metrics. Exploratory analysis can be a useful tool for identifying patterns and trends in the data that may not be immediately apparent, and for generating hypotheses about the underlying causes of these patterns.

# Importing essential libraries for the EDA

# In[11]:


pip install statsmodels


# In[5]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from patsy import dmatrices
import seaborn as sb
import io
#word count
from nbformat import read, NO_CONVERT


# Importing the college excel file 'College.xlsx' & displaying last 5 rows

# In[6]:


df=pd.read_excel("College regression.xlsx")
df.tail()


# In[7]:


df.rename(columns = {"S.F.Ratio":"SF_Ratio"}, inplace=True)
df.rename(columns = {"Grad.Rate":"Grad_Rate"}, inplace=True)
df.columns


# # Q1) Generating Descriptive statistics for the dataset

# In[8]:


#Using describe function
df.describe()


# To print the inforation about the dataframes containing number of columns, column labels, column data types, memory usage, range index, and the number of cells in each columns(non-null values)

# In[9]:


df.info()


# In[10]:


df.rename(columns = {"S.F.Ratio":"SF_Ratio"}, inplace=True)
df.rename(columns = {"Grad.Rate":"Grad_Rate"}, inplace=True)
df.columns


# In order to visualise multiple pairwise bivariate distributions in a dataset,

# Plot numerical data as pairs

# In[11]:


sb.pairplot(df);


# Investigating any null-values from the data set

# In[188]:


df.isnull().values.any()


# From the above, we got to know that there are null-values, so we'll check total number of null-values.

# In[189]:


#Summing up all the null values from all columns
df.isnull().sum()


# In[190]:


#Total number of null values
df.isnull().sum().sum()


# # Q2) Rectifying any  null-values from the dataset

# Create new dataframe by filling all the null values with the median values in order to, avoid outliers.

# In[191]:


correct_df = df
correct_df['Apps']=correct_df['Apps'].fillna(correct_df['Apps'].median())
df["Apps"].isnull().sum()


# Filling the null-values with Median values using .fillna function 

# In[192]:


correct_df['Accept']=correct_df['Accept'].fillna(correct_df['Accept'].median())
correct_df['Enroll']=correct_df['Enroll'].fillna(correct_df['Enroll'].median())
correct_df['Students']=correct_df['Students'].fillna(correct_df['Students'].median())
correct_df['SF_Ratio']=correct_df['SF_Ratio'].fillna(correct_df['SF_Ratio'].median())
correct_df['Expend']=correct_df['Expend'].fillna(correct_df['Expend'].median())
correct_df['Grad_Rate']=correct_df['Grad_Rate'].fillna(correct_df['Grad_Rate'].median())


# After, summing up all the null values with median values, remaining null-values are:

# In[196]:


df.isnull().sum().sum() 


# 'Keene State College' being the only Public college from the "Private" null-value column & discarding all null-values!

# In[194]:


pk = df[df["Institution"]=="Keene State College"]
pk


# Dropping the 'Keene State College' row being the only 'Public college' in private column

# In[109]:


df.drop(290,inplace=True, axis=0)


# Replacing null values in "Private" column with "Yes" as they're 'Private' instuitions

# In[110]:


df["Private"].fillna("Yes", inplace=True)


# Now, dropping the remaing null-values from the 'PhD' column

# In[111]:


df.dropna(inplace=True, axis=0)


# Data now is cleaned with no null-values

# In[112]:


df.isnull().sum()


# In[113]:


#Checking their number of rows & columns now.
df.shape


# # Question no.3) Illustrating visualizations for the different data cloumns

# Plotting histogram to check student distribution,

# from the visualisation we infered that from the intervals 0 to 38000, the student value increases from 0-5000
# but after 5000, gradually the value starts decreasing over the period of time till 40000

# In[114]:


sb.displot(x="Students", data=df, bins=9)


# Plotting histogram to check S.F.Ratio distribution,
# 

# From the visualization we can infere that,

# values between 9 to 15, the S.F ratio values are at its peak, then it starts decreasing gradually over period of time till 40

# In[195]:


sb.displot(x="SF_Ratio", data=df, bins=6)


# Plotting scatter plot, for the Y-axis is S.F.Ratio & X-axis is Students

# After visualizing, we can infere that

# There is positive linear relationship between number of applications being accepted & number of applications received by the university

# Therefore, we can say that, as the number of university applications increases, number of acceptance rates increases accordingly.

# In[116]:


sb.scatterplot(x="Accept", y= "Apps", data=df)
plt.xlabel=("Total number of acceptance rate")
plt.ylabel=("Total number Applications")
plt.title("Acceptance v/s Number of Applications")


# Plotting Barplot, in order to check relationship between one categorical variable & continuous variable

# In[130]:


plt.figure(figsize=(7,5))
sb.barplot(data = df, x = "Private", y = "Apps")
plt.show()


# From the visualization, we can infere that, more number of applications are received for the Public univeristy than the less number of Private university.

# # Q4) Displaying the unique values of categorical variables of Private columns.

# In[9]:


#Displaying unique values using .unique function
df["Private"].unique()


# To know about the frequency of two values and allocating them according their values.

# In[151]:


df["Private"].value_counts().to_frame()


# # Q5) To build contigency table, we use .crosstab function for two categorial tables

# In[152]:


#Building a contigency table of the two categorial tables
chi_test = pd.crosstab(df["Private"],df["PhD"])
chi_test


# Visualizing contigency table using stacked bars

# In[157]:


chi_test.plot(kind="bar", stacked=True, rot=0)


# Inspecting differences between two categorical variables of p-values, then comparing it with the significance level

# In[154]:


chi2, p_val, dof, expected = stats.chi2_contingency(chi_test)
print(f"pvalue: {p_val}")


# From the above data we can infer that, after procuring the p-value, the significance level value is greater than P-value.
# Accordingly, we have to reject Null Hypothesis & accept Alternate Hypothesis.
# By accepting alternate hypothesis, we can say that there is dependency between Private & PhD.

# # Q6) Retriving subsets of rows from the "Institution" using new_df as the new dataframe from the dataset

# In[197]:


new_df = df[(df["Private"]=="No") & (df["Grad_Rate"]<80) & (df["Students"]>1000) & (df["Apps"]<1000)]
new_df


# Therefore, we created new variable were we stored all the subsets.

# Generating descriptive statistics for the new dataframe created.

# In[198]:


new_df.describe()


# # Q7) Creating two new subsets eprivate & nopvt for "Private" & "Enroll" columns and calculating mean individually, followed independent Sample T-test 

# Formulating their mean values now

# In[3]:


eprivate = df[df["Private"]== "Yes"]["Enroll"]
eprivate.mean()


# In[4]:


noprivate = df[df["Private"]== "No"]["Enroll"]
noprivate.mean()


# Examining independent sample T-test

# In[201]:


t_val, p_val = stats.ttest_ind(epvt, npvt)
print(f"T-value: {t_val}, p_value: {p_val}")


# After running Independent sample T-test & procuring P-values, we can infere that we're rejecting Null-hypothesis after seeing significant difference between mean values of two subsets because P-values < 0.05

# # Question no.8) Segregating two specific columns of categorical variables Private & Enroll & displaying summarized information using .groupby function

# In[202]:


df6= df.groupby(["Private","Enroll"]).sum()
df6


# In[165]:


df6.mean()


# # Q9) In order to find out linearity by predicting linear relationship between depenedent and independent variables.

# In[219]:


sb.pairplot(df);


# In[221]:


df["Enroll"].corr(df["Grad_Rate"],method='pearson')


# In[207]:


df["Expend"].corr(df["Grad_Rate"],method='pearson')


# In[208]:


df["Apps"].corr(df["Grad_Rate"],method='pearson')


# In[220]:


df["SF_Ratio"].corr(df["Grad_Rate"],method='pearson')


# In[222]:


df["Students"].corr(df["Grad_Rate"],method='pearson')


# In[218]:


#Multi-collinearity
model=sm.OLS.from_formula("Grad_Rate ~ Students + Expend + Apps + Enroll + SF_Ratio", data=df).fit()
model.summary()


# After examining and analyzing linear regression models with different variables, we can say that model is parsimonous following normality, linearity, indpendence of errors and Homoscedasticity.

# We can infer that, that the graduation rate is dependent on the other variables and has a substantial impact on the following predcition using 'College.xlsx' file.
# Thus, after iterations,the adjusted R value is 0.242 which is dependent on Applications, Enroll, Students, SF_Ratio, Expend is directly proportional to the positive substantial impact. 
# An adjusted R value of 0.242 means that the model explains about 24.2% of the variability in the dependent variable. This may be considered a relatively low fit, depending on the specific context and the expected accuracy of the model.

# In[11]:


get_ipython().run_cell_magic('javascript', '', 'var nb = IPython.notebook;\nvar kernel = IPython.notebook.kernel;\nvar command = "NOTEBOOK_FULL_PATH = \'" + nb.notebook_path + "\'";\nkernel.execute(command);')


# In[12]:


with io.open(NOTEBOOK_FULL_PATH.split("/")[-1], 'r', encoding='utf-8') as f:
    nb = read(f, NO_CONVERT)

word_count = 0
for cell in nb.cells:
    if cell.cell_type == "markdown":
        word_count += len(cell['source'].replace('#', '').lstrip().split(' '))
print(f"Word count: {word_count}")

