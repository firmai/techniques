
# What are common characteristics of employees lost in attrition compared to those who stay in IBM's fictional dataset? 
## We will be using point plots, box plots, kernel density diagrams, means, standard deviations, and z-tests to explore this question.

----------


## Set Up Dataset


```python
from pandas import read_csv
data = read_csv("data/attrition.csv")
```


```python
target = "Attrition"
```


```python
feature_by_dtype = {}
for c in data.columns:
    
    if c == target: continue
    
    data_type = str(data[c].dtype)
    
    if data_type not in feature_by_dtype.keys():
         feature_by_dtype[data_type] = [c]
    else:
        feature_by_dtype[data_type].append(c)

feature_by_dtype
feature_by_dtype.keys()
```




    dict_keys(['int64', 'object'])




```python
objects = feature_by_dtype["object"]
```


```python
remove = ["Over18"]
```


```python
import pandas as pd

```


```python
pd.options.display.max_columns = None
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobRole</th>
      <th>JobSatisfaction</th>
      <th>MaritalStatus</th>
      <th>MonthlyIncome</th>
      <th>MonthlyRate</th>
      <th>NumCompaniesWorked</th>
      <th>Over18</th>
      <th>OverTime</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>Female</td>
      <td>94</td>
      <td>3</td>
      <td>2</td>
      <td>Sales Executive</td>
      <td>4</td>
      <td>Single</td>
      <td>5993</td>
      <td>19479</td>
      <td>8</td>
      <td>Y</td>
      <td>Yes</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>Male</td>
      <td>61</td>
      <td>2</td>
      <td>2</td>
      <td>Research Scientist</td>
      <td>2</td>
      <td>Married</td>
      <td>5130</td>
      <td>24907</td>
      <td>1</td>
      <td>Y</td>
      <td>No</td>
      <td>23</td>
      <td>4</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>Male</td>
      <td>92</td>
      <td>2</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>3</td>
      <td>Single</td>
      <td>2090</td>
      <td>2396</td>
      <td>6</td>
      <td>Y</td>
      <td>Yes</td>
      <td>15</td>
      <td>3</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>Female</td>
      <td>56</td>
      <td>3</td>
      <td>1</td>
      <td>Research Scientist</td>
      <td>3</td>
      <td>Married</td>
      <td>2909</td>
      <td>23159</td>
      <td>1</td>
      <td>Y</td>
      <td>Yes</td>
      <td>11</td>
      <td>3</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>Male</td>
      <td>40</td>
      <td>3</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>2</td>
      <td>Married</td>
      <td>3468</td>
      <td>16632</td>
      <td>9</td>
      <td>Y</td>
      <td>No</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
categorical_features = [f for f in objects if f not in remove]
```


```python
int64s = feature_by_dtype["int64"]
## handeling feature types in dictionary
```


```python
remove.append("StandardHours")
remove.append("EmployeeCount")
```


```python
count_features = []
for i in [i for i in int64s if len(data[i].unique()) < 20 and i not in remove]:
    count_features.append(i)
```


```python
count_features = count_features #+ ["TotalWorkingYears", "YearsAtCompany", "HourlyRate"]
```


```python
remove.append("EmployeeNumber")
```


```python
numerical_features = [i for i in int64s if i not in remove]
```

----------


# Numerical Features


```python
data[numerical_features].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobSatisfaction</th>
      <th>MonthlyIncome</th>
      <th>MonthlyRate</th>
      <th>NumCompaniesWorked</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>1102</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>94</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>5993</td>
      <td>19479</td>
      <td>8</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>279</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>61</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5130</td>
      <td>24907</td>
      <td>1</td>
      <td>23</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1373</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>92</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>2090</td>
      <td>2396</td>
      <td>6</td>
      <td>15</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>1392</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>56</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2909</td>
      <td>23159</td>
      <td>1</td>
      <td>11</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>591</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>40</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>3468</td>
      <td>16632</td>
      <td>9</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



----------

# Python Source Code


```python
def display_ttest(data, category, numeric):
    output = {}
    s1 = data[data[category] == data[category].unique()[0]][numeric]
    s2 = data[data[category] == data[category].unique()[1]][numeric]
    from scipy.stats import ttest_ind
    t, p = ttest_ind(s1,s2)
    from IPython.display import display
    from pandas import DataFrame
    display(DataFrame(data=[{"t-test statistic" : t, "p-value" : p}], columns=["t-test statistic", "p-value"], index=[category]).round(2))

def display_ztest(data, category, numeric):
    output = {}
    s1 = data[data[category] == data[category].unique()[0]][numeric]
    s2 = data[data[category] == data[category].unique()[1]][numeric]
    from statsmodels.stats.weightstats import ztest
    z, p = ztest(s1,s2)
    from IPython.display import display
    from pandas import DataFrame
    display(DataFrame(data=[{"z-test statistic" : z, "p-value" : p}], columns=["z-test statistic", "p-value"], index=[category]).round(2))
    
def display_cxn_analysis(data, category, numeric, target):
    
    from seaborn import boxplot, kdeplot, set_style, distplot, countplot
    from matplotlib.pyplot import show, figure, subplots, ylabel, xlabel, subplot, suptitle
    
    not_target = [a for a in data[category].unique() if a != target][0]
    
    pal = {target : "yellow",
          not_target : "darkgrey"}
    

    set_style("whitegrid")
    figure(figsize=(12,5))
    suptitle(numeric + " by " + category)

    # ==============================================
    
    p1 = subplot(2,2,2)
    boxplot(y=category, x=numeric, data=data, orient="h", palette = pal)
    p1.get_xaxis().set_visible(False)

    # ==============================================
    
    if(numeric in count_features):
        p2 = subplot(2,2,4)
        
        s2 = data[data[category] == not_target][numeric]
        s2 = s2.rename(not_target) 
        countplot(s2, color = pal[not_target])
        
        s1 = data[data[category] == target][numeric]
        s1 = s1.rename(target)
        ax = countplot(s1, color = pal[target])
        
        ax.set_yticklabels([ "{:.0f}%".format((tick/len(data)) * 100) for tick in ax.get_yticks()])
        
        ax.set_ylabel("Percentage")
        ax.set_xlabel(numeric)
        
    else:
        p2 = subplot(2,2,4, sharex=p1)
        s1 = data[data[category] == target][numeric]
        s1 = s1.rename(target)
        kdeplot(s1, shade=True, color = pal[target])
        #distplot(s1,kde=False,color = pal[target])

        s2 = data[data[category] == not_target][numeric]
        s2 = s2.rename(not_target)  
        kdeplot(s2, shade=True, color = pal[not_target])
        #distplot(s2,kde=False,color = pal[not_target])

        #ylabel("Density Function")
        ylabel("Distribution Plot")
        xlabel(numeric)
    
    # ==============================================
    
    p3 = subplot(1,2,1)
    from seaborn import pointplot
    from matplotlib.pyplot import rc_context

    with rc_context({'lines.linewidth': 0.8}):
        pp = pointplot(x=category, y=numeric, data=data, capsize=.1, color="black", marker="s")
        
    
    # ==============================================
    
    show()
    
    #display p value
    
    if(data[category].value_counts()[0] > 30 and data[category].value_counts()[1] > 30):
        display_ztest(data,category,numeric)
    else:
        display_ttest(data,category,numeric)
    
    #Means, Standard Deviation, Absolute Distance
    table = data[[category,numeric]]
    
    means = table.groupby(category).mean()
    stds = table.groupby(category).std()
    
    s1_mean = means.loc[data[category].unique()[0]]
    s1_std = stds.loc[data[category].unique()[0]]
    
    s2_mean = means.loc[data[category].unique()[1]]
    s2_std = means.loc[data[category].unique()[1]]
    
    print("%s Mean: %.2f (+/- %.2f)" % (category + " == " + str(data[category].unique()[0]),s1_mean, s1_std))
    print("%s Mean : %.2f (+/- %.2f)" % (category + " == " + str(data[category].unique()[1]), s2_mean, s2_std))
    print("Absolute Mean Diferrence Distance: %.2f" % abs(s1_mean - s2_mean))
```


```python
def get_p_value(s1,s2):
    
    from statsmodels.stats.weightstats import ztest
    from scipy.stats import ttest_ind
    
    if(len(s1) > 30 & len(s2) > 30):
        z, p = ztest(s1,s2)
        return p
    else:
        t, p = ttest_ind(s1,s2)
        return p
    
def get_p_values(data, category, numerics):
    
    output = {}
    
    for numeric in numerics:
        s1 = data[data[category] == data[category].unique()[0]][numeric]
        s2 = data[data[category] == data[category].unique()[1]][numeric]
        row = {"p-value" : get_p_value(s1,s2)}
        output[numeric] = row
    
    from pandas import DataFrame
    
    return DataFrame(data=output).T

def get_statistically_significant_numerics(data, category, numerics):
    df = get_p_values(data, category, numerics)
    return list(df[df["p-value"] < 0.05].index)

def get_statistically_non_significant_numerics(data, category, numerics):
    df = get_p_values(data, category, numerics)
    return list(df[df["p-value"] >= 0.05].index)
    
def display_p_values(data, category, numerics):
    from IPython.display import display
    display(get_p_values(data, category, numerics).round(2).sort_values("p-value", ascending=False))
```


```python
### TESTING

#Well this simply sees if there is a statistical difference between numeric feature's..
# .. distribution between the two attrition classes (Yes and No)
output = {}

for numeric in numerical_features:
    s1 = data[data[target] == data[target].unique()[0]][numeric]
    s2 = data[data[target] == data[target].unique()[1]][numeric]
    
    from statsmodels.stats.weightstats import ztest
    from scipy.stats import ttest_ind
    
    if(len(s1) > 30 & len(s2) > 30):
        # for this task this is always going to be true 
        # test for mean based on normal distribution, the samples are assumed
        # to be independent.
        z, p = ztest(s1,s2)
    else:
        t, p = ttest_ind(s1,s2)
    
    row = {"p-value" : p}
    output[numeric] = row

df = pd.DataFrame(data=output).T
df_sig = df[df["p-value"] < 0.05]

### TEST PASSED
```


```python
significant = get_statistically_significant_numerics(data,target,numerical_features) 
ns = get_statistically_non_significant_numerics(data,target,numerical_features)
```

----------

# Statistically Significant Numerical Features


```python
i = iter(significant)
```

## The fictional company on average loses staff that are 3 - 4 years younger than those who stay.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_28_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-6.18</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 33.61 (+/- 9.69)
    Attrition == No Mean : 37.56 (+/- 37.56)
    Absolute Mean Diferrence Distance: 3.95


## Employees lost in attrition tend to have lower daily rates than those who stay.
 - Each of the group are 180 degrees flipped from each other in their kernel density diagram


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_30_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-2.17</td>
      <td>0.03</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 750.36 (+/- 401.90)
    Attrition == No Mean : 812.50 (+/- 812.50)
    Absolute Mean Diferrence Distance: 62.14


## Employees lost in attrition tend to have longer commute distances than those who stay.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_32_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>2.99</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 10.63 (+/- 8.45)
    Attrition == No Mean : 8.92 (+/- 8.92)
    Absolute Mean Diferrence Distance: 1.72


# Employees lost in attrition are less satisfied with their work environment on average than those who stay.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_34_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-3.98</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 2.46 (+/- 1.17)
    Attrition == No Mean : 2.77 (+/- 2.77)
    Absolute Mean Diferrence Distance: 0.31


## Employees lost in attrition are less involved with their jobs on average than those who stay.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_36_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-5.02</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 2.52 (+/- 0.77)
    Attrition == No Mean : 2.77 (+/- 2.77)
    Absolute Mean Diferrence Distance: 0.25


## Employees lost in attrition tend to be lower in job level than those who stay.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_38_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-6.57</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 1.64 (+/- 0.94)
    Attrition == No Mean : 2.15 (+/- 2.15)
    Absolute Mean Diferrence Distance: 0.51


## Employees who stay have more job satisfication than employees lost in attrition


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_40_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-3.99</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 2.47 (+/- 1.12)
    Attrition == No Mean : 2.78 (+/- 2.78)
    Absolute Mean Diferrence Distance: 0.31


## Employees lost in attrition tend to have lower monthly average income on average than those who stay.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_42_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-6.2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 4787.09 (+/- 3640.21)
    Attrition == No Mean : 6832.74 (+/- 6832.74)
    Absolute Mean Diferrence Distance: 2045.65


## Employees who stay tend to have more stock options than those lost in attrition.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_44_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-5.3</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 0.53 (+/- 0.86)
    Attrition == No Mean : 0.85 (+/- 0.85)
    Absolute Mean Diferrence Distance: 0.32


## Employees lost in attrition had less total working years than those who stay.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_46_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-6.65</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 8.24 (+/- 7.17)
    Attrition == No Mean : 11.86 (+/- 11.86)
    Absolute Mean Diferrence Distance: 3.62


## Employees lost in attrition had less training opportunities than those who stay.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_48_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-2.28</td>
      <td>0.02</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 2.62 (+/- 1.25)
    Attrition == No Mean : 2.83 (+/- 2.83)
    Absolute Mean Diferrence Distance: 0.21


## Employees lost in attrition had poorer work-life balance on average than those who stay.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_50_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-2.45</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 2.66 (+/- 0.82)
    Attrition == No Mean : 2.78 (+/- 2.78)
    Absolute Mean Diferrence Distance: 0.12


## Employees who stay had longer organization tenure than those lost in attrition by 2 years on average.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_52_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-5.2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 5.13 (+/- 5.95)
    Attrition == No Mean : 7.37 (+/- 7.37)
    Absolute Mean Diferrence Distance: 2.24


## Employees who stayed had 1 - 2 more years in their current role than those lost in attrition.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_54_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-6.23</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 2.90 (+/- 3.17)
    Attrition == No Mean : 4.48 (+/- 4.48)
    Absolute Mean Diferrence Distance: 1.58


## Employees lost in attrition had less time with their current manager by 1 - 2 years on average than those who stay.


```python
display_cxn_analysis(data, target, next(i), "Yes")
```


![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_56_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>z-test statistic</th>
      <th>p-value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Attrition</th>
      <td>-6.06</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Attrition == Yes Mean: 2.85 (+/- 3.14)
    Attrition == No Mean : 4.37 (+/- 4.37)
    Absolute Mean Diferrence Distance: 1.52


## Employees who stay are more satisfied with their work environment on average than those who leave.

----------
# Non-Significant Features


```python
ns
```




    ['Education',
     'HourlyRate',
     'MonthlyRate',
     'NumCompaniesWorked',
     'PercentSalaryHike',
     'PerformanceRating',
     'RelationshipSatisfaction',
     'YearsSinceLastPromotion']




```python
### Some Additional Visualisations
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

dataset = data

# Define a set of graphs, 3 by 5, usin the matplotlib library
f, axes = plt.subplots(5, 3, figsize=(24, 36), sharex=False, sharey=False)

# Define a few seaborn graphs, which for the most part only need the "dataset", the "x and "y" axis and the position. 
# You can also show a third value and expand your analysis by setting the "hue" property.
sns.swarmplot(x="EducationField", y="MonthlyIncome", data=dataset, hue="Gender", ax=axes[0,0])
axes[0,0].set( title = 'Monthly income against Educational Field')

sns.pointplot(x="PerformanceRating", y="JobSatisfaction", data=dataset, hue="Gender", ax=axes[0,1])
axes[0,1].set( title = 'Job satisfaction against Performance Rating')

sns.barplot(x="NumCompaniesWorked", y="PerformanceRating", data=dataset, ax=axes[0,2])
axes[0,2].set( title = 'Number of companies worked against Performance rating')

sns.barplot(x="JobSatisfaction", y="EducationField", data=dataset, ax=axes[1,0])
axes[1,0].set( title = 'Educational Field against Job Satisfaction')

sns.barplot(x="YearsWithCurrManager", y="JobSatisfaction", data=dataset, ax=axes[1,1])
axes[1,1].set( title = 'Years with current Manager against Job Satisfaction')

sns.pointplot(x="JobSatisfaction", y="MonthlyRate", data=dataset, ax=axes[1,2])
axes[1,2].set( title = 'Job Satisfaction against Monthly rate')

sns.barplot(x="WorkLifeBalance", y="DistanceFromHome", data=dataset, ax=axes[2,0])
axes[2,0].set( title = 'Distance from home against Work life balance')

sns.pointplot(x="OverTime", y="WorkLifeBalance", hue="Gender", data=dataset, jitter=True, ax=axes[2,1])
axes[2,1].set( title = 'Work life balance against Overtime')

sns.pointplot(x="OverTime", y="RelationshipSatisfaction", hue="Gender", data=dataset, ax=axes[2,2])
axes[2,2].set( title = 'Overtime against Relationship satisfaction')

sns.pointplot(x="MaritalStatus", y="YearsInCurrentRole", hue="Gender", data=dataset, ax=axes[3,0])
axes[3,0].set( title = 'Marital Status against Years in current role')

sns.pointplot(x="Age", y="YearsSinceLastPromotion", hue="Gender", data=dataset, ax=axes[3,1])
axes[3,1].set( title = 'Age against Years since last promotion')

sns.pointplot(x="OverTime", y="PerformanceRating", hue="Gender", data=dataset, ax=axes[3,2])
axes[3,2].set( title = 'Performance Rating against Overtime')

sns.barplot(x="Gender", y="PerformanceRating", data=dataset, ax=axes[4,0])
axes[4,0].set( title = 'Performance Rating against Gender')

sns.barplot(x="Gender", y="JobSatisfaction", data=dataset, ax=axes[4,1])
axes[4,1].set( title = 'Job satisfaction against Gender')

sns.countplot(x="Attrition", data=dataset, ax=axes[4,2])
axes[4,2].set( title = 'Attrition distribution')
```




    [<matplotlib.text.Text at 0x10b8ab9e8>]




![png](Attrition%20Distribution%20Stats_files/Attrition%20Distribution%20Stats_61_1.png)

