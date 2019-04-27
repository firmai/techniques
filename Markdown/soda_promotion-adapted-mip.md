

```python
import pandas
df_hist = pandas.read_excel("soda_sales_historical_data.xlsx")
df_hist[:5]

df_hist[df_hist["Product"]=="11 Down"]

from pandas import DataFrame, get_dummies
categorical_columns = ['Product','Easter Included','Super Bowl Included', 
                       'Christmas Included', 'Other Holiday']
df_hist = get_dummies(df_hist, prefix={k:"dmy_%s"%k for k in categorical_columns},
                      columns = list(categorical_columns))
df_hist[:5]
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
      <th>Sales</th>
      <th>Cost Per Unit</th>
      <th>4 Wk Avg Temp</th>
      <th>4 Wk Avg Humidity</th>
      <th>Sales M-1 weeks</th>
      <th>Sales M-2 weeks</th>
      <th>Sales M-3 weeks</th>
      <th>Sales M-4 Weeks</th>
      <th>Sales M-5 weeks</th>
      <th>dmy_Product_11 Down</th>
      <th>...</th>
      <th>dmy_Product_Koala Kola</th>
      <th>dmy_Product_Mr. Popper</th>
      <th>dmy_Product_Popsi Kola</th>
      <th>dmy_Easter Included_No</th>
      <th>dmy_Easter Included_Yes</th>
      <th>dmy_Super Bowl Included_No</th>
      <th>dmy_Super Bowl Included_Yes</th>
      <th>dmy_Christmas Included_No</th>
      <th>dmy_Christmas Included_Yes</th>
      <th>dmy_Other Holiday_No</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>51.9</td>
      <td>1.6625</td>
      <td>80.69</td>
      <td>69.19</td>
      <td>17.0</td>
      <td>22.4</td>
      <td>13.5</td>
      <td>14.5</td>
      <td>28.0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.8</td>
      <td>2.2725</td>
      <td>80.69</td>
      <td>69.19</td>
      <td>2.4</td>
      <td>2.2</td>
      <td>2.0</td>
      <td>1.4</td>
      <td>0.5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3385.6</td>
      <td>1.3475</td>
      <td>80.69</td>
      <td>69.19</td>
      <td>301.8</td>
      <td>188.8</td>
      <td>101.4</td>
      <td>81.6</td>
      <td>213.8</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>63.5</td>
      <td>1.6600</td>
      <td>80.69</td>
      <td>69.19</td>
      <td>73.8</td>
      <td>69.4</td>
      <td>72.8</td>
      <td>75.4</td>
      <td>57.4</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>181.1</td>
      <td>1.8725</td>
      <td>80.69</td>
      <td>69.19</td>
      <td>23.1</td>
      <td>22.6</td>
      <td>22.1</td>
      <td>19.9</td>
      <td>23.2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn import model_selection
experiments = {"Algorithm":["Ordinary Least Squares", "Regression Tree", 
                            "Big Random Forest", "Random Forest", 
                            "Bagging"], 
               "Objects" : [lambda : LinearRegression(), 
                            lambda : DecisionTreeRegressor(), 
                            lambda : RandomForestRegressor(n_estimators=100), 
                            lambda : RandomForestRegressor(), 
                            lambda : BaggingRegressor()], 
               "Predictions":[[] for _ in range(5)]}
actuals = []
```


```python
from sklearn.model_selection import train_test_split
for _ in range (4):
    train_X, test_X, train_y, test_y = (
        train_test_split(df_hist.drop("Sales", axis=1), 
                         df_hist["Sales"], test_size=0.25))
    for i, obj_factory in enumerate(experiments["Objects"]):
        obj = obj_factory()
        obj.fit(y=train_y,X=train_X)
        experiments["Predictions"][i] += list(obj.predict(test_X))
    actuals += list(test_y)
actuals = pandas.Series(actuals)
experiments["Predictions"] = list(map(pandas.Series, experiments["Predictions"]))
```


```python
experiments["Results"] = []
for o in experiments["Objects"]:
    experiments["Results"].append(
        model_selection.cross_val_score(o(), y=df_hist['Sales'], 
                                        X=df_hist.drop("Sales", axis=1),
                                        cv=5).mean())
DataFrame(experiments).drop(["Objects", "Predictions"], 
                            axis=1).set_index("Algorithm")

fitted = (experiments["Objects"]
          [experiments["Algorithm"].index("Big Random Forest")]().
          fit(y=df_hist["Sales"], X=df_hist.drop("Sales", axis=1)))
```


```python
df_superbowl_original = pandas.read_excel("super_bowl_promotion_data.xlsx")
df_superbowl = get_dummies(df_superbowl_original, 
                           prefix={k:"dmy_%s"%k for k in categorical_columns},
                           columns = list(categorical_columns))
assert "Sales" not in df_superbowl.columns 
assert {"Sales"}.union(df_superbowl.columns).issubset(set(df_hist.columns))
len(df_superbowl)
```




    36




```python
for fld in set(df_hist.columns).difference(df_superbowl.columns, {"Sales"}):
    assert fld.startswith("dmy_")
    df_superbowl[fld] = 0
    
df_superbowl = df_superbowl[list(df_hist.drop("Sales", axis=1).columns)]

predicted = fitted.predict(df_superbowl)

```


```python
forecast_sales = df_superbowl_original[["Product", "Cost Per Unit"]].copy()
forecast_sales["Sales"] = predicted

forecast_sales = forecast_sales.groupby(['Product','Cost Per Unit']).mean()
```


```python
from __future__ import print_function
from ortools.constraint_solver import pywrapcp as cp

soda_family = {'11 Down': 'Clear', 'AB Root Beer': 'Dark', 
               'Alpine Stream': 'Clear', 'Bright': 'Clear', 
               'Crisp Clear': 'Clear', 'DC Kola': 'Dark',
               'Koala Kola': 'Dark', 'Mr. Popper': 'Dark', 
               'Popsi Kola': 'Dark'}
family  = set(soda_family[j] for j in soda_family)
soda    = set(j for j in soda_family)
max_prom = {f:2 for f in family}

product_prices = set(forecast_sales.index.values)
normal_price = {b:0 for b in soda}
for b,p in product_prices:
    normal_price[b] = max(normal_price[b],p)
```
for j in range(num_tasks):
    ## This in effect sets constraints on the domain
    t.append(solver.IntVar(0, 1, "x[%i,%i]" % (i, j)))from ortools.constraint_solver import pywrapcp as cp

solver = cp.Solver("nothing_fancy")


```python
from ortools.linear_solver import pywraplp  ## See this is Linear Programming

solver = pywraplp.Solver('SolveIntegerProblem',
                       pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)


### The thing is, once you use this the X_array looks wrong

```


```python
## More Specific Keys
forecast_sales_r =forecast_sales.reset_index()

x_array = []

"""x_array["Product", "Price"]"""
x_product = []
for i, product in enumerate(forecast_sales_r.drop_duplicates("Product")["Product"].values):
    t = []
    for j, price in enumerate(forecast_sales_r[forecast_sales_r["Product"]==product]["Cost Per Unit"].values):
        ## This in effect sets constraints on the domain
        t.append(solver.IntVar(0, 1, "x[%s,%s]" % (product, price)))
    x_array.extend(t) ## produces a flat array
    x_product.append(t)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-3cbfa3621057> in <module>()
          1 ## More Specific Keys
    ----> 2 forecast_sales_r =forecast_sales.reset_index()
          3 
          4 x_array = []
          5 


    NameError: name 'forecast_sales' is not defined



```python
## x_product, x_array
investment = solver.IntVar(0, 300, "investment")
sales = solver.IntVar(0.0, solver.infinity(), 'sales')
revenue = solver.IntVar(0.0, solver.infinity(), 'revenue')

```


```python
# Total cost
solver.Add(sales == solver.Sum(x_array * forecast_sales_r.Sales), name="sales")
```




    <ortools.linear_solver.pywraplp.Constraint; proxy of <Swig Object of type 'operations_research::MPConstraint *' at 0x118d54b70> >




```python
solver.Add(revenue == solver.Sum(forecast_sales.Sales[b,p] * p * 
                                 x_array[i] for i, (b,p) in enumerate(product_prices)), 
                name='revenue')
```




    <ortools.linear_solver.pywraplp.Constraint; proxy of <Swig Object of type 'operations_research::MPConstraint *' at 0x1190841b0> >




```python
solver.Add(investment == 
                solver.Sum(max(0,forecast_sales.Sales[b,p] - 
                            forecast_sales.Sales[b,normal_price[b]]) *
                        normal_price[b] * x_array[i] 
                        for i, (b,p) in enumerate(product_prices)),
                name='investment')
```




    <ortools.linear_solver.pywraplp.Constraint; proxy of <Swig Object of type 'operations_research::MPConstraint *' at 0x1190843c0> >


Msoda_family

[x_array[i] for i, (b,p) in enumerate(product_prices)]

solver.Add((solver.Sum(x_array[i] for i, (b,p) in enumerate(product_prices) if (soda_family[b] == f) and (p != normal_price[b]) ) <= max_prom[f] for f in family), name='MaxProm')

model.addConstrs((select_price.sum(b,'*') == 1 for b in soda), name='OnePrice')

soda

```python
solver.Maximize(sales)
```


```python
sol = solver.Solve()
```


```python
print('Total cost = ', solver.Objective().Value())

```

    Total cost =  -0.0



```python
sol
```




    0




```python
sales
```


```python
db = solver.Phase(x_array,
                solver.CHOOSE_FIRST_UNBOUND, # The VAR strategy
                solver.ASSIGN_MIN_VALUE)     # The Value strategy

# Create a solution collector.
collector = solver.LastSolutionCollector()
# Add decision variables
#collector.Add(x_array)

for i in range(num_workers):
    collector.Add(x_worker[i])
```


```python
decision_builder = solver.Phase([x, y],
                                  solver.CHOOSE_FIRST_UNBOUND,
                                  solver.ASSIGN_MIN_VALUE)
```
