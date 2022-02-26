---
jupyter:
  colab:
    name: Project 3.ipynb
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="LTs7BG93CbmP"}
# Machine Learning - Boosting

## Masaccio Braun

In the past, we have used machine learning methods that implement a
single *learner* i.e. linear regressor, decision tree regressor, etc.,
as well as learning methods that implement multiple learners i.e. random
forest regressor. Random forest models are known as \'ensemble\' methods
because they use a group of learners to enhance the performance of each
individual learner and create a stronger, aggregate model. However, many
of the individual learners in the ensemble tend to be *weak learners*,
such that some tend to make poor predictions based on the observations
they are specifically trained on. One way to combat this is through
**boosting**. Boosting is a derivative of bagging, in which weak
learners are trained simultaneously through a random selection, each
having equal *weight* in predictive capability. Boosting differs in that
the models have differing weights, since they are trained sequentially
to compensate for weak learners by taking the efficacy of the previous
model and increasing the weights of the data that the previous model had
the highest error with.

![boostingvsbagging](vertopal_d786ce807ccb45dc81ee007bf6b50e52/51c38e45f67558ed3101666399d0813b72394fd7.png)

Source:
<https://towardsdatascience.com/what-is-boosting-in-machine-learning-2244aa196682>

There are a multitude of methods that can be implemented to boost a
model. In this exploration, I will test a random forest booster and a
decision tree booster for locally weighted regression and a version of
extreme gradient boosting regression, and compare the results of their
predictions in two multivariate analyses.
:::

::: {.cell .code execution_count="37" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="cT6eOh_GNtFk" outputId="317d34ee-c6e4-4aec-ea64-5bf8cdcbf4d8"}
``` {.python}
# Update statsmodels
! pip install statsmodels==0.13.2
```

::: {.output .stream .stdout}
    Requirement already satisfied: statsmodels==0.13.2 in /usr/local/lib/python3.7/dist-packages (0.13.2)
    Requirement already satisfied: scipy>=1.3 in /usr/local/lib/python3.7/dist-packages (from statsmodels==0.13.2) (1.4.1)
    Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.7/dist-packages (from statsmodels==0.13.2) (21.3)
    Requirement already satisfied: pandas>=0.25 in /usr/local/lib/python3.7/dist-packages (from statsmodels==0.13.2) (1.3.5)
    Requirement already satisfied: patsy>=0.5.2 in /usr/local/lib/python3.7/dist-packages (from statsmodels==0.13.2) (0.5.2)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from statsmodels==0.13.2) (1.21.5)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=21.3->statsmodels==0.13.2) (3.0.7)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.25->statsmodels==0.13.2) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.25->statsmodels==0.13.2) (2018.9)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from patsy>=0.5.2->statsmodels==0.13.2) (1.15.0)
:::
:::

::: {.cell .code execution_count="1" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="lXnt67te04-C" outputId="c21913aa-57b2-41fd-8eed-27c379c1fd2d"}
``` {.python}
# Imports

# General libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Linear algebra
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr

# Interpolators
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator

# Scalers
from sklearn.preprocessing import StandardScaler

# Cross Validation
from sklearn.model_selection import KFold

# Model Evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2

# Regressors
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.nonparametric.kernel_regression import KernelReg
from xgboost import XGBRegressor

# Neural Network
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
```

::: {.output .stream .stderr}
    /usr/local/lib/python3.7/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
:::
:::

::: {.cell .code execution_count="2" id="YKCq8GOt3tf4"}
``` {.python}
# High-resolution images
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
mpl.rcParams['figure.dpi'] = 120
```
:::

::: {.cell .code execution_count="3" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="UX7Al8yf3phd" outputId="9da905c4-d083-45ea-d616-e9b47d6f0c24"}
``` {.python}
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

::: {.output .stream .stdout}
    Mounted at /content/drive
:::
:::

::: {.cell .code execution_count="4" id="9JFXnqie3pk_"}
``` {.python}
# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
```
:::

::: {.cell .markdown id="BUf-LB0lCfUD"}
### Boston Housing Prices

The first dataset I will try to model will be the Boston Housing Prices
dataset, which can be found here:
<https://www.kaggle.com/prasadperera/the-boston-housing-dataset>
:::

::: {.cell .code execution_count="5" colab="{\"height\":270,\"base_uri\":\"https://localhost:8080/\"}" id="1O3cHZTd3poB" outputId="ef115222-7916-472b-82ec-b1c19c969a39"}
``` {.python}
# Load data
houses = pd.read_csv('drive/MyDrive/Data/Boston Housing Prices.csv')
houses.head()
```

::: {.output .execute_result execution_count="5"}
```{=html}
  <div id="df-ef4c7c52-ab5d-49c8-b926-3ba678aca45d">
    <div class="colab-df-container">
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
      <th>town</th>
      <th>tract</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>crime</th>
      <th>residential</th>
      <th>industrial</th>
      <th>river</th>
      <th>nox</th>
      <th>rooms</th>
      <th>older</th>
      <th>distance</th>
      <th>highway</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>lstat</th>
      <th>cmedv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nahant</td>
      <td>2011</td>
      <td>-70.955002</td>
      <td>42.255001</td>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>no</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.199997</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.300000</td>
      <td>4.98</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Swampscott</td>
      <td>2021</td>
      <td>-70.949997</td>
      <td>42.287498</td>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>no</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.900002</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.799999</td>
      <td>9.14</td>
      <td>21.600000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Swampscott</td>
      <td>2022</td>
      <td>-70.935997</td>
      <td>42.283001</td>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>no</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.099998</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.799999</td>
      <td>4.03</td>
      <td>34.700001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Marblehead</td>
      <td>2031</td>
      <td>-70.928001</td>
      <td>42.292999</td>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>no</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.799999</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.700001</td>
      <td>2.94</td>
      <td>33.400002</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Marblehead</td>
      <td>2032</td>
      <td>-70.921997</td>
      <td>42.298000</td>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>no</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.200001</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.700001</td>
      <td>5.33</td>
      <td>36.200001</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ef4c7c52-ab5d-49c8-b926-3ba678aca45d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ef4c7c52-ab5d-49c8-b926-3ba678aca45d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ef4c7c52-ab5d-49c8-b926-3ba678aca45d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="DWTWDD5hP4U2"}
For the multivariate analysis, I will choose 3 feature variables to
predict the median value of owner occupied homes: average number of
rooms per dwelling, per capita crime rate by town, and full-value
property tax rate. To get an idea of the immediate linear correlation
coefficients of the feature and response variables, I will create a
heatmap.
:::

::: {.cell .code execution_count="57" colab="{\"height\":852,\"base_uri\":\"https://localhost:8080/\"}" id="0L_zDpj53pq9" outputId="6bcc64dd-3932-4d46-c712-6cf7345aafc2"}
``` {.python}
# Select variables and plot data
x = houses[['rooms', 'crime', 'tax']].values
y = houses['cmedv'].values

# Correlation Heatmap
irrel_col = ['town','tract','longitude','latitude','residential','industrial','river','nox','distance','older','highway','ptratio','lstat']
houses_select = houses.drop(columns=irrel_col)
c = houses_select.corr()
mask = np.triu(c)

plt.figure(figsize=(8,8))
sns.heatmap(c, mask=mask, annot=True, annot_kws={'fontsize':8,'weight':'bold'}, cmap='BuPu', vmin=0, vmax=1,
            yticklabels=[' '] + list(houses_select.columns[1:]), xticklabels=houses_select.columns[:-1])
plt.tick_params(size=0, labelsize=14, pad=20)
plt.title('Pairwise Linear Correlations', fontsize=16)
plt.savefig('houses_heat.png')
plt.show()
```

::: {.output .display_data}
![](vertopal_d786ce807ccb45dc81ee007bf6b50e52/25f9d79aa57e51d123a8b78bb0cc4035ce7994a8.png){height="835"
width="786"}
:::
:::

::: {.cell .markdown id="2g_4dVJwChRt"}
The only feature that seems to hold any significant linear correlation
to median house price is the number of rooms. Just to get an idea of the
distribution of the variables, I will create a pairplot of the relevant
data fields.
:::

::: {.cell .code execution_count="7" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="6lOnC5V-3pwN" outputId="82222522-5e22-4e9d-cb49-b07d3135fa3b"}
``` {.python}
plt.figure(figsize=(8,8))
sns.pairplot(houses_select)
plt.savefig('houses_pair.png')
plt.show()
```

::: {.output .display_data}
    <Figure size 960x960 with 0 Axes>
:::

::: {.output .display_data}
![](vertopal_d786ce807ccb45dc81ee007bf6b50e52/521d5d28703d9daccd0edd091f769f906f0442ae.png){height="1180"
width="1182"}
:::
:::

::: {.cell .code execution_count="8" id="9GB1eum7Ef2h"}
``` {.python}
# Standardization
ss = StandardScaler()
```
:::

::: {.cell .code execution_count="9" id="U7MHMSxdEf5X"}
``` {.python}
# Cross Validation
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=410)
```
:::

::: {.cell .markdown id="PmBEpZP0CmR9"}
To compare against the boosted models, I will be testing an ordinary
least-squares linear regression, a decision tree regression, and a
random forest regression, as well as an artificial neural network,
though I suspect this will perform the poorly.
:::

::: {.cell .code execution_count="10" id="eMt-SfmaKXR0"}
``` {.python}
# Least Squares Regression
lsr = LinearRegression()
```
:::

::: {.cell .code execution_count="11" id="DdiYS7UpD0Fp"}
``` {.python}
# Decision Tree Regression
dtr = DecisionTreeRegressor(max_depth=2, random_state=410)
```
:::

::: {.cell .code execution_count="12" id="GH3KO88YD-qe"}
``` {.python}
# Random Forest Regression
rfr = RandomForestRegressor(n_estimators=50, max_depth=2)
```
:::

::: {.cell .code execution_count="16" id="fEEiIBmXMFjj"}
``` {.python}
# Neural Network
seq = Sequential()

seq.add(Dense(32, activation="relu"))
seq.add(Dense(64, activation="relu"))
seq.add(Dense(128, activation="relu"))
seq.add(Dense(1, activation="linear"))

seq.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-2))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=800)
```
:::

::: {.cell .code execution_count="14" id="jXKBkJ1X3p6A"}
``` {.python}
# Locally Weighted Regression
def lwr(X, y, xnew, kern, tau, intercept, boost=None):
    n = len(X) 
    yest = np.zeros(n)

    if len(y.shape)==1: 
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)])

    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) 
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
```
:::

::: {.cell .markdown id="C8ElKZMcCvP1"}
## Locally Weighted Regression Boosting
:::

::: {.cell .code execution_count="15" id="0j67GP9f3p8J"}
``` {.python}
# Boosted Locally Weighted Regression
def blwr(X, y, xnew, kern, tau, intercept, boost):
  Fx = lwr(X,y,X,kern,tau,intercept)
  new_y = y - Fx
  boost.fit(X,new_y)
  output = boost.predict(xnew) + lwr(X,y,xnew,kern,tau,intercept)
  return output 
```
:::

::: {.cell .markdown id="Oj_Jh_F0Cvzv"}
## Extreme Gradient Boosting

A derivative of gradient boosting. In this method, a set of parallel
decision trees are fitted to the residuals of the previous trees\'
predictions, so it is an ensemble model similar to random forest.
However, instead of relying on traditional decision trees, XGB uses
*XGBoost trees*, which are created by calculating *similarity scores*
between the observations of the leaf nodes. Furthermore, XGB implements
regularization to prevent overfitting of the individual decision trees.
With XGB, the process of sequentially creating weak models is done so as
a *gradient descent* algorithm using an objective cost function, in
which the algorithm iterates until the cost function is at close to or
equal to zero, or at least at the lowest minima. In order to minimize
the loss function,we take its partial derivative with respect to its
slope and its intercept, then subtracting this from the slope beta
(denoted theta in the case of the equation below). The partial
derivative is scaled by a desired learning rate (hyperparameter) and
this process is repeated until a convergence at the minimum.

![xgbcostfunctionderivative](vertopal_d786ce807ccb45dc81ee007bf6b50e52/10255763d97490ec18f218c5d815c13a9dc8d915.png)

Source:
<https://medium.com/analytics-vidhya/what-makes-xgboost-so-extreme-e1544a4433bb>
:::

::: {.cell .code execution_count="13" id="7zCnKQdHLCNc"}
``` {.python}
# Extreme Gradient Regression
xgbr = XGBRegressor(objective='reg:squarederror', n_estimators=100, reg_lambda=20, alpha=1, gamma=10, max_depth=3)
```
:::

::: {.cell .code execution_count="22" id="k8pWPy0wEf_W"}
``` {.python}
# Model Set 1
  # Linear (lsr), Decision Tree (dtr), Random Forest (rfr), Extreme Gradient (xgr), 
def RunModel1(model, x , y, scaler, split):
  mse_model = []
  mae_model = []
  r2_model = []

  for idxtrain, idxtest in kf.split(x):
    xtrain = x[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = x[idxtest]
    xtrain = ss.fit_transform(xtrain)
    xtest = ss.transform(xtest)

    model.fit(xtrain, ytrain)
    yhat = model.predict(xtest)

    mse_model.append(mse(ytest, yhat))
    mae_model.append(mae(ytest, yhat))
    r2_model.append(r2(ytest, yhat))

  return np.mean(mse_model), np.mean(mae_model), np.mean(r2_model)
```
:::

::: {.cell .code execution_count="23" id="wsXVETnDIylR"}
``` {.python}
# Model Set 2
  # Loess, Boosted Loess
def RunModel2(model, x , y, kern, tau, scaler, split, boost=None):
  mse_model = []
  mae_model = []
  r2_model = []

  for idxtrain, idxtest in kf.split(x):
    xtrain = x[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = x[idxtest]
    xtrain = ss.fit_transform(xtrain)
    xtest = ss.transform(xtest)

    yhat = model(xtrain,ytrain, xtest, kern, tau=tau, intercept=True, boost=boost)

    mse_model.append(mse(ytest, yhat))
    mae_model.append(mae(ytest, yhat))
    r2_model.append(r2(ytest, yhat))

  return np.mean(mse_model), np.mean(mae_model), np.mean(r2_model)
```
:::

::: {.cell .code execution_count="24" id="09vAua6eIyoE"}
``` {.python}
# Model Set 3
  # Neural Network
def RunNN(model, x , y, scaler, split, val_split=0.25, epoch=500, batch=20):
  mse_model = []
  mae_model = []
  r2_model = []

  for idxtrain, idxtest in kf.split(x):
    xtrain = x[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = x[idxtest]
    xtrain = ss.fit_transform(xtrain)
    xtest = ss.transform(xtest)

    model.fit(xtrain, ytrain, validation_split=val_split, epochs=epoch, batch_size=batch, verbose=0, callbacks=[es])
    yhat = seq.predict(xtest)

    mse_model.append(mse(ytest, yhat))
    mae_model.append(mae(ytest, yhat))
    r2_model.append(r2(ytest, yhat))

  return np.mean(mse_model), np.mean(mae_model), np.mean(r2_model)
```
:::

::: {.cell .code execution_count="26" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="un1_Q_sGIyxu" outputId="60e454c1-1d2f-4e37-deb3-3655276a199a"}
``` {.python}
mse_lsr, mae_lsr, r2_lsr = RunModel1(lsr, x, y, ss, kf)
print('The Cross-validated Mean Squared Error for Least-Squares Regression is : ' + str(mse_lsr))
print('The Cross-validated Mean Absolute Error for Least-Squares Regression is : ' + str(mae_lsr))
print('The Cross-validated Coefficient of Determination for Least-Squares Regression is : ' + str(r2_lsr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Least-Squares Regression is : 37.23940104890824
    The Cross-validated Mean Absolute Error for Least-Squares Regression is : 4.020970004960848
    The Cross-validated Coefficient of Determination for Least-Squares Regression is : 0.5478240461818417
:::
:::

::: {.cell .code execution_count="27" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="bNIxecxSPvis" outputId="a757e604-4228-48b4-9c56-4423cb297ae5"}
``` {.python}
mse_dtr, mae_dtr, r2_dtr = RunModel1(dtr, x, y, ss, kf)
print('The Cross-validated Mean Squared Error for Decision Tree Regression is : ' + str(mse_dtr))
print('The Cross-validated Mean Absolute Error for Decision Tree Regression is : ' + str(mae_dtr))
print('The Cross-validated Coefficient of Determination for Decision Tree Regression is : ' + str(r2_dtr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Decision Tree Regression is : 35.18217909437036
    The Cross-validated Mean Absolute Error for Decision Tree Regression is : 4.063618925443233
    The Cross-validated Coefficient of Determination for Decision Tree Regression is : 0.5680960234893112
:::
:::

::: {.cell .code execution_count="28" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="9vbMtQ-gPvlO" outputId="2e21e9e9-b28e-46c9-8fd6-aa5abe51d941"}
``` {.python}
mse_rfr, mae_rfr, r2_rfr = RunModel1(rfr, x, y, ss, kf)
print('The Cross-validated Mean Squared Error for Random Forest Regression is : ' + str(mse_rfr))
print('The Cross-validated Mean Absolute Error for Random Forest Regression is : ' + str(mae_rfr))
print('The Cross-validated Coefficient of Determination for Random Forest Regression is : ' + str(r2_rfr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Random Forest Regression is : 31.9639600875443
    The Cross-validated Mean Absolute Error for Random Forest Regression is : 3.8578840078128542
    The Cross-validated Coefficient of Determination for Random Forest Regression is : 0.6121888076614384
:::
:::

::: {.cell .code execution_count="34" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Ey9U_yFtadp2" outputId="3bd8e492-0997-46db-e11a-9a2274a8c2ec"}
``` {.python}
mse_seq, mae_seq, r2_seq = RunNN(seq, x, y, ss, kf)
print('The Cross-validated Mean Squared Error for Neural Network is : ' + str(mse_seq))
print('The Cross-validated Mean Absolute Error for Neural Network is : ' + str(mae_seq))
print('The Cross-validated Coefficient of Determination for Neural Network is : ' + str(r2_seq))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Neural Network is : 85.57445466742146
    The Cross-validated Mean Absolute Error for Neural Network is : 5.030748784209109
    The Cross-validated Coefficient of Determination for Neural Network is : -0.1203951433644184
:::
:::

::: {.cell .code execution_count="31" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Bpwd9uFhPvqL" outputId="dc1a2bfc-6ca0-473f-8a65-a342bbea9659"}
``` {.python}
mse_lwr, mae_lwr, r2_lwr = RunModel2(lwr, x, y, Tricubic, 0.9, ss, kf)
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : ' + str(mse_lwr))
print('The Cross-validated Mean Absolute Error for Locally Weighted Regression is : ' + str(mae_lwr))
print('The Cross-validated Coefficient of Determination for Locally Weighted Regression is : ' + str(r2_lwr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Locally Weighted Regression is : 26.597270697712815
    The Cross-validated Mean Absolute Error for Locally Weighted Regression is : 3.3010239544041284
    The Cross-validated Coefficient of Determination for Locally Weighted Regression is : 0.6845775749624277
:::
:::

::: {.cell .code execution_count="32" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="yEPe6LkAZOu3" outputId="68f655f6-c0be-4c24-ce72-5cc592bb1a4d"}
``` {.python}
mse_blwr, mae_blwr, r2_blwr = RunModel2(blwr, x, y, Quartic, 0.9, ss, kf, dtr)
print('The Cross-validated Mean Squared Error for Locally Weighted Regression Boosted with Decision Trees is : ' + str(mse_blwr))
print('The Cross-validated Mean Absolute Error for Locally Weighted Regression Boosted with Decision Trees is : ' + str(mae_blwr))
print('The Cross-validated Coefficient of Determination for Locally Weighted Regression Boosted with Decision Trees is : ' + str(r2_blwr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Locally Weighted Regression Boosted with Decision Trees is : 26.943911216871744
    The Cross-validated Mean Absolute Error for Locally Weighted Regression Boosted with Decision Trees is : 3.2946931437035536
    The Cross-validated Coefficient of Determination for Locally Weighted Regression Boosted with Decision Trees is : 0.6813133952301352
:::
:::

::: {.cell .code execution_count="33" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="YUuFpIi-Pvsk" outputId="b89387de-a721-4913-dab4-3fe89724c228"}
``` {.python}
mse_blwr, mae_blwr, r2_blwr = RunModel2(blwr, x, y, Quartic, 0.9, ss, kf, boost=rfr)
print('The Cross-validated Mean Squared Error for Locally Weighted Regression Boosted with Random Forest is : ' + str(mse_blwr))
print('The Cross-validated Mean Absolute Error for Locally Weighted Regression Boosted with Random Forest is : ' + str(mae_blwr))
print('The Cross-validated Coefficient of Determination for Locally Weighted Regression Boosted with Random Forest is : ' + str(r2_blwr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Locally Weighted Regression Boosted with Random Forest is : 26.68176106594833
    The Cross-validated Mean Absolute Error for Locally Weighted Regression Boosted with Random Forest is : 3.2491331849574685
    The Cross-validated Coefficient of Determination for Locally Weighted Regression Boosted with Random Forest is : 0.6858306843421442
:::
:::

::: {.cell .code execution_count="30" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="eK2wCvQhPvnl" outputId="317dfb2c-01e9-4a94-eb15-755bfeab0aab"}
``` {.python}
mse_xgbr, mae_xgbr, r2_xgbr = RunModel1(xgbr, x, y, ss, kf)
print('The Cross-validated Mean Squared Error for Extreme Gradient Regression is : ' + str(mse_xgbr))
print('The Cross-validated Mean Absolute Error for Extreme Gradient Regression is : ' + str(mae_xgbr))
print('The Cross-validated Coefficient of Determination for Extreme Gradient Regression is : ' + str(r2_xgbr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Extreme Gradient Regression is : 26.548227007953603
    The Cross-validated Mean Absolute Error for Extreme Gradient Regression is : 3.217740893794725
    The Cross-validated Coefficient of Determination for Extreme Gradient Regression is : 0.6886189666427519
:::
:::

::: {.cell .markdown id="QsI5V3DfC0Ol"}
Overall, all of the models performed relatively poorly. Locally weighted
regression performed the best out of the standard nonparametric models,
performing marginally better than its primary competitor, random forest
regression. On a side note, the ordinary linear regression did not fare
so badly against the decision tree regression, which was a suprising
result. The artificial neural network is a useless model and is unable
to account for any of the explained variation in the data with a
negative R\^2 score. Strangely, the boosted versions of the locally
weighted regression, both the decision tree and random forest, performed
slightly poorer than the standard model, which is to say the current
state that I constructed added no predicitve power. Unfortunately, the
same can be said for the extreme gradient boosting, which performed the
best out of all of the models. Overall, the boosted models did perform
marginally better than the standard models; however, I suspect that they
are ill-equipped to make solid predictions for this particular data.
:::

::: {.cell .markdown id="Kki_Q2pCC0UR"}
### Concrete Data

The second dataset I will try to model will be the Boston Housing Prices
dataset, which can be found here:
<https://www.kaggle.com/elikplim/concrete-compressive-strength-data-set>
:::

::: {.cell .code execution_count="41" colab="{\"height\":206,\"base_uri\":\"https://localhost:8080/\"}" id="ghJGk4vW3qBn" outputId="dca4e9af-6276-4a57-a55b-6d356a760dbb"}
``` {.python}
# Load Data
concrete = pd.read_csv('drive/MyDrive/Data/concrete.csv')
concrete.head()
```

::: {.output .execute_result execution_count="41"}
```{=html}
  <div id="df-d90a33d5-988f-42dd-b78e-53e078367cfb">
    <div class="colab-df-container">
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
      <th>cement</th>
      <th>slag</th>
      <th>ash</th>
      <th>water</th>
      <th>superplastic</th>
      <th>coarseagg</th>
      <th>fineagg</th>
      <th>age</th>
      <th>strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>79.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>61.89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
      <td>40.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
      <td>41.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
      <td>44.30</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d90a33d5-988f-42dd-b78e-53e078367cfb')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d90a33d5-988f-42dd-b78e-53e078367cfb button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d90a33d5-988f-42dd-b78e-53e078367cfb');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="XkpeOfyLR5YI"}
Like before, I will choose 3 feature variables to predict the
compressive strength of the concrete: the amount of the cement
component, the amount of the slag component, and the amount of the water
component. To get an idea of the immediate linear correlation
coefficients and distributions of the feature and response variables, I
will again create a heatmap and a pairplot.
:::

::: {.cell .code execution_count="56" colab="{\"height\":852,\"base_uri\":\"https://localhost:8080/\"}" id="RDvh62G9IWX8" outputId="500c81a5-1742-4380-a955-8e35c104b4cc"}
``` {.python}
# Select variables and plot data
x = concrete[['cement','slag','water']].values
y = concrete['strength'].values

# Correlation Heatmap
irrel_col = ['ash','superplastic','coarseagg','fineagg','age']
concrete_select = concrete.drop(columns=irrel_col)
c = concrete_select.corr()
mask = np.triu(c)

plt.figure(figsize=(8,8))
sns.heatmap(c, mask=mask, annot=True, annot_kws={'fontsize':8,'weight':'bold'}, cmap='PuBu', vmin=0, vmax=1,
            yticklabels=[' '] + list(concrete_select.columns[1:]), xticklabels=concrete_select.columns[:-1])
plt.tick_params(size=0, labelsize=14, pad=20)
plt.title('Pairwise Linear Correlations', fontsize=16)
plt.savefig('concrete_heat.png')
plt.show()
```

::: {.output .display_data}
![](vertopal_d786ce807ccb45dc81ee007bf6b50e52/bf69307732d65a8a18d3aa7c936045d2d0a6becd.png){height="835"
width="786"}
:::
:::

::: {.cell .code execution_count="47" colab="{\"height\":1000,\"base_uri\":\"https://localhost:8080/\"}" id="1Uoxiv0E-6TU" outputId="c56ea2d0-e91f-4d98-c878-598af5a105b0"}
``` {.python}
plt.figure(figsize=(8,8))
sns.pairplot(concrete_select)
plt.savefig('concrete_pair.png')
plt.show()
```

::: {.output .display_data}
    <Figure size 960x960 with 0 Axes>
:::

::: {.output .display_data}
![](vertopal_d786ce807ccb45dc81ee007bf6b50e52/a75ef388cf021512a6a8a68f9fbd2976abc0a765.png){height="1180"
width="1182"}
:::
:::

::: {.cell .code execution_count="48" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="G0jbXPPlPU3E" outputId="8949a66d-1508-4567-9280-df6c62b17b05"}
``` {.python}
mse_lsr, mae_lsr, r2_lsr = RunModel1(lsr, x, y, ss, kf)
print('The Cross-validated Mean Squared Error for Least-Squares Regression is : ' + str(mse_lsr))
print('The Cross-validated Mean Absolute Error for Least-Squares Regression is : ' + str(mae_lsr))
print('The Cross-validated Coefficient of Determination for Least-Squares Regression is : ' + str(r2_lsr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Least-Squares Regression is : 167.78526376603222
    The Cross-validated Mean Absolute Error for Least-Squares Regression is : 10.429406509600094
    The Cross-validated Coefficient of Determination for Least-Squares Regression is : 0.38806908236831456
:::
:::

::: {.cell .code execution_count="49" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="hxSbTvXeAMQx" outputId="080e0791-4712-4559-a107-f2f1ef80989b"}
``` {.python}
mse_dtr, mae_dtr, r2_dtr = RunModel1(dtr, x, y, ss, kf)
print('The Cross-validated Mean Squared Error for Decision Tree Regression is : ' + str(mse_dtr))
print('The Cross-validated Mean Absolute Error for Decision Tree Regression is : ' + str(mae_dtr))
print('The Cross-validated Coefficient of Determination for Decision Tree Regression is : ' + str(r2_dtr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Decision Tree Regression is : 196.55510778069464
    The Cross-validated Mean Absolute Error for Decision Tree Regression is : 11.42242710516928
    The Cross-validated Coefficient of Determination for Decision Tree Regression is : 0.283131231488064
:::
:::

::: {.cell .code execution_count="50" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="juXMgjHMAQ4B" outputId="c2b29868-6a22-4fcf-8f23-9ce112d4ea6f"}
``` {.python}
mse_rfr, mae_rfr, r2_rfr = RunModel1(rfr, x, y, ss, kf)
print('The Cross-validated Mean Squared Error for Random Forest Regression is : ' + str(mse_rfr))
print('The Cross-validated Mean Absolute Error for Random Forest Regression is : ' + str(mae_rfr))
print('The Cross-validated Coefficient of Determination for Random Forest Regression is : ' + str(r2_rfr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Random Forest Regression is : 185.20401382173023
    The Cross-validated Mean Absolute Error for Random Forest Regression is : 11.121562873963494
    The Cross-validated Coefficient of Determination for Random Forest Regression is : 0.3260469998620026
:::
:::

::: {.cell .code execution_count="55" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="iz1VrCrZCWEY" outputId="968d205c-ab7e-42c3-961a-d710db50f77e"}
``` {.python}
mse_seq, mae_seq, r2_seq = RunNN(seq, x, y, ss, kf)
print('The Cross-validated Mean Squared Error for Neural Network is : ' + str(mse_seq))
print('The Cross-validated Mean Absolute Error for Neural Network is : ' + str(mae_seq))
print('The Cross-validated Coefficient of Determination for Neural Network is : ' + str(r2_seq))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Neural Network is : 185.4012886376533
    The Cross-validated Mean Absolute Error for Neural Network is : 11.223063258865505
    The Cross-validated Coefficient of Determination for Neural Network is : 0.32458313308094616
:::
:::

::: {.cell .code execution_count="52" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ELKlnX2wBRtM" outputId="e8923e21-6f0a-4730-ce26-e11831cc2c52"}
``` {.python}
mse_lwr, mae_lwr, r2_lwr = RunModel2(lwr, x, y, Tricubic, 0.9, ss, kf)
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : ' + str(mse_lwr))
print('The Cross-validated Mean Absolute Error for Locally Weighted Regression is : ' + str(mae_lwr))
print('The Cross-validated Coefficient of Determination for Locally Weighted Regression is : ' + str(r2_lwr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Locally Weighted Regression is : 146.4227649591804
    The Cross-validated Mean Absolute Error for Locally Weighted Regression is : 9.768092527277217
    The Cross-validated Coefficient of Determination for Locally Weighted Regression is : 0.4654909181670496
:::
:::

::: {.cell .code execution_count="53" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="PJ0ObzztCEqZ" outputId="0f450be6-18b8-439b-f2d2-f564ddc9bb7b"}
``` {.python}
mse_blwr, mae_blwr, r2_blwr = RunModel2(blwr, x, y, Quartic, 0.9, ss, kf, dtr)
print('The Cross-validated Mean Squared Error for Locally Weighted Regression Boosted with Decision Trees is : ' + str(mse_blwr))
print('The Cross-validated Mean Absolute Error for Locally Weighted Regression Boosted with Decision Trees is : ' + str(mae_blwr))
print('The Cross-validated Coefficient of Determination for Locally Weighted Regression Boosted with Decision Trees is : ' + str(r2_blwr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Locally Weighted Regression Boosted with Decision Trees is : 141.5780947490901
    The Cross-validated Mean Absolute Error for Locally Weighted Regression Boosted with Decision Trees is : 9.551662686619025
    The Cross-validated Coefficient of Determination for Locally Weighted Regression Boosted with Decision Trees is : 0.4829019098259071
:::
:::

::: {.cell .code execution_count="54" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="B3tzVepMBqhN" outputId="8d994d76-8c69-4b27-a257-6d505d43feab"}
``` {.python}
mse_blwr, mae_blwr, r2_blwr = RunModel2(blwr, x, y, Quartic, 0.9, ss, kf, boost=rfr)
print('The Cross-validated Mean Squared Error for Locally Weighted Regression Boosted with Random Forest is : ' + str(mse_blwr))
print('The Cross-validated Mean Absolute Error for Locally Weighted Regression Boosted with Random Forest is : ' + str(mae_blwr))
print('The Cross-validated Coefficient of Determination for Locally Weighted Regression Boosted with Random Forest is : ' + str(r2_blwr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Locally Weighted Regression Boosted with Random Forest is : 142.51517506407464
    The Cross-validated Mean Absolute Error for Locally Weighted Regression Boosted with Random Forest is : 9.620449887420262
    The Cross-validated Coefficient of Determination for Locally Weighted Regression Boosted with Random Forest is : 0.47975481741057535
:::
:::

::: {.cell .code execution_count="51" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="GZaziG1-AY31" outputId="15555f66-6dea-4313-c826-35f4b378ba12"}
``` {.python}
mse_xgbr, mae_xgbr, r2_xgbr = RunModel1(xgbr, x, y, ss, kf)
print('The Cross-validated Mean Squared Error for Extreme Gradient Regression is : ' + str(mse_xgbr))
print('The Cross-validated Mean Absolute Error for Extreme Gradient Regression is : ' + str(mae_xgbr))
print('The Cross-validated Coefficient of Determination for Extreme Gradient Regression is : ' + str(r2_xgbr))
```

::: {.output .stream .stdout}
    The Cross-validated Mean Squared Error for Extreme Gradient Regression is : 142.3439960678086
    The Cross-validated Mean Absolute Error for Extreme Gradient Regression is : 9.609343509785178
    The Cross-validated Coefficient of Determination for Extreme Gradient Regression is : 0.4814444923220341
:::
:::

::: {.cell .markdown id="LDQErmtaPKkP"}
Like before, all of the models performed relatively poorly, though much
more poorly than with the first data. Again, locally weighted regression
performed the best out of the standard nonparametric models. However,
even more strangely than with the last data, ordinary linear regression
performed marginally better than both the decision tree and random
forest regressions, though I suspect with a little optimization, these
two could surpass it. The artificial neural network handled this data
much better than the last, and achieved a result that competes with the
nonparametric models. In this case, both of the boosted versions of the
locally weighted regression did perform slightly better than the
standard model. The extreme gradient regression again performed the best
out of all of the models. Overall, the boosted models perform marginally
better than the standard models again. I suspect most of the differences
in the results between the two data are result of my variable selection.
:::

::: {.cell .markdown id="ckJG63f7cn5E"}
## Limitations

As previously mentioned, all of the tested models were ill-suited to
both data, and the next priority task is to perform adequate variable
selection. Furthermore, it would be beneficial to perform optimization
through hyperparameter selection loops, to determine the best
construction of each model for each dataset. Overall, I can conclude
that boosting is effective, and for this to become more obvious, I
suspect that I need to improve it with repeated boostings.
:::

::: {.cell .code id="x9uvK0jSdMOw"}
``` {.python}
```
:::
