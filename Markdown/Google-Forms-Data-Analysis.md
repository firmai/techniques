
# Visualizing Google Forms Data with Seaborn

This is the second part of an article from [Practical Business Python](htp://pbpython.com) describing how to retrieve and analyze data from a Google Form.

Please review [part 1](http://pbpython.com/pandas-google-forms-part1.html) for the details of how to set up authentication and get the data into the pandaqs dataframe.

The full article corresponding to this notebook is [here](http://pbpython.com/pandas-google-forms-part2.html)

## Setup

Bring in our standard imports as well as the authentication libraries we will need to get access to our form.


```python
import gspread
from oauth2client.client import SignedJwtAssertionCredentials
import pandas as pd
import json
```

Import Ipython display as well as graphing libraries. For this article, we will be using [seaborn](http://stanford.edu/~mwaskom/software/seaborn/index.html).


```python
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

Setup authentication process to pull in the survey data stored in the Google Sheet.


```python
SCOPE = ["https://spreadsheets.google.com/feeds"]
SECRETS_FILE = "Pbpython-key.json"
SPREADSHEET = "PBPython User Survey (Responses)"
# Based on docs here - http://gspread.readthedocs.org/en/latest/oauth2.html
# Load in the secret JSON key (must be a service account)
json_key = json.load(open(SECRETS_FILE))
# Authenticate using the signed key
credentials = SignedJwtAssertionCredentials(json_key['client_email'],
                                            json_key['private_key'], SCOPE)
```

Now open up the file and read all data in a DataFrame


```python
gc = gspread.authorize(credentials)
# Open up the workbook based on the spreadsheet name
workbook = gc.open(SPREADSHEET)
# Get the first sheet
sheet = workbook.sheet1
# Extract all data into a dataframe
results = pd.DataFrame(sheet.get_all_records())
results.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>How frequently do you use the following tools? [Javascript]</th>
      <th>How frequently do you use the following tools? [Python]</th>
      <th>How frequently do you use the following tools? [R]</th>
      <th>How frequently do you use the following tools? [Ruby]</th>
      <th>How frequently do you use the following tools? [SQL]</th>
      <th>How frequently do you use the following tools? [VBA]</th>
      <th>How useful is the content on practical business python?</th>
      <th>How would you like to be notified about new articles on this site?</th>
      <th>Timestamp</th>
      <th>What suggestions do you have for future content?</th>
      <th>What version of python would you like to see used for the examples on the site?</th>
      <th>Which OS do you use most frequently?</th>
      <th>Which python distribution do you primarily use?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Once a month</td>
      <td>A couple times a week</td>
      <td>Infrequently</td>
      <td>Never</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3</td>
      <td>RSS</td>
      <td>6/9/2015 23:22:43</td>
      <td></td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Included with OS - Mac</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>A couple times a week</td>
      <td>Never</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>3</td>
      <td>Reddit</td>
      <td>6/10/2015 1:19:08</td>
      <td></td>
      <td>2.7</td>
      <td>Windows</td>
      <td>Anaconda</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Infrequently</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>Daily</td>
      <td>Never</td>
      <td>2</td>
      <td>Planet Python</td>
      <td>6/10/2015 1:40:29</td>
      <td></td>
      <td>3.4+</td>
      <td>Windows</td>
      <td>Official python.org binaries</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Never</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>A couple times a week</td>
      <td>Once a month</td>
      <td>3</td>
      <td>Planet Python</td>
      <td>6/10/2015 1:55:46</td>
      <td></td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Official python.org binaries</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3</td>
      <td>Leave me alone - I will find it if I need it</td>
      <td>6/10/2015 4:10:17</td>
      <td></td>
      <td>I don't care</td>
      <td>Mac</td>
      <td>Anaconda</td>
    </tr>
  </tbody>
</table>
</div>



We need to do some cleanup to make the data easier to analyze.


```python
# Do some minor cleanups on the data
# Rename the columns to make it easier to manipulate
# The data comes in through a dictionary so we can not assume order stays the
# same so must name each column
column_names = {'Timestamp': 'timestamp',
                'What version of python would you like to see used for the examples on the site?': 'version',
                'How useful is the content on practical business python?': 'useful',
                'What suggestions do you have for future content?': 'suggestions',
                'How frequently do you use the following tools? [Python]': 'freq-py',
                'How frequently do you use the following tools? [SQL]': 'freq-sql',
                'How frequently do you use the following tools? [R]': 'freq-r',
                'How frequently do you use the following tools? [Javascript]': 'freq-js',
                'How frequently do you use the following tools? [VBA]': 'freq-vba',
                'How frequently do you use the following tools? [Ruby]': 'freq-ruby',
                'Which OS do you use most frequently?': 'os',
                'Which python distribution do you primarily use?': 'distro',
                'How would you like to be notified about new articles on this site?': 'notify'
                }
results.rename(columns=column_names, inplace=True)
results.timestamp = pd.to_datetime(results.timestamp)
results.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>freq-js</th>
      <th>freq-py</th>
      <th>freq-r</th>
      <th>freq-ruby</th>
      <th>freq-sql</th>
      <th>freq-vba</th>
      <th>useful</th>
      <th>notify</th>
      <th>timestamp</th>
      <th>suggestions</th>
      <th>version</th>
      <th>os</th>
      <th>distro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Once a month</td>
      <td>A couple times a week</td>
      <td>Infrequently</td>
      <td>Never</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3</td>
      <td>RSS</td>
      <td>2015-06-09 23:22:43</td>
      <td></td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Included with OS - Mac</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>A couple times a week</td>
      <td>Never</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>3</td>
      <td>Reddit</td>
      <td>2015-06-10 01:19:08</td>
      <td></td>
      <td>2.7</td>
      <td>Windows</td>
      <td>Anaconda</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Infrequently</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>Daily</td>
      <td>Never</td>
      <td>2</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:40:29</td>
      <td></td>
      <td>3.4+</td>
      <td>Windows</td>
      <td>Official python.org binaries</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Never</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>A couple times a week</td>
      <td>Once a month</td>
      <td>3</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:55:46</td>
      <td></td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Official python.org binaries</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3</td>
      <td>Leave me alone - I will find it if I need it</td>
      <td>2015-06-10 04:10:17</td>
      <td></td>
      <td>I don't care</td>
      <td>Mac</td>
      <td>Anaconda</td>
    </tr>
  </tbody>
</table>
</div>



There are a small number of free form comments. Let's strip those out and remove them from the results.


```python
suggestions = results[results.suggestions.str.len() > 0]["suggestions"]
```

Since there are only a small number of comments, just print them out.
However, if we had more comments and wanted to do more analysis we certainly good.


```python
for index, row in suggestions.iteritems():
    display(row)
```


    'A bit more coverage on how to make presentations - which in a lot of corporations just means powerpoint slides with python, from a business analyst perspective, of course'



    'Add some other authors to the website which can publish equally relevant content. Would  be nice to see more frequent updates if possible, keep up the good work!'



    'How to produce graphics using Python, Google Forms.'



    'Awesome site - keep up the good work'



    'Great job on the site.  Nice to see someone writing about actual Python use cases. So much writing is done elsewhere about software development without the connection to actual business work.'


Drop the suggestions. We won't use them any more.


```python
results.drop("suggestions", axis=1, inplace=True)
results.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>freq-js</th>
      <th>freq-py</th>
      <th>freq-r</th>
      <th>freq-ruby</th>
      <th>freq-sql</th>
      <th>freq-vba</th>
      <th>useful</th>
      <th>notify</th>
      <th>timestamp</th>
      <th>version</th>
      <th>os</th>
      <th>distro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Once a month</td>
      <td>A couple times a week</td>
      <td>Infrequently</td>
      <td>Never</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3</td>
      <td>RSS</td>
      <td>2015-06-09 23:22:43</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Included with OS - Mac</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>A couple times a week</td>
      <td>Never</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>3</td>
      <td>Reddit</td>
      <td>2015-06-10 01:19:08</td>
      <td>2.7</td>
      <td>Windows</td>
      <td>Anaconda</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Infrequently</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>Daily</td>
      <td>Never</td>
      <td>2</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:40:29</td>
      <td>3.4+</td>
      <td>Windows</td>
      <td>Official python.org binaries</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Never</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>A couple times a week</td>
      <td>Once a month</td>
      <td>3</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:55:46</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Official python.org binaries</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3</td>
      <td>Leave me alone - I will find it if I need it</td>
      <td>2015-06-10 04:10:17</td>
      <td>I don't care</td>
      <td>Mac</td>
      <td>Anaconda</td>
    </tr>
  </tbody>
</table>
</div>



## Explore the data

For Numeric columns, start with describe to see what we have


```python
results.describe()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>useful</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>55.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.072727</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.790090</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>



Because we only have 1, 2, 3 as options the numeric results aren't telling us that much. I am going to convert the number to more useful descriptions.


```python
results['useful'] = results['useful'].map({1: '1-low', 2: '2-medium', 3: '3-high'})
results.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>freq-js</th>
      <th>freq-py</th>
      <th>freq-r</th>
      <th>freq-ruby</th>
      <th>freq-sql</th>
      <th>freq-vba</th>
      <th>useful</th>
      <th>notify</th>
      <th>timestamp</th>
      <th>version</th>
      <th>os</th>
      <th>distro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Once a month</td>
      <td>A couple times a week</td>
      <td>Infrequently</td>
      <td>Never</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3-high</td>
      <td>RSS</td>
      <td>2015-06-09 23:22:43</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Included with OS - Mac</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>A couple times a week</td>
      <td>Never</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>3-high</td>
      <td>Reddit</td>
      <td>2015-06-10 01:19:08</td>
      <td>2.7</td>
      <td>Windows</td>
      <td>Anaconda</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Infrequently</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>Daily</td>
      <td>Never</td>
      <td>2-medium</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:40:29</td>
      <td>3.4+</td>
      <td>Windows</td>
      <td>Official python.org binaries</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Never</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>A couple times a week</td>
      <td>Once a month</td>
      <td>3-high</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:55:46</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Official python.org binaries</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3-high</td>
      <td>Leave me alone - I will find it if I need it</td>
      <td>2015-06-10 04:10:17</td>
      <td>I don't care</td>
      <td>Mac</td>
      <td>Anaconda</td>
    </tr>
  </tbody>
</table>
</div>



Value counts give us an easy distribution view into the raw numbers


```python
results["version"].value_counts()
```




    2.7             23
    3.4+            18
    I don't care    14
    dtype: int64



Use normalize to see it by percentage.


```python
results.os.value_counts(normalize=True)
```




    Windows    0.381818
    Linux      0.363636
    Mac        0.254545
    dtype: float64



While the numbers are useful, wouldn't it be nicer to visually show the results?

Seaborn's [factorplot](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.factorplot.html) is helpful for showing this kind of categorical data.

Because factorplot is so powerful, I'll build up step by step to show how it can be used for complex data analysis.

First, look at number of users by OS.


```python
sns.factorplot("os", data=results, palette="BuPu")
```




    <seaborn.axisgrid.FacetGrid at 0x7f23a34045d0>




![png](Google-Forms-Data-Analysis_files/Google-Forms-Data-Analysis_29_1.png)


It is easy to order the results using x_order


```python
sns.factorplot("os", x_order=["Linux", "Windows", "Mac"], data=results, palette="BuPu")
```




    <seaborn.axisgrid.FacetGrid at 0x7f23b1374990>




![png](Google-Forms-Data-Analysis_files/Google-Forms-Data-Analysis_31_1.png)


Do a similar plot on python version


```python
sns.factorplot("version", data=results, palette="BuPu")
```




    <seaborn.axisgrid.FacetGrid at 0x7f23a3250550>




![png](Google-Forms-Data-Analysis_files/Google-Forms-Data-Analysis_33_1.png)


This is useful but wouldn't it be better to compare with OS and preferred python version? This is where factorplot starts to show more versatility. The key component is to use hue to automatically slice the data by python version (in this case).


```python
sns.factorplot("os", hue="version", x_order=["Linux", "Windows", "Mac"], data=results, palette="Paired")
```




    <seaborn.axisgrid.FacetGrid at 0x7f23a3404cd0>




![png](Google-Forms-Data-Analysis_files/Google-Forms-Data-Analysis_35_1.png)


Because seaborn knows how to work with dataframes, we just need to pass in the column names for the various arguments and it will do the analysis and presentation.

How about if we try to see if there is any relationship between how useful the site is and OS/Python choice? We can add the useful column into the plot using col.


```python
sns.factorplot("version", hue="os", data=results, col="useful", palette="Paired")
```




    <seaborn.axisgrid.FacetGrid at 0x7f23a3078ed0>




![png](Google-Forms-Data-Analysis_files/Google-Forms-Data-Analysis_37_1.png)


If we can add a column, we can also add a row and seaborn takes care of the rest.

In looking at the data, we have two different versions of winpython so clean that up first.


```python
results['distro'] = results['distro'].str.replace('WinPython', 'winpython')
```


```python
results.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>freq-js</th>
      <th>freq-py</th>
      <th>freq-r</th>
      <th>freq-ruby</th>
      <th>freq-sql</th>
      <th>freq-vba</th>
      <th>useful</th>
      <th>notify</th>
      <th>timestamp</th>
      <th>version</th>
      <th>os</th>
      <th>distro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Once a month</td>
      <td>A couple times a week</td>
      <td>Infrequently</td>
      <td>Never</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3-high</td>
      <td>RSS</td>
      <td>2015-06-09 23:22:43</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Included with OS - Mac</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>A couple times a week</td>
      <td>Never</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>3-high</td>
      <td>Reddit</td>
      <td>2015-06-10 01:19:08</td>
      <td>2.7</td>
      <td>Windows</td>
      <td>Anaconda</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Infrequently</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>Daily</td>
      <td>Never</td>
      <td>2-medium</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:40:29</td>
      <td>3.4+</td>
      <td>Windows</td>
      <td>Official python.org binaries</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Never</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>A couple times a week</td>
      <td>Once a month</td>
      <td>3-high</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:55:46</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Official python.org binaries</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3-high</td>
      <td>Leave me alone - I will find it if I need it</td>
      <td>2015-06-10 04:10:17</td>
      <td>I don't care</td>
      <td>Mac</td>
      <td>Anaconda</td>
    </tr>
  </tbody>
</table>
</div>



We can also look at the distros. Since there is some overlap with the distros and os, let's only look at a subset of distros. For instance, someone using winpython is not going to be using it on a Mac.


```python
results['distro'].value_counts()
```




    Anaconda                        22
    Official python.org binaries    13
    Included with OS - Linux        11
    Included with OS - Mac           4
    winpython                        3
    Docker Python image              1
    3rd party packager               1
    dtype: int64



The most meaningful data would be looking at the Anaconda and Official python.org binaries. Let's filter all of our data only on these two values.


```python
results_distro = results[results["distro"].isin(["Anaconda", "Official python.org binaries"])]
results_distro.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>freq-js</th>
      <th>freq-py</th>
      <th>freq-r</th>
      <th>freq-ruby</th>
      <th>freq-sql</th>
      <th>freq-vba</th>
      <th>useful</th>
      <th>notify</th>
      <th>timestamp</th>
      <th>version</th>
      <th>os</th>
      <th>distro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>A couple times a week</td>
      <td>Never</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>3-high</td>
      <td>Reddit</td>
      <td>2015-06-10 01:19:08</td>
      <td>2.7</td>
      <td>Windows</td>
      <td>Anaconda</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Infrequently</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>Daily</td>
      <td>Never</td>
      <td>2-medium</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:40:29</td>
      <td>3.4+</td>
      <td>Windows</td>
      <td>Official python.org binaries</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Never</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>A couple times a week</td>
      <td>Once a month</td>
      <td>3-high</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:55:46</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Official python.org binaries</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3-high</td>
      <td>Leave me alone - I will find it if I need it</td>
      <td>2015-06-10 04:10:17</td>
      <td>I don't care</td>
      <td>Mac</td>
      <td>Anaconda</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td></td>
      <td></td>
      <td>A couple times a week</td>
      <td></td>
      <td>1-low</td>
      <td>Feedly</td>
      <td>2015-06-10 04:53:49</td>
      <td>2.7</td>
      <td>Windows</td>
      <td>Official python.org binaries</td>
    </tr>
  </tbody>
</table>
</div>



Now do our factorplot with multiple columns and rows using row and col.


```python
sns.factorplot("version", hue="os", data=results_distro, col="useful", row="distro", margin_titles=True, sharex=False)
```




    <seaborn.axisgrid.FacetGrid at 0x7f23a2ca6510>




![png](Google-Forms-Data-Analysis_files/Google-Forms-Data-Analysis_46_1.png)


## Responses over time

We know that we have 55 results now. It would be interesting to see how those results came in over time. Using this method, we can very simply look at this by any time period we want.

The seaborn's [timeseries](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.tsplot.html) supports this type of analysis and much more.

For ease of calculating responses over time, add a count colum for each response.


```python
results["count"] = 1
results.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>freq-js</th>
      <th>freq-py</th>
      <th>freq-r</th>
      <th>freq-ruby</th>
      <th>freq-sql</th>
      <th>freq-vba</th>
      <th>useful</th>
      <th>notify</th>
      <th>timestamp</th>
      <th>version</th>
      <th>os</th>
      <th>distro</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Once a month</td>
      <td>A couple times a week</td>
      <td>Infrequently</td>
      <td>Never</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3-high</td>
      <td>RSS</td>
      <td>2015-06-09 23:22:43</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Included with OS - Mac</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>A couple times a week</td>
      <td>Never</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>3-high</td>
      <td>Reddit</td>
      <td>2015-06-10 01:19:08</td>
      <td>2.7</td>
      <td>Windows</td>
      <td>Anaconda</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Infrequently</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>Daily</td>
      <td>Never</td>
      <td>2-medium</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:40:29</td>
      <td>3.4+</td>
      <td>Windows</td>
      <td>Official python.org binaries</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Never</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>A couple times a week</td>
      <td>Once a month</td>
      <td>3-high</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:55:46</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Official python.org binaries</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3-high</td>
      <td>Leave me alone - I will find it if I need it</td>
      <td>2015-06-10 04:10:17</td>
      <td>I don't care</td>
      <td>Mac</td>
      <td>Anaconda</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



To get totals over time, set our index to the timestamp


```python
total_results = results.set_index('timestamp')
total_results.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>freq-js</th>
      <th>freq-py</th>
      <th>freq-r</th>
      <th>freq-ruby</th>
      <th>freq-sql</th>
      <th>freq-vba</th>
      <th>useful</th>
      <th>notify</th>
      <th>version</th>
      <th>os</th>
      <th>distro</th>
      <th>count</th>
    </tr>
    <tr>
      <th>timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-06-09 23:22:43</th>
      <td>Once a month</td>
      <td>A couple times a week</td>
      <td>Infrequently</td>
      <td>Never</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3-high</td>
      <td>RSS</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Included with OS - Mac</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-06-10 01:19:08</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>A couple times a week</td>
      <td>Never</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>3-high</td>
      <td>Reddit</td>
      <td>2.7</td>
      <td>Windows</td>
      <td>Anaconda</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-06-10 01:40:29</th>
      <td>Infrequently</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>Daily</td>
      <td>Never</td>
      <td>2-medium</td>
      <td>Planet Python</td>
      <td>3.4+</td>
      <td>Windows</td>
      <td>Official python.org binaries</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-06-10 01:55:46</th>
      <td>Never</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>A couple times a week</td>
      <td>Once a month</td>
      <td>3-high</td>
      <td>Planet Python</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Official python.org binaries</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2015-06-10 04:10:17</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3-high</td>
      <td>Leave me alone - I will find it if I need it</td>
      <td>I don't care</td>
      <td>Mac</td>
      <td>Anaconda</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Use pandas TimeGrouper to summarize the data by day and do a cumulative sum. We could easily do this for any time period too.


```python
running_results = total_results.groupby(pd.TimeGrouper('D'))["count"].count().cumsum()
running_results
```




    timestamp
    2015-06-09     1
    2015-06-10    17
    2015-06-11    22
    2015-06-12    26
    2015-06-13    27
    2015-06-14    30
    2015-06-15    33
    2015-06-16    34
    2015-06-17    35
    2015-06-18    41
    2015-06-19    46
    2015-06-20    49
    2015-06-21    49
    2015-06-22    50
    2015-06-23    51
    2015-06-24    52
    2015-06-25    52
    2015-06-26    53
    2015-06-27    53
    2015-06-28    53
    2015-06-29    53
    2015-06-30    53
    2015-07-01    53
    2015-07-02    55
    Freq: D, Name: count, dtype: int64



To label the x-axis we need to define our time range


```python
step = pd.Series(range(0,len(running_results)), name="Days")
sns.tsplot(running_results, value="Total Responses", time=step, color="husl")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f23a45befd0>




![png](Google-Forms-Data-Analysis_files/Google-Forms-Data-Analysis_55_1.png)


## Heatmaps and Clustermaps

The final section of data to analyze is the frequency with which readers are using different technology. I am going to use a [heatmap](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.heatmap.html#seaborn.heatmap) to look for any interesting insights.

Let's look at the data again.


```python
results.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>freq-js</th>
      <th>freq-py</th>
      <th>freq-r</th>
      <th>freq-ruby</th>
      <th>freq-sql</th>
      <th>freq-vba</th>
      <th>useful</th>
      <th>notify</th>
      <th>timestamp</th>
      <th>version</th>
      <th>os</th>
      <th>distro</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Once a month</td>
      <td>A couple times a week</td>
      <td>Infrequently</td>
      <td>Never</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3-high</td>
      <td>RSS</td>
      <td>2015-06-09 23:22:43</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Included with OS - Mac</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>A couple times a week</td>
      <td>Never</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>3-high</td>
      <td>Reddit</td>
      <td>2015-06-10 01:19:08</td>
      <td>2.7</td>
      <td>Windows</td>
      <td>Anaconda</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Infrequently</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>Daily</td>
      <td>Never</td>
      <td>2-medium</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:40:29</td>
      <td>3.4+</td>
      <td>Windows</td>
      <td>Official python.org binaries</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Never</td>
      <td>Daily</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>A couple times a week</td>
      <td>Once a month</td>
      <td>3-high</td>
      <td>Planet Python</td>
      <td>2015-06-10 01:55:46</td>
      <td>2.7</td>
      <td>Mac</td>
      <td>Official python.org binaries</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Once a month</td>
      <td>Daily</td>
      <td>Infrequently</td>
      <td>Infrequently</td>
      <td>Once a month</td>
      <td>Never</td>
      <td>3-high</td>
      <td>Leave me alone - I will find it if I need it</td>
      <td>2015-06-10 04:10:17</td>
      <td>I don't care</td>
      <td>Mac</td>
      <td>Anaconda</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
results["freq-py"].value_counts()
```




    Daily                    36
    A couple times a week    15
    Once a month              3
                              1
    dtype: int64



What we need to do is construct a single DataFrame with all the value_counts for the specific technology.
First we will create a list containing each value count.


```python
all_counts = []
for tech in ["freq-py", "freq-sql", "freq-r", "freq-ruby", "freq-js", "freq-vba"]:
    all_counts.append(results[tech].value_counts())
display(all_counts)
```


    [Daily                    36
     A couple times a week    15
     Once a month              3
                               1
     dtype: int64, A couple times a week    18
     Daily                    13
     Infrequently             13
     Never                     5
     Once a month              4
                               2
     dtype: int64, Never                    24
     Infrequently             16
     Once a month              5
     Daily                     4
                               3
     A couple times a week     3
     dtype: int64, Never           45
     Infrequently     7
                      2
     Once a month     1
     dtype: int64, Never                    18
     Once a month             16
     Infrequently             12
     Daily                     5
     A couple times a week     3
                               1
     dtype: int64, Never           37
     Infrequently     7
     Once a month     6
     Daily            3
                      2
     dtype: int64]


Now, concat the lists along axis=1.

Fill in any nan values with 0 too.


```python
tech_usage = pd.concat(all_counts, keys=["Python", "SQL", "R", "Ruby", "javascript", "VBA"], axis=1)
tech_usage = tech_usage.fillna(0)
tech_usage
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Python</th>
      <th>SQL</th>
      <th>R</th>
      <th>Ruby</th>
      <th>javascript</th>
      <th>VBA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th></th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>A couple times a week</th>
      <td>15</td>
      <td>18</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Daily</th>
      <td>36</td>
      <td>13</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Infrequently</th>
      <td>0</td>
      <td>13</td>
      <td>16</td>
      <td>7</td>
      <td>12</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Never</th>
      <td>0</td>
      <td>5</td>
      <td>24</td>
      <td>45</td>
      <td>18</td>
      <td>37</td>
    </tr>
    <tr>
      <th>Once a month</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>16</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



We have a nice table but there are a few problems.

First, we have one column with blank values that we don't want.

Secondly, we would like to order from Daily -> Never. Use reindex to accomplish both tasks.


```python
tech_usage = tech_usage.reindex(["Daily", "A couple times a week", "Once a month", "Infrequently", "Never"])
tech_usage
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Python</th>
      <th>SQL</th>
      <th>R</th>
      <th>Ruby</th>
      <th>javascript</th>
      <th>VBA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Daily</th>
      <td>36</td>
      <td>13</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>A couple times a week</th>
      <td>15</td>
      <td>18</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Once a month</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Infrequently</th>
      <td>0</td>
      <td>13</td>
      <td>16</td>
      <td>7</td>
      <td>12</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Never</th>
      <td>0</td>
      <td>5</td>
      <td>24</td>
      <td>45</td>
      <td>18</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>



Now that the data is in the correct table format, we can create a heatmap.


```python
sns.heatmap(tech_usage, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f23a35d66d0>




![png](Google-Forms-Data-Analysis_files/Google-Forms-Data-Analysis_68_1.png)


So, what does this tell us?

Not surprisingly, most people use python very frequently.

Additionally, it looks like very few survey takers are using Ruby or VBA.

A variation of the heatmap is the [clustermap](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.clustermap.html#seaborn.clustermap). The main feature it does is that it tries to reorganize the data to more easily see relationships/clusters.


```python
sns.clustermap(tech_usage, annot=True)
```




    <seaborn.matrix.ClusterGrid at 0x7f23a275e110>




![png](Google-Forms-Data-Analysis_files/Google-Forms-Data-Analysis_71_1.png)


At first glance, it may seem to be a repeat but you'll notice that the order of the axes are different.

For instance, python and SQL are clusterd in the lower right with higher usage and Ruby and VBA have a cluster in the upper left with lower usage.
