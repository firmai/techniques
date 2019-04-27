

```python
import quantecon as qe

```


```python
### I don't recommend pyliferisk

import pyliferisk.lifecontingencies as lc
from pyliferisk.mortalitytables import SPAININE2004
```


```python
tariff=lc.MortalityTable(nt=SPAININE2004)
experience=lc.MortalityTable(nt=SPAININE2004,perc=85)

# Print the omega (limiting age) of the both mortality tables:
print(tariff.w)
print(experience.w)

# Print the qx at 50 years old:
## QX is the rate of mortality.
print(tariff.qx[50])
print(experience.qx[50])
```

    101
    102
    3.113
    2.6460500000000002



```python
from pyliferisk.lifecontingencies import Actuarial
from pyliferisk.mortalitytables import SPAININE2004, GKM95

mt=Actuarial(nt=SPAININE2004,i=0.02)
lc.annuity(mt,50,10,12,['g',0.03],-15)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-9-7e45544755e0> in <module>()
          3 
          4 mt=Actuarial(nt=SPAININE2004,i=0.02)
    ----> 5 lc.annuity(mt,50,10,12,['g',0.03],-15)
    

    ~/anaconda/envs/py36/lib/python3.6/site-packages/pyliferisk/lifecontingencies.py in annuity(mt, x, n, p, m, *args)
        450     wh_l = False
        451 
    --> 452     if isinstance(n,basestring) or n == 99:
        453         wh_l = True
        454     else:


    NameError: name 'basestring' is not defined

