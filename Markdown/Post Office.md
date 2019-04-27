"""

  Post office problem in Google CP Solver.

  Problem statement:
  http://www-128.ibm.com/developerworks/linux/library/l-glpk2/

  From Winston 'Operations Research: Applications and Algorithms':
  '''
  A post office requires a different number of full-time employees working
  on different days of the week [summarized below]. Union rules state that
  each full-time employee must work for 5 consecutive days and then receive
  two days off. For example, an employee who works on Monday to Friday
  must be off on Saturday and Sunday. The post office wants to meet its
  daily requirements using only full-time employees. Minimize the number
  of employees that must be hired.

  To summarize the important information about the problem:

    * Every full-time worker works for 5 consecutive days and takes 2 days off
    * Day 1 (Monday): 17 workers needed
    * Day 2 : 13 workers needed
    * Day 3 : 15 workers needed
    * Day 4 : 19 workers needed
    * Day 5 : 14 workers needed
    * Day 6 : 16 workers needed
    * Day 7 (Sunday) : 11 workers needed

  The post office needs to minimize the number of employees it needs
  to hire to meet its demand.
  '''

"""

```python
from __future__ import print_function
from ortools.constraint_solver import pywrapcp


# Create the solver.
solver = pywrapcp.Solver('Post office problem')

#
# data
#

# days 0..6, monday 0
n = 7
days = list(range(n))
need = [17, 13, 15, 19, 14, 16, 11]

# Total cost for the 5 day schedule.
# Base cost per day is 100.
# Working saturday is 100 extra
# Working sunday is 200 extra.
cost = [500, 600, 800, 800, 800, 800, 700]

#
# variables
#

# No. of workers starting at day i
x = [solver.IntVar(0, 1000, 'x[%i]' % i) for i in days]

total_cost = solver.IntVar(0, 20000, 'total_cost')
num_workers = solver.IntVar(0, 1000, 'num_workers')

#
# constraints
# variable not in consraint is not a constraint
solver.Add(total_cost == solver.ScalProd(x, cost))
solver.Add(num_workers == solver.Sum(x))

for i in days:
    s = solver.Sum([x[j] for j in days
                    if j != (i + 5) % n and j != (i + 6) % n])
    solver.Add(s >= need[i])

    
# objective
objective = solver.Minimize(total_cost, 1)

#
# search and result
#
db = solver.Phase(x,
                solver.CHOOSE_MIN_SIZE_LOWEST_MIN,
                solver.ASSIGN_MIN_VALUE)

solver.NewSearch(db, [objective])

num_solutions = 0

while solver.NextSolution():
    num_solutions += 1
    print('num_workers:', num_workers.Value())
    print('total_cost:', total_cost.Value())
    print('x:', [x[i].Value() for i in days])

solver.EndSearch()

print()
print('num_solutions:', num_solutions)
print('failures:', solver.Failures())
print('branches:', solver.Branches())
print('WallTime:', solver.WallTime())

## These are the number of workers starting.
```

    num_workers: 31
    total_cost: 19800
    x: [4, 12, 0, 0, 1, 0, 14]
    num_workers: 30
    total_cost: 19300
    x: [4, 11, 0, 0, 2, 0, 13]
    num_workers: 29
    total_cost: 18800
    x: [4, 10, 0, 0, 3, 0, 12]
    num_workers: 28
    total_cost: 18300
    x: [4, 9, 0, 0, 4, 0, 11]
    num_workers: 27
    total_cost: 17800
    x: [4, 8, 0, 0, 5, 0, 10]
    num_workers: 26
    total_cost: 17300
    x: [4, 7, 0, 0, 6, 0, 9]
    num_workers: 25
    total_cost: 16800
    x: [4, 6, 0, 0, 7, 0, 8]
    num_workers: 24
    total_cost: 16300
    x: [4, 5, 0, 0, 8, 0, 7]
    num_workers: 23
    total_cost: 15800
    x: [6, 2, 2, 0, 7, 2, 4]
    
    num_solutions: 9
    failures: 3795
    branches: 7605
    WallTime: 79

