#
# This file solves the netflow model with a static (i.e. hard coded) small data example.
#

from netflow import solve, input_schema
dat = input_schema.TicDat(
    commodities = ['Pencils', 'Pens'],
    nodes = ['Detroit', 'Denver', 'Boston', 'New York', 'Seattle'],
    arcs = {('Detroit', 'Boston'):   100,
            ('Detroit', 'New York'):  80,
            ('Detroit', 'Seattle'):  120,
            ('Denver',  'Boston'):   120,
            ('Denver',  'New York'): 120,
            ('Denver',  'Seattle'):  120},

    cost = {('Pencils', 'Detroit', 'Boston'):   10,
            ('Pencils', 'Detroit', 'New York'): 20,
            ('Pencils', 'Detroit', 'Seattle'):  60,
            ('Pencils', 'Denver',  'Boston'):   40,
            ('Pencils', 'Denver',  'New York'): 40,
            ('Pencils', 'Denver',  'Seattle'):  30,
            ('Pens',    'Detroit', 'Boston'):   20,
            ('Pens',    'Detroit', 'New York'): 20,
            ('Pens',    'Detroit', 'Seattle'):  80,
            ('Pens',    'Denver',  'Boston'):   60,
            ('Pens',    'Denver',  'New York'): 70,
            ('Pens',    'Denver',  'Seattle'):  30 },
    inflow = {  ('Pencils', 'Detroit'):   50,
                ('Pencils', 'Denver'):    60,
                ('Pencils', 'Boston'):   -50,
                ('Pencils', 'New York'): -50,
                ('Pencils', 'Seattle'):  -10,
                ('Pens',    'Detroit'):   60,
                ('Pens',    'Denver'):    40,
                ('Pens',    'Boston'):   -40,
                ('Pens',    'New York'): -30,
                ('Pens',    'Seattle'):  -30 })

solution =  solve(dat)

if solution :
    print('\nCost: %g' % solution.parameters["Total Cost"]["Value"])
    print('\nFlow:')
    for (h,i,j),r in solution.flow.items():
        print('%s %s %s %g' % (h, i, j, r["Quantity"]))
else :
    print('\nNo solution')
