# Simplest netflow example using amplpy, Gurobi and ticdat

from ticdat import TicDatFactory, standard_main
from amplpy import AMPL

input_schema = TicDatFactory (
     commodities = [["Name"],[]],
     nodes  = [["Name"],[]],
     arcs = [["Source", "Destination"],["Capacity"]],
     cost = [["Commodity", "Source", "Destination"], ["Cost"]],
     inflow = [["Commodity", "Node"],["Quantity"]]
)

solution_schema = TicDatFactory(
        flow = [["Commodity", "Source", "Destination"], ["Quantity"]],
        parameters = [["Key"],["Value"]])

def solve(dat):
    assert input_schema.good_tic_dat_object(dat)

    # copy the data over to amplpy.DataFrame objects, renaming the data fields as needed
    dat = input_schema.copy_to_ampl(dat, field_renamings={("arcs", "Capacity"): "capacity",
            ("cost", "Cost"): "cost", ("inflow", "Quantity"): "inflow"})

    ampl = AMPL()
    ampl.setOption('solver', 'gurobi')
    ampl.eval("""
    set NODES;
    set ARCS within {i in NODES, j in NODES: i <> j};
    set COMMODITIES;

    param capacity {ARCS} >= 0;
    param cost {COMMODITIES,ARCS} > 0;
    param inflow {COMMODITIES,NODES};

    var Flow {COMMODITIES,ARCS} >= 0;

    minimize TotalCost:
       sum {h in COMMODITIES, (i,j) in ARCS} cost[h,i,j] * Flow[h,i,j];

    subject to Capacity {(i,j) in ARCS}:
       sum {h in COMMODITIES} Flow[h,i,j] <= capacity[i,j];

    subject to Conservation {h in COMMODITIES, j in NODES}:
       sum {(i,j) in ARCS} Flow[h,i,j] + inflow[h,j] = sum {(j,i) in ARCS} Flow[h,j,i];
    """)

    input_schema.set_ampl_data(dat, ampl, {"nodes": "NODES", "arcs": "ARCS",
                                           "commodities": "COMMODITIES"})
    ampl.solve()

    if ampl.getValue("solve_result") != "infeasible":
        sln = solution_schema.copy_from_ampl_variables(
            {('flow' ,'Quantity'):ampl.getVariable("Flow")})
        sln.parameters["Total Cost"] = ampl.getObjective('TotalCost').value()
        return sln

# when run from the command line, will read/write json/xls/csv/db/sql/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)