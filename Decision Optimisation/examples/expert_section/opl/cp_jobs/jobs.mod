/*********************************************
 * OPL job scheduling problem organized into tabular, ticdat compliant format
 * See https://ibm.co/2rGVyet for reference problem and sample data.
 *********************************************/

/* ------------------------ begin data initialization section ---------------------- */
using CP;
include "ticdat_jobs.mod";

/* Tasks and 'areaA', 'areaS' are hard coded into the mathematics section,
 * so we hard code them here as well
 */

{string} Jobs = {j | <j,m1,d1,m2,d2> in inp_jobs};
{string} Tasks  = {"loadA","unload1","process1","load1","unload2","process2","load2","unloadS"};

{string} Machines = inp_machines;
{string} States = Machines union {"areaA","areaS"};

int Index[s in States] = ord(States, s);

tuple jobRecord {
    string machine1;
    int    durations1;
    string machine2;
    int    durations2;
}
jobRecord job[Jobs] = [j: <m1,ftoi(d1),m2,ftoi(d2)> | <j,m1,d1,m2,d2> in inp_jobs];

{string} inputParameterNames = {k | <k,v> in inp_parameters};
float parameters[inputParameterNames] = [k:v | <k,v> in inp_parameters];
int loadDuration = ftoi(parameters["Load Duration"]);
/* ------------------------ end data initialization section ------------------------ */

/* ------------------------ begin core mathematics section ------------------------- */
dvar interval act[Jobs][Tasks];

stateFunction trolleyPosition;

minimize max(j in Jobs) endOf(act[j]["unloadS"]);
subject to {
   // durations
   forall(j in Jobs) {
     lengthOf(act[j]["loadA"])    == loadDuration;
     lengthOf(act[j]["unload1"])  == loadDuration;
     lengthOf(act[j]["process1"]) == job[j].durations1;
     lengthOf(act[j]["load1"])    == loadDuration;
     lengthOf(act[j]["unload2"])  == loadDuration;
     lengthOf(act[j]["process2"]) == job[j].durations2;
     lengthOf(act[j]["load2"])    == loadDuration;
     lengthOf(act[j]["unloadS"])  == loadDuration;
   };

   // precedence
   forall(j in Jobs)
        forall(ordered t1, t2 in Tasks)
          endBeforeStart(act[j][t1], act[j][t2]);

   // no-overlap on machines
   forall (m in Machines) {
     noOverlap( append(
               all(j in Jobs: job[j].machine1==m) act[j]["process1"],
               all(j in Jobs: job[j].machine2==m) act[j]["process2"])
            );
   }

    // state constraints
    forall(j in Jobs) {
        alwaysEqual(trolleyPosition, act[j]["loadA"],   Index["areaA"]);
        alwaysEqual(trolleyPosition, act[j]["unload1"], Index[job[j].machine1]);
        alwaysEqual(trolleyPosition, act[j]["load1"],   Index[job[j].machine1]);
        alwaysEqual(trolleyPosition, act[j]["unload2"], Index[job[j].machine2]);
        alwaysEqual(trolleyPosition, act[j]["load2"],   Index[job[j].machine2]);
        alwaysEqual(trolleyPosition, act[j]["unloadS"], Index["areaS"]);
    };
 };

/* ------------------------ end core mathematics section --------------------------- */


/* ------------------------ begin ticdat output section ---------------------------- */
include "ticdat_jobs_output.mod";

execute {

  for (var j in Jobs){
     for (var t in Tasks){
        sln_act.add(j,t,act[j][t]['start'], act[j][t]['end']);
     }
  }

  writeOutputToFile();
}
/* ------------------------ end ticdat output section ------------------------------ */