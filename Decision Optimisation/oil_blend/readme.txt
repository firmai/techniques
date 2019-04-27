An expected use of the ticdat.opl_run functionality would be to add Python connectivity to
pre-existing OPL projects. In order to streamline this process, its important to simplify the
extent to which the current .mod file needs to be edited in order to communicate with ticdat.
The TicDatFactory.opl_prepend attribute anticipates this concern by giving you the ability
to prepend a short string onto the names of the data structures created by the auto-generated
.mod files. This allows you to define the input and solution schemas with whatever table names
make the most sense, without concern that such names will collide with variable and tuple names
in your current .mod file.

As a result, the core mathematical logic of your .mod files can be preserved completely. The
porting process only requires you to edit the data initialization section at the beginning of
your .mod file, and add a simple "write to ticdat" section at the end of your .mod file.
(Of course, a ticdat-based Python configuration file is also required).

To demonstrate, please consider the OPL oil blending example at https://goo.gl/kqXmQE. This
example contains an oil.mod file that solves the oil blending problem, and an oil.dat file
that defines a sample data set. We used this oil.dat as the starting point for the ticdat
compatible oil_blend.mod file presented here. Specifically, oil_blend.mod contains three sections.

 * A data initialization section. This was created via a simple editing process of the first 22
   lines of oil.mod. In oil_blend.mod, Gasolines, Oils, Gas, Oil, MaxProduction and ProdCost are
   all populated from the the relevant data provided by inp_gas, inp_oil, and inp_parameters. When
   comparing to oil.mod, you can see that just the six lines that use ...; were altered, and the
   inputParameterNames, parameters data structures were added. The rest of the code in this
   section is exactly the same as in oil.mod.

 * A core mathematical section. This section of oil_blend.mod performs the actual optimization. It
   is copied over unaltered from oil.mod.

 * A ticdat output section. This section of the code was added to populate the sln_ data
   structures created for the solution schema. It computes a series of KPIs to be stored in the
   solution parameters table, and then populates all three solution tables in an execute block.

As you can see, even though the input schema contains tables named gas and oil, the resulting
auto-generated code populates data structures named inp_gas and inp_oil. This avoids naming
collisions with the gas and oil variables defined in oil.mod, and precludes any need to rename 
these variables in oil_blend.mod. Similarly, the parameters tables for the input and solution 
schema correspond to inp_parameters and sln_parameters, and thus it is easy to distinguish between 
them in oil_blend.mod.
