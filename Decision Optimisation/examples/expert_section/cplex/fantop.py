#
# Author: Pete Cacioppi Opalytics.com https://lnkd.in/bUhfyNn
#
# Solves a fantasy football drafting problem. Tries to maximize the weighted
# expected points of a draft, while obeying min/max restrictions for different
# positions (to include a maximum-flex-players constraint).
#
# Pre-computes the expected draft position of each player, so as to prevent
# creating a draft plan based on unrealistic expectations of player availability
# at each round.
#
# The current draft standing can be filled in as you go in the drafted table.
# A user can thus re-optimize for each of his draft picks.
#
# Uses the ticdat package to simplify file IO and provide command line functionality.
# Can read from .csv, Access, Excel or SQLite files. Self validates the input data
# before solving to prevent strange errors or garbage-in, garbage-out problems.

from ticdat import TicDatFactory, standard_main


# this version of the file uses CPLEX
from docplex.mp.model import Model

# ------------------------ define the input schema --------------------------------
input_schema = TicDatFactory (
 parameters = [["Key"],["Value"]],
 players = [['Player Name'],
            ['Position', 'Average Draft Position', 'Expected Points', 'Draft Status']],
 roster_requirements = [['Position'],
                       ['Min Num Starters', 'Max Num Starters', 'Min Num Reserve', 'Max Num Reserve',
                        'Flex Status']],
 my_draft_positions = [['Draft Position'],[]]
)

# add foreign key constraints (optional, but helps with preventing garbage-in, garbage-out)
input_schema.add_foreign_key("players", "roster_requirements", ['Position', 'Position'])

# set data types (optional, but helps with preventing garbage-in, garbage-out)
input_schema.set_data_type("parameters", "Key", number_allowed = False,
                          strings_allowed = ["Starter Weight", "Reserve Weight",
                                             "Maximum Number of Flex Starters"])
input_schema.set_data_type("parameters", "Value", min=0, max=float("inf"),
                          inclusive_min = True, inclusive_max = False)
input_schema.set_data_type("players", "Average Draft Position", min=0, max=float("inf"),
                          inclusive_min = False, inclusive_max = False)
input_schema.set_data_type("players", "Expected Points", min=-float("inf"), max=float("inf"),
                          inclusive_min = False, inclusive_max = False)
input_schema.set_data_type("players", "Draft Status",
                          strings_allowed = ["Un-drafted", "Drafted By Me", "Drafted By Someone Else"])
for fld in ("Min Num Starters",  "Min Num Reserve", "Max Num Reserve"):
    input_schema.set_data_type("roster_requirements", fld, min=0, max=float("inf"),
                          inclusive_min = True, inclusive_max = False, must_be_int = True)
input_schema.set_data_type("roster_requirements", "Max Num Starters", min=0, max=float("inf"),
                      inclusive_min = False, inclusive_max = True, must_be_int = True)
input_schema.set_data_type("roster_requirements", "Flex Status", number_allowed = False,
                          strings_allowed = ["Flex Eligible", "Flex Ineligible"])
input_schema.set_data_type("my_draft_positions", "Draft Position", min=0, max=float("inf"),
                          inclusive_min = False, inclusive_max = False, must_be_int = True)
# ---------------------------------------------------------------------------------


# ------------------------ define the output schema -------------------------------
solution_schema = TicDatFactory(
        parameters = [["Key"],["Value"]],
        my_draft = [['Player Name'], ['Draft Position', 'Position', 'Planned Or Actual',
                                     'Starter Or Reserve']])
# ---------------------------------------------------------------------------------


# ------------------------ create a solve function --------------------------------
def solve(dat):
    assert input_schema.good_tic_dat_object(dat)
    assert not input_schema.find_foreign_key_failures(dat)
    assert not input_schema.find_data_type_failures(dat)

    expected_draft_position = {}
    # for our purposes, its fine to assume all those drafted by someone else are drafted
    # prior to any players drafted by me
    for player_name in sorted(dat.players,
                              key=lambda _p: {"Un-drafted":dat.players[_p]["Average Draft Position"],
                                              "Drafted By Me":-1,
                                              "Drafted By Someone Else":-2}[dat.players[_p]["Draft Status"]]):
        expected_draft_position[player_name] = len(expected_draft_position) + 1
    assert max(expected_draft_position.values()) == len(set(expected_draft_position.values())) == len(dat.players)
    assert min(expected_draft_position.values()) == 1

    already_drafted_by_me = {player_name for player_name,row in dat.players.items() if
                            row["Draft Status"] == "Drafted By Me"}
    can_be_drafted_by_me = {player_name for player_name,row in dat.players.items() if
                            row["Draft Status"] != "Drafted By Someone Else"}

    m = Model('fantop')
    my_starters = {player_name:m.binary_var(name="starter_%s"%player_name)
                  for player_name in can_be_drafted_by_me}
    my_reserves = {player_name:m.binary_var(name="reserve_%s"%player_name)
                  for player_name in can_be_drafted_by_me}


    for player_name in can_be_drafted_by_me:
        if player_name in already_drafted_by_me:
            m.add_constraint(my_starters[player_name] + my_reserves[player_name] == 1,
                             ctname="already_drafted_%s"%player_name)
        else:
            m.add_constraint(my_starters[player_name] + my_reserves[player_name] <= 1,
                             ctname="cant_draft_twice_%s"%player_name)

    for i,draft_position in enumerate(sorted(dat.my_draft_positions)):
        m.add_constraint(m.sum(my_starters[player_name] + my_reserves[player_name]
                                for player_name in can_be_drafted_by_me
                                if expected_draft_position[player_name] < draft_position) <= i,
                         ctname = "at_most_%s_can_be_ahead_of_%s"%(i,draft_position))

    my_draft_size = m.sum(my_starters[player_name] + my_reserves[player_name]
                          for player_name in can_be_drafted_by_me)
    m.add_constraint(my_draft_size >= len(already_drafted_by_me) + 1,
                     ctname = "need_to_extend_by_at_least_one")
    m.add_constraint(my_draft_size <= len(dat.my_draft_positions), ctname = "cant_exceed_draft_total")

    for position, row in dat.roster_requirements.items():
        players = {player_name for player_name in can_be_drafted_by_me
                   if dat.players[player_name]["Position"] == position}
        starters = m.sum(my_starters[player_name] for player_name in players)
        reserves = m.sum(my_reserves[player_name] for player_name in players)
        m.add_constraint(starters >= row["Min Num Starters"], ctname = "min_starters_%s"%position)
        m.add_constraint(starters <= row["Max Num Starters"], ctname = "max_starters_%s"%position)
        m.add_constraint(reserves >= row["Min Num Reserve"], ctname = "min_reserve_%s"%position)
        m.add_constraint(reserves <= row["Max Num Reserve"], ctname = "max_reserve_%s"%position)

    if "Maximum Number of Flex Starters" in dat.parameters:
        players = {player_name for player_name in can_be_drafted_by_me if
                   dat.roster_requirements[dat.players[player_name]["Position"]]["Flex Status"] == "Flex Eligible"}
        m.add_constraint(m.sum(my_starters[player_name] for player_name in players)
                    <= dat.parameters["Maximum Number of Flex Starters"]["Value"],
                    ctname = "max_flex")

    starter_weight = dat.parameters["Starter Weight"]["Value"] if "Starter Weight" in dat.parameters else 1
    reserve_weight = dat.parameters["Reserve Weight"]["Value"] if "Reserve Weight" in dat.parameters else 1
    m.maximize(m.sum(dat.players[player_name]["Expected Points"] *
                               (my_starters[player_name] * starter_weight + my_reserves[player_name] * reserve_weight)
                               for player_name in can_be_drafted_by_me))

    if not m.solve():
        print("No draft at all is possible!")
        return

    sln = solution_schema.TicDat()
    def almostone(x):
        return abs(m.solution.get_value(x)-1) < 0.0001
    picked = sorted([player_name for player_name in can_be_drafted_by_me
                     if almostone(my_starters[player_name]) or almostone(my_reserves[player_name])],
                    key=lambda _p: expected_draft_position[_p])
    assert len(picked) <= len(dat.my_draft_positions)
    if len(picked) < len(dat.my_draft_positions):
        print("Your model is over-constrained, and thus only a partial draft was possible")

    draft_yield = 0
    for player_name, draft_position in zip(picked, sorted(dat.my_draft_positions)):
        draft_yield += dat.players[player_name]["Expected Points"] * \
                       (starter_weight if almostone(my_starters[player_name]) else reserve_weight)
        assert draft_position <= expected_draft_position[player_name]
        sln.my_draft[player_name]["Draft Position"] = draft_position
        sln.my_draft[player_name]["Position"] = dat.players[player_name]["Position"]
        sln.my_draft[player_name]["Planned Or Actual"] = "Actual" if player_name in already_drafted_by_me else "Planned"
        sln.my_draft[player_name]["Starter Or Reserve"] = \
            "Starter" if almostone(my_starters[player_name]) else "Reserve"
    sln.parameters["Total Yield"] = draft_yield
    sln.parameters["Feasible"] = len(sln.my_draft) == len(dat.my_draft_positions)

    return sln
# ---------------------------------------------------------------------------------

# ------------------------ provide stand-alone functionality ----------------------
# when run from the command line, will read/write json/xls/csv/db/mdb files
if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)
# ---------------------------------------------------------------------------------