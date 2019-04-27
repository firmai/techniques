

```python
import numpy as np
import quantecon.game_theory as gt
```



In our module, a normal form game and a player are represented by the classes NormalFormGame and Player, respectively.

A Player carries the player's payoff function and implements in particular a method that returns the best response action(s) given an action of the opponent player, or a profile of actions of the opponents if there are more than one.

A NormalFormGame is in effect a container of Player instances.



```python
matching_pennies_bimatrix = [[(1, -1), (-1, 1)],
                             [(-1, 1), (1, -1)]]
g_MP = gt.NormalFormGame(matching_pennies_bimatrix)

print(g_MP)
print("Player Instance; \n",g_MP.players[1])  # Player instance for player 1
print("Player 1's payoff array\n",g_MP.players[1].payoff_array)  # Player 1's payoff array
print("payoff profile for action profile (0, 0)\n",g_MP[0, 0])  # payoff profile for action profile (0, 0)



```

    2-player NormalFormGame with payoff profile array:
    [[[ 1, -1],  [-1,  1]],
     [[-1,  1],  [ 1, -1]]]
    Player Instance; 
     Player in a 2-player normal form game with payoff array:
    [[-1,  1],
     [ 1, -1]]
    Player 1's payoff array
     [[-1  1]
     [ 1 -1]]
    payoff profile for action profile (0, 0)
     [ 1 -1]


The game_theory module also supports games with more than two players.

Let us consider the following version of N
-player Cournot Game.


```python
from quantecon import cartesian


def cournot(a, c, N, q_grid):
    """
    Create a `NormalFormGame` instance for the symmetric N-player Cournot game
    with linear inverse demand a - Q and constant marginal cost c.

    Parameters
    ----------
    a : scalar
        Intercept of the demand curve

    c : scalar
        Common constant marginal cost

    N : scalar(int)
        Number of firms

    q_grid : array_like(scalar)
        Array containing the set of possible quantities

    Returns
    -------
    NormalFormGame
        NormalFormGame instance representing the Cournot game

    """
    q_grid = np.asarray(q_grid)
    payoff_array = \
        cartesian([q_grid]*N).sum(axis=-1).reshape([len(q_grid)]*N) * (-1) + \
        (a - c)
    payoff_array *= q_grid.reshape([len(q_grid)] + [1]*(N-1))
    payoff_array += 0  # To get rid of the minus sign of -0

    player = gt.Player(payoff_array)
    return gt.NormalFormGame([player for i in range(N)])
```

Here's a simple example with three firms, marginal cost 20, and inverse demand function 80−Q, where the feasible quantity values are assumed to be 10 and 15.


```python


a, c = 80, 20
N = 3
q_grid = [10, 15]  # [1/3 of Monopoly quantity, Nash equilibrium quantity]

g_Cou = cournot(a, c, N, q_grid)


```


```python
print(g_Cou)
```

    3-player NormalFormGame with payoff profile array:
    [[[[300, 300, 300],   [250, 250, 375]],
      [[250, 375, 250],   [200, 300, 300]]],
    
     [[[375, 250, 250],   [300, 200, 300]],
      [[300, 300, 200],   [225, 225, 225]]]]


Finding Nash equilibria

There are several algorithms implemented to compute Nash equilibria:

Brute force
Find all pure-action Nash equilibria of an N player game (if any).
Sequential best response
Find one pure-action Nash equilibrium of an N player game (if any).
Support enumeration
Find all mixed-action Nash equilibria of a two-player nondegenerate game.
Vertex enumeration
Find all mixed-action Nash equilibria of a two-player nondegenerate game.
Lemke-Howson
Find one mixed-action Nash equilibrium of a two-player game.
McLennan-Tourky
Find one mixed-action Nash equilibrium of an N player game.

For more variety of algorithms, one should look at Gambit.

```python
def print_pure_nash_brute(g):
    """
    Print all pure Nash equilibria of a normal form game found by brute force.
    
    Parameters
    ----------
    g : NormalFormGame
    
    """
    NEs = gt.pure_nash_brute(g)
    num_NEs = len(NEs)
    if num_NEs == 0:
        msg = 'no pure Nash equilibrium'
    elif num_NEs == 1:
        msg = '1 pure Nash equilibrium:\n{0}'.format(NEs)
    else:
        msg = '{0} pure Nash equilibria:\n{1}'.format(num_NEs, NEs)

    print('The game has ' + msg)
```


```python
print_pure_nash_brute(g_Cou)
```

    The game has 1 pure Nash equilibrium:
    [(1, 1, 1)]



```python
def sequential_best_response(g, init_actions=None, tie_breaking='smallest',
                             verbose=True):
    """
    Find a pure Nash equilibrium of a normal form game by sequential best
    response.

    Parameters
    ----------
    g : NormalFormGame

    init_actions : array_like(int), optional(default=[0, ..., 0])
        The initial action profile.

    tie_breaking : {'smallest', 'random'}, optional(default='smallest')

    verbose: bool, optional(default=True)
        If True, print the intermediate process.

    """
    N = g.N  # Number of players
    a = np.empty(N, dtype=int)  # Action profile
    if init_actions is None:
        init_actions = [0] * N
    a[:] = init_actions

    if verbose:
        print('init_actions: {0}'.format(a))

    new_a = np.empty(N, dtype=int)
    max_iter = np.prod(g.nums_actions)

    for t in range(max_iter):
        new_a[:] = a
        for i, player in enumerate(g.players):
            if N == 2:
                a_except_i = new_a[1-i]
            else:
                a_except_i = new_a[np.arange(i+1, i+N) % N]
            new_a[i] = player.best_response(a_except_i,
                                            tie_breaking=tie_breaking)
            if verbose:
                print('player {0}: {1}'.format(i, new_a))
        if np.array_equal(new_a, a):
            return a
        else:
            a[:] = new_a

    print('No pure Nash equilibrium found')
    return None
```

A Cournot game with linear demand is known to be a potential game, for which sequential best response converges to a Nash equilibrium.

Let us try a bigger instance:



```python


a, c = 80, 20
N = 3
q_grid = np.linspace(0, a-c, 13)  # [0, 5, 10, ..., 60]
g_Cou = cournot(a, c, N, q_grid)


```


```python
a_star = sequential_best_response(g_Cou)  # By default, start with (0, 0, 0)
print('Nash equilibrium indices: {0}'.format(a_star))
print('Nash equilibrium quantities: {0}'.format(q_grid[a_star]))
```

    init_actions: [0 0 0]
    player 0: [6 0 0]
    player 1: [6 3 0]
    player 2: [6 3 1]
    player 0: [4 3 1]
    player 1: [4 3 1]
    player 2: [4 3 2]
    player 0: [3 3 2]
    player 1: [3 3 2]
    player 2: [3 3 3]
    player 0: [3 3 3]
    player 1: [3 3 3]
    player 2: [3 3 3]
    Nash equilibrium indices: [3 3 3]
    Nash equilibrium quantities: [15. 15. 15.]



```python


# Start with the largest actions (12, 12, 12)
sequential_best_response(g_Cou, init_actions=(12, 12, 12))


```

    init_actions: [12 12 12]
    player 0: [ 0 12 12]
    player 1: [ 0  0 12]
    player 2: [0 0 6]
    player 0: [3 0 6]
    player 1: [3 1 6]
    player 2: [3 1 4]
    player 0: [3 1 4]
    player 1: [3 2 4]
    player 2: [3 2 3]
    player 0: [3 2 3]
    player 1: [3 3 3]
    player 2: [3 3 3]
    player 0: [3 3 3]
    player 1: [3 3 3]
    player 2: [3 3 3]





    array([3, 3, 3])




```python
# The limit action profile is indeed a Nash equilibrium:
g_Cou.is_nash(a_star)
```




    True




```python
# In fact, the game has other Nash equilibria (because of our choice of grid points and parameter values):

print_pure_nash_brute(g_Cou)
```

    The game has 7 pure Nash equilibria:
    [(2, 3, 4), (2, 4, 3), (3, 2, 4), (3, 3, 3), (3, 4, 2), (4, 2, 3), (4, 3, 2)]




Next, let us study the All-Pay Acution, where, unlike standard auctions, bidders pay their bids regardless of whether or not they win. Situations modeled as all-pay auctions include job promotion, R&D, and rent seeking competitions, among others.

Here we consider a version of All-Pay Auction with complete information, symmetric bidders, discrete bids, bid caps, and "full dissipation" (where the prize is materialized if and only if there is only one bidder who makes a highest bid).



```python
from numba import jit


def all_pay_auction(r, c, N, dissipation=True):
    """
    Create a `NormalFormGame` instance for the symmetric N-player
    All-Pay Auction game with common reward `r` and common bid cap `e`.

    Parameters
    ----------
    r : scalar(float)
        Common reward value.

    c : scalar(int)
        Common bid cap.

    N : scalar(int)
        Number of players.

    dissipation : bool, optional(default=True)
        If True, the prize fully dissipates in case of a tie. If False,
        the prize is equally split among the highest bidders (or given
        to one of the highest bidders with equal probabilities).

    Returns
    -------
    NormalFormGame
        NormalFormGame instance representing the All-Pay Auction game.

    """
    player = gt.Player(np.empty((c+1,)*N))
    populate_APA_payoff_array(r, dissipation, player.payoff_array)
    return gt.NormalFormGame((player,)*N)


@jit(nopython=True)
def populate_APA_payoff_array(r, dissipation, out):
    """
    Populate the payoff array for a player in an N-player All-Pay
    Auction game.

    Parameters
    ----------
    r : scalar(float)
        Reward value.

    dissipation : bool, optional(default=True)
        If True, the prize fully dissipates in case of a tie. If False,
        the prize is equally split among the highest bidders (or given
        to one of the highest bidders with equal probabilities).

    out : ndarray(float, ndim=N)
        NumPy array to store the payoffs. Modified in place.

    Returns
    -------
    out : ndarray(float, ndim=N)
        View of `out`.

    """
    nums_actions = out.shape
    N = out.ndim
    for bids in np.ndindex(nums_actions):
        out[bids] = -bids[0]
        num_ties = 1
        for j in range(1, N):
            if bids[j] > bids[0]:
                num_ties = 0
                break
            elif bids[j] == bids[0]:
                if dissipation:
                    num_ties = 0
                    break
                else:
                    num_ties += 1
        if num_ties > 0:
            out[bids] += r / num_ties
    return out
```


```python
# Consider the two-player case with the following parameter values:

N = 2
c = 5  # odd
r = 8

g_APA_odd = all_pay_auction(r, c, N)
print(g_APA_odd)
```

    2-player NormalFormGame with payoff profile array:
    [[[ 0.,  0.],  [ 0.,  7.],  [ 0.,  6.],  [ 0.,  5.],  [ 0.,  4.],  [ 0.,  3.]],
     [[ 7.,  0.],  [-1., -1.],  [-1.,  6.],  [-1.,  5.],  [-1.,  4.],  [-1.,  3.]],
     [[ 6.,  0.],  [ 6., -1.],  [-2., -2.],  [-2.,  5.],  [-2.,  4.],  [-2.,  3.]],
     [[ 5.,  0.],  [ 5., -1.],  [ 5., -2.],  [-3., -3.],  [-3.,  4.],  [-3.,  3.]],
     [[ 4.,  0.],  [ 4., -1.],  [ 4., -2.],  [ 4., -3.],  [-4., -4.],  [-4.,  3.]],
     [[ 3.,  0.],  [ 3., -1.],  [ 3., -2.],  [ 3., -3.],  [ 3., -4.],  [-5., -5.]]]



```python
## no pure nash

gt.pure_nash_brute(g_APA_odd)
```




    []



As pointed out by Dechenaux et al. (2006), there are three Nash equilibria when the bid cap c is odd (so that there are an even number of actions for each player):


```python
gt.support_enumeration(g_APA_odd)

```




    [(array([0.5 , 0.  , 0.25, 0.  , 0.25, 0.  ]),
      array([0.  , 0.25, 0.  , 0.25, 0.  , 0.5 ])),
     (array([0.  , 0.25, 0.  , 0.25, 0.  , 0.5 ]),
      array([0.5 , 0.  , 0.25, 0.  , 0.25, 0.  ])),
     (array([0.125, 0.125, 0.125, 0.125, 0.125, 0.375]),
      array([0.125, 0.125, 0.125, 0.125, 0.125, 0.375]))]




```python
c = 6  # even
g_APA_even = all_pay_auction(r, c, N)
gt.support_enumeration(g_APA_even)
```




    [(array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.25 ]),
      array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.25 ]))]


