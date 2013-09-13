'''
Author: Chase Coleman and Spencer Lyon

This file contains a class that takes a markov matrix and determines
several important pieces of information about it such as:
Stationarity, ergodicity, etc...

Reference: Recursive Macroeconomic Theory:
           Thomas Sargent and Lars Ljungqvist
           3rd Edition

TODO:
* Add ergodic sets (Possibly done)
* Find good definition for ergodicity
'''
import numpy as np
import scipy.linalg as la
from numpy.linalg import matrix_power


class DMarkov(object):
    """
    This class takes a discrete transition matrix P, an initial
    distribution, number of periods to be simulated and returns
    key information about the matrix P, and a simulation of the
    markov chain and returns the chain as a series of states
    (states don't use python indexing: Namely the first state is 1)

    Attributes
    ----------
    P : np.ndarray : float
        The transition matrix

    pi_0 : np.ndarray : float
        The initial probability distribution

    pers : scalar : int
        The number of periods to simulate the markov chain

    statdists : np.ndarray : float
        An array with stationary distributions as columns

    ergodic_sets : list : lists
        A list of lists where each list in the main list
        has one of the ergodic sets.

    ergodic_stat_dists : np.ndarray : floats
        An array with the ergodic stationary distributions
        as columns


    Methods
    -------
    stationarity : This method finds stationary distributions

    ergodicity : This method determines which distributions
                 are ergodic

    simulate_chain : Simulates the markov chain for a given
                     initial distribution

    """
    def __init__(self, P, pi_0=None, pers=None):
        """
        Parameters
        ----------
        P : np.ndarray : float
            The transition matrix

        pi_0 : np.ndarray : float
            The initial probability distribution

        pers : scalar : int
            The number of periods to simulate the markov chain
        """
        self.P = P
        self.n = P.shape[0]
        self.pi_0 = pi_0
        self.pers = pers

        # double check that P is a square matrix
        if P.shape[0] != P.shape[1]:
            raise ValueError('The transition matrix must be square!')

        # Double check that the rows of P sum to one
        if np.all(np.sum(P, axis=1) != np.ones(P.shape[0])):
            raise ValueError('The rows must sum to 1. P is a trans matrix')

        self.stationarity()
        self._find_erg_sets()

    def __repr__(self):
        msg = "Markov process with transition matrix \n P = \n {0} \n \
               \n with ergodic sets E = {1} and \n \
               \n stationary distributions \n pi = \n {2}"
        return msg.format(self.P, self.ergodic_sets, self.statdists)

    def __str__(self):
        return str(self.__repr__)

    def stationarity(self):
        """
        This method determines the stationary distributions of P.
        These are the eigenvectors that correspond to the unit eigen-
        values of the matrix P' (They satisfy pi_{t+1}' = pi_{t}' P)

        Parameters
        ----------
        self : object

        Returns
        -------
        stat_dists : np.ndarray : float
            This is an array that has the stationary distributions as
            its columns.

        absorb_states : np.ndarray : ints
            This is a vector that says which of the states are
            absorbing states
        """
        P = self.P

        # Compute the eigenvalues and eigenvectors
        eigvals, eigvecs = la.eig(P.T)

        # Find the inex of where the unit eig-vals are
        index = np.where(abs(eigvals - 1.) < 1e-10)[0]

        # Pull out corresponding eig-vecs
        uniteigvecs = eigvecs[:, index]

        # Scale so that the stationary distributions sum to 1
        statdists = uniteigvecs/np.sum(uniteigvecs, axis=0)
        self.statdists = statdists

        return statdists

    def ergodicity(self, distributions_to_check):
        """
        This method determines which of the stationary distribution are
        ergodic.  If an initial distribution is given then it also will
        evaluate whether (P, pi_0) is an ergodic markov chain.

        Parameters
        ----------
        self : object

        distributions_to_check : string
            A string.  Either "stationary" or "initial" or "all".  It
            determines whether the method checks the ergodicity of
            the stationary distributions or initial distributions or
            possibly both.

        Returns
        -------
        ergodic_stat_dists : np.ndarray : Boolean
            This is an array that contains all of the stationary
            distributions that are also ergodic.

        ergodic_pi_0 : scalar : Boolean
            If an initial probability distribution was
            given then this will tell you whether the
            initial distribution is ergodic or not
        """
        if distributions_to_check == "stationary":
            ergodic_stat_dists = np.zeros(self.statdists.shape[1])
            erg_dists = 0
            for i in xrange(self.statdists.shape[1]):
                ergodic_stat_dists[i] = self._check_erg(self.statdists[:, i])
                if ergodic_stat_dists[i] == True:
                    erg_dists += 1
            temp1 = self.statdists[:, np.where(ergodic_stat_dists == True)[0]]
            self.ergodic_stat_dists = temp1

        if distributions_to_check == "initial":
            if self.pi_0 == None:
                raise ValueError('No initial distribution given. Cant compute')
            ergodic_pi_0 = self._check_erg(self.pi_0)
            if ergodic_pi_0 == True:
                self.ergodic_pi_0 = "(P, pi_0) is ergodic"
            else:
                self.ergodic_pi_0 = "(P, pi_0) is not ergodic"

        if distributions_to_check == "all":
            self.ergodicity("stationary")
            self.ergodicity("initial")

    def _check_erg(self, dists):
        """
        This method is called by ergodicity.  It checks whether a
        group of distributions are ergodic.

        Parameters
        ----------
        self : object

        dists : np.ndarray : float
            Distribution to check for ergodicity with transition mat P

        Returns : scalar : Boolean
            Whether (P, dists) is ergodic or not
        """
        P = self.P

        # Find the invariant functions : Solve (P - I)\bar{y} = 0
        eigvals, eigvecs = la.eig(P)

        # Find the inex of where the unit eig-vals are
        index = np.where(abs(eigvals - 1.) < 1e-10)[0]

        # Pull out corresponding eig-vecs
        invar_funcs = eigvecs[:, index]

        # Find out which states are possible (i.e. where the
        # probability of dists != 0)
        index_1 = np.where(dists > 0.)[0]

        # Check to see if y_i = y_j for all i,j where dists(i) and
        # dists(j) != 0
        if np.allclose(invar_funcs[index_1[0], :], invar_funcs[index_1, :]):
            return True
        else:
            return False

    def simulate_chain(self, dist, pers=500):
        """
        This method takes a distribution and simulates it for pers
        periods (default number of periods is 500).

        Parameters
        ----------
        dist : np.ndarray : float
            The starting distribution of the simulation.

        pers : scalar : int
            The number of periods that the simulation should last

        Returns
        -------
        mc_siml : np.ndarray : ints
            This matrix holds the states (numbered 1, 2, ..., n) for
            each period of the simulation
        """
        P = self.P

        # Take the ith power of P for every period
        matP_power = lambda i : matrix_power(P, i)
        mats = np.array(map(matP_power, np.arange(pers)))

        # Now multiply dist by P^i to get the pi_t
        probs = dist.dot(mats)
        cum_probs = np.cumsum(probs, axis=1)

        # Draw a random number (between 0 and 1) for every per so we
        # can randomly determine the state.
        rand_draw = np.random.rand(pers)

        # determine state for every period
        mc_simul = np.zeros(pers)
        for i in xrange(pers):
            mc_simul[i] = np.where(cum_probs[i, :] > rand_draw[i])[0][0]

        self.simulated_chain = mc_simul

        return mc_simul

    def _find_erg_sets(self):
        """
        This method finds all of the ergodic sets for a given transition
        matrix P.  It saves the ergodic sets in the class object.

        Note: This method is still undergoing testing and improvement. If
        you find a bug or an example where it doesn't work please let us
        know so we can amend our code to do the right thing.

        Parameters
        ----------
        self

        Returns
        -------
        None
        """
        P = self.P
        num_states = P.shape[0]

        # We are going to use graph theory to find ergodic sets
        # We first find where there is positive probability of
        # the sets
        pos_prob_P = np.where(P > 0)

        # Put a one in a matrix everywhere there is positive
        # probability of going from state i to state j
        connect_mat = np.zeros(P.shape)
        connect_mat[pos_prob_P] = 1.
        sets_mat = np.zeros(P.shape)

        # We want to know whether they can get from any given
        # state to the other so we need to add each exponent
        # of connect_mat^n for n up to the num_states.  This is
        # equivalent to (I - P)^{-1}(I - P^n)
        # TODO: If matrix is singular that doesn't work
        for i in xrange(num_states):
            sets_mat += matrix_power(connect_mat, i+1)

        # Now we know where they really are connected in num_states steps
        true_connected = np.zeros(P.shape)
        true_index = np.where(sets_mat > 0)
        true_connected[true_index] = 1.

        # Create a dictionary where every state will have a set
        # we will use these sets to establish which sets of
        # elements are in the same ergodic set
        conn_states = {'state_%s' %st : set() for st in xrange(num_states)}

        possible_ergodic = set(np.arange(num_states))
        trans_states = set()
        ergodic_sets = []
        num_erg_sets = 0


        # We need to first remove the transient states
        # TODO: Think more about this loop.  Can this be vectorized?

        # Find all of the connected both ways states and the ones that
        # aren't connected both ways should be transient
        for row in possible_ergodic:
            for col in possible_ergodic:
                if abs(true_connected[row, col] - 1) < 1e-10 \
                and abs(true_connected[col, row] - 1) < 1e-10:
                    conn_states['state_%s' %row].add(col)

                elif abs(true_connected[row, col] - 1) > 1e-10 \
                and abs(true_connected.T[row, col] - 1) < 1e-10:
                    trans_states.add(col)

        # Take the transient states out of the possible ergodic states
        for i in trans_states:
            possible_ergodic.remove(i)

        # Now we need to see whether the sets match up like we want them to
        for i in possible_ergodic:
            check_equiv = []
            for t in conn_states['state_%s' %i]:
                if conn_states['state_%s' %i] == conn_states['state_%s' %t]:
                    check_equiv.append(t)

            if conn_states['state_%s' %i] == set(check_equiv) \
            and check_equiv not in ergodic_sets:
                ergodic_sets.append(check_equiv)

        self.ergodic_sets = ergodic_sets
        return None


P = np.array([[.2, .7, .1, 0., 0., 0.],
              [.4, .1, .5, 0., 0., 0.],
              [0., .5, .5, 0., 0., 0.],
              [0., .5, 0., .4, 0., .1],
              [0., 0., 0., 0., 1., 0.],
              [.1, 0., 0., .1, 0., .8]])

P2 = np.array([[ .5,  .5,  0.,  0. ],
               [ .5,  .5,  0.,  0. ],
               [ 0.,  0.,  .5,  .5 ],
               [ 0.,  0.,  .5,  .5 ]])


MC = DMarkov(P)
print("Ergodic sets are", MC.ergodic_sets)
print("Stationary distributions are", MC.statdists)
MC.simulate_chain(MC.statdists[:, 0], 100)