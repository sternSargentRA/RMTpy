'''
Author: Chase Coleman and Spencer Lyon

This file contains a class that takes a markov matrix and determines
several important pieces of information about it.

Reference: Recursive Macroeconomic Theory:
           Thomas Sargent and Lars Ljungqvist
           3rd Edition

TODO:
* Add ergodic sets (Possibly done)
* Determine ergodicity of (P, pi)
* Check for other things to do w/ a markov chain
'''
import numpy as np
import scipy.linalg as la
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt


class DMarkov(object):
    """
    This class takes a discrete transition matrix P, an initial
    distribution, number of periods to be simulated and returns
    key information about the matrix P, and a simulation of the
    markov chain and returns the chain as a series of states

    Attributes
    ----------
    P : np.ndarray : float
        The transition matrix

    pi_0 : np.ndarray : float
        The initial probability distribution

    invar_dists : np.ndarray : float
        An array with invariant distributions as columns

    ergodic_sets : list : lists
        A list of lists where each list in the main list
        has one of the ergodic sets.


    Methods
    -------
    invariant_distributions : This method finds invariant
                              distributions

    simulate_chain : Simulates the markov chain for a given
                     initial distribution

    """
    def __init__(self, P, pi_0=None):
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

        # double check that P is a square matrix
        if P.shape[0] != P.shape[1]:
            raise ValueError('The transition matrix must be square!')

        # Double check that the rows of P sum to one
        if np.all(np.sum(P, axis=1) != np.ones(P.shape[0])):
            raise ValueError('The rows must sum to 1. P is a trans matrix')

        self.invariant_distributions()
        self._find_erg_sets()

    def __repr__(self):
        msg = "Markov process with transition matrix \n P = \n {0} \n \
               \n with ergodic sets E = {1} and \n \
               \n invariant distributions \n pi = \n {2}"
        return msg.format(self.P, self.ergodic_sets, self.invar_dists)

    def __str__(self):
        return str(self.__repr__)

    def invariant_distributions(self):
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
        invar_dists = uniteigvecs/np.sum(uniteigvecs, axis=0)
        self.invar_dists = invar_dists

        return invar_dists

    def simulate_chain(self, dist, plot=False, pers=500):
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

        if plot==True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(np.arange(pers), self.simulated_chain)
            ax.set_title("Markov Simulation for Transition Matrix")
            ax.set_xlabel("Period")
            ax.set_ylabel("State")
            ax.set_ylim((0, self.n))

            plt.savefig("simulated_chain.png")

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


# Example of how to use this class

# define the transition matrix
P = np.array([[.2, .7, .1, 0., 0., 0.],
              [.4, .1, .5, 0., 0., 0.],
              [0., .5, .5, 0., 0., 0.],
              [0., .5, 0., .4, 0., .1],
              [0., 0., 0., 0., 1., 0.],
              [.1, 0., 0., .1, 0., .8]])

# Create an object (MC) that is the object created by
# DMarkov
MC = DMarkov(P)

# If you type print(MC) or just MC then it should return
# several valuable pieces of information about the markov
# chain
print(MC)

# Simulate the chain using the first invariant distribution for
# 100 periods and plot it (plot gets saved in directory)
MC.simulate_chain(MC.invar_dists[:, 0], True, 100)