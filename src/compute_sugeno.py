import numpy as np
from itertools import combinations


def compute_sugeno(densities):

    N = densities.shape[0]                       # get the number of densities

    densities = densities + np.finfo(float).eps  # condition the densities

    coefficients = np.zeros(N)                   # hold onto the polynomial

    coefficients[N-1] = np.sum(densities)        # compute the first term

    ##################################################################
    # compute the rest of the terms.
    ##################################################################
    for i in range(N, 1, -1):
        combos = list(combinations(range(0, N), i))  # generate the combinations
        number_of_combinations = combos.__len__()    # How many elements were generated?
        elements = combos[0].__len__()
        coefficients[(N - 1) - i + 1] = 0

        ##################################################################
        # For each coefficient, compute its product, summing them all up
        ##################################################################
        for j in range(0, number_of_combinations):
            tmp = 1                                       # tmp used for product calculation

            ##################################################################
            # For each combo, compute the product
            ##################################################################
            for k in range(0, elements):
                tmp = tmp * densities[combos[j][k]]

            coefficients[(N-1)-i+1] = coefficients[(N-1)-i+1] + tmp

    coefficients[(N-1)] = coefficients[(N-1)] - 1

    s_lambda = np.max((np.roots(coefficients)))         # calculate lambda

    ##################################################################
    # Now calculate the full measure
    ##################################################################

    n_vars = 2**N - 1               # get the number of FM variables


    fm_binary = np.zeros(n_vars)    # where to store the measure values

    ##################################################################
    # First, get the densities
    ##################################################################

    for i in range(0, N):
        fm_binary[int(2**i)-1] = densities[i]

    ##################################################################
    # Next, get the rest of the values
    ##################################################################
    for i in range(2, N):
        combos = list(combinations(range(0, N), i))
        number_of_combinations = combos.__len__()

        for j in range(0, number_of_combinations):
            this_set_1 = combos[j][0]    # get the first element of this combination

            this_set_2 = combos[j][1:]  # get the rest of the elements

            # fetch the calculated values thus far

            val_1, term_1 = np.cumsum(2**(this_set_1)), fm_binary[val_1[-1]-1]
            val_2, term_2 = np.cumsum(2**np.asarray(this_set_2)), fm_binary[val_2[-1]-1]
            val_3 = np.cumsum(2**np.asarray(combos[j]))

            # fill in the rest of the values
            fm_binary[val_3[-1]-1] = term_1 + term_2 + s_lambda * term_1 * term_2

    fm_binary[-1] = 1       # last term is a 1

    return fm_binary



if __name__ == '__main__':
    FM = compute_sugeno(np.array([.1, .25, .5, .25, .8]))