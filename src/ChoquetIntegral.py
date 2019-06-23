import numpy as np
import itertools
from cvxopt import solvers, matrix


class ChoquetIntegral:

    def __init__(self):
        """Instantiation of a ChoquetIntegral.

           This sets up the ChI. It doesn't take any input parameters
           because you may want to use pass your own values in(as opposed
           to learning from data). To instatiate, use
           chi = ChoquetIntegral.ChoquetIntegral()
        """
        self.trainSamples, self.trainLabels = [], []
        self.testSamples, self.testLabels = [], []
        self.N, self.numberConstraints, self.M = 0, 0, 0
        self.g = 0
        self.fm = []
        self.type = []

    def train_chi_quad(self, x1, l1):
        """
        This trains this instance of your ChoquetIntegral w.r.t x1 and l1.

        :param x1: These are the training samples of size N x M(inputs x number of samples)
        :param l1: These are the training labels of size 1 x M(label per sample)

        """
        self.type = 'quad'
        self.trainSamples = x1
        self.trainLabels = l1
        self.N = self.trainSamples.shape[0]
        self.M = self.trainSamples.shape[1]
        print("Number Inputs : ", self.N, "; Number Samples : ", self.M)
        self.fm = self.produce_lattice()

    def chi_quad(self, x2):
        """
        This will produce an output for this instance of the ChI

        This will use the learned(or specified) Choquet integral to
        produce an output w.r.t. to the new input.

        :param x2: testing sample
        :return: output of the choquet integral.
        """
        if self.type == 'quad':
            n = len(x2)
            pi_i = np.argsort(x2)[::-1][:n] + 1
            ch = x2[pi_i[0] - 1] * (self.fm[str(pi_i[:1])])
            for i in range(1, n):
                latt_pti = np.sort(pi_i[:i + 1])
                latt_ptimin1 = np.sort(pi_i[:i])
                ch = ch + x2[pi_i[i] - 1] * (self.fm[str(latt_pti)] - self.fm[str(latt_ptimin1)])
            return ch
        print("If using sugeno measure, you need to use chi_sugeno.")

    def produce_lattice(self):
        """
            This method builds is where the lattice(or FM variables) will be learned.

            The FM values can be found via a quadratic program, which is used here
            after setting up constraint matrices. Refer to papers for complete overview.

        :return: Lattice, the learned FM variables.
        """


        fm_len = 2 ** self.N - 1  # nc
        E = np.zeros((fm_len, fm_len))  # D
        L = np.zeros(fm_len)  # f
        index_keys = self.get_keys_index()
        for i in range(0, self.M):  # it's going through one sample at a time.
            l = self.trainLabels[i]  # this is the labels
            fm_coeff = self.get_fm_class_img_coeff(index_keys, self.trainSamples[:, i], fm_len)  # this is Hdiff
            # print(fm_coeff)
            L = L + (-2) * l * fm_coeff
            E = E + np.matmul(fm_coeff.reshape((fm_len, 1)), fm_coeff.reshape((1, fm_len)))

        G, h, A, b = self.build_constraint_matrices(index_keys, fm_len)
        sol = solvers.qp(matrix(2 * E, tc='d'), matrix(L.T, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'), matrix(A, tc='d'), matrix(b, tc='d'))
        
        g = sol['x']
        Lattice = {}
        for key in index_keys.keys():
            Lattice[key] = g[index_keys[key]]
        return Lattice

    def build_constraint_matrices(self, index_keys, fm_len):
        """
        This method builds the necessary constraint matrices.



        :param index_keys: map to reference lattice components
        :param fm_len: length of the fuzzy measure
        :return: the constraint matrices
        """


        vls = np.arange(1, self.N + 1)
        line = np.zeros(fm_len)
        G = line
        line[index_keys[str(np.array([1]))]] = -1.
        h = np.array([0])
        for i in range(2, self.N + 1):
            line = np.zeros(fm_len)
            line[index_keys[str(np.array([i]))]] = -1.
            G = np.vstack((G, line))
            h = np.vstack((h, np.array([0])))
        for i in range(2, self.N + 1):
            parent = np.array(list(itertools.combinations(vls, i)))
            for latt_pt in parent:
                for j in range(len(latt_pt) - 1, len(latt_pt)):
                    children = np.array(list(itertools.combinations(latt_pt, j)))
                    for latt_ch in children:
                        line = np.zeros(fm_len)
                        line[index_keys[str(latt_ch)]] = 1.
                        line[index_keys[str(latt_pt)]] = -1.
                        G = np.vstack((G, line))
                        h = np.vstack((h, np.array([0])))

        line = np.zeros(fm_len)
        line[index_keys[str(vls)]] = 1.
        G = np.vstack((G, line))
        h = np.vstack((h, np.array([1])))
        
        # equality constraints
        A = np.zeros((1,fm_len))
        A[0,-1] = 1
        b = np.array([1]);

        return G, h, A, b

    def get_fm_class_img_coeff(self, Lattice, h, fm_len):  # Lattice is FM_name_and_index, h is the samples, fm_len
        """
        This creates a FM map with the name as the key and the index as the value

        :param Lattice: dictionary with FM
        :param h: sample
        :param fm_len: fm length
        :return: the fm_coeff
        """

        n = len(h)  # len(h) is the number of the samples
        fm_coeff = np.zeros(fm_len)
        pi_i = np.argsort(h)[::-1][:n] + 1
        for i in range(1, n):
            fm_coeff[Lattice[str(np.sort(pi_i[:i]))]] = h[pi_i[i - 1] - 1] - h[pi_i[i] - 1]
        fm_coeff[Lattice[str(np.sort(pi_i[:n]))]] = h[pi_i[n - 1] - 1]
        np.matmul(fm_coeff, np.transpose(fm_coeff))
        return fm_coeff

    def get_keys_index(self):
        """
        Sets up a dictionary for referencing FM.

        :return: The keys to the dictionary
        """


        vls = np.arange(1, self.N + 1)
        count = 0
        Lattice = {}
        for i in range(0, self.N):
            Lattice[str(np.array([vls[i]]))] = count
            count = count + 1
        for i in range(2, self.N + 1):
            A = np.array(list(itertools.combinations(vls, i)))
            for latt_pt in A:
                Lattice[str(latt_pt)] = count
                count = count + 1
        return Lattice

    def train_chi_sugeno(self, densities):
        """
        Trains a fm w.r.t the sugeno measure.
        The learned FM is in binary encoded format.
        """
        self.type = 'sugeno'
        self.N = densities.shape[0]  # get the number of densities
        densities = densities + np.finfo(float).eps  # condition the densities

        coefficients = np.zeros(self.N)  # hold onto the polynomial

        coefficients[self.N - 1] = np.sum(densities)  # compute the first term

        ##################################################################
        # compute the rest of the terms.
        ##################################################################
        for i in range(self.N, 1, -1):
            combos = list(itertools.combinations(range(0, self.N), i))  # generate the combinations
            number_of_combinations = combos.__len__()  # How many elements were generated?
            elements = combos[0].__len__()
            coefficients[(self.N - 1) - i + 1] = 0

            ##################################################################
            # For each coefficient, compute its product, summing them all up
            ##################################################################
            for j in range(0, number_of_combinations):
                tmp = 1  # tmp used for product calculation

                ##################################################################
                # For each combo, compute the product
                ##################################################################
                for k in range(0, elements):
                    tmp = tmp * densities[combos[j][k]]

                coefficients[(self.N - 1) - i + 1] = coefficients[(self.N - 1) - i + 1] + tmp

        coefficients[(self.N - 1)] = coefficients[(self.N - 1)] - 1

        s_lambda = np.max((np.roots(coefficients)))  # calculate lambda

        ##################################################################
        # Now calculate the full measure
        ##################################################################

        n_vars = 2 ** self.N - 1  # get the number of FM variables

        fm_binary = np.zeros(n_vars)  # where to store the measure values

        ##################################################################
        # First, get the densities
        ##################################################################

        for i in range(0, self.N):
            fm_binary[int(2 ** i) - 1] = densities[i]

        ##################################################################
        # Next, get the rest of the values
        ##################################################################
        for i in range(2, self.N):
            combos = list(itertools.combinations(range(0, self.N), i))
            number_of_combinations = combos.__len__()

            for j in range(0, number_of_combinations):
                this_set_1 = combos[j][0]  # get the first element of this combination
                this_set_2 = combos[j][1:]  # get the rest of the elements

                # fetch the calculated values thus far
                val_1 = np.cumsum(2 ** (this_set_1))
                term_1 =  fm_binary[val_1[-1] - 1]
                val_2 = np.cumsum(2 ** np.asarray(this_set_2))
                term_2 = fm_binary[val_2[-1] - 1]
                val_3 = np.cumsum(2 ** np.asarray(combos[j]))

                # fill in the rest of the values
                fm_binary[val_3[-1] - 1] = term_1 + term_2 + s_lambda * term_1 * term_2


        fm_binary[-1] = 1  # last term is a 1


        self.fm = fm_binary

    def chi_sugeno(self, x2):
        """
        The chi_sugeno compute the output with respect to the learned, sugeno fm.
        This relies on the FM being in the binary encoding format.


        """

        if self.type == 'sugeno':
            n = len(x2)
            sorted_inds = np.argsort(x2)[::-1][:n] + 1 # sorted indices
            sorted_vals = np.append(x2[sorted_inds-1], 0)
            inds = np.cumsum(2**(sorted_inds-1))
            res = 0
            val = 1
            for i in inds:
                res = res+self.fm[i-1] * (sorted_vals[val-1] - sorted_vals[val])
                val += 1


        print("If using sugeno measure, you need to use chi_sugeno.")

        return res


if __name__ == "__main__":
    chi_quad = ChoquetIntegral()

    train_data = np.random.randn(3,3)
    label_data = np.random.randn(3,1)
    chi_quad.train_chi_quad(train_data, label_data)
    print('Quadratic Program Fuzzy Measure : ', chi_quad.fm)

    chi_sugeno  = ChoquetIntegral()
    densities = np.array([.1, .2, .3])
    chi_sugeno.train_chi_sugeno(densities)
    chi_sugeno.fm = ([.3, .3, .6, .3, .6, .6, 1])

    print('Sugeno-Lambda Fuzzy Measure : ', chi_sugeno.fm)
    print(chi_sugeno.chi_sugeno(np.array([.4, .5, .3])))





    