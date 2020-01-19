import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from inconsistent_system_solver import InconsistentSystemSolver

from numpy.polynomial import Polynomial as pm
from functools import reduce

def basis_sh_chebyshev(degree):
    basis = [pm([-1, 2]), pm([1])]
    for i in range(degree):
        basis.append(pm([-2, 4])*basis[-1] - basis[-2])
    del basis[0]
    return basis

def basis_sh_legendre(degree):
    basis = [pm([1])]
    for i in range(degree):
        if i == 0:
            basis.append(pm([-1, 2]))
            continue
        basis.append((pm([-2*i - 1, 4*i + 2])*basis[-1] - i * basis[-2]) / (i + 1))
    return basis

def basis_laguerre(degree):
    basis = [pm([1])]
    for i in range(degree):
        if i == 0:
            basis.append(pm([1, -1]))
            continue
        basis.append(pm([2*i + 1, -1])*basis[-1] - i * i * basis[-2])
    return basis

def basis_hermite(degree):
    basis = [pm([0]), pm([1])]
    for i in range(degree):
        basis.append(pm([0,2])*basis[-1] - 2 * i * basis[-2])
    del basis[0]
    return basis

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def relu(x):
    return np.maximum(1e-6, x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def activation(type='relu'):
    def closure(x, type=type):
        if type == 'gelu':
            return x*sigmoid(1.702*x)
        elif type == 'sigmoid':
            return sigmoid(x)
        elif type == 'softplus':
            return np.log(1 + np.exp(x))
        else:
            return relu(x)
    return closure

class MultiplicativeModelSystem:
    def __init__(self, polynomes_type='Chebyshev', calculate_separately=False, degrees=[], activation_type='relu'):
        self.system_solver = InconsistentSystemSolver('lsqr')
        self.lambda_values = {}
        self.alpha_values = {}
        self.c_values = {}
        self.polynomes = {}
        self.F_functions = {}
        self.calculate_separately = calculate_separately
        self.polynomes_type = polynomes_type
        self.degrees = degrees
        self.activation = activation(activation_type)

        if polynomes_type == 'Legendre':
            self.basis_gen_function = basis_sh_legendre
        elif polynomes_type == 'Hermit':
            self.basis_gen_function = basis_hermite
        elif polynomes_type == 'Laguerre':
            self.basis_gen_function = basis_laguerre
        else:
            self.basis_gen_function = basis_sh_chebyshev

    def _parse_vectors(self, vectors, name='X'):
        return {f'{name}{i + 1}': vec for i, vec in enumerate(vectors)}

    def _get_poly(self, y_ind, x_ind, x_ind_coord, arg_x=0):
        def _closure_poly(arg_x):
            return reduce(lambda x, y: x * y, [
                (1 + self.activation(self.basises[x_ind][degree](arg_x))) ** self.lambda_values[y_ind][x_ind][x_ind_coord][
                    degree] for degree in
                range(len(self.basises[x_ind]))]) - 1

        return _closure_poly

    def _poly_transform(self, X, x_ind, y_ind):
        res = np.zeros_like(X)
        for value_ind in range(X.shape[0]):
            for x_ind_coord in range(X.shape[1]):
                res[value_ind][x_ind_coord] = self._get_poly(y_ind, x_ind, x_ind_coord)(X[value_ind][x_ind_coord])
        return res

    def _F_transform(self, X, x_ind, y_ind):
        X = self._poly_transform(X, x_ind, y_ind)
        res = np.ones(X.shape[0])
        for value_ind in range(X.shape[0]):
            for x_ind_coord in range(X.shape[1]):
                res[value_ind] *= (relu(1 + (X[value_ind][x_ind_coord]))) ** self.alpha_values[y_ind][x_ind][
                    x_ind_coord]
            res[value_ind] -= 1
        return res

    def predict(self, X):
        if self.c_values:
            results = np.ones((X[0].shape[0], len(self.c_values)))
            for y_ind in range(len(self.c_values)):
                for x_ind, x in enumerate(X):
                    x = self._F_transform(x, x_ind, y_ind)
                    results[:, y_ind] *= (relu(1 + x)) ** self.c_values[y_ind][x_ind]
                results[:, y_ind] -= 1

            return results

        else:
            print('Models not fitted yet!')

    def _normalize(self, X, Y):
        X_normalizers = []
        Y_normalizer = MinMaxScaler()
        Y = Y_normalizer.fit_transform(Y)
        X_normalized = []
        for x in X:
            scaler = MinMaxScaler()
            scaler.fit(x)
            X_normalized.append(scaler.transform(x))
            X_normalizers.append(scaler)

        self.X_normalizers = X_normalizers
        self.Y_normalizer = Y_normalizer
        return X_normalized, Y

    def _inverse_normalize(self, X, Y):
        X_inverse = []
        Y_inverse = self.Y_normalizer.inverse_transform(Y)
        for x, scaler in zip(X, self.X_normalizers):
            X_inverse.append(scaler.inverse_transform(x))
        return X_inverse, Y_inverse

    def fit(self, X, Y):
        calculate_separately = self.calculate_separately
        degrees = self.degrees
        self.basises = [self.basis_gen_function(d) for d in degrees]
        X, Y = self._normalize(X, Y)
        Y = Y.T
        for i in range(len(Y)):
            self.lambda_values[i] = {}
            self.polynomes[i] = {}
            self.alpha_values[i] = {}
            self.F_functions[i] = {}
            self.c_values[i] = {}
            for j in range(len(X)):
                self.lambda_values[i][j] = {}
                self.polynomes[i][j] = {}
                self.alpha_values[i][j] = {}
                for k in range(X[j].shape[1]):
                    self.lambda_values[i][j][k] = {}

        for value in self._lambda_solver(X, Y, calculate_separately=calculate_separately):
            pass
            #yield 30*value
        for value in self._alpha_solver(X, Y):
            pass
            #yield 20*value + 30
        for value in self._c_solver(X, Y):
            pass
            #yield 20*value + 50

    def _lambda_solver(self, X, bqi, calculate_separately=True):
        X, bqi = self._parse_vectors(X, name='X'), self._parse_vectors(bqi, name='bqi')

        pr_bar_ind = 0
        if calculate_separately:
            for x_ind, (basis, (x_name, x)) in enumerate(zip(self.basises, X.items())):
                print(x_name)
                for y_ind, (bqi_name, b) in enumerate(bqi.items()):
                    pr_bar_ind += 1
                    yield pr_bar_ind / (len(X.items()) * len(bqi.items()))
                    print(bqi_name)
                    x_poly = np.hstack([np.log(1 + self.activation(poly(x))) for poly in basis])
                    results = self.system_solver.solve(x_poly, np.log(1 + b))
                    for poly_degree, res in enumerate(chunks(results, x.shape[1])):
                        for x_ind_coord, l in enumerate(res):
                            self.lambda_values[y_ind][x_ind][x_ind_coord][poly_degree] = l

                    for x_ind_coord in range(x.shape[1]):
                        self.polynomes[y_ind][x_ind][x_ind_coord] = self._get_poly(y_ind, x_ind, x_ind_coord)
                    print(results)
        else:
            pr_bar_ind = 0
            for y_ind, (bqi_name, b) in enumerate(bqi.items()):
                # print(bqi_name)
                x_dims = [x.shape[1] * len(basis) for basis, (k, x) in zip(self.basises, X.items())]
                merged_x = np.hstack(
                    [np.hstack([np.log(1 + self.activation(poly(x))) for poly in basis]) for basis, (k, x) in
                     zip(self.basises, X.items())])
                results = self.system_solver.solve(merged_x, np.log(1 + b))

                # print('MAX NORM DIFF: ', (merged_x @ results - np.log(1 + b)).max())
                # print(results.max(), results.min())
                curr_x_ind = 0
                for x_ind, x_dim in enumerate(x_dims):
                    pr_bar_ind += 1
                    yield pr_bar_ind/(len(x_dims)*len(bqi.items()))
                    res_curr = results[curr_x_ind: curr_x_ind + x_dim]
                    # print(res_curr)
                    for poly_degree, res in enumerate(chunks(res_curr, list(X.items())[x_ind][1].shape[1])):
                        #  print(poly_degree, res)
                        for x_ind_coord, l in enumerate(res):
                            self.lambda_values[y_ind][x_ind][x_ind_coord][poly_degree] = l
                    curr_x_ind += x_dim

                    for x_ind_coord in range(list(X.items())[x_ind][1].shape[1]):
                        self.polynomes[y_ind][x_ind][x_ind_coord] = self._get_poly(y_ind, x_ind, x_ind_coord)
                    # print(results)

    def _alpha_solver(self, X, bqi):
        X, bqi = self._parse_vectors(X, name='X'), self._parse_vectors(bqi, name='bqi')
        pr_bar_ind = 0
        for y_ind, (bqi_name, b) in enumerate(bqi.items()):
            x_dims = [x.shape[1] for k, x in X.items()]
            merged_x = np.hstack(
                [np.log(relu(1 + self._poly_transform(x, x_ind, y_ind))) for x_ind, (k, x) in enumerate(X.items())])
            results = self.system_solver.solve(merged_x, np.log(1 + b))
            # print(results)
            # print((merged_x @ results - np.log(1 + b)).max())

            curr_x_ind = 0
            for x_ind, x_dim in enumerate(x_dims):
                pr_bar_ind += 1
                yield pr_bar_ind / (len(x_dims) * len(bqi.items()))
                res_current = results[curr_x_ind: curr_x_ind + x_dim]

                for x_ind_coord, alpha in enumerate(res_current):
                    self.alpha_values[y_ind][x_ind][x_ind_coord] = alpha

                # self.F_functions[y_ind][x_ind] = reduce(lambda x, y: x*y,\
                #   [(1 + softplus(self._get_poly(y_ind, x_ind, x_ind_coord)))**self.alpha_values[y_ind][x_ind][x_ind_coord] \
                #                                   for x_ind_coord in range(x_dim)])

    def _c_solver(self, X, Y):
        X, Y = self._parse_vectors(X, name='X'), self._parse_vectors(Y, name='Y')

        pr_bar_ind = 0
        for y_ind, (bqi_name, y) in enumerate(Y.items()):
            X_merged = np.array(
                [np.log(relu(1 + self._F_transform(x, x_ind, y_ind))) for x_ind, (k, x) in enumerate(X.items())]).T
            results = self.system_solver.solve(X_merged, np.log(1 + y))

            # print(results)
            print((X_merged @ results - np.log(1 + y)).max())
            for x_ind, c in enumerate(results):
                pr_bar_ind += 1
                yield pr_bar_ind / (len(results) * len(Y.items()))
                self.c_values[y_ind][x_ind] = c

    def get_logs(self):
        if self.c_values:
            file_str = ''
            file_str += 'Ψ functions (lambda values):\n'
            for y_ind in range(len(self.lambda_values)):
                file_str += f'For Y{y_ind + 1}\n'
                for x_ind in range(len(self.lambda_values[y_ind])):
                    for x_ind_coord in range(len(self.lambda_values[y_ind][x_ind])):
                        file_str += f'1 + Ψ{x_ind}{x_ind_coord} = ' + \
                                    ' * '.join(f'(1 + T{degree}(x{x_ind}{x_ind_coord}))^' + \
                                    str(self.lambda_values[y_ind][x_ind][x_ind_coord][degree]) \
                                     for degree in range(len(self.lambda_values[y_ind][x_ind][x_ind_coord]))) + \
                                    '\n'

            file_str += 'Φ functions (alpha values):\n'
            for y_ind in range(len(self.alpha_values)):
                for x_ind in range(len(self.alpha_values[y_ind])):
                    file_str += f'1 + Φ{y_ind}{x_ind} = ' + \
                                ' * '.join(f'(1 + Ψ{y_ind}{x_ind}(x{x_ind}{x_ind_coord}))^' + \
                                    str(self.alpha_values[y_ind][x_ind][x_ind_coord])  \
                                           for x_ind_coord in
                                           range(len(self.alpha_values[y_ind][x_ind]))) + \
                                '\n'
            file_str += 'F functions (c values):\n'
            for y_ind in range(len(self.c_values)):
                file_str += f'1 + F{y_ind} = ' + \
                            ' * '.join(f'(1 + Φ{y_ind}{x_ind})^' + str(self.c_values[y_ind][x_ind]) for x_ind in
                                       range(len(self.c_values[y_ind]))) + \
                            '\n'
            return file_str
        else:
            print('Models not fitted yet!')

    def get_final_results(self, X, Y):
        X_scaled, Y_scaled = self._normalize(X, Y)
        Y_preds_scaled = self.predict(X_scaled)

        Y_preds = self.Y_normalizer.inverse_transform(Y_preds_scaled)

        Y_diff = np.abs(Y_preds - Y)
        Y_diff_scaled = np.abs(Y_preds_scaled - Y_scaled)
        return {
            'X': X, 'Y': Y, 'Y_scaled': Y_scaled, 'Y_preds': Y_preds, 'Y_preds_scaled': Y_preds_scaled,
            'Y_err': Y_diff, 'Y_err_scaled': Y_diff_scaled
        }


def get_results(params):
    add_solver = MultiplicativeModelSystem(polynomes_type=params['method'], calculate_separately=params['lambda_from_3sys'], \
                                           degrees=params['X_degree'], activation_type=params['activation'])
    for v in add_solver.fit(params['X'], params['y']):
        yield v
    logs = add_solver.get_logs()
    yield 90
    final_res = add_solver.get_final_results(params['X'], params['y'])
    final_res['logs'] = logs
    yield 100
    yield final_res