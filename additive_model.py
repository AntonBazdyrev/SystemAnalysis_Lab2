import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from inconsistent_system_solver import InconsistentSystemSolver
from numpy.polynomial import Polynomial as pm

def basis_sh_chebyshev(degree):
    basis = [pm([-1, 2]), pm([1])]
    for i in range(degree):
        basis.append(pm([-2, 4])*basis[-1] - basis[-2])
    del basis[0]
    return basis


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class AdditiveModelSystem:
    def __init__(self, polynomes_type='Chebyshev', calculate_separately=False, degrees=[]):
        self.system_solver = InconsistentSystemSolver()
        self.lambda_values = {}
        self.alpha_values = {}
        self.c_values = {}
        self.polynomes = {}
        self.F_functions = {}
        self.calculate_separately = calculate_separately
        self.polynomes_type = polynomes_type
        self.degrees = degrees

    def _parse_vectors(self, vectors, name='X'):
        return {f'{name}{i + 1}': vec for i, vec in enumerate(vectors)}

    def _get_poly(self, y_ind, x_ind, x_ind_coord):
        return sum(self.lambda_values[y_ind][x_ind][x_ind_coord][degree] * self.basises[x_ind][degree] for degree in
                   range(len(self.basises[x_ind])))

    def _poly_transform(self, X, x_ind, y_ind):
        res = np.zeros_like(X)
        for value_ind in range(X.shape[0]):
            for x_ind_coord in range(X.shape[1]):
                res[value_ind][x_ind_coord] = self._get_poly(y_ind, x_ind, x_ind_coord)(X[value_ind][x_ind_coord])
        return res

    def _F_transform(self, X, x_ind, y_ind):
        X = self._poly_transform(X, x_ind, y_ind)
        res = np.zeros(X.shape[0])
        for value_ind in range(X.shape[0]):
            for x_ind_coord in range(X.shape[1]):
                res[value_ind] += self.alpha_values[y_ind][x_ind][x_ind_coord] * X[value_ind][x_ind_coord]
        return res

    def predict(self, X):
        if self.c_values:
            results = np.zeros((X[0].shape[0], len(self.c_values)))
            for y_ind in range(len(self.c_values)):
                for x_ind, x in enumerate(X):
                    x = self._F_transform(x, x_ind, y_ind)
                    results[:, y_ind] += self.c_values[y_ind][x_ind] * x
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
        self.basises = [basis_sh_chebyshev(d) for d in degrees]
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

        self._lambda_solver(X, Y, calculate_separately=calculate_separately)
        self._alpha_solver(X, Y)
        self._c_solver(X, Y)

    def _lambda_solver(self, X, bqi, calculate_separately=True):
        X, bqi = self._parse_vectors(X, name='X'), self._parse_vectors(bqi, name='bqi')

        if calculate_separately:
            for x_ind, (basis, (x_name, x)) in enumerate(zip(self.basises, X.items())):
                print(x_name)
                for y_ind, (bqi_name, b) in enumerate(bqi.items()):
                    print(bqi_name)
                    x_poly = np.hstack([poly(x) for poly in basis])
                    results = self.system_solver.solve(x_poly, b)
                    for poly_degree, res in enumerate(chunks(results, x.shape[1])):
                        for x_ind_coord, l in enumerate(res):
                            self.lambda_values[y_ind][x_ind][x_ind_coord][poly_degree] = l

                    for x_ind_coord in range(x.shape[1]):
                        self.polynomes[y_ind][x_ind][x_ind_coord] = self._get_poly(y_ind, x_ind, x_ind_coord)
                    print(results)
        else:
            print('else')
            for y_ind, (bqi_name, b) in enumerate(bqi.items()):
                print(bqi_name)
                x_dims = [x.shape[1] * len(basis) for basis, (k, x) in zip(self.basises, X.items())]
                merged_x = np.hstack(
                    [np.hstack([poly(x) for poly in basis]) for basis, (k, x) in zip(self.basises, X.items())])
                results = self.system_solver.solve(merged_x, b)
                print(results)
                print('MAX NORM DIFF: ', (merged_x @ results - b).max())

                curr_x_ind = 0
                for x_ind, x_dim in enumerate(x_dims):
                    res_curr = results[curr_x_ind: curr_x_ind + x_dim]
                    # print(res_curr)
                    for poly_degree, res in enumerate(chunks(res_curr, list(X.items())[x_ind][1].shape[1])):
                        #  print(poly_degree, res)
                        for x_ind_coord, l in enumerate(res):
                            self.lambda_values[y_ind][x_ind][x_ind_coord][poly_degree] = l
                    curr_x_ind += x_dim

                    for x_ind_coord in range(list(X.items())[x_ind][1].shape[1]):
                        self.polynomes[y_ind][x_ind][x_ind_coord] = self._get_poly(y_ind, x_ind, x_ind_coord)
                    print(results)

    def _alpha_solver(self, X, bqi):
        X, bqi = self._parse_vectors(X, name='X'), self._parse_vectors(bqi, name='bqi')
        for y_ind, (bqi_name, b) in enumerate(bqi.items()):
            x_dims = [x.shape[1] for k, x in X.items()]
            merged_x = np.hstack([self._poly_transform(x, x_ind, y_ind) for x_ind, (k, x) in enumerate(X.items())])
            results = self.system_solver.solve(merged_x, b)
            print(results)
            print((merged_x @ results - b).max())

            curr_x_ind = 0
            for x_ind, x_dim in enumerate(x_dims):
                res_current = results[curr_x_ind: curr_x_ind + x_dim]

                for x_ind_coord, alpha in enumerate(res_current):
                    self.alpha_values[y_ind][x_ind][x_ind_coord] = alpha

                self.F_functions[y_ind][x_ind] = sum(self.alpha_values[y_ind][x_ind][x_ind_coord] * \
                                                     self._get_poly(y_ind, x_ind, x_ind_coord) \
                                                     for x_ind_coord in range(x_dim))

    def _c_solver(self, X, Y):
        X, Y = self._parse_vectors(X, name='X'), self._parse_vectors(Y, name='Y')

        for y_ind, (bqi_name, y) in enumerate(Y.items()):
            X_merged = np.array([self._F_transform(x, x_ind, y_ind) for x_ind, (k, x) in enumerate(X.items())]).T
            results = self.system_solver.solve(X_merged, y)
            for x_ind, c in enumerate(results):
                self.c_values[y_ind][x_ind] = c

    def get_logs(self):
        if self.c_values:
            file_str = ''
            file_str += 'Ψ functions (lambda values):\n'
            for y_ind in range(len(self.lambda_values)):
                file_str += f'For Y{y_ind + 1}\n'
                for x_ind in range(len(self.lambda_values[y_ind])):
                    for x_ind_coord in range(len(self.lambda_values[y_ind][x_ind])):
                        file_str += f'Ψ{x_ind}{x_ind_coord} = ' + \
                                    ' + '.join(str(self.lambda_values[y_ind][x_ind][x_ind_coord][degree]) + \
                                               f'*T{degree}(x{x_ind}{x_ind_coord})' for degree in
                                               range(len(self.lambda_values[y_ind][x_ind][x_ind_coord]))) + \
                                    '\n'

            file_str += 'Φ functions (alpha values):\n'
            for y_ind in range(len(self.alpha_values)):
                for x_ind in range(len(self.alpha_values[y_ind])):
                    file_str += f'Φ{y_ind}{x_ind} = ' + \
                                ' + '.join(str(self.alpha_values[y_ind][x_ind][x_ind_coord]) + \
                                           f'*Ψ{y_ind}{x_ind}(x{x_ind}{x_ind_coord})' for x_ind_coord in
                                           range(len(self.alpha_values[y_ind][x_ind]))) + \
                                '\n'
            file_str += 'F functions (c values):\n'
            for y_ind in range(len(self.c_values)):
                file_str += f'F{y_ind} = ' + \
                            ' + '.join(str(self.c_values[y_ind][x_ind]) + f'*Φ{y_ind}{x_ind}' for x_ind in
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
    add_solver = AdditiveModelSystem(polynomes_type=params['method'], calculate_separately=params['lambda_from_3sys'], degrees=params['X_degree'])
    add_solver.fit(params['X'], params['y'])
    logs = add_solver.get_logs()
    final_res = add_solver.get_final_results(params['X'], params['y'])
    final_res['logs'] = logs
    return logs