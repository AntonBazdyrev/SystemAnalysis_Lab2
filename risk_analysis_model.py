import pandas as pd
import numpy as np
from scipy.stats import norm

from multiplicative_model import MultiplicativeModelSystem
from additive_model import AdditiveModelSystem


EMERGENCY_VALUES = 'data/emergency_values.csv'
CRUSH_VALUES = 'data/crush_values.csv'


def get_risk_function(EMERGENCY_VALUES=EMERGENCY_VALUES, CRUSH_VALUES=CRUSH_VALUES):
    em = pd.read_csv(EMERGENCY_VALUES)['values'].values
    cr = pd.read_csv(CRUSH_VALUES)['values'].values

    def risk_function(pred, sigma, ind):
        if pred >= em[ind]:
            return 0
        elif pred <= cr[ind]:
            return 1
        else:
            return norm.cdf((cr[ind] - pred)/(sigma + 1e-7), 0, 1)

    return risk_function


class RiskAnalysisModel:
    def __init__(self, predictor, risk_function=get_risk_function()):
        self.predictor = predictor
        self.risk_function = risk_function

    def get_results(self, X, Y, window=20):
        curr_pos = window
        diffs = []
        risks = []
        y_preds = []
        y_true = []
        print(Y.shape)
        for i in range(curr_pos, Y.shape[0]):
            print(i)
            X_observed, Y_observed = [x[i - curr_pos: i] for x in X], Y[i - curr_pos: i]
            self.predictor.fit(X_observed, Y_observed)
            #print('FIIIT')
            X_test = [normalizer.transform(x[i: i+1]) for x, normalizer in zip(X, self.predictor.X_normalizers)]
            Y_test_scaled = self.predictor.Y_normalizer.transform(Y[i: i+1])
            Y_pred_scaled = self.predictor.predict(X_test)
            Y_preds = self.predictor.Y_normalizer.inverse_transform(Y_pred_scaled)
            Y_test = self.predictor.Y_normalizer.inverse_transform(Y_test_scaled)
            diffs.append((Y_test - Y_preds).flatten())
            Y_preds, Y_test = Y_preds.flatten(), Y_test.flatten()
            y_preds.append(Y_preds)
            y_true.append(Y_test)
            sigma = np.array(diffs).std(axis=0)
            r = []
            for ind, (pred, s) in enumerate(zip(Y_preds, sigma)):
                r.append(self.risk_function(pred, s, ind))
                #print('END')
            risks.append(r)
            yield np.vstack(y_true), np.vstack(y_preds), np.vstack(risks)
        #return y_true, y_preds, risks

def get_risks_results(params):
    add_solver = MultiplicativeModelSystem(polynomes_type=params['method'], calculate_separately=params['lambda_from_3sys'], \
                                           degrees=params['X_degree'], activation_type=params['activation'])

    risk_model = RiskAnalysisModel(add_solver)
    for y_true, y_preds, risks in risk_model.get_results(params['X'], params['y']):
        yield {
        'X': params['X'], 'Y': y_true, 'Y_scaled': y_true, 'Y_preds': y_preds, 'Y_preds_scaled': y_preds,
        'Y_err': risks, 'Y_err_scaled': risks, 'logs': 'tyrgew'
    }

    #print(y_true)
    return {
        'X': params['X'], 'Y': y_true, 'Y_scaled': y_true, 'Y_preds': y_preds, 'Y_preds_scaled': y_preds,
        'Y_err': risks, 'Y_err_scaled': risks, 'logs': 'tyrgew'
    }

