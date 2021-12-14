# import necessary class/functions
from SVM import *

# Grid search, uncomment if you want to reproduce the results
# gamma_lst = sorted(np.round(np.concatenate([[1/(X_train.shape[1]*np.var(X_train))],np.linspace(0.01, 10e-5, 10)]),4))
# C_lst = sorted(np.concatenate([np.linspace(1e-3, 1,10 ), np.linspace(1, 11,11)]))
# parameter_scores = hyper_parameter_tune(X_train, y_train, gamma_lst, C_lst)

# get data
X_train, X_test, y_train, y_test = get_data('binary')
# get best parametes
C,gamma = get_best_params()
# prepare the arguments
args = {'X':X_train,
        'y':y_train,
        'C': C,
        'q':'Full problem',
        'tol':10e-4,
        'gamma':gamma,
        'kernel_type': 'rbf',
        'seed':1902392
        }
# initialize the model, fit, predict, get all needed statistics
svm = SVM(args)
_ = svm.fit()
svm.get_info(X_test, y_test)