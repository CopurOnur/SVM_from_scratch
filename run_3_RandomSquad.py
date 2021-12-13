# import necessary class/functions
from MVP import *

# get data
X_train, X_test, y_train, y_test = get_data('binary')
# get best parametes
C, gamma = get_best_params()
# prepare the arguments
q = 2
args = {'X':X_train,
        'y':y_train,
        'C': C,
        'q':q,
        'tol':10e-4,
        'gamma':gamma,
        'kernel_type': 'rbf',
        'seed':1902392
        }
# initialize the model, fit, predict, get all needed statistics
svm = MVP(args)
_ = svm.fit()
svm.get_info(X_test, y_test)