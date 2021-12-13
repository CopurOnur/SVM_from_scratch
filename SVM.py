# import libraries 
import time
import os
import gzip
import numpy as np
from collections import defaultdict

from scipy.optimize import minimize
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# import data uploading
import sys
sys.path.append('../')
from data import *


class SVM():
    def __init__(self, args):
        # assign global parameters
        self.X, self.y = args['X'], args['y']
        self.C, self.q, self.tol, self.gamma = args['C'], args['q'], args['tol'], args['gamma']
        self.kernel_type, self.seed = args['kernel_type'], args['seed']
        
        # parameters to maintain asked information
        self.t = 0
        self.nit = 0
        self.nfev = 0
        
        self.size = self.X.shape[0]
        # determine the randomnes
        np.random.seed(self.seed)
        
        # initialize alphas uniformly from the interval [0, C]
        self.alphas = self._initialize_weights()
        # create main matricies
        self.Ky, self.K = self.pairwise_matrix()
        # get initial pbjective value
        self.get_initial_vall_obj()
        
        # constrains
        
        # Inequality contrains
        # we decided to split alpha_help and alpha_constraints because 
        # in this case it would be easier to pass jacobian of the jacobian of contr.
        self.alpha_help = np.concatenate((np.zeros(self.size), self.C * np.ones(self.size)))
        self.alpha_constraints = np.vstack((-np.eye(self.size), np.eye(self.size)))
        
        
        # assing the constrains
        self.constraints = ({'type': 'ineq', 'fun': lambda x: self.inequality_constraints(x), 'jac': lambda x: -self.alpha_constraints},
                            {'type': 'eq', 'fun': lambda x: self.equality_constraints(x), 'jac': lambda x: self.y})
    
    #######################################################################################################################################
    """
    Create constrains and initialize the alpha
    
    """
    # make inequality constrains
    def inequality_constraints(self, x):
        return self.alpha_help - np.dot(self.alpha_constraints, x)
    # make equality cinstraints
    def equality_constraints(self, x):
        return np.dot(x, self.y)
    
    
    # alpha initialization 
    def _initialize_weights(self):
        # determine the randomness
        np.random.seed(self.seed)
        alphas = np.random.uniform(low = 0, high = self.C, size = self.size)
        return alphas
    
    
    #######################################################################################################################################
    # kernel functions
    
    # rbf kernel 
    def gaussian_kernel(self, x, y):
        return np.exp(-self.gamma * np.sum(np.square(x - y)))
    #  polynomial kernel 
    def polynomial_kernel(self, x, y):
        return (np.matmul(x, y) + 1)**self.gamma
    # linear kernel
    def linear_kernel(self, x, y):
        return np.dot(x,y)
    
    # general function for kernel counting 
    def kernel(self, x, y):
        if self.kernel_type == 'rbf':
            return self.gaussian_kernel(x,y)
        
        elif self.kernel_type == 'Polynomial':
            return self.polynomial_kernel(x,y)
        
        else:
            return self.linear_kernel(x,y) 
        
    # functions to compute the full matricies
    def kernel_gaus_matrix(self, X):
        return np.exp(-self.gamma*X)
    
    def kernel_linear_metrix(self):
        return np.matmul(self.X, self.X.T)
    
    # general function to count the matrix Q
    def pairwise_matrix(self):
        # matrix of labels using broadcast
        Y = self.y*self.y.reshape(-1,1)
        if self.kernel_type == 'rbf':
            K = self.kernel_gaus_matrix(pairwise_distances(self.X, metric='euclidean')**2)
        elif self.kernel_type == 'Linear':
            K = self.kernel_linear_metrix()
        
        
        return Y*K, K
    
    ####################################################################################################################################
            
    """
    Optimization part:
    
    Loss function
    Jacobian
    Fit method
    Interceiot method
    Predict
    
    """
    
    def loss(self,alphas):
        '''
        This functions counts the loss of objective function

        Parameters
        ----------
        alphas : alphas

        Returns
        -------
        loss of objective function

        '''

        return -(alphas.sum() - 0.5 * np.dot(alphas.T, np.dot(self.Ky, alphas)))  # negative

    def jac(self,x):
        """
        Calculate the Jacobian of the loss function (for the QP solver)
        
        """
        return np.dot(x.T, self.Ky) - np.ones(x.shape[0], dtype=np.float64)
    
    def fit(self):
        '''
        Fit function, with SLSQP solver

        Returns
        -------
        Assign optimized alphas to global parameter alphas

        '''
        
        self.time = time.time()
        # minimize function
        self.res = minimize(self.loss, self.alphas, jac=self.jac ,constraints=self.constraints, method='SLSQP', options={})
        # all values less then 10e-10 and then round the values 6 decimals
        self.alphas = np.round(self.res.x*(self.res.x>10e-15)*1,10)
        
        # maitain needed information
        self.nit += self.res.nit
        self.nfev += self.res.nfev
        self.time = time.time() - self.time
        
        return self.res
    
    def get_interceipt(self):
        # support vectors idxs
        self.support_idx = np.where(0 < self.alphas)[0]  
        # index of support vectors that happen to lie on the margin
        self.margin_idx = np.where((0 < self.alphas) & (self.alphas < self.C))[0]  
        
        # find the intercept term, b 
        self.alpha_y = self.alphas * self.y
        #support_a_times_t = self.alpha_y[self.support_idx]  
        cum_b = 0
        # calculate the interceipt
        for idx in self.support_idx:
            # evaluate the kernel between x_n and support vectors; fill the rest with zeros
            kernel_eval = np.array([self.kernel(x_m, self.X[idx]) if a_m > 0 else 0 for x_m, a_m in zip(self.X, self.alphas)])
            b = self.y[idx] - self.alpha_y.dot(kernel_eval)
            cum_b += b
        self.b = cum_b / (len(self.support_idx))
        
    
    def predict(self, X_test, y_test):
        '''
        Parameters
        ----------
        X_test : X test set
        y_test : true labels


        Returns
        -------
        y : return the distance to the hyperplane.

        '''
        # get interceipt
        self.get_interceipt()
        # prepare predict array
        y = np.empty(len(X_test))
        for i, s in enumerate(X_test):  
            # evaluate the kernel between new data point and support vectors; 0 if not a support vector
            kernel_eval = np.array([self.kernel(s, x_m) if a_m > 0 else 0 for x_m, a_m in zip(self.X, self.alphas)])
            y[i] = self.alpha_y.dot(kernel_eval) + self.b
        return y
    
    '''
    During the grid search we have found that time taking to calculate the 
    interceipt is quite big, so we decided to improve the speed of the cal-
    culation of interceipt
    '''
    def get_interceipt_fast(self):
        
        self.support_idx = np.where(0 < self.alphas)[0]  # support vectors idxs
        
        # still need to find the intercept term, b (unfortunately this name collides with the earlier Ax=b);
        self.alpha_y = self.alphas * self.y
        #support_a_times_t = self.alpha_y[self.support_idx]
        cum_b = 0
        
        # get needed rows of the matrix Q with respect of support indexes
        # so the main difference that we dont need to recount the values
        # from matrix Q
        kernel_eval = np.zeros((self.K.shape[0], self.K.shape[1]))
        kernel_eval[self.support_idx, :] = self.K[self.support_idx, :]

        for idx in self.support_idx:
            b = self.y[idx] - self.alpha_y.dot(kernel_eval[idx,:])
            cum_b += b
        self.b = cum_b / (len(self.support_idx))
        
    # the main difference that instead of recomputing values for interceipt we just take them from the matrix Q
    def predictt(self, X_test, y_test):
        '''
        Parameters
        ----------
        X_test : X test set
        y_test : true labels

        Returns
        -------
        y : return the distance to the hyperplane.

        '''
        # use fast variation of calculating the interceipt
        self.get_interceipt_fast()
        y = np.empty(len(X_test))
        for i, s in enumerate(X_test):  
            # evaluate the kernel between new data point and support vectors; 0 if not a support vector
            kernel_eval = np.array([self.kernel(s, x_m) if a_m > 0 else 0 for x_m, a_m in zip(self.X, self.alphas)])
            y[i] = self.alpha_y.dot(kernel_eval) + self.b
        return y

        
    
    
    ####################################################################################################################################
    
    """
    Help methods
    
    """
    
    # Get index sets
    def index_sets(self):
        # calculating help sets to get right R and S sets
        L_plus_set  = (self.alphas == 0) & (self.y > 0)
        L_minus_set = (self.alphas == 0) & (self.y < 0)
        U_plus_set  = (self.alphas == self.C) & (self.y > 0)
        U_minus_set = (self.alphas == self.C) & (self.y < 0)
        margin_set  = (self.alphas > 0) & (self.alphas < self.C)
        # assign R and S sets to the global variables
        self.R = np.where(L_plus_set  | U_minus_set | margin_set)[0]
        self.S = np.where(L_minus_set | U_plus_set  | margin_set)[0]
    
    # the product of -grad * 1/labels with respect to indexes
    def m_grad_y(self, idx):
        return -self.grad[idx] / self.y[idx]
    
    # getting the Working set for decomposition algorithm
    def Working_set(self):
        # get index sets
        self.index_sets()        
        # get q/2 indexes from S and R indxes with respect to gradient
        self.R_grad_idx = np.array(list(map(lambda x: x[1],sorted(list(zip(self.m_grad_y(self.R), self.R)), key=lambda x:x[0], reverse=True)[:self.q//2])))
        self.S_grad_idx = np.array(list(map(lambda x: x[1],sorted(list(zip(self.m_grad_y(self.S), self.S)), key=lambda x:x[0])[:self.q//2])))
        self.W = sorted(np.concatenate([self.R_grad_idx, self.S_grad_idx]))
        self.notW = np.setxor1d(np.arange(len(self.y)), self.W)
    
    # calculate the final weights of the hyperplane W: Wx+b    
    def get_Weights(self):
        self.Weights = np.dot(self.alphas*self.y, self.X)
    
    # calculate m, M to check the KKT
    def m(self):
        return np.max(self.m_grad_y(self.R))
    def M(self):
        return np.min(self.m_grad_y(self.S))
    
    # Check KKT (works only for full SVM because full gradient have to be counted) 
    def check_KKT(self):
        self.grad = np.dot(self.Ky, self.alphas) - np.ones(len(self.alphas))
        self.index_sets()
        
        
        print(" m(R)-M(S) = ", np.round(self.m() - self.M(), 6))
        return "Satisfied"
        
    def get_initial_vall_obj(self):
        self.init_value_obj = self.loss(self.alphas)
        
    # get full set of information about the training and testing process    
    def get_info(self, X_test, y_test):
        '''
        This functions to print all needed information 
        the try, expcept technique is used because
        we use this function in Q1, Q2, Q3 and sometimes
        the procedures to get needed values a little bit different

        Parameters
        ----------
        X_test : test set
        y_test : test label set
        Returns
        -------
        Print all needed information

        '''
        print('Kernel: ', self.kernel_type)
        print('hyperparameter C = ', self.C)
        print('q = ', self.q)
        try:
            print('Initial value of objective func.', self.init_value_obj)
            
        except: 
            pass
        
        try:
            print('Final value of objective func. = ', self.loss(self.alphas))
        except: 
            try:
                print('Final value of objective func. = ', self.loss(self.alphas[self.W]))
            except:
                pass
            pass
        try:
            print('Number of itterations = ', self.nit)
            print('Number of function evaluations = ', self.nfev)
        except: 
            pass
        try:
            print('KKT', self.check_KKT())
        except:
            print('m(R)- M(S) =', self.m() - self.M()) 
            
        print('Time to train = ', self.time)
        try:
            print('Number of itterations k = ', self.k)
        
        except:
            pass
        # print train accuracy 
        predict = self.predict(self.X, self.y)
        num_wrong = np.sum((predict * self.y) < 0)
        print('Train Classification accuracy : ' + str(1 - num_wrong / self.y.shape[0]))
        
        # print Test accuracy
        predict = self.predict( X_test, y_test)
        num_wrong = np.sum((predict * y_test) < 0)
        print('Test Classification accuracy : ' + str(1 - num_wrong / y_test.shape[0]))
        
        #print("Confusion matrix", confusion_matrix(y_test, np.sign(predict * y_test)))
        print("Confusion matrix", confusion_matrix(y_test,np.sign(predict)))

########################################################################################################################

########################################### CROSS VALIDATION FUNCTION ##################################################

########################################################################################################################
import pickle 
import matplotlib.pyplot as plt


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def cross_validate(X, y, gamma, c, folds = 2):
    start_time = time.time()

    av_acc_test = 0
    av_acc_train = 0
    kf = KFold(n_splits = folds, shuffle = True, random_state = 1902392)
    for train_index, test_index in kf.split(X):
        args = {'X': X[train_index],
                'y': y[train_index],
                'C': c,
                'q': None,
                'tol': 10e-4,
                'gamma': gamma,
                'kernel_type': 'rbf',
                'seed': 0
                }
        
        svm = SVM(args)
        res2 = svm.fit()
        predictions_test = svm.predict(X[test_index], y[test_index])
        num_wrong_test = np.sum((predictions_test * y[test_index]) < 0)
        
        av_acc_test += (1 - num_wrong_test / y[test_index].shape[0]) / folds

        predictions_train = svm.predict(X[train_index], y[train_index])
        num_wrong_train = np.sum((predictions_train * y[train_index]) < 0)
        av_acc_train += (1 - num_wrong_train / y[train_index].shape[0])/folds

    print('Parameter C = ', c)
    print('Parameter gamma = ', gamma)
    print('CV ratio:', svm.support_idx.shape[0]/svm.X.shape[0])
    print('Classification accuracy cv train: ' + str(av_acc_train))
    print('Classification accuracy cv test: ' + str(av_acc_test))
    print("Cross validation time is : ", time.time() - start_time)
    
    return av_acc_train, av_acc_test, time.time() - start_time, svm.support_idx.shape[0]/svm.X.shape[0]




########################################################################################################################

########################################### HYPER-PARAMETER TUNNING ####################################################

########################################################################################################################

def hyper_parameter_tune(gamma_list, c_list):
    '''
    

    Parameters
    ----------
    gamma_list : gamma space list
    c_list : C space list

    Returns
    -------
    parameter_scores : dict where the key is the parameters and values needed statistics

    '''
    parameter_scores = {}
    for gamma in gamma_list:
        for c in c_list:
            acc_train, acc_test, t, SV = cross_validate(X_train, y_train, gamma, c)
            parameter_scores["g: "+str(gamma) + " c: " + str(c)] = [acc_train, acc_test, t, SV]
            save_obj(parameter_scores, 'grid_search_final')
    return parameter_scores


########################################################################################################################

################################################ Graphics ##############################################################

########################################################################################################################
def plot_graphs(df, X, sort_by,condition = 'C'):
    '''
    This functions make the plots which we have discussed in the report
    Generaly we are not going to plot, just wanted to provide with the code

    Parameters
    ----------
    df : dataframe with parameters
    X : name of X axis (have to coinside withone column from df)
    sort_by : sort parameter (have to coinside withone column from df)
    condition : parameter on which to be conditioned

    

    '''
    # Y axis is always accuracies (Y1, Y2 have to coinside with columns from df)
    Y1 = 'Acc_train'
    Y2 = 'Acc_test'
    
    # unique values in condition column
    condition_unique = df[condition].unique()
    # make the plots
    if len(condition_unique) == 20:
        nrows, ncols = 5, 4
        fig, axs = plt.subplots(nrows, ncols, figsize = (20,16))
        fig.tight_layout(pad=3.0)
    if len(condition_unique) != 20:
        nrows, ncols = 5, 4
        fig, axs = plt.subplots(nrows, ncols, figsize = (20,16))
        fig.tight_layout(pad=3.0)
    i = 0
    for row in range(nrows):
        for col in range(ncols):
            a = df[df[condition] == condition_unique[i]].sort_values(sort_by)
            axs[row, col].plot(a[X], a[Y1], color =  'b', label = 'Train')
            axs[row, col].plot(a[X], a[Y2], color = 'orange', label = 'Test')
            
            axs[row, col].set_title(condition + " = " + str(condition_unique[i]), size=14)
            axs[row, col].set_ylabel('Accuracy', size=14)
            axs[row, col].set_xlabel(X, size=14)
            axs[row, col].set_ylim([0,1.1])
            axs[row, col].legend(loc = 3, prop={'size': 16})
            
            i+=1
    plt.show()
    
# the best parameters which we have found for rbf kernel        
def get_best_params():
    C = 5
    gamma = 0.0012
    return C, gamma