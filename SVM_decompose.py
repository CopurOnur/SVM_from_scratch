# add path to upload the class which we will inherit 
import sys
sys.path.append('../Q1/')

from SVM import *




class SVM_decompose(SVM):
    def __init__(self, args):
        # assign global parameters
        self.X, self.y = args['X'], args['y']
        self.C, self.q, self.tol, self.gamma = args['C'], args['q'], args['tol'], args['gamma']
        self.kernel_type, self.seed = args['kernel_type'], args['seed']
        
        # parameters to maintain asked information
        self.time = 0
        self.nit = 0
        self.nfev = 0
        
        
        self.full_size = self.X.shape[0]
        # determine the randomnes
        np.random.seed(self.seed)
        
        # initialize alphas to 0
        self._initial_alphas_()
        # initialize gradient
        self.grad = -np.ones(len(self.y))
        # initialize working set
        self.Working_set()
        
    ##############################################################################################################
    """
    Create constrains and initialize the alpha
    
    """
    def _initial_alphas_(self):
        self.alphas =  np.zeros(self.full_size) 
    
    # this time we decided to change a little bit constaints, otherwise should call it explicitly while now implicitly
    def constaints(self):
        eq_constraints = lambda alpha: np.dot(alpha, self.y[self.W])\
            + np.dot(self.alphas[self.notW], self.y[self.notW])
    
        return [{'type': 'ineq', 'fun': lambda alpha: alpha}, 
                {'type': 'ineq', 'fun': lambda alpha: -(alpha - self.C)},
                {'type': 'eq', 'fun': eq_constraints}]
    
    ##############################################################################################################
    """
    Optimizaton part
    
    Note: calculation of matrix Q was changes,
    now Q have the shape (working set, N-number of points in training set)
    
    """
    # make matrix sizes (size Working set, number of samples in training data)
    def linear_sub_matrix(self):
        return np.matmul(self.X[self.W], self.X.T)
       
    
    def pairwise_matrix(self):
        # matrix of labels using broadcast
        Y = self.y[self.W].reshape(-1,1)*self.y
        # rbf kernel
        if self.kernel_type == 'rbf':
            K = self.kernel_gaus_matrix(pairwise_distances(self.X[self.W], self.X, metric='euclidean')**2)
        # linear kernel
        elif self.kernel_type == 'Linear':
            K = self.linear_sub_matrix()
        return K*Y, K
            
    # get matrixes
    def get_submatrix(self):
        '''
        Returns
        -------
        Get submaprices from the main matrix
        Make matrices:
        Q_WW size: (working set, working set)
        Q_WnotW: (working set, N)

        '''
        self.Q_WW = np.array([i[self.W] for i in self.Q])
        self.Q_WnotW = np.array([i[self.notW] for i in self.Q])
    

    def loss(self, alphas):
        '''
        This functions counts the loss of objective function

        Parameters
        ----------
        alphas : alphas of working set

        Returns
        -------
        loss of objective function

        '''
        self.first = (1/2)*np.dot(np.dot(alphas, self.Q_WW), alphas)
        self.sec = np.dot(np.dot(alphas, self.Q_WnotW), self.alphas[self.notW])
        self.four = -(sum(alphas) + sum(self.alphas[self.notW]))
        return self.first + self.sec + self.four

    
    
    def minimize(self):
        '''
        Minimize function

        Returns
        -------
        Upgrates alphas from working set

        '''
        
        x0 = self.alphas[self.W]
        self.res = minimize(self.loss, x0, constraints=self.constaints(), method='SLSQP', options={})
        
        # updating of the gradients
        self.update_grad()
        
        # all values less then 10e-10 and then round the values 6 decimals
        self.alphas[self.W] = np.round(self.res.x*(self.res.x>10e-10)*1,6)
        
        
    def update_grad(self):
        self.grad += np.sum(self.Q.T * (self.res.x - self.alphas[self.W]).reshape(1, -1), axis=1)
    

    def fit(self):
        '''
        Fit function

        Returns
        -------
        Fit function loops till the KKT satisfied
        
        Due to computantional stability we decided to add 
        max_it parameter. In out case we dacided to assing 
        max_it = 600. The main idea of parameter to keep 
        track how many itterations have been done, if 
        KKT still not satisfied with respect to current 
        tolerance, so then we are strating to increase the 
        tolerance. We decided to call this procedure 
        "tolerance warm up".

        '''
        # keep track of the number of iterations
        self.k = 0
        # due to computantional stability we
        max_it = 600
        self.time = time.time()
        # main loop
        while self.m() - self.M() > self.tol :
            # get working set
            self.Working_set()
            # recound the main matrix
            self.Q, _ = self.pairwise_matrix()
            # get submatrixes
            self.get_submatrix()
            # minimize the subproblem
            self.minimize()
            # keep track of needed parameters
            self.k += 1
            self.nit += self.res.nit
            self.nfev += self.res.nfev
            
            # sometimes due to numerical stability we can start increase the threshold to satisfy kkt
            if self.k > max_it:
                self.tol = self.tol*(self.k-max_it)
            
        self.time = time.time() - self.time
        
    
# the best parameters which we have found for rbf kernel        
def get_best_params():
    C = 5
    gamma = 0.0012
    return C, gamma