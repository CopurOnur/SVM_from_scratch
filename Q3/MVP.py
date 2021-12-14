# add path to upload the class which we will inherit 
import sys
sys.path.append('../Q2/')

from SVM_decompose import *

class MVP(SVM_decompose):
    def __init__(self, args):
        # assign global parameters
        self.X, self.y = args['X'], args['y']
        self.C, self.q, self.tol, self.gamma = args['C'], args['q'], args['tol'], args['gamma']
        self.kernel_type, self.seed = args['kernel_type'], args['seed']
        
        # parameters to maintain asked information
        self.time = 0
        self.full_size = self.X.shape[0]
        # determine the randomnes
        np.random.seed(self.seed)
        
        # initialize alphas to 0
        self._initial_alphas_()
        # initialize gradient
        self.grad = -np.ones(len(self.y))
        # initialize index sets
        self.index_sets() 
        
    def get_MVP_pair(self):
        '''
        Calculate MVP pair

        Returns
        -------
        Assign the indexes of mvp pair (I,J sets)

        '''
        I_lst = list(zip(self.m_grad_y(self.R),self.R))
        self.I = sorted(I_lst, key = lambda x: x[0], reverse = True)[0][1]
        J_lst = list(zip(self.m_grad_y(self.S),self.S))
        self.J = sorted(J_lst, key = lambda x: x[0])[0][1]
        
        # update d
        self.get_d()
        
    def get_d(self):
        '''
        Function to make vector d
        only position from sets I and J are not equal to zero

        Returns
        -------
        Vector d

        '''
        self.d = np.zeros(len(self.y))
        self.d[self.I] = self.y[self.I]
        self.d[self.J] = -self.y[self.J]
        
        
    def get_t(self):
        '''
        Calculate max feasible, calculate t*, then take the minimum between them.
        
        Returns
        -------
        Minimum between max feasible, calculate t*
        
        '''
        
        # get MVP pair
        self.get_MVP_pair()
        idx = np.where(self.d!=0)[0]
        t_max_feass = np.min(np.array([self.C - self.alphas[i] if self.d[i]>0 else self.alphas[i]/np.abs(self.d[i]) for i in idx]))
        # calculate t*
        numerator = np.sum(-self.grad * self.d)
        denominator = np.dot(np.dot(self.d[self.W], self.Q_WW), self.d[self.W])
        t_star = numerator/denominator

        self.t = np.min([t_star, t_max_feass])
        
    def step(self):
        '''
        Do MVP step

        Returns
        -------
        None.

        '''
        self.index_sets() 
        
        # update MVP and update d 
        self.get_MVP_pair()
    
    
        # update workingset
        self.W = np.sort(np.array([self.I, self.J]))
        self.notW = np.sort(np.setxor1d(np.arange(len(self.y)), self.W))
        
        # update matrixes
        self.Q, _ = self.pairwise_matrix()
        self.get_submatrix()
        
        self.get_t()
        new_alpha = self.alphas + self.t*self.d
        
        # update gradient 
        coef = new_alpha[self.W] - self.alphas[self.W]
        
        self.grad += np.sum(self.Q.T*coef, axis = 1)
        
        # update alphas
        self.alphas = new_alpha
        self.alphas = np.round(self.alphas,6)
        
    def fit(self):
        '''
        Fit function

        Returns
        -------
        Fit function loops till the KKT satisfied
        
        Again due to computantional stability we decided to add 
        max_it parameter. This time we decided to make it greater
        because for one step we are changing just 2 variables instead of 
        optional parameter q.
        

        '''
        
        self.time = time.time()
        # maintain the number of iterations
        self.k = 0
        max_it = 1200
        # main loop
        while self.m() - self.M() > self.tol:
            
            self.step()
            self.k += 1
            if self.k > max_it:
                self.tol = self.tol*(self.k - max_it)
        self.time = time.time() - self.time
    
    # computationally more stable version of computing the sets 
    def index_sets(self):
        '''
        Basically this function computes the same R and S set,
        we implemented this function because it computationaly
        more stable
        

        Returns
        -------
        R, S sets 

        '''
        self.R = np.where(((self.alphas < self.C) & (self.y > 0)) | ((self.alphas > 0) & (self.y < 0)))[0]
        self.S = np.where(((self.alphas < self.C) & (self.y < 0)) | ((self.alphas > 0) & (self.y > 0)))[0]


    
    def oaa_predict(self, X_test, alphas, b, alpha_y):
        '''
        Function to make predictions in the question 4
        We have used One Against All technique. We need this 
        function, because in the next question we have to compute
        the predictions M time, where M is the number of classes 
        in training dataset.

        Parameters
        ----------
        X_test : test dataset
        alphas : alphas
        b : interceipt
        alpha_y : the product of alphas and y where y is labels on which MVP was trained

        Returns
        -------
        y : return the distance to the hyperplane.

        '''
        y = np.empty(X_test.shape[0])
        for i, s in enumerate(X_test):  
            # evaluate the kernel between new data point and support vectors; 0 if not a support vector
            kernel_eval = np.array([self.kernel(s, x_m) if a_m > 0 else 0 for x_m, a_m in zip(self.X, alphas)])
            y[i] = alpha_y.dot(kernel_eval) + b
        return y   

