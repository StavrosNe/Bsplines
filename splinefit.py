import numpy as np
import matplotlib.pyplot as plt

class bspline:
    def __init__(self, degree:int = None, 
                 grid:int = None,
                 input:np.ndarray = None):
        
        self.max_degree = degree
        self.intervals = grid
        self.imax = self.intervals + self.max_degree -1
        self.X = input
        assert self.X.ndim == 1
        self.knot_vector = self.make_knot_vector()
        
    def basis_function(self,x, i, p, knot_vector)->np.ndarray:
        # im applying the Cox deBoor reqursion formula for bspline basis functions
        # p is the degree of the basis function
        V = knot_vector
        x = np.asarray(x)  
        
        if p == 0:
            if i != self.imax-1:
                return np.where((V[i] <= x) & (x < V[i+1]), 1, 0)
            else:
                return np.where((V[i] <= x) & (x <= V[i+1]), 1, 0)
        
        else:
            # Avoid division by zero error
            if V[i+p] == V[i]:
                A1 = 0
            else:
                A1 = (x - V[i]) / (V[i+p] - V[i])
                
            if V[i+p+1] == V[i+1]:
                A2 = 0
            else:
                A2 = (V[i+p+1] - x) / (V[i+p+1] - V[i+1])
            
            B1 = self.basis_function(x, i, p-1, knot_vector)
            B2 = self.basis_function(x, i+1, p-1, knot_vector)
            
            N = A1 * B1 + A2 * B2

        # evaluates Ni,p at all x data points
        return N
    
    def basis_function_derivative(self,x, i, p, knot_vector)->np.ndarray:

        V = knot_vector
        x = np.asarray(x)  
        
        if p == 0:
            return np.zeros_like(x)
        else:
            # Avoid division by zero error
            if V[i+p] == V[i]:
                A1d = 0
                A1 = 0
            else:
                A1d = 1/(V[i+p] - V[i])
                A1 = (x - V[i]) / (V[i+p] - V[i])
                
            if V[i+p+1] == V[i+1]:
                A2d = 0
                A2 = 0
            else:
                A2d = (-1)/(V[i+p+1] - V[i+1])
                A2 = (V[i+p+1] - x) / (V[i+p+1] - V[i+1])
            
            B1 = self.basis_function(x, i, p-1, knot_vector)
            B2 = self.basis_function(x, i+1, p-1, knot_vector)
            
            B1d = self.basis_function_derivative(x, i, p-1, knot_vector)
            B2d = self.basis_function_derivative(x, i+1, p-1, knot_vector)


            Nd = A1d * B1 + A1 * B1d+ A2d * B2 + A2 * B2d

            return Nd
        
    def evaluate(self,C:np.ndarray = None)->np.ndarray:
        """
        X is a 1d vector
        """
        imax = self.imax
        assert len(C) == imax
        N = []
        try:
            for i in range(imax):  
                B = self.basis_function(self.X,i,self.max_degree,self.knot_vector)
                Ni = B.reshape((len(B), 1))
                N.append(Ni)

        except AssertionError as error:
            pass

        N = np.hstack(N)

        Y_hat = np.dot(N,C)

        return N,Y_hat
    
    def derivative_evaluate(self,C:np.ndarray = None)->np.ndarray:
        """
        X is a 1d vector
        """
        imax = self.imax
        assert len(C) == imax
        
        N_der = []
        try:
            for i in range(imax):  
                B = self.basis_function_derivative(self.X,i,self.max_degree,self.knot_vector)
                Ni = B.reshape((len(B), 1))
                N_der.append(Ni)

        except AssertionError as error:
            pass

        N_der = np.hstack(N_der)

        Y_hat_der = np.dot(N_der,C)

        return Y_hat_der
        
    def make_knot_vector(self)->np.ndarray:
        """
        creates a clamped knot vector, 
        depending on the input min and max value.
        X has shape (m,1),
        where m is the number of examples
        """
        def k_v_create(tmin,tmax, degree:int = None, intervals:int = None ):
            h = 1e-5
            k = degree
            vector = np.linspace(tmin-h,tmax+h,intervals)
            prev_int = np.full(k, tmin-h)
            after_int = np.full(k, tmax+h)

            knot_vector = np.concatenate((prev_int, vector, after_int))
            return knot_vector
        
        def find_domain(X):
            min_value = np.min(X)
            max_value = np.max(X)
    
            return min_value,max_value
        
        min_value,max_value = find_domain(self.X)

        knot_vector = k_v_create(min_value,max_value,degree=self.max_degree,
                                 intervals=self.intervals)
    

        return knot_vector
    

class BSpline():
    def __init__(self, degree: int = None, grid: int = None, 
                 input: np.ndarray = None, target: np.ndarray = None ):
        
        assert grid>=1

        self.input = input
        self.max_degree = degree
        self.grid = grid
        self.Y = target

        self.imax = self.grid + self.max_degree -1

        self.spline = bspline(degree=self.max_degree, grid=self.grid, input=input)


    def init_control_points(self):
        self.C = np.random.normal(0,1,self.imax)
    
    def update_control_points(self,learning_rate):
        self.C += -learning_rate*self.dC

    def output(self):
        Y_hat = self.Y_hat
        return Y_hat

    def show_progress(self,epoch:int = None):
        plt.figure(figsize=(10, 6))

        plt.plot(self.input,self.Y,label=f'Data')

        plt.plot(self.input, self.Y_hat, 
                 label=f'B-Spline of degree {self.max_degree}', 
                 color='black', linewidth=2)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Fit results in epoch:{epoch}')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot(X:np.ndarray=None,
             Y:np.ndarray=None,
             Y_hat:np.ndarray=None):
        
        plt.figure(figsize=(10, 6))

        plt.plot(X,Y,label=f'Data')

        plt.plot(X,Y_hat, 
                 label=f'B-Spline',linestyle = '--', 
                 color='black', linewidth=2)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Fit results')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_spline_components(self):
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8)) 

        n = self.N.shape[1]
        for i in range(n):
            sub_spline = self.C[i]*self.N[:,i]
            ax1.plot(self.input,sub_spline)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('BSpline Components')
        ax1.grid(True)


        ax2.plot(self.input, self.Y_hat, label=f'B-Spline of degree {self.max_degree}', color='black', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('BSpline Curve')
        ax2.legend()
        ax2.grid(True)


        plt.tight_layout()
        plt.show()


    
    def loss_function(self):
        self.mse = 0.5*np.mean((self.Y_hat - self.Y) ** 2)

    def backward(self)->np.array:
        m = len(self.Y_hat)
        dydc = self.N  
        dldy = np.transpose((1/m)*(self.Y_hat - self.Y)) 
        dldc = np.dot(dldy,dydc) 
        
        self.dC = np.transpose(dldc)

        assert self.dC.shape == self.C.shape

    def forward(self):
        self.N,self.Y_hat = self.spline.evaluate(C = self.C)

    def train(self,epochs:int = None, learning_rate = None, show_learning:bool = False):
        self.init_control_points()
        for i in range(1,epochs+1):
            self.forward()
            self.backward()
            self.update_control_points(learning_rate)
        
            if i%200 == 0 or i == 1:
                self.loss_function()
                print(f'Epoch {i}/{epochs} ->mse: {self.mse}')
                if show_learning == True:
                    self.show_progress(epoch = i)

X = np.linspace(0,20,400)

Y = np.sin(X) + 0.2   + 0.1*X + np.cos(0.3*X) + 2*np.cos(0.1*X)

spline = BSpline(degree=3, grid=12, input=X, target=Y)
spline.train(epochs=1000, learning_rate=1, show_learning=False)
Y_hat = spline.output()
spline.plot(X=X,Y=Y,Y_hat=Y_hat)
spline.plot_spline_components()