#coding: utf-8
import numpy as np
from ._kernel import kernel_handler
from ..utils import flush_progress_bar

class BaseSVM():
    def __init__(self, kernel="gaussian", **kernelargs):
        self.kernel = kernel_handler(kernel, **kernelargs)
        self.isZero = None
        self.N = None; self.M = None;
        self.K = None # gram matrix: shape=(N,N)
        self.SVidx = None
        self.b = None # bias parameter
        # Memorize all data while training otherwise only support data.
        self.x_train = None; self.y_train = None
        self.a = None # Lagrange multiplier

    def calcuBias(self):
        return np.mean([self.y_train[n]-np.sum([self.a[m]*self.y_train[m]*self.K[n,m] for m in self.SVidx]) for n in self.SVidx])

    def y(self, x):
        return sum([self.a[i]*self.y_train[i]*self.kernel(self.x_train[i],x) for i in self.SVidx]) + self.b

    def predict(self, X):
        return np.array([1 if self.y(x)>0 else -1 for x in X]).astype(int)

    def accuracy(self, x_train, y_train):
        return np.mean(self.predict(x_train) == self.formatting_y(y_train, verbose=0))

    @staticmethod
    def formatting_y(y_train, verbose=1):
        y_train = np.copy(y_train)
        valid_train = np.array([-1,1])
        unique_cls = np.unique(y_train)
        if len(unique_cls) > 2:
            raise ValueError("If you want to classify more than 2 class, please use MultipleSVM` instead.")
        if not np.all(np.in1d(valid_train, unique_cls)):
            for i,t in enumerate(unique_cls):
                y_train[y_train==t] = valid_train[i]
                if verbose>0: print(f"Convert {t} to {valid_train[i]:>2} to suit for the SVM train data format.")
        return y_train

    def fit(self, x_train, y_train, max_iter=500, zero_eps=1e-2, sparse_memorize=True, verbose=1):
        """
        @param x_train : (ndarray) shape=(N,M)
        @param t_train : (ndarray) shape=(N,)
        @param kernel  : (function) k(x,x')
        @param max_iter: (int)
        @param zero_eps: (float) if |x|<zero_eps, x is considered equal to 0.
        """
        y_train = self.formatting_y(y_train)
        self.isZero = lambda x:abs(x)<zero_eps
        self.N, self.M = x_train.shape
        self.K = np.array([[self.kernel(x_train[i], x_train[j]) for j in range(self.N)] for i in range(self.N)])
        self.x_train = x_train; self.y_train = y_train

        # Initialization.
        self.a = np.random.uniform(low=0, high=1, size=self.N)
        self.SVidx = self.isSV()
        self.b = self.calcuBias()

        # Optimization.
        self.SMO(max_iter=max_iter, verbose=verbose)

        # Memorize only support vector data.
        if sparse_memorize:
            self.sparseMemorize()

        return self

    def SMO(self, max_iter, verbose=1):
        N = self.N
        idxes = np.arange(N)
        for it in range(max_iter):
            changed = False
            np.random.shuffle(idxes)
            for i in idxes:
                # 1. Find an a_i that violates KKT condition.
                if self.checkKKT(i): continue
                # 2. Select an a_j that is expected to maximize the step size.
                j = self.choose_second(i)
                # Update a_i and a_j & memorize whether updated or not.
                changed = self.update(i, j) or changed
                # Update the bias parameter.
                self.b = self.calcuBias()
                self.SVidx = self.isSV()
            if not changed: break
            flush_progress_bar(it, max_iter, metrics={"Rate of Support Vector": 100*len(self.SVidx)/N}, verbose=verbose)
        if verbose: print()

    def sparseMemorize(self):
        self.N = np.count_nonzero(self.SVidx)
        self.x_train = self.x_train[self.SVidx]
        self.y_train = self.y_train[self.SVidx]
        self.a = self.a[self.SVidx]
        self.SVidx = np.arange(self.N)

class SVC(BaseSVM):
    def __init__(self, kernel="gaussian", C=10, **kernelargs):
        super().__init__(kernel=kernel, kernelargs=kernelargs)
        self.C = C

    def isSV(self):
        """whether x_train[i] is a support vector or not."""
        return np.arange(self.N)[np.logical_not(np.logical_or(self.isZero(self.a), self.isZero(self.C-self.a)))]

    def checkKKT(self, i):
        """whether x_train[i] satisfy the KKT condition or not."""
        Cs = self.y_train[i]*self.y(self.x_train[i]) # Complementary slackness
        if self.isZero(self.C-self.a[i]):
            return Cs>=1 and self.isZero(self.a[i]*(Cs-1))
        else:
            return Cs<=1

    def choose_second(self, i):
        yi = self.y(self.x_train[i])
        expected_step_size_rank = [abs(self.y(self.x_train[j]) - yi) if not self.isZero(self.a[j]) else 0 for j in range(self.N)]
        return np.argmax(expected_step_size_rank)

    def update(self, i, j):
        if i == j: return False
        ti = self.y_train[i]; yi = self.y(self.x_train[i])
        tj = self.y_train[j]; yj = self.y(self.x_train[j])
        c = ti*self.a[i] + tj*self.a[j]
        # clipping
        if ti == tj:
            l = max(0, c/ti-self.C); h = max(self.C, c/ti)
        else:
            l = max(0, c/ti); h = min(self.C, self.C+c/ti)

        denominator = self.K[i,i]-2*self.K[i,j]+self.K[j,j]
        numerator = 1-ti*tj+ti*(yj-yi)

        #=== Dealing with Overflow. ===
        if denominator==0:
            ai_next = h if np.sign(numerator)==1 else l
        else:
            delta_i  = numerator/denominator
            if np.inf==abs(delta_i):
                ai_next = h if np.sign(delta_i)==1 else l
            else:
                ai_next = self.a[i] + delta_i
                ai_next = l if ai_next<l else h if ai_next>h else ai_next

        if self.isZero(ai_next - self.a[i]): return False

        # Update
        self.a[i] = ai_next
        self.a[j] = (c-ti*self.a[i])/tj
        return True

class hardSVC(BaseSVM):
    def __init__(self, kernel="gaussian", **kernelargs):
        super().__init__(kernel=kernel, kernelargs=kernelargs)

    def isSV(self):
        """whether x_train[i] is a support vector or not."""
        return np.arange(self.N)[np.logical_not(self.isZero(self.a))]

    def checkKKT(self, i):
        Cs = self.y_train[i]*self.y(self.x_train[i]) # Complementary slackness
        return self.isZero(Cs-1) and self.isZero(self.a[i]*(Cs-1))

    def choose_second(self, i):
        yi = self.y(self.x_train[i])
        expected_step_size_rank = [abs(self.y(self.x_train[j]) - yi) if not self.isZero(self.a[j]) else 0 for j in range(self.N)]
        return np.argmax(expected_step_size_rank)

    def update(self, i, j):
        if i == j: return False
        ti = self.y_train[i]; yi = self.y(self.x_train[i])
        tj = self.y_train[j]; yj = self.y(self.x_train[j])
        c = ti*self.a[i] + tj*self.a[j]
        # clipping
        if ti == tj:
            l = 0; h = c/ti
        else:
            l = max(0, c/ti); h = 10*c/ti

        denominator = self.K[i,i]-2*self.K[i,j]+self.K[j,j]
        numerator = 1-ti*tj+ti*(yj-yi)

        #=== Dealing with Overflow. ===
        if denominator==0:
            ai_next = h if np.sign(numerator)==1 else l
        else:
            delta_i  = numerator/denominator
            if np.inf==abs(delta_i):
                ai_next = h if np.sign(delta_i)==1 else l
            else:
                ai_next = self.a[i] + delta_i
                ai_next = l if ai_next<l else h if ai_next>h else ai_next

        if self.isZero(ai_next - self.a[i]): return False

        # Update
        self.a[i] = ai_next
        self.a[j] = (c-ti*self.a[i])/tj
        return True

class MultipleSVM():
    def __init__(self, kernel="gaussian", C=10, **kernelargs):
        self.weekSVMs = []
        self.kernel = kernel
        self.C = C
        self.kernelargs = kernelargs

    def fit(self, x_train, y_train, max_iter=500, zero_eps=1e-2, sparse_memorize=True):
        """
        @param x_train : (ndarray) shape=(N,M)
        @param t_train : (ndarray) shape=(N,)
        @param max_iter: (int)
        @param zero_eps: (float) if |x|<zero_eps, x is considered equal to 0.
        """
        self.weekSVMs = []
        for cls in np.unique(y_train):
            print(f"[{cls} vs others]")
            y_train_for_week = np.ones_like(y_train, dtype=int)
            y_train_for_week[np.where(y_train != cls)] = -1
            weekSVM = SVC(kernel=self.kernel, **self.kernelargs)
            weekSVM.fit(x_train, y_train_for_week, max_iter=max_iter, zero_eps=zero_eps, sparse_memorize=sparse_memorize)
            self.weekSVMs.append(weekSVM)

    def predict(self, X):
        return np.asarray([np.argmax([weekSVM.y(x) for weekSVM in self.weekSVMs]) for x in X], dtype=int)

    def accuracy(self, x_train, y_train):
        return np.mean(self.predict(x_train) == y_train)

# class SVR(BaseSVM):
#     def __init__(self, kernel="gaussian", **kernelargs):
#         super().__init__(kernel=kernel, kernelargs=kernelargs)
#
#     def calcuBias(self):
#         return np.mean([self.y_train[n]-np.sum([self.a[m]*self.y_train[m]*self.K[n,m] for m in self.SVidx]) for n in self.SVidx])
#
#     def y(self, x):
#         return sum([self.a[i]*self.y_train[i]*self.kernel(self.x_train[i],x) for i in self.SVidx]) + self.b
#
#     def predict(self, X):
#         return np.array([1 if self.y(x)>0 else -1 for x in X]).astype(int)


class RVM():
    """Relevance Vector Machine
    """
    def __init__(self, initial_alpha, initial_beta):
        self.initial_alpha = initial_alpha
        self.initial_beta  = initial_beta

    def fit(self, x_train, y_train, max_iter=500, zero_eps=1e-2, sparse_memorize=True, add_bias=False):
        """
        @param x_train : (ndarray) shape=(N,M)
        @param t_train : (ndarray) shape=(N,)
        @param max_iter: (int)
        @param zero_eps: (float) if |x|<zero_eps, x is considered equal to 0.
        """
        N,M = x_train.shape
        if add_bias:
            x_train = np.c_[np.copy(x_train), np.ones(shape=N)].shape
            M+=1
        self.phi = x_train
        self.alpha = np.full(shape=M, fill_value=self.initial_alpha)
        self.beta
