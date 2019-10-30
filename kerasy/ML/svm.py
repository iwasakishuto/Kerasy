#coding: utf-8
import numpy as np
from ._kernel import kernel_handler

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

    def fit(self, x_train, y_train, max_iter=1000, zero_eps=1e-2, sparse_memorize=True):
        """
        @param x_train : (ndarray) shape=(N,M)
        @param t_train : (ndarray) shape=(N,)
        @param kernel  : (function) k(x,x')
        @param max_iter: (int)
        @param zero_eps: (float) if |x|<zero_eps, x is considered equal to 0.
        """
        self.isZero = lambda x:abs(x)<zero_eps
        self.N, self.M = x_train.shape
        self.K = np.array([[self.kernel(x_train[i], x_train[j]) for j in range(self.N)] for i in range(self.N)])
        self.x_train = x_train; self.y_train = y_train

        # Initialization.
        self.a = np.random.uniform(low=0, high=1, size=self.N)
        self.SVidx = self.isSV()
        self.b = self.calcuBias()

        # Optimization.
        self.SMO(max_iter=max_iter)

        # Memorize only support vector data.
        if sparse_memorize:
            self.sparseMemorize()

        return self

    def SMO(self, max_iter):
        idxes = np.arange(self.N)
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
        delta_i  = (1-ti*tj+ti*(yj-yi))/(self.K[i,i]-2*self.K[i,j]+self.K[j,j])
        ai_next = self.a[i] + delta_i
        c = ti*self.a[i] + tj*self.a[j]

        # clipping
        if ti == tj:
            l = max(0, c/ti-self.C); h = max(self.C, c/ti)
        else:
            l = max(0, c/ti); h = min(self.C, self.C+c/ti)
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
        delta_i  = (1-ti*tj+ti*(yj-yi))/(self.K[i,i]-2*self.K[i,j]+self.K[j,j])
        ai_next = self.a[i] + delta_i
        c = ti*self.a[i] + tj*self.a[j]

        # clipping
        if ti == tj: ai_next = 0 if ai_next<0 else c/ti if ai_next>c/ti else ai_next
        else: ai_next = max(c/ti, 0) if ai_next<max(c/ti, 0) else ai_next

        if self.isZero(ai_next - self.a[i]): return False

        # Update
        self.a[i] = ai_next
        self.a[j] = (c-ti*self.a[i])/tj
        return True

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
