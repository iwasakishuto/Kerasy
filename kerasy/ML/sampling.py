#coding: utf-8
import time
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

from ..utils import flush_progress_bar
from ..utils import measure_complexity

class BaseSampler():
    def __init__(self, p, q=None, domain=None, qargs={}):
        self.p = p
        self.domain = domain
        if q is not None:
            self._proposal_is_valid(q)
            self.qloc   = qargs.pop("loc", 0)
            self.qscale = qargs.pop("scale", 1)
            self.qargs = qargs
            self.q = lambda x: stats.__dict__.get(q).pdf((x-self.qloc)/self.qscale, **self.qargs)/self.qscale
            self.q_sampler = np.random.__dict__.get(q)

    def _proposal_is_valid(self, q):
        if isinstance(q, str) and (np.random.__dict__.get(q) is None or stats.__dict__.get(q) is None):
            raise ValueError(f"np.random.{q} and stats.{q} should be defined.")

    def plot(self, *args, **kwargs):
        raise NotImplementedError()

    def hist(self, *args, **kwargs):
        raise NotImplementedError()

    def sampler(self, *args, **kwargs):
        raise NotImplementedError()
        """
        self.n_gen = self.n_reject = 0
        np.random.seed(random_state)
        x = "initialization"
        while True:
            x = func(x)
            if cond1:
                yield x
            else:
                self.n_reject += 1
            self.n_gen += 1
        """

    def sample(self, n, initial_x=None, burnin=0, random_state=None, verbose=1):
        n = int(n)
        buff = np.empty(shape=(n,self.M))
        max_iter=n+burnin
        for i,x in enumerate(self.sampler(initial_x=initial_x, random_state=random_state)):
            if i==max_iter: break
            flush_progress_bar(i, max_iter, barname=self.__class__.__name__, verbose=verbose)
            if i<burnin: continue
            buff[i-burnin] = x
        return buff

class RejectionSampler(BaseSampler):
    def __init__(self, p, q, domain, k=1, **qargs):
        """
        @params p: (func) The objective distribution (wish to sample from this).
                          It is impractical to sample directly from p(z)
                          but that we can evaluate p(z) easily for any given value of z.
        @params q: (str) Proposal distribution q. It is easy to draw samples from q(z).
        @params domain: (ndarray) shape=(M,N)
            - M is the dimension of the domain.
            - N is the number of points representing the domain.
        """
        super().__init__(p=p, q=q, domain=domain, qargs=qargs)
        self.M = 1 if domain.ndim==1 else domain.shape[0]
        self.k = k
        # validation check.
        pz = p(domain); qz=k*self.q(domain)
        if np.any(pz>qz):
            self.plot(domain=domain)
            plt.show()
            raise ValueError("`q`(z) >= p(z) must hold in all domains.")

    def plot(self, domain=None, ax=None):
        if ax is None: fig, ax = plt.subplots()
        domain = self.domain if domain is None else domain
        ax.plot(domain, self.p(domain), label="p: Objective distribution", color="blue")
        ax.plot(domain, self.k*self.q(domain), label="q: Proposal distribution",  color="red")
        ax.set_title("The relation between\n'Objective Distribution' and 'Proposal Distribution'"), ax.legend()
        return ax

    def hist(self, n_samples, burnin=0, domain=None, ax=None, bins=50, random_state=None, verbose=1):
        if ax is None: fig, ax = plt.subplots()
        domain = self.domain if domain is None else domain
        self.n_gen = self.n_reject = 0
        s = time.time()
        samples = self.sample(n_samples, burnin=burnin, random_state=random_state, verbose=verbose)
        p_time = time.time()-s
        ax.plot(domain, self.p(domain), label="p: Objective distribution", color="blue")
        ax.hist(samples, density=True, bins=bins, label="Sampling Result", color="blue", alpha=0.5)
        ax.set_title(f"Sampling Histogram\nRejection rate: {100*self.n_reject/self.n_gen:.3f}%, Processing time: {p_time:.3f}[s]")
        return ax

    def sampler(self, initial_x=None, random_state=None):
        self.n_gen = self.n_reject = 0
        np.random.seed(random_state)
        while True:
            x = self.qscale*self.q_sampler(**self.qargs, size=(self.M))+self.qloc
            prob = self.p(x)/(self.k*self.q(x))
            if np.random.uniform() < prob:
                yield x
            else:
                self.n_reject += 1
            self.n_gen += 1

class MHSampler(BaseSampler):
    """ Metropolis-Hastings algorithm. """
    def __init__(self, p, q, **qargs):
        super().__init__(p=p, q=q, qargs=qargs)
        self.M = len(self.q_sampler(**qargs))

    def sampler(self, initial_x=None, random_state=None):
        self.n_gen = self.n_reject = 0
        np.random.seed(random_state)
        x = self.qscale*self.q_sampler(**self.qargs)+self.qloc if initial_x is None else initial_x
        while True:
            new_x = x + self.qscale*self.q_sampler(**self.qargs)+self.qloc
            prob = min(self.p(new_x)*self.q(x-new_x) / (self.p(x)*self.q(new_x-x)), 1)
            if np.random.uniform() <= prob:
                x = new_x
                yield x
            else:
                self.n_reject += 1
            self.n_gen += 1

    def plot(self, n_samples, *domain, initial_x=None, burnin=0, ax=None, random_state=None, verbose=1):
        if ax is None: fig, ax = plt.subplots()
        s = time.time()
        samples = self.sample(n_samples+burnin, initial_x=initial_x, burnin=0, random_state=random_state, verbose=verbose)
        p_time = time.time()-s
        print("Plotting...")
        if self.M == 1:
            if len(domain)>0: ax.plot(domain, self.p(domain), label="p: Objective distribution", color="black")
            ax.plot(samples[:burnin], color="blue", label="samples")
            if burnin>0: ax.plot(samples[burnin:], color="red", label="burn-in")
            ax.set_title("Random Walk Results"), ax.legend()
        elif self.M == 2:
            if len(domain)>0:
                X,Y = domain
                if X.ndim==1: X,Y = np.meshgrid(X, Y)
                Z = np.vectorize(lambda x,y: self.p([x,y]))(X, Y)
                ax.pcolor(X, Y, Z, alpha=0.3)
            ax.plot(samples[burnin:, 0], samples[burnin:, 1], color="blue", label="samples")
            if burnin>0: ax.plot(samples[:burnin, 0], samples[:burnin, 1], color="red", label="burn-in")
            ax.scatter(samples[0, 0], samples[0,1], color="black", label="Initial", s=1000, marker="*")
            ax.set_title(f"Random Walk Results\nRejection rate: {100*self.n_reject/self.n_gen:.3f}%, Processing time: {p_time:.3f}[s]"), ax.legend()
        else:
            raise NotImplementedError()
        return ax

    def scatter(self, n_samples, *domain, initial_x=None, burnin=0, ax=None, random_state=None, verbose=1):
        if ax is None: fig, ax = plt.subplots()
        s = time.time()
        samples = self.sample(n_samples+burnin, initial_x=initial_x, burnin=0, random_state=random_state, verbose=verbose)
        p_time = time.time()-s
        print("Plotting...")
        if self.M == 2:
            if len(domain)>0:
                X,Y = domain
                if X.ndim==1: X,Y = np.meshgrid(X, Y)
                Z = np.vectorize(lambda x,y: self.p([x,y]))(X, Y)
                ax.pcolor(X, Y, Z, alpha=0.3)
            ax.scatter(samples[burnin:, 0], samples[burnin:, 1], color="blue", label="samples", s=1)
            if burnin>0: ax.scatter(samples[:burnin, 0], samples[:burnin, 1], color="red", label="burn-in", s=1)
            ax.scatter(samples[0, 0], samples[0,1], color="black", label="Initial", s=1000, marker="*")
            ax.set_title(f"Random Walk Results\nRejection rate: {100*self.n_reject/self.n_gen:.3f}%, Processing time: {p_time:.3f}[s]"), ax.legend()
        else:
            raise NotImplementedError()
        return ax

class GibbsMsphereSampler():
    def __init__(self, M, r=1):
        self.M = M
        self.r = r

    def sampler(self, initial_x=None, random_state=None):
        r2_lim = np.square(self.r)
        x = np.zeros(shape=self.M) if initial_x is None else initial_x
        np.random.seed(random_state)
        while True:
            r2 = np.dot(x,x)
            for m in range(self.M):
                r2-=np.square(x[m])
                x[m]=np.random.uniform(-np.sqrt(r2_lim-r2),np.sqrt(r2_lim-r2))
                r2+=np.square(x[m])
            yield x

    def sample(self, n, initial_x=None, burnin=0, random_state=None, verbose=1):
        n = int(n)
        buff = np.empty(shape=(n,self.M))
        max_iter=n+burnin
        for i,x in enumerate(self.sampler(initial_x=initial_x, random_state=random_state)):
            if i==max_iter: break
            flush_progress_bar(i, max_iter, barname=self.__class__.__name__, verbose=verbose)
            if i<burnin: continue
            buff[i-burnin,:] = x
        return buff

    @classmethod
    def plot2D(cls, n, r=1, initial_x=None, burnin=0, random_state=None, verbose=-1, ax=None):
        model = cls(M=2, r=r)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        s = time.time()
        X,Y = model.sample(n, initial_x=initial_x, burnin=burnin, random_state=random_state, verbose=verbose).T
        p_time = time.time()-s
        ax.scatter(X,Y,s=1000//n+1)

        x = np.linspace(-r, r, int(r*100))
        ax.scatter(X,Y,s=1000//n+1, color="blue", alpha=0.5)
        ax.plot(x,  np.sqrt(np.square(r)-np.square(x)), color="red")
        ax.plot(x, -np.sqrt(np.square(r)-np.square(x)), color="red")
        ax.set_title(f"Processing time: {p_time:.3f}[s]"); ax.grid()
        ax.set_xlim(-r, r); ax.set_ylim(-r,r)
        return ax

    @classmethod
    def plot3D(cls, n, r=1, initial_x=None, burnin=0, random_state=None, verbose=-1, ax=None):
        model = cls(M=3, r=r)
        if ax is None:
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, projection='3d')
        s = time.time()
        X,Y,Z = model.sample(n, initial_x=initial_x, burnin=burnin, random_state=random_state, verbose=verbose).T
        p_time = time.time()-s
        ax.scatter(X,Y,Z,s=1000//n+1, color="blue", alpha=0.5)
        ax.set_title(f"Processing time: {p_time:.3f}[s]"); ax.grid()
        ax.set_xlim(-r, r); ax.set_ylim(-r,r); ax.set_zlim(-r,r)
        return ax

    @classmethod
    def plot_p_times(cls, n, D, r=1, repetitions=5):
        for d in range(D):
            model = cls(d+1)
            flush_progress_bar(d, D, metrics={"processing_time": measure_complexity(model.sample, n, verbose=-1, repetitions_=repetitions)}, verbose=2)
