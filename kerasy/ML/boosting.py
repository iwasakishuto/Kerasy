#coding: utf-8
import numpy as np

class BaseBoosting():
    def __init__(self, Models, Masks=None):
        """
        @params Models: (list,tuple) Each model
            - Input shape  = (None, ?)
            - Output shape = (None, M)
        @params Masks: (list,tuple): i-th model's input shape=(D[Masks[i]])
        """
        self.num_Models = len(Models)
        self.Models = Models
        self.Masks = Masks
        self.alpha = np.zeros(shape=len(Models))

    def fit(self, train_x, train_y, T):
        """
        @param train_x: shape=(N,D)
        @param train_y: shape=(N,M)
        @param T      : (int) Iteration Counts.
        """
        N,*D = train_x.shape
        self.output_shape = train_y.shape[1:]

        self.alpha = np.zeros(shape=(self.num_Models,)) # shape=(num_Models,)
        Masks = np.ones(shape=(self.num_Models, *D), dtype=bool) if self.Masks is None else self.Masks

        H = np.asarray(
            [model.predict(train_x[:,mask].reshape(N,-1)) for mask,model in zip(Masks,self.Models)], dtype=float
        ).reshape(self.num_Models,-1) # H.shape=(num_models, N×prod(M))
        H_abs = np.sqrt(np.sum(np.square(H), axis=1)) # Habs.shape=(num_models, )

        train_y = train_y.reshape(-1) # shape=(N×prod(M))
        for t in range(T):
            y_loss = train_y - self.alpha.dot(H) # (N×prod(M))-(num_Models,)@(num_models,N×prod(M))=(N×prod(M))
            m_idx  = np.argmax(np.sum(y_loss*H, axis=1) / H_abs)
            alpha  = (1/H_abs[m_idx]**2) * np.sum(H[m_idx, :]*y_loss)
            self.alpha[m_idx] += alpha
        self.Masks = Masks

    def predict(self, X):
        """
        @param X: shape=(N', D)
        """
        N_,*D = X.shape
        predictions = self.alpha.dot(np.asarray([
            model.predict(X[:,mask].reshape(N_,-1)) for mask,model in zip(Masks,self.Models)]
        ).reshape(self.num_Models,-1)).reshape(N_,*self.output_shape)
        return predictions

class L2Boosting(BaseBoosting):
    def __init__(self, Models, Masks):
        super().__init__(Models, Masks)
