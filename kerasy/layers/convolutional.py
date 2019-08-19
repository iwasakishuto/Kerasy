# coding: utf-8
import numpy as np

class Conv2D():
    def __init__(self, filters, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu',
                 kernel_initializer='random_normal', bias_initializer='zeros', data_format='channels_last'):
        """
        @param filters           : (int) the dimensionality of the output space.
        @param kernel_size       : (int,int) 
        @param strides           : (int,int) 
        @param padding           : (str) "valid" or "same".
        @param activation        : (str) Activation function to use.
        @param kernel_initializer: (str) Initializer for the `kernel` weights matrix.
        @param bias_initializer  : (str) Initializer for the bias vector.
        """
        self.OF = filters # Output filters.
        self.kh, self.kw = kernel_size # kernel size.
        self.sh, self.sw = strides 
        self.padding = padding
        self.h = ActivationFunc(activation)
        self.kernel_initializer = Initializer(kernel_initializer)
        self.bias_initializer   = Initializer(bias_initializer)
        
    def build(self, input_shape):
        """ @params input_shape: (H,W,F) of input image. """
        self.H, self.W, self.F = input_filters
        try: 
            if padding=="same":
                self.OH=self.H; self.OW=self.W
                self.ph = ((self.sh-1)*self.H+self.kh-self.sh)//2
                self.pw = ((self.sw-1)*self.W+self.kw-self.sw)//2
            elif padding=="valid":
                self.OH=self.H-self.kh; self.OW=self.W-self.kw
                while self.OH%self.sh != 0:
                    self.OH-=1
                while self.OW%self.sw != 0:
                    self.OW-=1
                self.ph, self.pw = (0,0)
        except:
            print("Can't understand 'padding=f{self.padding}'")     

        self.kernel = self.kernel_initializer(shape=(self.kh, self.kw, self.F, self.OF)),
        self.bias   = self.bias_initializer(shape=(self.OF,))
        
    def _generator(self, image):
        """ @param image: (ndarray) must be a 3-D array. (Height, Width, Channel) """
        #=== padding ===
        try:            
            if self.padding =="same":
                z_in = np.zeros(shape=(self.H+2*self.ph,self.W+2*self.pw))
                z_in[self.ph:self.H+self.ph,self.pw:self.W+self.pw,:] = image
            elif self.padding == "valid": 
                z_in = image[:self.OH,:self.OW,:]
        except:
            print("Can't understand 'padding=f{self.padding}'")
            
        for i in range(0,self.H-self.kh+1,self.sh):
            for j in range(0,self.W-self.kw+1,self.sw):
                clip_image = image[i:(i+self.kh), j:(j+self.kw), :]
                yield clip_image,i,j
    
    def forward(self, z_in):
        self.z = z_in # Memorize!!
        z_out = np.zeros(shape=(self.OH, self.OW, self.OF))
        for f in range(self.OF):
            for clip_image,i,j in self._generator(z_in):
                z_out[i][j][f] = np.sum(clip_image * self.kernel[:,:,:,f], axis=(1,2))

        return z_out + self.bias
    
    def backprop(self, dEdz_out, lr=1e-3):
        dEdz_in = np.zeros(shape=self.z.shape);
        
        for crip_image,i,j,c in self._generator(self.input):
            Filter_delta[f] += Pre_deltat[i,j,f] * crip_image
            
        # Update filters.
        self.filters -= lr * Filter_delta
        # Propagate the loss gradients.
#         Next_delta = np.zeros(self.input.shape)
        
#         for crip_image, i, j in self._generator(self.input):
#             h, w, _ = image.shape
#             ph,pw = self.pool_size
            
#         return Next_delta