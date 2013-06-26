from __future__ import division
import numpy as np
na = np.newaxis
import abc

from util.stats import sample_mniw, sample_invwishart

'''predictive samplers for basic distributions'''

class PredictiveDistribution(object):
    __metaclass__ = abc.ABCMeta

    def sample_next(self,*args,**kwargs):
        val = self._sample(*args,**kwargs)
        self._update_hypparams(val)
        return val

    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def _update_hypparams(self,x):
        pass

    @abc.abstractmethod
    def _sample(self):
        pass

###############
#  Durations  #
###############

class Poisson(PredictiveDistribution):
    def __init__(self,alpha_0,beta_0):
        self.alpha_n = alpha_0
        self.beta_n = beta_0

    def _update_hypparams(self,k):
        self.alpha_n += k
        self.beta_n += 1

    def _sample(self):
        return np.random.poisson(np.random.gamma(self.alpha_n,1./self.beta_n))+1

    def copy(self):
        return Poisson(self.alpha_n,self.beta_n)


class NegativeBinomial(PredictiveDistribution): # TODO
    pass

##################
#  Observations  #
##################

class FixedNoiseDiagonal(PredictiveDistribution):
    def __init__(self,variances):
        self.scales = np.sqrt(variances)

    def _update_hypparams(self,y):
        pass

    def _sample(self):
        return self.scales*np.random.randn(self.scales.shape[0])

    def copy(self):
        return self

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__,self.weights)

    def __repr__(self):
        return str(self)


class FixedNoise(PredictiveDistribution):
    def __init__(self,noisechol):
        self.noisechol = noisechol

    def _update_hypparams(self,y):
        pass

    def _sample(self):
        return self.noisechol.dot(np.random.randn(self.noisechol.shape[0]))

    def copy(self):
        return self

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__,self.noisechol)

    def __repr__(self):
        return str(self)


class InverseWishartNoise(PredictiveDistribution):
    def __init__(self,n_0,S_0):
        self.S_0 = S_0
        self.n_n = n_0

        self.yyt = np.zeros(S_0.shape)

    def _update_hypparams(self,y):
        self.n_n += 1
        self.yyt += y[:,na] * y

    def _sample(self):
        Sigma = sample_invwishart(self.S_0 + self.yyt,self.n_n)
        return np.linalg.cholesky(Sigma).dot(np.random.randn(Sigma.shape[0]))

    def copy(self):
        new = self.__new__(self.__class__)
        new.n_n = self.n_n
        new.S_0 = self.S_0
        new.yyt = self.yyt.copy()
        return new

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__,(self.n_n,self.S_0 + self.yyt))

    def __repr__(self):
        return str(self)


class MNIWAR(PredictiveDistribution):
    '''Conjugate Matrix-Normal-Inverse-Wishart prior'''
    def __init__(self,n_0,sigma_0,M,K):
        # hyperparameters
        self.n = n_0
        self.sigma_0 = sigma_0
        self.M_n = M
        self.K_n = K
        self.sigma_n = sigma_0.copy()

        # statistics
        self.Sytyt = K.copy()
        self.Syyt = M.dot(K)
        self.Syy = M.dot(K).dot(M.T)

        # temporary variables to cut down on mallocs
        self.Sy_yt = np.empty(self.sigma_0.shape)
        self._ylags = np.zeros(self.M_n.shape[1])

        # error handling
        self._broken = False

    def _update_hypparams(self,y):
        ylags = self._ylags # gets info passed from previous _sample call, state!

        self.Syy += y[:,na] * y
        self.Sytyt += ylags[:,na] * ylags
        self.Syyt += y[:,na] * ylags

        M_n = np.linalg.solve(self.Sytyt,self.Syyt.T).T
        np.dot(-M_n,self.Syyt.T,out=self.Sy_yt)
        self.Sy_yt += self.Syy

        self.n += 1
        np.add(self.Sy_yt,self.sigma_0,out=self.sigma_n)
        self.M_n = M_n
        self.K_n = self.Sytyt

        try:
            pass
            # assert np.allclose(self.sigma_n,self.sigma_n.T) and (np.linalg.eigvals(self.sigma_n) > 0).all()
            # assert np.allclose(self.K_n,self.K_n.T) and (np.linalg.eigvals(self.K_n) > 0).all()
        except AssertionError:
            print 'WARNING: particle exploded'
            self._broken = True

    def _sample(self,lagged_outputs):
        if not self._broken:
            try:
                ylags = self._pad_ylags(lagged_outputs)
                A,sigma = sample_mniw(self.n,self.sigma_n,self.M_n,np.linalg.inv(self.K_n))
                return A.dot(ylags) + np.linalg.cholesky(sigma).dot(np.random.randn(sigma.shape[0]))
            except np.linalg.LinAlgError:
                print 'WARNING: particle broke'
                self._broken = True
        return -99999*np.ones(self.M_n.shape[0])


    def _pad_ylags(self,lagged_outputs):
        ylags = self._ylags
        ylags[...] = 0

        # plug in lagged data
        temp = np.array(lagged_outputs)
        temp.shape = (-1,)
        ylags[:temp.shape[0]] = temp

        # plug in affine drift
        ylags[-1] = 1

        return ylags

    def copy(self):
        new = self.__new__(self.__class__)
        new.n = self.n
        new.sigma_0 = self.sigma_0
        new.sigma_n = self.sigma_n.copy()
        new.M_n = self.M_n
        new.K_n = self.K_n
        new.Sytyt = self.Sytyt.copy()
        new.Syyt = self.Syyt.copy()
        new.Syy = self.Syy.copy()
        new.Sy_yt = self.Sy_yt.copy()
        new._ylags = self._ylags.copy()
        new._broken = self._broken
        return new

    def __str__(self):
        return '\n'.join(map(str,sample_mniw(self.n,self.sigma_n,self.M_n,np.linalg.inv(self.K_n))))

    def __repr__(self):
        return str(self)

