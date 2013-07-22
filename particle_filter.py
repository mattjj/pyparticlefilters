from __future__ import division
import numpy as np
na = np.newaxis
from collections import deque
import abc, warnings

from util.general import ibincount

DEBUG = False

# this is a great reference on techniques:
# http://www.cs.berkeley.edu/~pabbeel/cs287-fa11/slides/particle-filters++_v2.pdf

class ParticleFilter(object):
    def __init__(self,ndim,cutoff,log_likelihood_fn,initial_particles):
        assert len(initial_particles) > 0

        self.particles = initial_particles
        self.log_likelihood_fn = log_likelihood_fn
        self.cutoff = cutoff

        self.numsteps = 0
        self.log_weights = np.zeros(len(initial_particles))
        self.weights_norm = np.ones(len(initial_particles))

        self._Nsurvive_history = []
        self._Neff_history = []

        self._locs = np.empty((len(initial_particles),ndim))

    def step(self,data,resample_method='lowvariance',particle_kwargs={}):
        for idx, particle in enumerate(self.particles):
            self._locs[idx] = particle.sample_next(**particle_kwargs)
        self.log_weights += self.log_likelihood_fn(self.numsteps,data,self._locs)

        if self._Neff < self.cutoff:
            self._resample(resample_method)
            resampled = True
        else:
            resampled = False

        self.numsteps += 1

        return resampled

    def change_numparticles(self,newnum,resample_method='lowvariance'):
        if newnum != len(self.particles):
            self._resample(resample_method,num=newnum)

    def inject_particles(self,particles_to_inject,particle_kwargs={}):
        warnings.warn('untested, probably broken')
        # breaks posterior estimation, but good for tracking if the proposal
        # model doesn't have much meaning!

        # attaches to random histories
        # need to weight likelihood

        self.particles_were_injected = True

        if self.numsteps > 0:
            new_weights_norm = np.empty(len(particles_to_inject))
            new_log_weights = np.empty(len(particles_to_inject))
            copy_sources = self._lowvariance_sources(len(particles_to_inject))
            for i,(p,copy_index) in enumerate(zip(particles_to_inject,copy_sources)):
                p.track = self.particles[copy_index].track[:-1]

                new_weights_norm[i] = self.weights_norm[i]/2.
                self.weights_norm[i] /= 2

                new_log_weights[i] = self.log_weights[i] - np.log(2)
                self.log_weights[i] -= np.log(2)

        self._locs = np.concatenate((self._locs,[p.sample_next(**particle_kwargs) for p in particles_to_inject]))
        self.particles += particles_to_inject
        # TODO TODO weight likelihoods! these could be shitty darts and this
        # sample_next must be weighted

        self.weights_norm = np.concatenate((self.weights_norm,new_weights_norm))
        self.log_weights = np.concatenate((self.log_weights,new_log_weights))

    @property
    def _Neff(self):
        self.weights_norm = np.exp(self.log_weights - np.logaddexp.reduce(self.log_weights))
        self.weights_norm /= self.weights_norm.sum()
        Neff = 1./np.sum(self.weights_norm**2)

        self._Neff_history.append((self.numsteps,Neff))

        if DEBUG:
            print Neff

        return Neff

    def _resample(self,method,num=None):
        num = (num if num is not None else len(self.particles))

        assert method in ['lowvariance','independent']
        if method is 'lowvariance':
            sources = self._lowvariance_sources(num)
        if method is 'independent':
            sources = self._independent_sources(num)

        self.particles = [self.particles[i].copy() for i in sources]

        self.log_weights = np.repeat(np.logaddexp.reduce(self.log_weights) - np.log(num),num)
        self.weights_norm = np.repeat(1./num, num)
        if len(self._locs) != len(self.particles):
            self._locs = np.empty((len(self.particles),self._locs.shape[1]))

        self._Nsurvive_history.append((self.numsteps,len(np.unique(sources))))

        if DEBUG:
            print self._Nsurvive_history[-1][1]

    def _independent_sources(self,num):
        return ibincount(np.random.multinomial(num,self.weights_norm))

    def _lowvariance_sources(self,num):
        r = np.random.rand()/num
        bins = np.concatenate(((0,),np.cumsum(self.weights_norm)))
        return ibincount(np.histogram(r+np.linspace(0,1,num,endpoint=False),bins)[0])

    def __getstate__(self):
        result = self.__dict__.copy()
        del result['log_likelihood_fn']
        return result


######################
#  Particle objects  #
######################

# particle objects should look like this to interface with the ParticleFilter

class Particle(object):
    __metaclass__ = abc.ABCMeta
    __slots__ = ('track',)

    def __init__(self,maxtracklen=None):
        self.track = deque(maxlen=maxtracklen) # faster for pushing one at a time

    @abc.abstractmethod
    def sample_next(self,*args,**kwargs):
        pass # must return something that can be put into a row of the _locs ndarray

    @abc.abstractmethod
    def copy(self):
        new = self.__new__(self.__class__)
        new.track = self.track.__copy__() # shallow copy of the deque
        return new

# the next two classes reflect a standard particle organization

class ForwardSampler(object):
    __metaclass__ = abc.ABCMeta
    __slots__ = tuple()

    @abc.abstractmethod
    def initial_pos(self):
        pass

    @abc.abstractmethod
    def sample_next(self,track,*args,**kwargs):
        pass

    def copy(self):
        return self # optional, for copying any internal state

class BasicParticle(Particle):
    __slots__ = ('forward_sampler',)

    def __init__(self,forward_sampler,maxtracklen=None):
        super(BasicParticle,self).__init__(maxtracklen)
        self.forward_sampler = forward_sampler
        self.track.append(forward_sampler.initial_pos())

    def sample_next(self,**kwargs):
        state, emission = self.forward_sampler.sample_next(track=self.track,**kwargs)
        self.track.append((state,emission))
        return emission

    def copy(self):
        new = super(BasicParticle,self).copy()
        new.forward_sampler = self.forward_sampler.copy()
        return new

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__,self.foward_sampler)

###############
#  Utilities  #
###############

def topktracks(pf,k):
    indices = np.argsort(pf.weights_norm)[:-(k+1):-1]
    return [pf.particles[i].track for i in indices], pf.weights_norm[indices]

def meantrack(pf):
    track = np.array(pf.particles[0].track)*pf.weights_norm[0,na]
    for p,w in zip(pf.particles[1:],pf.weights_norm[1:]):
        track += np.array(p.track) * w
    return track

