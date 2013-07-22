from __future__ import division
import numpy as np
na = np.newaxis
from matplotlib import pyplot as plt
plt.interactive(True)

import predictive_models as pm
import predictive_distributions as pd
import particle_filter as pf

# TODO TODO this file is allllll broken

COLORS = ['r','g','c','m','k']

##############
#  examples  #
##############

def dumb_randomwalk_fixednoise():
    nlags = 2
    noisechol = 20*np.eye(2)
    initial_particles = [
            pf.AR(
                    num_ar_lags=nlags,
                    previous_outputs=[np.zeros(2)]*nlags,
                    baseclass=lambda: \
                        pm.RandomWalk(noiseclass=lambda: pd.FixedNoise(noisechol=noisechol))
                    ) for itr in range(10000)]

    def plotfunc(particles,weights):
        plottopk(particles,weights,5)
        plotmeanpath(particles,weights)

    return interactive(initial_particles,2500,plotfunc)

def smart_hdphsmm_mniwar_adaptive():
    nlags = 2
    MNIWARparams = (
                3,
                10*np.eye(2),
                np.zeros((2,2*nlags+1)),
                np.diag((10,)*(2*nlags) + (0.1,))
                )

    initial_particles = [
            pf.AR(
                num_ar_lags=nlags,
                previous_outputs=[np.zeros(2),np.zeros(2)],
                baseclass=lambda: \
                        pm.HDPHMMSampler(
                            alpha=3.,gamma=4.,
                            obs_sampler_factory=lambda: pd.MNIWAR(*MNIWARparams),
                            )
                ) for itr in range(2500)]

    def plotfunc(particles,weights):
        for p in topk(particles,weights,5):
            t = np.array(p.track)
            plt.plot(t[:,0],t[:,1],'r-')
            stateseq = np.array(p.sampler.stateseq)
            for i in range(len(set(stateseq))):
                plt.plot(t[stateseq == i,0],t[stateseq == i,1],COLORS[i % len(COLORS)] + 'o')

    return interactive(initial_particles,500,plotfunc)

##############
#  back-end  #
##############

def interactive(initial_particles,cutoff,plotfunc):
    sigma = 25.
    def loglikelihood(_,locs,data):
        return -np.sum((locs - data)**2,axis=1)/(2*sigma**2)

    plt.clf()

    points = [np.zeros(2)]

    particlefilter = pf.ParticleFilter(2,cutoff,loglikelihood,initial_particles)

    plt.ioff()

    pts = np.array(points)
    plt.plot(pts[:,0],pts[:,1],'bo-')
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.draw()
    plt.ion()

    while True:
        out = plt.ginput()
        if len(out) == 0:
            break
        else:
            out = np.array(out[0])
            points.append(out)

            plt.ioff()

            plt.clf()

            particlefilter.step(out,resample_method='lowvariance')
            particlefilter.change_numparticles(5000) # TESTING

            plotfunc(particlefilter.particles,particlefilter.weights_norm)

            pts = np.array(points)
            plt.plot(pts[:,0],pts[:,1],'bo--')

            plt.xlim(-100,100)
            plt.ylim(-100,100)
            plt.draw()
            plt.ion()

    return particlefilter

###########
#  utils  #
###########

def topk(items,scores,k):
    return [items[idx] for idx in np.argsort(scores)[:-(k+1):-1]]

def plottopk(particles,weights,k):
    for p in topk(particles,weights,k):
        t = np.array(p.track)
        plt.plot(t[:,0],t[:,1],'rx-',alpha=0.3)
        print p

def plotmeanpath(particles,weights):
    track = np.array(particles[0].track)*weights[0,na]
    for p,w in zip(particles[1:],weights[1:]):
        track += np.array(p.track) * w
    plt.plot(track[:,0],track[:,1],'k^:')

##########
#  main  #
##########

if __name__ == '__main__':
    dumb_randomwalk_fixednoise()

