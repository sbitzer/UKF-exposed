# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:59:03 2015

@author: bitzer
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import UKF_aux as aux


class BAttM(object):
    """Basic implementation of the Bayesian Attractor Model."""

    def __init__(self):
        """Initialise parameters (a 2D case)."""

        # dimensionality of state space
        self._nd = 2
        # dimensionality of observations
        self.nD = 2

        # dynamics: parameters
        self.inhib = 1.7;
        self.hopk = 100.0;
        self.hopg = 10.0;
        self.leak = self.inhib / (2 * self.hopg);
        self.hopslope = 1;
        self.hopL = ( -self.inhib * np.ones((self.nd, self.nd)) +
                      self.inhib * np.eye(self.nd) );

        # observations: parameters
        self.oslope = .7
        self.oshift = self.hopg / 2
        self.set_standard_oM()

        # sampled noise added in functions (standard deviations)
        self.q = 0.0 # dynamics uncertainty
        self.r = 0.0 # observation uncertainty

        # uncertainties during inference
        self.Q = np.eye(self.nd)
        self.R = np.eye(self.nD)


    def set_standard_oM(self):
        """Distributes observation prototypes around the unit circle."""

        rho = np.linspace(0, 2*np.pi, self.nd, endpoint=False)
        self.oM = np.vstack( (np.cos(rho), np.sin(rho)) )


    @property
    def nd(self):
        """Getter for dimensionality of state space."""
        return self._nd


    @nd.setter
    def nd(self, nd):
        """Setter for dimensionality of state space.

        Changing the dimensionality of state space will also update the
        connection matrix between state variables and the observation
        prototypes.
        """

        self._nd = nd

        # update connection matrix
        self.hopL = ( -self.inhib * np.ones((nd, nd)) +
                      self.inhib * np.eye(nd) );

        # update observation prototypes
        self.set_standard_oM()


    def dynfun(self, Z):
        """Dynamics function."""

        sigz = 1 / ( 1 + np.exp( -self.hopslope * (Z - self.hopg) ) )
        dZ = self.hopk * (self.hopL.dot(sigz) + self.leak * (self.hopg - Z))
        if self.q > 0:
            dZ = dZ + self.q * np.random.normal(size=dZ.shape)

        return dZ


    def obsfun(self, Z):
        """Observation function."""

        alpha = 1 / ( 1 + np.exp( -self.oslope * (Z - self.oshift) ) )
        X = self.oM.dot(alpha)
        if self.r > 0:
            X = X + self.r * np.random.normal(size=X.shape)

        return X


    def findSaddle(self):
        """Estimates location of the saddle point of the Hopfield dynamics."""

        oldq = self.q
        self.q = 0

        z = np.ones((self.nd))

        zold = z - 1
        while linalg.norm(z - zold) > 1e-7:
            zold = z
            z = z + 0.01 * self.dynfun(z)

        self.q = oldq

        return z


class UT(object):
    """Implements the Unscented Transform (min set)."""

    _name = 'min set'

    def __init__(self, D):
        """Initialise weights."""

        # dimensionality in original space, will compute weights
        self.D = D

    @property
    def name(self):
        self._name

    @property
    def D(self):
        return self._D


    @D.setter
    def D(self, D):
        self._D = D

        self.setWeights()


    @property
    def N(self):
        """Number of sigma points."""

        return self.D + 1


    def setWeights(self):
        # weights for reconstructing the mean
        self.wm = np.hstack((1.0, np.zeros((self.D))))

        # weights for reconstructing the covariance
        self.wc = np.hstack((0.0, np.ones((self.D)) / self.D))


    def constructSigmaPoints(self, mean, cov):
        """Construct sigma points."""

        SP = mean[:, None].repeat(self.N, axis=1)
        SP[:, 1:] = SP[:, 1:] + np.sqrt(self.D) * linalg.cholesky(cov)

        return SP


    def performUT(self, mean, cov, trfun):
        """Performs the unscented transform."""

        # construct sigma points
        SP = self.constructSigmaPoints(mean, cov)

        # transform sigma points
        SP_tr = trfun(SP)

        # reconstruct mean and covariance from transformed sigma points
        return self.reconstruct(SP_tr)


    def performUTforUKF(self, mean, cov, dynfun, obsfun):
        """Performs the unscented transform twice for use within UKF."""

        # construct sigma points
        SP = self.constructSigmaPoints(mean, cov)

        # transform through dynamics function
        SPdyn = dynfun(SP)

        # reconstruct mean
        meandyn = SPdyn.dot(self.wm)

        # error between sigma points and mean
        errdyn = SPdyn - meandyn[:, None]

        # reconstruct covariance from errors
        covdyn = errdyn.dot((self.wc * errdyn).T)

        # same procedure with transform through observation function
        SPobs = obsfun(SPdyn)
        meanobs = SPobs.dot(self.wm)
        errobs = SPobs - meanobs[:, None]
        covobs = errobs.dot((self.wc * errobs).T)

        # compute cross-covariance between dynamic states and observations
        xcov = errdyn.dot((self.wc * errobs).T)

        return meandyn, covdyn, meanobs, covobs, xcov


    def reconstruct(self, S):
        """Reconstruct mean and covariance from sigma points.

        (D, N) = shape(S) where D is the dimensionality of the sigma points
                          and N is their number
        """
        mean = S.dot(self.wm)

        Sm = S - mean[:, None]
        cov = Sm.dot((self.wc * Sm).T)

        return mean, cov


    def printparams(self):
        return ''


    def __str__(self):
        desc = 'Unscented Transform (%s with %d sigma points)' % (
            self.name, self.N)

        parstr = self.printparams()

        if len(parstr) > 0:
            desc = desc + '\n' + parstr

        return desc


class UT_base(UT):
    """Implements the Unscented Transform (base set)."""

    _name = 'base set'

    @property
    def N(self):
        """Number of sigma points."""

        return self.D * 2


    def setWeights(self):
        self.wm = np.ones((self.N)) / self.N
        self.wc = self.wm


    def constructSigmaPoints(self, mean, cov):
        SP = mean[:, None].repeat(self.N, axis=1)
        L = np.sqrt(self.D) * linalg.cholesky(cov)
        SP[:, :self.D] = SP[:, :self.D] + L
        SP[:, self.D:] = SP[:, self.D:] - L

        return SP


class UT_scaled(UT):
    """Implements the Unscented Transform (Gauss and scaled sets)."""

    _name = 'scaled set'

    def __init__(self, D):
        """Initialise parameters and weights."""

        # these default parameters implement the Gauss set
        self._alpha = np.sqrt(3)
        self._kappa = 1.0
        self._beta = 2.0

        # dimensionality in original space, will compute weights
        self.D = D


    @property
    def name(self):
        if np.allclose([np.sqrt(3), 1.0, 2.0],
                       [self.alpha, self.kappa, self.beta]):
            return 'Gauss set'
        else:
            return self._name


    @property
    def N(self):
        """Number of sigma points."""

        return self.D * 2 + 1


    @property
    def alpha(self):
        """Scale parameter."""

        return self._alpha


    @alpha.setter
    def alpha(self, alpha):
        self._alpha = float(alpha)

        self.setWeights()


    @property
    def kappa(self):
        """Another scale parameter."""

        return self._kappa


    @kappa.setter
    def kappa(self, kappa):
        self._kappa = float(kappa)

        self.setWeights()


    @property
    def beta(self):
        """Scale correction parameter."""

        return self._beta


    @beta.setter
    def beta(self, beta):
        self._beta = float(beta)

        self.setWeights()


    def setWeights(self):
        a2k = self.alpha**2 * self.kappa
        self.wm = np.ones((self.N)) / (2 * a2k)
        self.wc = np.copy(self.wm)

        self.wm[0] = (a2k - self.D) / a2k
        self.wc[0] = self.wm[0] + 1 - self.alpha**2 + self.beta


    def constructSigmaPoints(self, mean, cov):
        SP = mean[:, None].repeat(self.N, axis=1)
        L = self.alpha * np.sqrt(self.kappa) * linalg.cholesky(cov)
        SP[:, 1:self.D+1] = SP[:, 1:self.D+1] + L
        SP[:, self.D+1:] = SP[:, self.D+1:] - L

        return SP


    def printparams(self):
        return 'alpha = %5.3f\nbeta  = %5.3f\nkappa = %5.3f' % (self.alpha,
                                                                self.beta,
                                                                self.kappa)


class UT_mean(UT):
    """Implements the Unscented Transform (mean set)."""

    _name = 'mean set'

    def __init__(self, D):
        """Initialise parameters and weights."""

        # this default value implements a Gauss set in 2D
        self._w0 = 1.0 / 3.0

        # dimensionality in original space, will compute weights
        self.D = D


    @property
    def D(self):
        return self._D


    @D.setter
    def D(self, D):
        self._D = D
        self._kappa = D / (1 - self.w0)

        self.setWeights()


    @property
    def N(self):
        return 2 * self.D + 1


    @property
    def w0(self):
        return self._w0


    @w0.setter
    def w0(self, w0):
        self._w0 = w0
        self._kappa = self.D / (1 - w0)

        self.setWeights()


    def setWeights(self):
        self.wm = np.r_[self.w0, np.ones(2 * self.D) / (2 * self._kappa)]
        self.wc = self.wm


    def constructSigmaPoints(self, mean, cov):
        SP = mean[:, None].repeat(self.N, axis=1)
        L = np.sqrt(self._kappa) * linalg.cholesky(cov)
        SP[:, 1:self.D+1] = SP[:, 1:self.D+1] + L
        SP[:, self.D+1:] = SP[:, self.D+1:] - L

        return SP


    def printparams(self):
        return 'w0 = %5.3f with kappa = %5.3f' % (self.w0, self._kappa)


class UKF(object):
    """Implementation of the Unscented Kalman Filter."""

    def __init__(self, model, dt, UT=None):
        # a dynamic generative model
        self.model = model

        # time resolution of internal integration
        self.dt = dt

        # number of dimensions of augmented state variable
        self.na = model.nd * 2 + model.nD

        # this is the unscented transform used for filtering
        if UT is None:
            self.UT = UT_scaled(self.na)
        else:
            self.UT = UT


    def obsfun(self, Za):
        return (self.model.obsfun(Za[:self.model.nd, :]) +
                Za[2*self.model.nd:, :])


    def dynfun(self, Za, dt, nsteps):
        # simple Euler integration
        for st in range(nsteps):
            dZ = (self.model.dynfun(Za[:self.model.nd, :]) +
                  Za[self.model.nd:2*self.model.nd, :])

            Za[:self.model.nd, :] = Za[:self.model.nd, :] + dt * dZ

        return Za


    def run(self, Obs, Time, mean0, cov0):
        from scipy.linalg import block_diag

        # augment the state
        meana = np.r_[mean0, np.zeros(self.na-self.model.nd)]
        cova = block_diag(cov0, self.model.Q / self.dt, self.model.R)

        nT = Time.size
        dTime = np.diff(np.r_[0.0, Time])

        mean = np.zeros((self.model.nd, nT))
        cov = np.zeros((self.model.nd, self.model.nd, nT))
        for t in range(nT):
            # determine how many steps into the future you have to simulate
            nsteps = int(np.ceil(dTime[t] / self.dt))
            dtt = dTime[t] / nsteps
            dynfun = lambda x: self.dynfun(x, dtt, nsteps)

            mdyn, cdyn, mobs, cobs, cdynobs = self.UT.performUTforUKF(meana,
                cova, dynfun, self.obsfun)

            K = cdynobs[:self.model.nd, :].dot(linalg.inv(cobs))

            perr = Obs[:, t] - mobs

            mean[:, t] = mdyn[:self.model.nd] + K.dot(perr)

            cov[:, :, t] = cdyn[:self.model.nd, :self.model.nd] - \
                           K.dot(cdynobs[:self.model.nd, :].T)

            # ensure symmetry
            cov[:, :, t] = (cov[:, :, t] + cov[:, :, t].T) / 2.0

            if t < nT-1:
                meana[:self.model.nd] = mean[:, t]
                cova[:self.model.nd, :self.model.nd] = cov[:, :, t]

        return mean, cov


if __name__ == "__main__":
    bam = BAttM()
    bam.nd = 3
    Z = np.vstack((np.ones(bam.nd) * bam.oshift,             # point in linear region
               bam.findSaddle(),                         # saddle point of dynamics
               np.hstack((bam.hopg, np.zeros(bam.nd-1))) # a stable fixed point
              )).T

    C = 8**2 * np.eye(bam.nd)

    num = Z.shape[1]
    nsample = 10000

    # make full dynamics function
    # (model function is only the continuous change dz/dt)
    dt = 0.05
    dynfun = lambda x: x + dt * bam.dynfun(x)

    # initialise output arrays
    meantrue = [np.zeros((bam.nD, num)), np.zeros((bam.nd, num))]
    covtrue = [np.zeros((bam.nD, bam.nD, num)), np.zeros((bam.nd, bam.nd, num))]
    Strue = [np.zeros((bam.nD, nsample, num)), np.zeros((bam.nd, nsample, num))]
    for i in range(num):
        meantrue[0][:, i], covtrue[0][:, :, i], Strue[0][:, :, i] = \
            aux.naiveSamplingEstimate(Z[:, i], C, bam.obsfun, nsample)
        meantrue[1][:, i], covtrue[1][:, :, i], Strue[1][:, :, i] = \
            aux.naiveSamplingEstimate(Z[:, i], C, dynfun, nsample)

#    axes = aux.plotSamples(Strue[:, :200, :, 1], meantrue[:, :, 1],
#                           covtrue[:, :, :, 1], ylabels=['dynamics function'],
#                           titles=['[5, 5]', 'saddle point', 'stable fixed point'])

    pi = 1

    ut = UT_mean(bam.nd)
#    ut = UT_base(bam.nd)
    meanUT, covUT = ut.performUT(Z[:, pi], C, dynfun)

    axes, leg = aux.plotHDSamples(Strue[1][:, :, pi], meantrue[1][:, pi],
                             covtrue[1][:, :, pi], meanUT=meanUT, covUT=covUT,
                             utlabels=[ut._name])

    DKL = aux.KLGauss(meantrue[1][:, pi], covtrue[1][:, :, pi], meanUT, covUT)
    print "DKL = %5.3f" % (DKL,)

    nsample = 2 + bam.nd ** np.arange(1, 10)
    print 'number of samples: ' + nsample.__str__()
    nrep = 20
    trfuns = (bam.obsfun, dynfun)
    funlabels = ('observation function', 'dynamics function')
    pointlabels = ('[5, 5]', 'saddle', 'stable FP')
    desKL = np.array([10.0, 1.0, 0.1]);

#    axes = aux.plotSamplingKLs(Z, C, meantrue, covtrue, nsample, nrep, trfuns,
#                               desKL, funlabels, pointlabels)

#    ut_mean = UT_scaled(bam.nd)
#
#    bam.Q = 5**2 * np.eye(bam.nd)
#    bam.R = 1**2 * np.eye(bam.nD)
#
#    ukf = UKF(bam, dt)
#
#    mean0 = bam.findSaddle()
#    cov0 = dt * 5**2 * np.eye(bam.nd)
#
#    Time = np.arange(dt, 2.52, dt)
#    t2 = Time.size / 2
##    Obs = np.c_[np.outer(bam.oM[:, 0], np.ones(t2)),
##                np.outer(bam.oM[:, 1], np.ones(t2))]
#    Obs = np.c_[np.outer(bam.oM[:, 0], np.ones(t2)),
#                np.zeros((bam.nd, t2))]
#
#    M, C = ukf.run(Obs, Time, mean0, cov0)
#
#    aux.plot_stdshade(M, C, X=Time)

