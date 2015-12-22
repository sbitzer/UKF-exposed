# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:09:30 2015

@author: bitzer
"""

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.patches import Ellipse

obscols = ['0.0', '0.5']
datacol = '0.4'
#approxcols = ['#1b9e77', '#d95f02', '#7570b3']
utcols = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
datacol2 = utcols[1]


def WassDist(mean, Sigma, mean2, Sigma2):
    # ensure that means are 1D arrays
    mean = mean.squeeze()
    mean2 = mean2.squeeze()

    # check that covariances are positive definite
    # by computing matrix square roots
    try:
        SC  = linalg.cholesky(Sigma);
        SC2 = linalg.cholesky(Sigma2);
    except linalg.LinAlgError:
        return np.nan
    else:
        return np.sqrt(linalg.norm(mean - mean2, ord=2) ** 2 +
                       linalg.norm(SC - SC2, ord='fro') ** 2)


#def KLGauss(mean, Sigma, mean2, Sigma2, eps=1e-12):
#    # ensure that means are 1D arrays
#    mean = mean.squeeze()
#    mean2 = mean2.squeeze()
#
#    # check that covariances are positive definite
#    try:
#        linalg.cholesky(Sigma);
#        linalg.cholesky(Sigma2);
#    except linalg.LinAlgError:
#        return np.nan
#
#    # compute difference in means
#    meandiff = mean - mean2
#
#    # ignoring tiny differences in dimensions with tiny variance
#    var = Sigma.diagonal()
#    var2 = Sigma2.diagonal()
#    tinyind = ((var < eps) * (var2 < eps)).nonzero()[0]
#    if tinyind.size > 0:
#        if meandiff[tinyind] < eps:
#            meandiff[tinyind] = 0.0
#
#    # compute inverse of Sigma2
#    Sigma2inv = linalg.inv(Sigma2)
#
#    # compute log determinants
#    (sign, logdet) = linalg.slogdet(Sigma)
#    sldet = sign * logdet
#    (sign, logdet) = linalg.slogdet(Sigma2)
#    sldet2 = sign * logdet
#
#    return ( Sigma2inv.dot(Sigma).trace() +
#             meandiff.dot(Sigma2inv).dot(meandiff) -
#             mean.size + sldet2 - sldet )


def naiveSamplingEstimate(mean, cov, trfun, N=None):
    # if no covariance, mean contains samples
    if cov is None:
        S = mean
    # else sample from Gaussian using numpy
    else:
        S = np.random.multivariate_normal(mean, cov, int(N)).T

    # project through transform function
    Str = trfun(S)

    # estimate transformed mean
    meantr = np.mean(Str, 1)

    # estimate transformed covariance
    covtr = np.cov(Str)

    return meantr, covtr, Str


# function to plot trajectories
def plottr(Z, X, T):
    if len(Z.shape) < 3:
        Z = Z[:, :, None]
        X = X[:, :, None]

    ntr = Z.shape[2]

    fig, axes = plt.subplots(2, ntr, sharex='all', sharey='row', figsize=(10.0, 5.0))
    if ntr == 1:
        axes = axes[:, None]

    for tr in range(ntr):
        axes[0, tr].plot(T, Z[:, :, tr].T)

        lines = axes[1, tr].plot(T, X[:, :, tr].T)

        for (line,col) in zip(lines, obscols):
            line.set_color(col)

        if tr == 0:
            axes[0, tr].set_ylabel('hidden state')
            axes[1, tr].set_ylabel('observation')

    return axes


def plotSamples(S, mean=None, cov=None, ylabels=None, titles=None, nplot=1000,
                meanUT=None, covUT=None, utlabels=None):
    from scipy.stats import norm
    import matplotlib.lines as mlines

    nd, nsample = S.shape[:2]
    if nsample < nplot:
        nplot = nsample

    # ensure that ndarrays have standard shape
    if len(S.shape) < 3:
        nset = 1
        nalt = 1
        S = S[:, :, None, None]
        mean = mean[:, None, None]
        cov = cov[:, :, None, None]
        if meanUT is not None:
            if len(meanUT.shape) == 1:
                meanUT = meanUT[:, None, None, None]
                covUT = covUT[:, :, None, None, None]
            else:
                meanUT = meanUT[:, None, :, None]
                covUT = covUT[:, :, None, :, None]
    else:
        nset = S.shape[2]
        if len(S.shape) < 4:
            nalt = 1
            S = S[:, :, :, None]
            mean = mean[:, :, None]
            cov = cov[:, :, :, None]
            if meanUT is not None:
                if len(meanUT.shape) == 2:
                    meanUT = meanUT[:, :, None, None]
                    covUT = covUT[:, :, :, None, None]
                else:
                    meanUT = meanUT[:, :, :, None]
                    covUT = covUT[:, :, :, :, None]
        else:
            nalt = S.shape[3]

    if meanUT is not None:
        nUT = meanUT.shape[2]
        if utlabels is None:
            utlabels = ['ut %d' % ut for ut in range(nUT)]

    if ylabels is None:
        ylabels = ['y' for i in range(nalt)]

    fig, axes = plt.subplots(nalt, nset, sharex='row', sharey='row',
                             figsize=(10.0, nalt*5.0))

    if nalt == 1:
        axes = np.array(axes, ndmin=2)

    for i in range(nalt):
        axes[i, 0].set_ylabel(ylabels[i])

        # determine range of values in each axis
        mins = S[:, :, :, i].min(axis=1).min(axis=1)
        maxes = S[:, :, :, i].max(axis=1).max(axis=1)

        for j in range(nset):
            # which variances are greater than some small threshold?
            varinds = (cov[:, :, j, i].diagonal() > 1e-10).nonzero()[0]

            # if the distribution is 1D (only one large variance)
            if varinds.size == 1:
                # get the dimension with large variance
                dim = varinds[0]

                # plot a histogram of the samples on the corresponding axis
                if dim == 0:
                    orient = 'vertical'
                else:
                    orient = 'horizontal'

                axes[i, j].set_aspect('auto')
                axes[i, j].hist(S[dim, :, j, i], bins=30, normed=True,
                                range=(mins[dim], maxes[dim]),
                                orientation=orient, color=datacol)

                # plot the estimated Gaussian pdf
                xx = np.linspace(mins[dim], maxes[dim], 100)
                axes[i, j].plot(xx, norm.pdf(xx, mean[dim, j, i],
                                np.sqrt(cov[dim, dim, j, i])), 'k', lw=1)

                # plot the UT approximations
                if meanUT is not None:
                    lh = []
                    for uti in range(nUT):
                        lh.append(axes[i, j].plot(xx,
                                  norm.pdf(xx, meanUT[dim, j, uti, i],
                                  np.sqrt(covUT[dim, dim, j, uti, i])),
                                  color=utcols[uti], lw=1,
                                  label=utlabels[uti])[0])

                    # legend
                    if j==0:
                        axes[i, j].legend(handles=lh, loc='upper left')
            else:
                axes[i, j].set_aspect('equal', adjustable='box-forced')
                axes[i, j].scatter(S[0, :nplot, j, i], S[1, :nplot, j, i],
                                   c=datacol, alpha=0.3)

                plot_cov_ellipse(cov[:, :, j, i], mean[:, j, i],
                                 volume=.9, ax=axes[i, j], lw=1)

                if meanUT is not None:
                    # for the legend I need to make some line objects
                    # with the corresponding colors
                    if j == 0:
                        lh = []

                    for uti in range(nUT):
                        ellip = plot_cov_ellipse(covUT[:, :, j, uti, i],
                                                 meanUT[:, j, uti, i],
                                                 volume=.9, ax=axes[i, j], lw=1)
                        ellip.set_edgecolor(utcols[uti])

                        if j == 0:
                            lh.append(mlines.Line2D([], [],
                                                    color=utcols[uti],
                                                    label=utlabels[uti]))

                    if j == 0:
                        axes[i, j].legend(handles=lh, loc='lower left')

            if i == 0 and titles is not None:
                axes[i, j].set_title(titles[j])

    return axes


def plotHDSamples(S, mean=None, cov=None, title=None, nplot=np.inf,
                meanUT=None, covUT=None, utlabels=None, S2=None):
    import matplotlib.lines as mlines

    nd, nsample = S.shape
    if nsample < nplot:
        nplot = nsample

    # ensure 1D mean
    mean = mean.squeeze()

    if meanUT is not None:
        assert(meanUT.shape[0] == nd)
        assert(covUT.shape[0] == nd)

        if len(meanUT.shape) == 1:
            meanUT = meanUT[:, None]
            covUT = covUT[:, :, None]

        assert(meanUT.shape[1] == covUT.shape[2])

        nut = meanUT.shape[1]
    else:
        nut = 0

    fig, axes = plt.subplots(nd-1, nd-1, sharex='all', sharey='all',
                             figsize=(10.0, 10.0),
                             subplot_kw={'aspect': 'equal', 'visible': False},
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    dimlabels = ['dim %d' % (dim+1,) for dim in range(nd)]

    for d1 in range(nd-1):
        for d2 in range(d1+1, nd):
            # select corresponding 2D subset
            indm = [d2, d1]
            indc = [[d2, d2, d1, d1], [d2, d1, d2, d1]]

            #axes[d1, d2].set_aspect('equal', adjustable='box-forced')
            axes[d1, d2-1].set_visible(True)
            axes[d1, d2-1].scatter(S[d2, :nplot], S[d1, :nplot],
                                   c=datacol, alpha=0.3)

            if S2 is not None:
                axes[d1, d2-1].scatter(S2[d2, :], S2[d1, :],
                                       c=datacol2, alpha=0.8)

            cov2 = cov[indc[0], indc[1]].reshape((2, 2))
            plot_cov_ellipse(cov2, mean[indm],
                             volume=.9, ax=axes[d1, d2-1], lw=1)

            if d1 == 0 and d2 == 1 and utlabels is not None:
                lh = []

            for uti in range(nut):
                covUT2 = covUT[indc[0], indc[1], uti].reshape((2, 2))
                ellip = plot_cov_ellipse(covUT2, meanUT[indm, uti],
                                         volume=.9, ax=axes[d1, d2-1], lw=1)
                ellip.set_edgecolor(utcols[uti])

                if d1 == 0 and d2 == 1 and utlabels is not None:
                    lh.append(mlines.Line2D([], [], color=utcols[uti],
                                            label=utlabels[uti]))

            if d2-1 == d1:
                axes[d1, d2-1].set(ylabel=dimlabels[d1],
                                   xlabel=dimlabels[d2])
                if d1==0:
                    if title is not None:
                        axes[d1, d2-1].set_title(title)
                    if utlabels is not None:
                        leg = axes[d1, d2-1].legend(handles=lh,
                                                    loc='upper right',
                                                    bbox_to_anchor=(0.9, -0.1))

    return axes, leg


def plotSamplingDis(mean0, cov0, mean, cov, nsample, nrep, trfuns,
                    desKL, funlabels=None, setlabels=None):
    if len(mean.shape) < 2:
        mean = mean[:, None, None]
        cov = cov[:, :, None, None]
    elif len(mean.shape) < 3:
        mean = mean[:, :, None]
        cov = cov[:, :, :, None]

    nd, nset, nfun = mean.shape

    assert(nfun == len(trfuns))

    if setlabels is None:
        setlabels = ['set %d'% (i,) for i in range(nset)]

    fig, axes = plt.subplots(1, nfun, sharex='row', sharey='row',
                             figsize=(10.0, 4.0))

    estNs = np.zeros((desKL.size, nset, nfun))
    for trfi, trfun in enumerate(trfuns):
        for s in range(nset):
            Dis = np.zeros((nrep, nsample.size))
            for i, ns in enumerate(nsample):
                for rep in range(nrep):
                    meantr, covtr, _ = naiveSamplingEstimate(mean0[:, s],
                                                             cov0, trfun, ns)
                    Dis[rep, i] = WassDist(mean[:, s, trfi],
                        cov[:, :, s, trfi], meantr, covtr)

            Dismean = Dis.mean(axis=0)
            Perc = np.percentile(Dis, [5, 95], axis=0)

            # plot mean with error bars
            axes[trfi].errorbar(nsample * (1.0 + s/(4.0*nset)), Dismean,
                                np.c_[Dismean - Perc[0, :], Perc[1, :] - Dismean].T,
                                label=setlabels[s])

            # estimate number of samples for desired accuracy
            estNs[:, s, trfi] = np.interp(desKL, Dismean[::-1], nsample[::-1],
                                          left=np.nan)

        if trfi == 0:
            axes[trfi].set_ylabel('$W$')
            axes[trfi].legend()

        axes[trfi].set_xscale('log')
        axes[trfi].set_yscale('log')
        axes[trfi].set_xlabel('number of samples')
        axes[trfi].set_xlim([nsample[0]-3, 2*nsample[-1]]);
        axes[trfi].set_ylim([0.001, 10]);

        if funlabels is not None:
            axes[trfi].set_title(funlabels[trfi])

    return estNs

def plotKLcontours(X, Y, KLs, titles=None):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), sharex='all',
                             sharey='all')

    if KLs.ndim < 3:
        KLs = KLs[:, :, None]

    nc = KLs.shape[2]

    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    bounds = lambda z, dz: np.r_[z - dz/2, z[-1] + dz/2]

    for c in range(nc):
#        cs = axes[c].contourf(X, Y, np.log10(KLs[:, :, c]), 100,
#                              vmin=-2.0, vmax=2.0, extend='both', cmap='Blues')
        cs = axes[c].pcolormesh(bounds(X, dx), bounds(Y, dy),
                                np.log10(KLs[:, :, c]), vmin=-3.0, vmax=1.0,
                                cmap='Blues')

        cb = fig.colorbar(cs, ax=axes[c], extend='both')
        cb.set_label('$W$')

        axes[c].plot(X[KLs[:, :, c].argmin(axis=1)], Y, '.', color='0.5')

        if titles is not None:
            axes[c].set_title(titles[c])

        axes[c].set(xlabel='$w_0$', ylim=(Y[0]-dy/2, Y[-1]+dy/2),
                    xlim=(X[0]-dx/2, X[-1]+dx/2))
        axes[c].plot(1/3.0*np.ones(2), axes[c].get_ylim(), color='0.4')

    axes[0].set_ylabel('log(std)')

    return axes


def plot_stdshade(mean, cov, stdmult=2.0, X=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # if one-dimensional
    if len(mean.shape) == 1:
        mean = mean[None, :]
        cov = cov[None, None, :]

    nd, N = mean.shape

    if X is None:
        X = np.arange(N)

    # extract standard deviations
    stds = np.zeros((nd, N))
    for i in range(nd):
        stds[i, :] = np.sqrt(cov[i, i, :])

    # apply multiplier
    stds = stdmult * stds

    stdh = []
    for i in range(nd):
        stdh.append(ax.fill(np.r_[X, X[::-1]],
                            np.r_[mean[i, :] + stds[i, :],
                                  mean[i, ::-1] - stds[i, ::-1]],
                            alpha=0.5)[0])

    mh = []
    for i in range(nd):
        mh.append(ax.plot(X, mean[i, :])[0])

    return mh, stdh


def plot_cov_ellipse(cov, pos, volume=.5, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.

    @author: Noah Haskell Silbert
    http://www.nhsilbert.net/source/2014/06/bivariate-normal-ellipse-plotting-in-python/
    """

    from scipy.stats import chi2

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    ax.add_artist(ellip)

    return ellip