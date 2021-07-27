# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-03-31 13:56:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 18:29:12

import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.core import getPulseTrainProtocol, LogBatch
from PySONIC.utils import logger, si_format, loadData
from PySONIC.plt import setNormalizer, XYMap

from MorphoSONIC.models import SennFiber, UnmyelinatedFiber
from MorphoSONIC.sources import GaussianAcousticSource
from MorphoSONIC.plt import spatioTemporalMap

from utils import getSubRoot, getCommandLineArguments, saveFigs, getAxesFromGridSpec

logger.setLevel(logging.INFO)

modroot = getSubRoot('modulation')


class FRvsPRFBatch(LogBatch):

    in_key = 'PRF'
    out_keys = ['FR']
    suffix = 'FR'
    unit = 'Hz'

    def __init__(self, fiber, source, PD, npulses, PRFs=None, nPRF=10, **kwargs):
        self.fiber = fiber
        self.source = source
        self.PD = PD
        self.npulses = npulses
        if PRFs is None:
            PRFs = self.getPRFrange(nPRF)
        super().__init__(PRFs, **kwargs)

    def compute(self, PRF):
        pp = getPulseTrainProtocol(self.PD, self.npulses, PRF)
        data, meta = self.fiber.simulate(self.source, pp)
        return self.fiber.getEndFiringRate(data)

    @property
    def sourcecode(self):
        codes = self.source.filecodes
        return f'{self.source.key}_{"_".join(codes.values())}'

    def corecode(self):
        return f'FRvsPRF_{self.fiber.modelcode}_{self.sourcecode}_PD{si_format(self.PD, 1)}s'

    def getPRFrange(self, n):
        ''' Get pulse-duration-dependent PRF range. '''
        PRF_max = 0.99 / self.PD  # Hz
        PRF_min = max(PRF_max / 100, 10)  # Hz
        return np.logspace(np.log10(PRF_min), np.log10(PRF_max), n)


class NormalizedFiringRateMap(XYMap):

    xkey = 'duty cycle'
    xfactor = 1e0
    xunit = '-'
    ykey = 'amplitude'
    yfactor = 1e0
    yunit = 'Pa'
    zkey = 'normalized firing rate'
    zunit = '-'
    zfactor = 1e0
    suffix = 'FRmap'

    def __init__(self, fiber, source, DCs, amps, npulses, PRF, root='.'):
        self.fiber = fiber
        self.source = source
        self.PRF = PRF
        self.npulses = npulses
        super().__init__(root, DCs, amps)

    @property
    def sourcecode(self):
        codes = self.source.filecodes
        if 'A' in codes:
            del codes['A']
        return f'{self.source.key}_{"_".join(codes.values())}'

    def corecode(self):
        return f'normFRmap_{self.fiber.modelcode}_{self.sourcecode}_PRF{si_format(self.PRF, 1)}Hz'

    @property
    def title(self):
        return f'Normalized firing rate map - {self.fiber}, {self.source}, {si_format(self.PRF)}Hz PRF'

    def compute(self, x):
        DC, A = x
        self.source.A = A
        pp = getPulseTrainProtocol(DC / self.PRF, self.npulses, self.PRF)
        data, meta = self.fiber.simulate(self.source, pp)
        return self.fiber.getEndFiringRate(data) / self.PRF

    def onClick(self, event):
        DC, A = self.getOnClickXY(event)
        self.source.A = A
        pp = getPulseTrainProtocol(DC / self.PRF, self.npulses, self.PRF)
        data, meta = self.fiber.simulate(self.source, pp)
        spatioTemporalMap(self.fiber, self.source, data, 'Qm', fontsize=fontsize)
        plt.show()


def plotFRvsPRF(PRFs, FRs, cmaps, ax=None):
    ''' Plot FR vs PRF across cell types for various PDs. '''
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 3.5))
    else:
        fig = None
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('pulse repetition frequency (Hz)')
    ax.set_ylabel('firing rate (Hz)')

    PRF_min = 1e0
    PRF_max = 0.
    for key, val in FRs.items():
        for k, v in val.items():
            imax = np.where(~np.isnan(v))[0][-1]
            PRF_max = max(PRFs[key][k][imax], PRF_max)
    PRF_range = np.array([PRF_min, PRF_max])
    ax.plot(PRF_range, PRF_range, '--', c='k')
    for x in [2]:
        ax.plot(PRF_range, PRF_range * x, '--', c='k')
        ax.plot(PRF_range, PRF_range / x, '--', c='k')
    ax.set_xlim(*PRF_range)
    ax.set_ylim(*PRF_range)

    for k, FRdict in FRs.items():
        nPDs = len(FRdict)
        _, sm = setNormalizer(plt.get_cmap(cmaps[k]), (0, 1))
        xstart = 0.4  # avoid white-ish colors
        clist = [sm.to_rgba((1 - xstart) / (nPDs - 1) * i + xstart) for i in range(nPDs)]
        for c, (PD_key, FR) in zip(clist, FRdict.items()):
            lbl = f'{k} - {PD_key}'
            if np.all(np.isnan(FR)):
                logger.info(f'{lbl}: all NaNs')
            else:
                PRF = PRFs[k][PD_key]
                if np.all(~np.isnan(FR[:3])):  # if starts with 3 real values
                    s0, s1, *_ = np.diff(FR) / np.diff(PRF)
                    if np.isclose(s1, s0, atol=1e-2):  # if first 2 slopes are similar
                        # Extend linear range to minimum PRF (assuming to vertical offset)
                        PRF, FR = np.hstack(([PRF_min], PRF)), np.hstack(([s0 * PRF_min], FR))
                ax.plot(PRF, FR, label=lbl, c=c)
                # ax.axvline(PRF.max(), c=c, linestyle='--')
    ax.legend(frameon=False)
    return fig


# Fiber models
a = 32e-9  # sonophore radius (m)
fs = 1.    # sonophore coverage fraction (-)
fibers = {
    'myelinated': SennFiber(10e-6, 21, a=a, fs=fs),
    'unmyelinated': UnmyelinatedFiber(0.8e-6, fiberL=5e-3, a=a, fs=fs),
}

PDs = {
    'myelinated': np.array([20e-6, 100e-6, 1e-3]),
    'unmyelinated': np.array([1e-3, 5e-3, 10e-3])
}

# US parameters
Fdrive = 500e3  # Hz
Adrive = 300e3  # Pa

# Define acoustic sources
sources = {}
for k, fiber in fibers.items():
    w = fiber.length / 5  # m
    sigma = GaussianAcousticSource.from_FWHM(w)  # m
    sources[k] = GaussianAcousticSource(0, sigma, Fdrive, Adrive)

# Pulsing parameters
npulses = 10
nPD = 10
nPRF = 50
ncombs = len(fibers) * nPD * nPRF

# Plot parameters
fontsize = 10
Qbounds = {
    'myelinated': (-175, 75),
    'unmyelinated': (-80, 36)
}
cmaps = {
    'unmyelinated': 'Blues',
    'myelinated': 'Oranges'
}
subset_colors = {
    'unmyelinated': 'C0',
    'myelinated': 'C1'
}
normFRbounds = {
    'myelinated': (0, 1),
    'unmyelinated': (0, 3.1)
}

map_PRFs = {
    'myelinated': [500, 2600],   # Hz
    'unmyelinated': [50., 200.]  # Hz
}

subsets = {
    'myelinated': [
        (100e-6, 500),
        (100e-6, 2600),
        (20e-6, 5000),
    ],
    'unmyelinated': [
        (10e-3, 15.),
        (5e-3, 15.),
        (1e-3, 400.)
    ]
}


if __name__ == '__main__':

    args = getCommandLineArguments()
    figs = {}

    # Create figure backbone
    nfibers = len(fibers)
    nQmaps = len(subsets['myelinated'])
    nPRFs = len(map_PRFs['myelinated'])
    y1 = 6
    y2 = y1 + 2
    y3 = nQmaps * y2
    y4 = 2 * y3 // 3
    ysep = 4
    nrows = y3 + y4 + ysep
    x1 = 6
    x2 = ((1 + nfibers) * x1 + 2) // (nfibers * nPRFs)
    ncols = (1 + nfibers) * x1 + 2 * nfibers
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(nrows, ncols)
    subplots = {'FRvsPRF': gs[:y3, :x1]}
    for i, k in enumerate(fibers.keys()):
        subplots[k] = {}
        # Top section
        x0 = x1 + i * (x1 + 2)
        yslice = lambda j: slice(j * y2 + 2, j * y2 + 2 + y1)
        subplots[k]['field'] = [gs[yslice(j), x0] for j in range(nQmaps)]
        xslice = slice(x0 + 1, x0 + x1 + 1)
        subplots[k]['stim'] = [gs[j * y2, xslice] for j in range(nQmaps)]
        subplots[k]['spikes'] = [gs[j * y2 + 1, xslice] for j in range(nQmaps)]
        subplots[k]['Qmmap'] = [gs[yslice(j), xslice] for j in range(nQmaps)]
        subplots[k]['Qmcbar'] = gs[:y3, x0 + x1 + 1]
        # Bottom section
        x0 = i * (nPRFs * x2 + 1)
        y0 = y3 + ysep
        subplots[k]['normFRmap'] = [gs[y0:, x0 + j * x2:x0 + (j + 1) * x2] for j in range(nPRFs)]
        subplots[k]['normFRcbar'] = gs[y0:, x0 + nPRFs * x2]
    axes = getAxesFromGridSpec(fig, subplots)
    figs['modulation'] = fig

    # FR vs PRF batches
    PRFs = {}
    FRs = {}
    for k, fiber in fibers.items():
        # For each pulse duration
        PRFs[k] = {}
        FRs[k] = {}
        for PD in PDs[k]:
            # Run FR batch
            PD_key = f'PD = {si_format(PD, 1)}s'
            frbatch = FRvsPRFBatch(fiber, sources[k], PD, npulses, nPRF=nPRF, root=modroot)
            PRFs[k][PD_key] = frbatch.inputs
            FRs[k][PD_key] = frbatch.run()
    FRax = axes['FRvsPRF']
    plotFRvsPRF(PRFs, FRs, cmaps, ax=FRax)

    # Spatiotemporal maps for fiber-specific subsets
    minFR = FRax.get_ylim()[0]
    subset_FRs = {}
    for k, fiber in fibers.items():
        subset_FRs[k] = []
        for imap, (PD, PRF) in enumerate(subsets[k]):
            key = f'PD = {si_format(PD, 1)}s, PRF = {si_format(PRF, 1)}Hz'
            pp = getPulseTrainProtocol(PD, npulses, PRF)
            fpath = fiber.simAndSave(sources[k], pp, overwrite=False, outputdir=modroot)
            data, meta = loadData(fpath)
            subaxes = {key: axes[k][key][imap] for key in ['field', 'stim', 'spikes', 'Qmmap']}
            subaxes['Qmcbar'] = axes[k]['Qmcbar']
            fig = spatioTemporalMap(
                fiber, sources[k], data, 'Qm', fontsize=fontsize,
                cmap=cmaps[k], zbounds=Qbounds[k], rasterized=True, axes=subaxes)
            fname = f'Qmap_{k}_{key}'.replace(' ', '').replace(',', '_').replace('=', '')
            subset_FRs[k].append(fiber.getEndFiringRate(data))
        subset_FRs[k] = np.array(subset_FRs[k])  # convert to array
        subset_FRs[k][np.isnan(subset_FRs[k])] = minFR  # convert nans to inferior ylim
        FRax.scatter([x[1] for x in subsets[k]], subset_FRs[k],
                     c=[subset_colors[k]], zorder=2.5)

    # Fiber-specific FR / PRF maps
    nperax = 40
    DCs = np.linspace(0.0, 1, nperax)
    amps = np.logspace(np.log10(10e3), np.log10(600e3), nperax)
    for k, fiber in fibers.items():
        for imap, PRF in enumerate(map_PRFs[k]):
            frmap = NormalizedFiringRateMap(
                fiber, sources[k], DCs, amps, npulses, PRF, root=modroot)
            frmap.run()
            frmap.render(
                yscale='log', cmap=cmaps[k], zbounds=normFRbounds[k], interactive=True,
                ax=axes[k]['normFRmap'][imap], cbarax=axes[k]['normFRcbar'],
                title=f'{si_format(PRF)}Hz', fs=fontsize)

    # Post-processing
    for k in fibers.keys():
        for ax in axes[k]['Qmmap'][:-1]:
            ax.set_xlabel(None)
        for ax in axes[k]['normFRmap']:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['0', '100'])
            ax.set_xlabel('Duty cycle (%)')
        ax.set_yticklabels([])
        ax.set_ylabel(None)

    if args.save:
        saveFigs(figs)
    plt.show()
