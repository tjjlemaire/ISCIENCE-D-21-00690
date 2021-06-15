# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-05-14 19:42:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-14 12:44:52

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.interpolate import RectBivariateSpline

from PySONIC.core import AcousticDrive, AcousticDriveArray
from PySONIC.multicomp import PassiveBenchmark
from PySONIC.utils import logger
from PySONIC.plt import PassiveDivergenceMap
from ExSONIC.models import SennFiber, UnmyelinatedFiber

from utils import getSubRoot, getCommandLineArguments, saveFigs

logger.setLevel(logging.INFO)

benchmarksroot = getSubRoot('benchmarks')


def getAmpPairs(A1, A2, alpha):
    return {
        'gradient+': [A1 * alpha, A2 / alpha],
        'gradient-': [(A1 + A2) / 2] * 2,
        'field+': [A1 * alpha, A2 * alpha],
        'field-': [A1 / alpha, A2 / alpha],
    }


def findDivmapContours(model_args, drives, covs, tau_ranges, levels, axes, mpi=False, pltmaps=False):
    # Create passive benchmark object
    subdir = f'passive_{"_".join(drives.filecodes.values())}'
    outdir = os.path.join(benchmarksroot, subdir)
    benchmark = PassiveBenchmark(*model_args, outdir=outdir)
    # Create divmap objects over time constants dense 2D space
    divmaps = {
        evmode: PassiveDivergenceMap(
            benchmark, *tau_ranges, [drives, covs], evmode, [])
        for evmode in levels.keys()}
    # Run missing simulations/evaluations if any of the divmaps are not complete
    if not all(x.isFinished() for x in divmaps.values()):
        benchmark.runSimsOverTauSpace(drives, covs, *tau_ranges, mpi=mpi)
    # Extract and plot threshold curves for relevant divergence levels
    for i, (evmode, divmap) in enumerate(divmaps.items()):
        if not divmap.isFinished():
            divmap.run()
        data = divmap.getOutput().T * divmap.zfactor
        x, y = np.log10(divmap.xvec), np.log10(divmap.yvec)
        # f = RectBivariateSpline(x, y, data)
        # norders = int(x[-1] - x[1]) + 1
        # nperorder = (x.size - 1) // norders
        # nextraorders = 1
        # x_extra = x[-nperorder * nextraorders:] + nextraorders
        # x = np.hstack((x, x_extra))
        # data = np.vstack((data, f(x_extra, y)))
        x, y = np.power(10., x), np.power(10., y)
        axes[i].contour(x, y, data.T, levels[evmode], colors=[cdict[k]])
        axes[i].contourf(x, y, data.T, [*levels[evmode], np.inf], colors=[cdict[k]], alpha=0.2)
        if pltmaps:
            divmap.render(
                fs=fs, title=subdir, zscale=zscale[evmode], zbounds=None,
                T=1 / Fdrive, levels=levels[evmode], interactive=True)


# Coupled sonophores model parameters
nnodes = 2
a = 32e-9  # m
covs = [0.8] * nnodes  # (-)

# Fibers references
fibers = {
    'MY': SennFiber(10e-6, 2),
    'UN': UnmyelinatedFiber(0.8e-6, nnodes=2)
}
fiberinsets = {
    k: (f.pneuron.Cm0 / f.pneuron.gLeak, f.pneuron.Cm0 / (f.ga_node_to_node * 1e4))
    for k, f in fibers.items()
}

# Stimulus drives
Fdrive = 500e3  # Hz
other_freqs = {'LF': 20e3, 'HF': 4e6}
ref_amps = [100e3, 50e3]  # Pa
other_amps_pairs = getAmpPairs(*ref_amps, alpha=2)
delta_phis = np.arange(1, 5) * np.pi / 4  # rad

# Passive point-neuron model parameters
Cm0 = 1e-2   # F/m2
ELeak = -70  # mV

# Time constants ranges
tau_bounds = (1e-7, 1e-3)  # s
densification_factor = 4
ntaus = {'sparse': 5}
ntaus['dense'] = (ntaus['sparse'] - 1) * densification_factor + ntaus['sparse']
tau_ranges = {k: np.logspace(*np.log10(tau_bounds), v) for k, v in ntaus.items()}  # s
tau_ranges = {k: [v, v] for k, v in tau_ranges.items()}

# Plot parameters
levels = {
    'ss': [1.],        # nC/cm2
    'transient': [5.]  # %
}
zscale = {'ss': 'log', 'transient': 'log'}
nrows = len(levels)
fs = 12
cbar_shrink_xratio = 10

# colors = plt.get_cmap('Paired').colors
# color_pairs = list(zip(colors[::2], colors[1::2]))
# cdict = {
#     'ref': 'k',
#     'field+': color_pairs[1][1],
#     'field-': color_pairs[1][0],
#     'gradient+': color_pairs[2][1],
#     'gradient-': color_pairs[2][0],
#     'LF': 'dimgrey',
#     'HF': 'darkgrey'
# }

colors = plt.get_cmap('tab10').colors
cdict = {
    'ref': 'k',
    'gradient+': colors[0],
    'gradient-': colors[1],
    'field+': colors[2],
    'field-': colors[3],
    'LF': colors[4],
    'HF': colors[5]
}

labels = ['ref'] + list(other_amps_pairs.keys()) + list(other_freqs.keys())
chandles = [mlines.Line2D([], [], color=cdict[k], label=k) for k in labels]

if __name__ == '__main__':

    args = getCommandLineArguments()
    figs = {}

    # Create figure backbone
    fig = plt.figure(constrained_layout=True, figsize=(9, 7))
    fig.suptitle('passive benchmark', fontsize=fs)
    gs = fig.add_gridspec(nrows, 2 * cbar_shrink_xratio + 1)
    subplots = {
        'map': [gs[i, :cbar_shrink_xratio] for i in range(nrows)],
        'cbar': [gs[i, cbar_shrink_xratio] for i in range(nrows)],
        'threshold': [gs[i, cbar_shrink_xratio + 1:] for i in range(nrows)]
    }
    axes = {k: [fig.add_subplot(x) for x in v] for k, v in subplots.items()}
    for ax, k in zip(axes['threshold'], levels.keys()):
        ax.set_xlabel('taum (s)', fontsize=fs)
        ax.set_ylabel('tauax (s)', fontsize=fs)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'threshold curves - {k}', fontsize=fs)
        ldg = ax.legend(handles=chandles, fontsize=fs, frameon=False,
                        loc='center left', bbox_to_anchor=(1.0, 0.5))
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)
    figs['passive_maps'] = fig

    # Reference condition
    drives = AcousticDriveArray([AcousticDrive(Fdrive, A) for A in ref_amps])
    # Create passive benchmark object
    subdir = f'passive_{"_".join(drives.filecodes.values())}'
    outdir = os.path.join(benchmarksroot, subdir)
    benchmark = PassiveBenchmark(a, nnodes, Cm0, ELeak, outdir=outdir)
    # Run simulations over time constants sparse 2D space and plot resulting signals
    # results = benchmark.runSimsOverTauSpace(drives, covs, *tau_ranges['sparse'], mpi=args.mpi)
    # figs['passive_signals'] = benchmark.plotSignalsOverTauSpace(*tau_ranges['sparse'], results)
    # Create divmap objects over time constants dense 2D space
    divmaps = {
        evmode: PassiveDivergenceMap(benchmark, *tau_ranges['dense'], [drives, covs], evmode, [])
        for evmode in levels.keys()}
    # Run missing simulations/evaluations if any of the divmaps are not complete
    if not all(x.isFinished() for x in divmaps.values()):
        benchmark.runSimsOverTauSpace(drives, covs, *tau_ranges['dense'], mpi=args.mpi)
    # Render divmaps
    for i, (evmode, divmap) in enumerate(divmaps.items()):
        if not divmap.isFinished():
            divmap.run()
        divmap.render(
            ax=axes['map'][i], cbarax=axes['cbar'][i], cbarlabel='horizontal', fs=fs,
            title=f'divmap - {evmode}', zscale=zscale[evmode], zbounds=None,
            T=1 / Fdrive, levels=levels[evmode], interactive=True)
        axes['threshold'][i].contour(
            *tau_ranges['dense'], divmap.getOutput() * divmap.zfactor,
            levels[evmode], colors=[cdict['ref']])

    # Other amplitude pairs
    for k, amps in other_amps_pairs.items():
        drives = AcousticDriveArray([AcousticDrive(Fdrive, A) for A in amps])
        findDivmapContours([a, nnodes, Cm0, ELeak], drives, covs, tau_ranges['dense'],
                           levels, axes['threshold'], mpi=args.mpi)

    # Other carrier frequencies
    for k, v in other_freqs.items():
        drives = AcousticDriveArray([AcousticDrive(v, A) for A in ref_amps])
        findDivmapContours([a, nnodes, Cm0, ELeak], drives, covs, tau_ranges['dense'],
                           levels, axes['threshold'], mpi=args.mpi)

    # Other phases
    amps = other_amps_pairs['gradient-']
    for delta_phi in delta_phis:
        phis = (np.pi - delta_phi, np.pi)
        drives = AcousticDriveArray([
            AcousticDrive(Fdrive, A, phi) for A, phi in zip(amps, phis)])
        findDivmapContours([a, nnodes, Cm0, ELeak], drives, covs, tau_ranges['dense'],
                           levels, axes['threshold'], mpi=args.mpi, pltmaps=True)

    # Add periodicity lines
    for ax in axes['threshold']:
        PassiveDivergenceMap.addInsets(ax, fiberinsets, fs)
        PassiveDivergenceMap.addPeriodicityLines(
            ax, 1 / Fdrive, color=cdict['ref'], pattern='upper-square')
        for fk, fv in other_freqs.items():
            PassiveDivergenceMap.addPeriodicityLines(
                ax, 1 / fv, color=cdict[fk], pattern='upper-square')

    # Save figures if specified
    if args.save:
        saveFigs(figs)

    plt.show()
