# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-05-14 19:42:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-17 23:03:23

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from PySONIC.core import AcousticDrive, AcousticDriveArray
from PySONIC.multicomp import PassiveBenchmark
from PySONIC.utils import logger, frac_format
from PySONIC.plt import PassiveDivergenceMap
from ExSONIC.models import SennFiber, UnmyelinatedFiber

from utils import getSubRoot, getCommandLineArguments, saveFigs

logger.setLevel(logging.DEBUG)

benchmarksroot = getSubRoot('benchmarks')


def getAmpPairs(A1, A2, alpha):
    field_mod = {
        'field-': [A1 / alpha, A2 / alpha],
        'field+': [A1 * alpha, A2 * alpha],
    }
    grad_mod = {
        'gradient-': [(A1 + A2) / 2] * 2,
        'gradient+': [A1 * alpha, A2 / alpha],
    }
    return field_mod, grad_mod


def plotDivergenceArea(ax, x, y, z, zthr, color):
    ax.contour(x, y, z.T, [zthr], colors=[color])
    # ax.contourf(x, y, z.T, [zthr, np.inf], colors=[color], alpha=0.2)


def findDivmapContours(model_args, drives, covs, tau_ranges, levels, axes,
                       mpi=False, mapaxes=None, label=None, taum_extra=None):
    # Create passive benchmark object
    subdir = f'passive_{"_".join(drives.filecodes.values())}'
    outdir = os.path.join(benchmarksroot, subdir)
    benchmark = PassiveBenchmark(*model_args, outdir=outdir)
    # Create divmap objects over time constants dense 2D space
    divmaps = {
        evmode: PassiveDivergenceMap(
            benchmark, *tau_ranges, [drives, covs], evmode, [])
        for evmode in levels.keys()}
    # Extract and plot threshold curves for relevant divergence levels
    for i, (evmode, divmap) in enumerate(divmaps.items()):
        if not divmap.isFinished():
            divmap.run(mpi=mpi)
        x, y, data = divmap.xvec, divmap.yvec, divmap.getOutput() * divmap.zfactor
        if taum_extra is not None:
            x, y, data = PassiveDivergenceMap.extrapolate(
                x, y, data, 'log', 'log', xextra=taum_extra)
        plotDivergenceArea(axes[i], x, y, data, levels[evmode][0], cdict[label])
        if mapaxes is not None:
            divmap.render(
                fs=fs, title=label, zscale=zscale[evmode], zbounds=zbounds[evmode], ax=mapaxes[i],
                T=1 / drives[0].f, levels=levels[evmode],
                interactive=True, minimal=True, plt_cbar=False)
            mapaxes[i].set_aspect(1.)
    return divmaps


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
other_fields, other_grads = getAmpPairs(*ref_amps, alpha=2)
rel_delta_phis = np.arange(1, 5) / 4
other_dphis = {f'd_phi = {frac_format(x, "PI")}': x * np.pi for x in rel_delta_phis}  # rad

# Passive point-neuron model parameters
Cm0 = 1e-2   # F/m2
ELeak = -70  # mV

# Time constants ranges
tau_bounds = (1e-7, 1e-3)  # s
densification_factor = 4
ntaus = {'sparse': 5}
ntaus['dense'] = (ntaus['sparse'] - 1) * densification_factor + ntaus['sparse']
tau_ranges = {k: np.logspace(*np.log10(tau_bounds), v) for k, v in ntaus.items()}  # s
# Extrapolation range in the taum direction
norders = int(np.diff(np.log10(tau_bounds)))
nperorder = (ntaus['dense'] - 1) // norders
taum_extra = np.power(10., np.log10(tau_ranges['dense'][-nperorder:]) + 1)
# taum_extra = None
# Expansion into 2D tau space
tau_ranges = {k: [v, v] for k, v in tau_ranges.items()}

# Plot parameters
levels = {
    'ss': [1.],        # nC/cm2
    'transient': [10.]  # %
}
zscale = {'ss': 'log', 'transient': 'log'}
zbounds = {'ss': (1e-1, 1e1), 'transient': (1e-1, 1e2)}
nrows = len(levels)
fs = 12
cbar_shrink_xratio = 10

colors = list(plt.get_cmap('Paired').colors)[:6] + list(plt.get_cmap('tab20c').colors)[4:8][::-1]
labels = ['ref']
for x in [other_fields, other_freqs, other_grads, other_dphis]:
    labels += list(x.keys())
cdict = dict(zip(labels, ['k'] + colors))

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

    # Create figure for detailed maps
    other_divmaps = []
    figs['detailed_passive_maps'], mapaxes = plt.subplots(
        2, len(labels), figsize=((len(labels) * 1.5, 4)),
        constrained_layout=True)
    for ax, label in zip(mapaxes[:, 0], levels.keys()):
        ax.set_ylabel(label, fontsize=fs)
    imapcol = 0

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
    # Render divmaps
    for i, (evmode, divmap) in enumerate(divmaps.items()):
        if not divmap.isFinished():
            divmap.run(mpi=args.mpi)
        divmap.render(
            ax=axes['map'][i], cbarax=axes['cbar'][i], cbarlabel='horizontal', fs=fs,
            title=f'divmap - {evmode}', zscale=zscale[evmode], zbounds=zbounds[evmode],
            T=1 / Fdrive, levels=levels[evmode], interactive=True)
        # x, y, data = divmap.xvec, divmap.yvec, divmap.getOutput() * divmap.zfactor
        # if taum_extra is not None:
        #     x, y, data = PassiveDivergenceMap.extrapolate(
        #         x, y, data, 'log', 'log', xextra=taum_extra)
        # plotDivergenceArea(axes['threshold'][i], x, y, data, levels[evmode][0], cdict['ref'])
    other_divmaps.append(findDivmapContours(
        [a, nnodes, Cm0, ELeak], drives, covs, tau_ranges['dense'], levels, axes['threshold'],
        mpi=args.mpi, mapaxes=mapaxes[:, imapcol], label='ref', taum_extra=taum_extra))
    imapcol += 1

    # Other field amplitudes
    for k, v in other_fields.items():
        drives = AcousticDriveArray([AcousticDrive(Fdrive, A) for A in v])
        other_divmaps.append(findDivmapContours(
            [a, nnodes, Cm0, ELeak], drives, covs, tau_ranges['dense'], levels, axes['threshold'],
            mpi=args.mpi, mapaxes=mapaxes[:, imapcol], label=k, taum_extra=taum_extra))
        imapcol += 1

    # Other carrier frequencies
    for k, v in other_freqs.items():
        drives = AcousticDriveArray([AcousticDrive(v, A) for A in ref_amps])
        other_divmaps.append(findDivmapContours(
            [a, nnodes, Cm0, ELeak], drives, covs, tau_ranges['dense'], levels, axes['threshold'],
            mpi=args.mpi, mapaxes=mapaxes[:, imapcol], label=k, taum_extra=taum_extra))
        imapcol += 1

    # Other field gradients
    for k, v in other_grads.items():
        drives = AcousticDriveArray([AcousticDrive(Fdrive, A) for A in v])
        other_divmaps.append(findDivmapContours(
            [a, nnodes, Cm0, ELeak], drives, covs, tau_ranges['dense'], levels, axes['threshold'],
            mpi=args.mpi, mapaxes=mapaxes[:, imapcol], label=k, taum_extra=taum_extra))
        imapcol += 1

    # Other phases
    amps = other_grads['gradient-']
    for k, v in other_dphis.items():
        phis = (np.pi - v, np.pi)
        drives = AcousticDriveArray([
            AcousticDrive(Fdrive, A, phi) for A, phi in zip(amps, phis)])
        other_divmaps.append(findDivmapContours(
            [a, nnodes, Cm0, ELeak], drives, covs, tau_ranges['dense'], levels, axes['threshold'],
            mpi=args.mpi, mapaxes=mapaxes[:, imapcol], label=k, taum_extra=taum_extra))
        imapcol += 1

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
