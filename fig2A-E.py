# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-05-14 19:42:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 18:29:06

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from PySONIC.core import AcousticDrive, AcousticDriveArray
from PySONIC.multicomp import PassiveBenchmark
from PySONIC.utils import logger, frac_format
from PySONIC.plt import PassiveDivergenceMap
from MorphoSONIC.models import SennFiber, UnmyelinatedFiber

from utils import getSubRoot, getCommandLineArguments, saveFigs, getAxesFromGridSpec

logger.setLevel(logging.INFO)

benchmarksroot = getSubRoot('benchmarks')


def renderDivmaps(model_args, drives, covs, tau_ranges, levels, axes, color,
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
    # For each divergence metrics
    for i, (evmode, divmap) in enumerate(divmaps.items()):
        divmap.run(mpi=mpi)
        # Render divergence area on associated variations axis (with taum extrapolation)
        divmap.render(
            ax=axes[i], fs=fs, zscale=zscale[evmode], zbounds=zbounds[evmode],
            levels=levels[evmode], render_mode='divarea', xextra=taum_extra, ccolor=color,
            minimal=True, title='')
        # If details enabled, render also on detailed map figure
        if mapaxes[i] is not None:
            divmap.render(
                fs=fs, title=label, zscale=zscale[evmode], zbounds=zbounds[evmode],
                ax=mapaxes[i], T=1 / drives[0].f, levels=levels[evmode],
                interactive=True, minimal=True, plt_cbar=False)
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

# Stimulus drive: reference condition and variations
Fdrive = 500e3  # Hz
ref_amps = [100e3, 50e3]  # Pa
alpha = 2
rel_delta_phis = np.arange(1, 5) / 4
variations = {
    'field': {
        'field-': [ref_amps[0] / alpha, ref_amps[1] / alpha],
        'field+': [ref_amps[0] * alpha, ref_amps[1] * alpha]
    },
    'freq': {
        'LF': 20e3,
        'HF': 4e6
    },
    'grad': {
        'grad-': [(ref_amps[0] + ref_amps[1]) / 2] * 2,
        'grad+': [ref_amps[0] * alpha, ref_amps[1] / alpha],
    },
    'phase': {
        frac_format(x, "PI"): x * np.pi for x in rel_delta_phis  # rad
    }
}
nvariations = sum(len(x) for x in variations.values())

# Passive point-neuron model parameters
Cm0 = 1e-2   # F/m2
ELeak = -70  # mV

# Time constants
tau_bounds = (1e-7, 1e-3)  # s
densification_factor = 4
ntaus = {'sparse': 5}
ntaus['dense'] = (ntaus['sparse'] - 1) * densification_factor + ntaus['sparse']
tau_ranges = {k: np.logspace(*np.log10(tau_bounds), v) for k, v in ntaus.items()}  # s
# Extrapolation range in the taum direction
norders = int(np.diff(np.log10(tau_bounds)))
nperorder = (ntaus['dense'] - 1) // norders
taum_extra = np.power(10., np.log10(tau_ranges['dense'][-nperorder:]) + 1)
# Expansion into 2D tau space
tau_ranges = {k: [v, v] for k, v in tau_ranges.items()}

# Plot parameters
levels = {
    'ss': [1.],         # nC/cm2
    'transient': [10.]  # %
}
zscale = {'ss': 'log', 'transient': 'log'}
zbounds = {'ss': (1e-1, 1e1), 'transient': (1e-1, 1e2)}
fs = 12
paired_colors = list(plt.get_cmap('Paired').colors)
tetrad_colors = list(plt.get_cmap('tab20c').colors)
cdict = {
    'ref': 'k',
    'field': paired_colors[:2],
    'freq': paired_colors[2:4],
    'grad': paired_colors[4:6],
    'phase': tetrad_colors[4:8][::-1]
}


if __name__ == '__main__':

    args = getCommandLineArguments()
    figs = {}

    # Main maps figure
    x = len(variations) // 2
    nrows = x * len(levels)
    mapncols = 5
    fig = plt.figure(constrained_layout=True, figsize=(9, 7))
    fig.suptitle('passive benchmark', fontsize=fs)
    gs = fig.add_gridspec(nrows, 2 * x * mapncols + 1)
    subplots = {
        'map': [gs[x * i:x * (i + 1), :x * mapncols] for i in range(nrows // x)],
        'cbar': [gs[x * i:x * (i + 1), x * mapncols] for i in range(nrows // x)],
    }
    subplots['thr'] = {}
    j0 = x * mapncols + 1
    for icat, k in enumerate(variations.keys()):
        ioffset, joffset = icat // x, (icat % x) * mapncols
        jslice = slice(j0 + joffset, j0 + joffset + mapncols)
        subplots['thr'][k] = [gs[ioffset, jslice], gs[x + ioffset, jslice]]
    axes = getAxesFromGridSpec(fig, subplots)
    figs['passive_maps'] = fig
    chandles = [mlines.Line2D([], [], color=cdict['ref'], label='ref')]
    for k, axlist in axes['thr'].items():
        for ax in axlist:
            ax.set_aspect(1)
        for i, x in enumerate(variations[k].keys()):
            chandles.append(mlines.Line2D([], [], color=cdict[k][i], label=x))

    # Legend on separate figure
    figs['passive_benchmark_legend'], ax = plt.subplots(figsize=(1.5, 3))
    ax.set_xticks([])
    ax.set_yticks([])
    for sk in ['top', 'bottom', 'left', 'right']:
        ax.spines[sk].set_visible(False)
    ax.legend(handles=chandles, fontsize=fs, frameon=False, loc='center left')

    # Create figure for detailed maps
    ndetailedmaps = nvariations + 1
    other_divmaps = []
    if args.details:
        figs['detailed_passive_maps'], mapaxes = plt.subplots(
            2, ndetailedmaps, figsize=((ndetailedmaps * 1.5, 4)), constrained_layout=True)
        for ax, label in zip(mapaxes[:, 0], levels.keys()):
            ax.set_ylabel(label, fontsize=fs)
        for ax in mapaxes.flatten():
            ax.set_aspect(1.)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        mapaxes = np.array([[None] * ndetailedmaps, [None] * ndetailedmaps])
    imapcol = 0
    iaxthr = 0

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
    # For each divergence metrics
    for i, (evmode, divmap) in enumerate(divmaps.items()):
        # Run divmap
        divmap.run(mpi=args.mpi)
        # Render full map on main map axis
        divmap.render(
            ax=axes['map'][i], cbarax=axes['cbar'][i], cbarlabel='horizontal', fs=fs,
            title=f'divmap - {evmode}', zscale=zscale[evmode], zbounds=zbounds[evmode],
            T=1 / Fdrive, levels=levels[evmode], interactive=True)
        # Render divergence area on all variations axes (with taum extrapolation)
        for axlist in axes['thr'].values():
            divmap.render(
                ax=axlist[i], fs=fs, zscale=zscale[evmode], zbounds=zbounds[evmode], minimal=True,
                levels=levels[evmode], render_mode='divarea', xextra=taum_extra, ccolor='k',
                title='')
        # If details enabled, render also on detailed map figure
        if args.details:
            divmap.render(
                fs=fs, title='ref', zscale=zscale[evmode], zbounds=zbounds[evmode],
                ax=mapaxes[i, 0], T=1 / Fdrive, levels=levels[evmode],
                interactive=True, minimal=True, plt_cbar=False)
    imapcol += 1

    # Other field amplitudes
    for c, (k, v) in zip(cdict['field'], variations['field'].items()):
        drives = AcousticDriveArray([AcousticDrive(Fdrive, A) for A in v])
        other_divmaps.append(renderDivmaps(
            [a, nnodes, Cm0, ELeak], drives, covs, tau_ranges['dense'], levels,
            axes['thr']['field'], c, mpi=args.mpi, mapaxes=mapaxes[:, imapcol], label=k,
            taum_extra=taum_extra))
        imapcol += 1

    # Other carrier frequencies
    for c, (k, v) in zip(cdict['freq'], variations['freq'].items()):
        drives = AcousticDriveArray([AcousticDrive(v, A) for A in ref_amps])
        other_divmaps.append(renderDivmaps(
            [a, nnodes, Cm0, ELeak], drives, covs, tau_ranges['dense'], levels,
            axes['thr']['freq'], c, mpi=args.mpi, mapaxes=mapaxes[:, imapcol], label=k,
            taum_extra=taum_extra))
        imapcol += 1

    # Other field gradients
    for c, (k, v) in zip(cdict['grad'], variations['grad'].items()):
        drives = AcousticDriveArray([AcousticDrive(Fdrive, A) for A in v])
        other_divmaps.append(renderDivmaps(
            [a, nnodes, Cm0, ELeak], drives, covs, tau_ranges['dense'], levels,
            axes['thr']['grad'], c, mpi=args.mpi, mapaxes=mapaxes[:, imapcol], label=k,
            taum_extra=taum_extra))
        imapcol += 1

    # Other phases
    amps = variations['grad']['grad-']
    for c, (k, v) in zip(cdict['phase'], variations['phase'].items()):
        phis = (np.pi - v, np.pi)
        drives = AcousticDriveArray([
            AcousticDrive(Fdrive, A, phi) for A, phi in zip(amps, phis)])
        other_divmaps.append(renderDivmaps(
            [a, nnodes, Cm0, ELeak], drives, covs, tau_ranges['dense'], levels,
            axes['thr']['phase'], c, mpi=args.mpi, mapaxes=mapaxes[:, imapcol], label=k,
            taum_extra=taum_extra))
        imapcol += 1

    # Add periodicity lines
    for k, axlist in axes['thr'].items():
        for ax in axlist:
            PassiveDivergenceMap.addInsets(ax, fiberinsets, fs)
            PassiveDivergenceMap.addPeriodicityLines(
                ax, 1 / Fdrive, color=cdict['ref'], pattern='upper-square')
    for ax in axes['thr']['freq']:
        for i, (fk, fv) in enumerate(variations['freq'].items()):
            PassiveDivergenceMap.addPeriodicityLines(
                ax, 1 / fv, color=cdict['freq'][i], pattern='upper-square')

    # Save figures if specified
    if args.save:
        saveFigs(figs)

    plt.show()
