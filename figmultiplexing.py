# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-09 13:30:58
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-22 16:19:09

import numpy as np
import logging
import matplotlib.pyplot as plt

from PySONIC.core import NeuronalBilayerSonophore, AcousticDrive, Batch
from PySONIC.core import getPulseTrainProtocol, PulsedProtocol, ProtocolArray
from PySONIC.utils import logger, si_format, loadData
from PySONIC.plt import XYMap
from ExSONIC.sources import GaussianAcousticSource
from ExSONIC.plt import spatioTemporalMap

from utils import getSubRoot, getCommandLineArguments, saveFigs, getNPulses, getFiber

logger.setLevel(logging.INFO)

mproot = getSubRoot('multiplexing')


def runSimAndSave(fiber, source, PRFs, PDs, factors, min_npulses, root):
    ''' Run simulation for a given combination of PRFs. '''
    npulses = getNPulses(min_npulses, PRFs)
    # Construct and combine pulsed protocols
    dual_pp = ProtocolArray(
        [f * getPulseTrainProtocol(PD, npls, PRF)
         for f, PD, npls, PRF in zip(factors, PDs, npulses, PRFs)],
        minimize_overlap=True)
    # Run simulation and save file, return filepath
    return fiber.simAndSave(source, dual_pp, outputdir=root, overwrite=False, full_output=False)


class DualProtocolFiringRateMap(XYMap):

    xkey = 'PRF1'
    xfactor = 1e0
    xunit = 'Hz'
    ykey = 'PRF2'
    yfactor = 1e0
    yunit = 'Hz'
    zkey = 'FR'
    zunit = 'Hz'
    zfactor = 1e0
    suffix = 'DualProtocolFRmap'

    def __init__(self, fiber, source, PDs, factors, PRF1, PRF2, min_npulses, root='.',):
        self.fiber = fiber
        self.source = source
        self.PDs = PDs
        self.factors = factors
        self.min_npulses = min_npulses
        super().__init__(root, PRF1, PRF2)

    @property
    def sourcecode(self):
        return f'{self.source.key}_{"_".join(self.source.filecodes.values())}'

    def corecode(self):
        PDstr = '_'.join([f'{si_format(x, 1, space="")}s' for x in self.PDs])
        fstr = '_'.join([f'{x:.2f}' for x in self.factors])
        dualcode = f"PDs{PDstr}_factors{fstr}"
        return f'FRmap_{self.fiber.modelcode}_{self.sourcecode}_{dualcode}'

    @property
    def title(self):
        PDstr = ', '.join([f'{si_format(x, 1)}s' for x in self.PDs])
        fstr = ', '.join([f'{x:.2f}' for x in self.factors])
        dualcode = f"PDs = ({PDstr}), factors = ({fstr})"
        return f'Firing rate map - {self.fiber}, {self.source}, {dualcode}'

    def compute(self, x):
        ''' Run simulation and return firing rate detected on end node. '''
        fpath = runSimAndSave(
            self.fiber, self.source, x, self.PDs, self.factors, self.min_npulses, self.root)
        data, _ = loadData(fpath)
        return self.fiber.getEndFiringRate(data)

    def onClick(self, event):
        x = self.getOnClickXY(event)
        fpath = runSimAndSave(
            self.fiber, self.source, x, self.PDs, self.factors, self.min_npulses, self.root)
        data, _ = loadData(fpath)
        ftype = 'MY' if self.fiber.is_myelinated else 'UN'
        spatioTemporalMap(self.fiber, self.source, data, 'Qm', fontsize=fontsize,
                          cmap=cmaps[ftype], zbounds=Qbounds[ftype])
        plt.show()


# Fiber objects
a = 32e-9       # m
fs = 0.8        # (-)


fibers = {'UN': getFiber('UN'), 'MY': getFiber('MY')}

# US parameters
Fdrive = 500e3  # Hz
w = 2e-3  # FWHM (m)
sigma = GaussianAcousticSource.from_FWHM(w)  # m

# Pulsing parameters
PDs = {'UN': 10e-3, 'MY': 0.1e-3}  # s
PDlist = list(PDs.values())
min_npulses = 10
PRF_bounds = [1e1, 1e2]
nperax = 10
PRF_ranges = {k: np.logspace(*np.log10([10., min(1 / PD, 1000.)]), nperax) for k, PD in PDs.items()}
PRFqueue = Batch.createQueue(*PRF_ranges.values())

# Plot parameters
Qbounds = {'UN': (-80, 36), 'MY': (-175, 75)}  # nC/cm2
cmaps = {'UN': 'Blues', 'MY': 'Oranges'}
subset_colors = {'UN': 'C0', 'MY': 'C1'}
fontsize = 10

# Get fiber-specific single-pulse threshold excitation amplitudes
# for their respective pulse durations
drive = AcousticDrive(Fdrive)
Athrs = {k: NeuronalBilayerSonophore(a, fiber.pneuron).titrate(
    drive, PulsedProtocol(PDs[k], 10e-3), fs=fs) for k, fiber in fibers.items()}
s = 'fiber-specific thresholds: '
s += ', '.join([f'A_{k} = {v * 1e-3:.1f} kPa'for k, v in Athrs.items()])
s += f' (ratio = {max(Athrs.values()) / min(Athrs.values()):.2f})'
logger.info(s)

# Determine acoustic source amplitude and pulsed protocols modulation factors
# from single-pulse thresholds
Amin = min(Athrs.values())
Adrive = 1.1 * Amin
factors = {k: v / Amin for k, v in Athrs.items()}
flist = list(factors.values())
source = GaussianAcousticSource(0., sigma, Fdrive, Adrive)

if __name__ == '__main__':

    args = getCommandLineArguments()
    figs = {}

    # Particular example:
    # Construct multiplexed protocol
    PRFs = {'UN': 50., 'MY': 200.}  # Hz
    npulses = dict(zip(PRFs.keys(), getNPulses(min_npulses, PRFs.values())))
    dual_pp = ProtocolArray(
        [f * getPulseTrainProtocol(PDs[k], npulses[k], PRFs[k]) for k, f in factors.items()],
        minimize_overlap=True)
    # Run simulations and plot spatio-temporal maps for each fiber
    FRs = {}
    for k, fiber in fibers.items():
        fpath = fiber.simAndSave(
            source, dual_pp, outputdir=mproot, overwrite=False, full_output=False)
        data, _ = loadData(fpath)
        fig = spatioTemporalMap(
            fiber, source, data, 'Qm', fontsize=fontsize, rasterized=True,
            zbounds=Qbounds[k], cmap=cmaps[k])
        FRs[k] = fiber.getEndFiringRate(data)
    s = 'induced firing rates: '
    s += ', '.join([f'FR_{k} = {v:.1f} Hz'for k, v in FRs.items()])
    logger.info(s)

    # # # For each fiber
    # for k, fiber in fibers.items():
    #     # Initialize dual protocol firing rate map
    #     frmap = DualProtocolFiringRateMap(
    #         fiber, source, PDlist, flist, *PRF_ranges.values(), min_npulses,
    #         root=mproot)
    #     # If map is not complete, run simulations over PRF 2D space and save files, then run map
    #     if not frmap.isFinished():
    #         queue = [[k, *x] for x in PRFqueue]
    #         def simfunc(k, *PRFs):
    #             return runSimAndSave(getFiber(k), source, PRFs, PDlist, flist, min_npulses, mproot)
    #         batch = Batch(simfunc, queue)
    #         batch.run(loglevel=logger.getEffectiveLevel(), mpi=args.mpi)
    #     # Render map
    #     frmap.run()
    #     fig = frmap.render(
    #         cmap=cmaps[k], interactive=True, title=k, xscale='log', yscale='log', zscale='log')
    #     figs[frmap.corecode()] = fig

    if args.save:
        saveFigs(figs)
    plt.show()
