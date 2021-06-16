# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-09 13:30:58
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-16 13:52:48

import numpy as np
import logging
import matplotlib.pyplot as plt
import time

from PySONIC.core import NeuronalBilayerSonophore, AcousticDrive, Batch
from PySONIC.core import getPulseTrainProtocol, PulsedProtocol, ProtocolArray
from PySONIC.utils import logger, si_format
from PySONIC.plt import XYMap
from ExSONIC.models import SennFiber, UnmyelinatedFiber
from ExSONIC.sources import GaussianAcousticSource
from ExSONIC.plt import spatioTemporalMap

from utils import getSubRoot, getCommandLineArguments, loadData

logger.setLevel(logging.INFO)

dproot = getSubRoot('dual protocol')


def getNPulses(min_npulses, PRFs):
    # Compute tstim to ensure a minimum number of pulses with the lowest PRF
    tstim = min_npulses / min(PRFs)
    # Compute the corresponding number of pulses with each PRF
    return [int(np.ceil(tstim * PRF)) - 1 for PRF in PRFs]


def runSimAndSave(fiber, source, PRFs, PDs, factors, min_npulses, root):
    ''' Run simulation for a given combination of PRFs. '''
    npulses = getNPulses(min_npulses, PRFs)
    # Construct and combine pulsed protocols
    dual_pp = ProtocolArray(
        [f * getPulseTrainProtocol(PD, npls, PRF)
         for f, PD, npls, PRF in zip(factors, PDs, npulses, PRFs)],
        minimize_overlap=True)
    # Run simulation and save file, return filepath
    return fiber.simAndSave(source, dual_pp, outputdir=root, overwrite=False)


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

    def __init__(self, fiber, source, PDs, factors, PRFs, min_npulses, root='.',):
        self.fiber = fiber
        self.source = source
        self.PDs = PDs
        self.factors = factors
        self.min_npulses = min_npulses
        super().__init__(root, PRFs, PRFs)

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
        spatioTemporalMap(self.fiber, self.source, data, 'Qm', fontsize=fontsize)
        plt.show()


# Fiber objects
a = 32e-9       # m
fs = 0.8        # (-)


def getFiber(k):
    ''' Generate fiber model '''
    if k == 'UN':
        return UnmyelinatedFiber(0.8e-6, fiberL=10e-3, a=a, fs=fs)
    elif k == 'MY':
        return SennFiber(10e-6, fiberL=10e-3, a=a, fs=fs)
    raise ValueError(f'invalid fiber key: {k}')


fibers = {'UN': getFiber('UN'), 'MY': getFiber('MY')}

# US parameters
Fdrive = 500e3  # Hz
w = 2e-3  # FWHM (m)
sigma = GaussianAcousticSource.from_FWHM(w)  # m

# Pulsing parameters
PDs = {'UN': 5e-3, 'MY': 0.1e-3}  # s
PDlist = list(PDs.values())
min_npulses = 10
PRF_bounds = [1e1, 1e2]
PRF_range = np.linspace(*PRF_bounds, 10)
PRFqueue = Batch.createQueue(PRF_range, PRF_range)

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

    # Construct, combine, and optimize pulsed protocols for particular example
    PRFs = {'UN': 40., 'MY': 100.}  # Hz
    npulses = dict(zip(PRFs.keys(), getNPulses(min_npulses, PRFs.values())))
    dual_pp = ProtocolArray(
        [f * getPulseTrainProtocol(PDs[k], npulses[k], PRFs[k]) for k, f in factors.items()],
        minimize_overlap=True)

    # # Run simulations and plot spatio-temporal maps for each fiber
    # FRs = {}
    # for k, fiber in fibers.items():
    #     fpath = fiber.simAndSave(source, dual_pp, outputdir=dproot, overwrite=False)
    #     data, _ = loadData(fpath)
    #     fig = spatioTemporalMap(
    #         fiber, source, data, 'Qm', fontsize=fontsize, rasterized=True,
    #         zbounds=Qbounds[k], cmap=cmaps[k])
    #     FRs[k] = fiber.getEndFiringRate(data)
    # print(FRs)

    # For each fiber
    for k, fiber in fibers.items():
        # Initialize dual protocol firing rate map
        frmap = DualProtocolFiringRateMap(
            fiber, source, PDlist, flist, PRF_range, min_npulses,
            root=dproot)
        # If map is not complete, run simulations over PRF 2D space and save files, then run map
        if not frmap.isFinished():
            queue = [[k, *x] for x in PRFqueue]
            def simfunc(k, *PRFs):
                return runSimAndSave(getFiber(k), source, PRFs, PDlist, flist, min_npulses, dproot)
            batch = Batch(simfunc, queue)
            batch.run(loglevel=logger.getEffectiveLevel(), mpi=args.mpi)
            frmap.run()
        # Render map
        fig = frmap.render(cmap=cmaps[k], interactive=True)
        figs[frmap.corecode] = fig

    plt.show()
