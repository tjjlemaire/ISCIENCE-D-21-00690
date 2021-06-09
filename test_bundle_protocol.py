# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-09 13:30:58
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-09 20:57:50
import logging

import matplotlib.pyplot as plt
from PySONIC.core import NeuronalBilayerSonophore, AcousticDrive
from PySONIC.core import getPulseTrainProtocol, PulsedProtocol
from PySONIC.utils import logger
from ExSONIC.core import SennFiber, UnmyelinatedFiber
from ExSONIC.core.sources import *
from ExSONIC.plt import spatioTemporalMap

from utils import getSubRoot, getCommandLineArguments, loadData

logger.setLevel(logging.DEBUG)

bundleroot = getSubRoot('bundle')

# Fiber objects
a = 32e-9       # m
fs = 0.8        # (-)
fibers = {
    # 'UN': UnmyelinatedFiber(0.8e-6, fiberL=10e-3, a=a, fs=fs),
    'MY': SennFiber(10e-6, 11, a=a, fs=fs),
}

# US parameters
Fdrive = 500e3  # Hz
w = 2e-3  # FWHM (m)
sigma = GaussianSource.from_FWHM(w)  # m

# Pulsing parameters
PRFs = {'UN': 20., 'MY': 50.}  # Hz
PDs = {'UN': 5e-3, 'MY': 0.1e-3}  # s
tstim = 110e-3  # s
npulses = {k: tstim * PRF - 1 for k, PRF in PRFs.items()}
DCs = {k: PDs[k] * PRF for k, PRF in PRFs.items()}  # (-)
tstarts = {k: 1 / PRF for k, PRF in PRFs.items()}

# Plot parameters
Qbounds = {'UN': (-80, 36), 'MY': (-175, 75)}  # nC/cm2
cmaps = {'UN': 'Blues', 'MY': 'Oranges'}
subset_colors = {'UN': 'C0', 'MY': 'C1'}
maponly = False
fontsize = 10

if __name__ == '__main__':

    args = getCommandLineArguments()

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
    source = GaussianAcousticSource(0., sigma, Fdrive, Adrive)

    # Construct and comboine pulsed protocols
    pps = {k: f * getPulseTrainProtocol(PDs[k], npulses[k], PRFs[k]) for k, f in factors.items()}
    pps['comb'] = sum(list(pps.values()))

    # # Plot protocols
    # ax = None
    # for k, pp in pps.items():
    #     fig = pp.plot(label=k, ax=ax, color=subset_colors.get(k, 'k'))
    #     if fig is not None:
    #         ax = fig.axes[0]

    # Run simulations and plot spatio-temporal maps
    for i, (k, fiber) in enumerate(fibers.items()):
        logger.info(f'{fiber}: length = {fiber.length * 1e3:.1f} mm')
        fpath = fiber.simAndSave(source, pps['comb'], outputdir=bundleroot, overwrite=False)
        data, meta = loadData(fpath)
        fig = spatioTemporalMap(
            fiber, source, data, 'Qm', fontsize=fontsize,
            zbounds=Qbounds[k], maponly=maponly, rasterized=True)

    plt.show()
