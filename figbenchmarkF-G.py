# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-05-14 19:42:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-18 18:37:15

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.core import NeuronalBilayerSonophore, AcousticDrive, PulsedProtocol
from PySONIC.multicomp import FiberBenchmark
from PySONIC.utils import logger, si_format
from PySONIC.plt import FiberDivergenceMap
from PySONIC.postpro import detectSpikes
from ExSONIC.models import SennFiber, UnmyelinatedFiber

from utils import getSubRoot, getCommandLineArguments, saveFigs

logger.setLevel(logging.INFO)

benchmarksroot = getSubRoot('benchmarks')

# Coupled sonophores model parameters
nnodes = 2
a = 32e-9  # m
covs = [0.8] * nnodes  # (-)

# Carrier frequency
Fdrive = 500e3  # Hz

# Fibers
fibers = {
    'MY': SennFiber(10e-6, 2),
    'UN': UnmyelinatedFiber(0.8e-6, nnodes=2)
}

# Fiber-specific pulse durations
tstims = {'MY': 1e-3, 'UN': 1e-2}

# Amplitude ranges
densification_factor = 4
namps = {'sparse': 5}
namps['dense'] = (namps['sparse'] - 1) * densification_factor + namps['sparse']
Ascale = 'log'

# Runtime simulation parameters
mpi = False

# Plot parameters
flip = {'MY': False, 'UN': True}
rel_gamma = 0.25
levels = [1.2]
zscale = 'lin'
zbounds = (0., 6.)

if __name__ == '__main__':

    args = getCommandLineArguments()
    figs = {}

    for k, fiber in fibers.items():
        # Determine amplitude range between half and double the single-node SONIC threshold
        nbls = NeuronalBilayerSonophore(a, fiber.pneuron)
        drive = AcousticDrive(Fdrive)
        pp = PulsedProtocol(tstims[k], 0.)
        Athr = nbls.titrate(drive, pp, fs=covs[0])
        Abounds = (Athr / 2, 2 * Athr)
        Aranges = {k: np.logspace(*np.log10(Abounds), v) for k, v in namps.items()}  # Pa

        # Simulate single node at 1.1 times threshold and determine gamma evaluation
        # parameters from extracted spike properties
        data, _ = nbls.simulate(drive.updatedX(1.1 * Athr), pp, fs=covs[0])
        _, properties = detectSpikes(data, key='Qm')
        spikewidth = properties['widths'][0]  # s
        spikeprom = properties['prominences'][0]  # C/cm2
        gamma_args = [rel_gamma * spikewidth, rel_gamma * spikeprom]

        # Create fiber benchmark object
        subdir = f'{fiber.modelcode}_f_{si_format(Fdrive, space="")}Hz'
        outdir = os.path.join(benchmarksroot, subdir)
        benchmark = FiberBenchmark(a, nnodes, fiber.pneuron, fiber.ga_node_to_node, outdir=outdir)

        # Run simulations over the amplitude sparse 2D space and plot resulting signals
        # results = benchmark.runSimsOverAmplitudeSpace(
        #     Fdrive, tstims[k], covs, Aranges['sparse'], mpi=args.mpi)
        # for pltfunc in ['plotQm', 'plotGamma']:
        #     key = f'{k}-{pltfunc[4:]}-signals'
        #     figs[key] = benchmark.plotSignalsOverAmplitudeSpace(
        #         Aranges['sparse'], results, *gamma_args, pltfunc=pltfunc)
        #     figs[key].suptitle(key)

        # Run simulations and plot divergence maps over amplitude dense 2D space
        divmap = FiberDivergenceMap(
            benchmark, Aranges['dense'], [Fdrive, tstims[k], covs], 'gamma', gamma_args)
        divmap.run(mpi=args.mpi)
        figs[f'{k}-divmap'] = divmap.render(
            zscale=zscale, zbounds=zbounds, levels=levels, flip=True,
            Ascale=Ascale)

    # Save figures if specified
    if args.save:
        saveFigs(figs)

    plt.show()
