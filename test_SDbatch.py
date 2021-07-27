# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-05 14:05:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-27 18:28:56

import numpy as np
import logging
import matplotlib.pyplot as plt

from PySONIC import PulsedProtocol, logger, AcousticDrive, CompTimeSeries
from MorphoSONIC.core import UnmyelinatedFiber, Node
from MorphoSONIC.batches import StrengthDurationBatch

logger.setLevel(logging.INFO)

# Fiber model
aref = 32e-9
fsref = 0.8
fiber = UnmyelinatedFiber(0.8e-6, fiberL=5e-3, a=aref, fs=fsref)

# Acoustic field
Fdrive = 500e3  # Hz
drive = AcousticDrive(Fdrive)
drive.key = 'A'

# Durations and offset
durations = np.logspace(-5, 0, 20)  # s
toffset = 10e-3  # s
pp = PulsedProtocol(durations[0], toffset)

a = 16e-9
coverages = np.linspace(0.0, 1.0, 21)[1:]  # (-)

fs = coverages[-16]
node = Node(fiber.pneuron, a=aref, fs=fs)

sd_batch = StrengthDurationBatch('A (Pa)', drive, node, durations, toffset, root='.')
thrs = sd_batch.run()

outs = []
for tstim, Athr in zip(durations, thrs):
    if not np.isnan(Athr):
        outs.append(node.simulate(drive.updatedX(Athr), PulsedProtocol(tstim, toffset)))

comp_plot = CompTimeSeries(outs, 'Qm')
fig = comp_plot.render()

plt.show()

print('ok')
