# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-21 13:50:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-07-09 12:04:27

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pickle

from PySONIC.utils import logger, loadData
from PySONIC.core import NeuronalBilayerSonophore, AcousticDrive
from PySONIC.core.protocols import getPulseTrainProtocol, PulsedProtocol, ProtocolArray
from ExSONIC.containers import circleContour, Bundle
from ExSONIC.sources import GaussianAcousticSource

from utils import getSubRoot, getCommandLineArguments, saveFigs, getNPulses, getFiber

logger.setLevel(logging.INFO)

bundle_root = getSubRoot('bundle')

# Generic sonophore parameters
a = 32e-9  # m
fs = 0.8  # (-)

# US parameters
Fdrive = 500e3  # Hz
w = 2e-3  # FWHM (m)
sigma = GaussianAcousticSource.from_FWHM(w)  # m

# Bundle parameters, extracted from morphological data of human sural nerves
# Reference: Jacobs, J.M., and Love, S. (1985). QUALITATIVE AND QUANTITATIVE MORPHOLOGY OF
# HUMAN SURAL NERVE AT DIFFERENT AGES. Brain 108, 897–924.
diameter = 100e-6  # m
bundle_contours = circleContour(diameter / 2, n=100)
length = 10e-3  # m
# Histogram distributions of fiber diameters per fiber type
fiberD_hists = {
    'UN': (  # Jacobs 1985, fig. 11D
        np.array([7, 11.5, 9.5, 16, 18.5, 24, 8.5, 3.5, 1, 0.5]),                 # weights
        np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]) * 1e-6  # edges
    ),
    'MY': (  # Jacobs 1985, fig. 9C
        np.array([6, 22, 20, 9, 3, 5.5, 7, 8, 9, 7.5, 3]),         # weights
        np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) * 1e-6  # edges
    )
}
target_un_to_my_ratio = 4.0  # UN:MY fiber count ratio (Jacobs 1985, Table 1)
target_pratio = 0.30  # Packing ratio (chosen to ensure representative populations of each type)
f_kwargs = dict(
    fiberD_hists=fiberD_hists,
    target_pratio=target_pratio,
    target_un_to_my_ratio=target_un_to_my_ratio,
    a=a,
    fs=fs
)

# Pulsing parameters
min_npulses = 10
PDs = {'UN': 10e-3, 'MY': 0.1e-3}  # s
PRFs = {'UN': 50., 'MY': 200.}  # Hz
npulses = dict(zip(PRFs.keys(), getNPulses(min_npulses, PRFs.values())))

# Get fiber-specific single-pulse threshold excitation amplitudes
# for their respective pulse durations
fibers = {k: getFiber(k, a=a, fs=fs, fiberL=length) for k in ['UN', 'MY']}
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
pp = ProtocolArray(
    [f * getPulseTrainProtocol(PDs[k], npulses[k], PRFs[k]) for k, f in factors.items()],
    minimize_overlap=True)


if __name__ == '__main__':

    args = getCommandLineArguments()
    figs = {}

    # Bundle model
    bundle = Bundle.get(bundle_contours, length, root=bundle_root, **f_kwargs)
    figs['diams'] = bundle.plotDiameterDistribution()
    figs['cross-section'] = bundle.plotCrossSection()
    figs['xoffsets'] = bundle.plotLongitudinalOffsets()

    def simfunc(fiber, pos):
        ''' Simulation function to apply to all bundle fibers. '''
        source.x0 = - pos[0]
        return fiber.simAndSave(
            source, pp, outputdir=bundle_root, overwrite=False, full_output=False)

    # Apply simulation to all fibers
    # fpaths = bundle.forall(simfunc, mpi=args.mpi)
    # figs['raster'] = bundle.rasterPlot(fpaths)

    fname = f'{bundle.filecode()}_FRdata.pkl'
    fpath = os.path.join(bundle_root, fname)
    print(fpath)
    if os.path.exists(fpath):
        with open(fpath, 'rb') as fh:
            fr_data = pickle.load(fh)
        print(fr_data)
    # else:
    #     fr_data = {k: [] for k in ['MY', 'UN']}
    #     for i, ((fk, _), fpath) in enumerate(zip(bundle.fibers[::-1], fpaths)):
    #         k = {True: 'MY', False: 'UN'}[fk.is_myelinated]
    #         data, _ = loadData(fpath)
    #         fr_data[k].append(fk.getEndFiringRate(data))
    #     fr_data = {k: np.array(v) for k, v in fr_data.items()}
    #     with open(fpath, 'wb') as fh:
    #         pickle.dump(fr_data, fh)
    figs['bundle_frdist'] = bundle.plotFiringRateDistribution(fr_data)

    if args.save:
        saveFigs(figs)
    plt.show()
