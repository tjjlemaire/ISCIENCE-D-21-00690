# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-21 13:50:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-24 19:40:20

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from PySONIC.utils import logger
from PySONIC.core import NeuronalBilayerSonophore, AcousticDrive
from PySONIC.core.protocols import getPulseTrainProtocol, PulsedProtocol, ProtocolArray
from ExSONIC.containers import circleContour, Bundle
from ExSONIC.sources import GaussianAcousticSource

from utils import getSubRoot, getCommandLineArguments, saveFigs, getNPulses, getFiber

logger.setLevel(logging.INFO)

bundle_root = getSubRoot('bundle')

# Distributions parameters per fiber type
fiberD_dist_params = {
    'MY': {
        'mean': 6.74e-6,  # um
        'std': 2e-6,      # um
        'min': 3.7e-6,    # um
        'max': 13.7e-6    # um
    },
    'UN': {
        'mean': 0.8e-6,   # um
        'std': 0.6e-6,  # um
        'max': 1.5e-6,    # um
        'min': 0.2e-6   # um
    }
}

# Distribution histogram per fiber type
fiberD_ref_hist = {}
for k, d in fiberD_dist_params.items():
    data = norm.rvs(size=1000, loc=d['mean'], scale=d['std'])
    data = data[data > d['min']]
    data = data[data < d['max']]
    fiberD_ref_hist[k] = np.histogram(data, bins=100)

# Generic sonophore parameters
a = 32e-9  # m
fs = 0.8  # (-)

# US parameters
Fdrive = 500e3  # Hz
w = 2e-3  # FWHM (m)
sigma = GaussianAcousticSource.from_FWHM(w)  # m

# Bundle parameters
n = 8
bundle_contours = circleContour(100e-6, n=n)
length = 10e-3  # m
# f_kwargs = dict(fiberD_hists=fiberD_ref_hist, pratio=0.3, a=a, fs=fs)
f_kwargs = dict(fiberD_hists=fiberD_ref_hist, pratio=0.01, a=a, fs=fs)

# # Pulsing parameters
# min_npulses = 10
# PDs = {'UN': 10e-3, 'MY': 0.1e-3}  # s
# PRFs = {'UN': 50., 'MY': 200.}  # Hz
# npulses = dict(zip(PRFs.keys(), getNPulses(min_npulses, PRFs.values())))

# # Get fiber-specific single-pulse threshold excitation amplitudes
# # for their respective pulse durations
# fibers = {k: getFiber(k, a=a, fs=fs, fiberL=length) for k in ['UN', 'MY']}
# drive = AcousticDrive(Fdrive)
# Athrs = {k: NeuronalBilayerSonophore(a, fiber.pneuron).titrate(
#     drive, PulsedProtocol(PDs[k], 10e-3), fs=fs) for k, fiber in fibers.items()}
# s = 'fiber-specific thresholds: '
# s += ', '.join([f'A_{k} = {v * 1e-3:.1f} kPa'for k, v in Athrs.items()])
# s += f' (ratio = {max(Athrs.values()) / min(Athrs.values()):.2f})'
# logger.info(s)

# # Determine acoustic source amplitude and pulsed protocols modulation factors
# # from single-pulse thresholds
# Amin = min(Athrs.values())
# Adrive = 1.1 * Amin
# factors = {k: v / Amin for k, v in Athrs.items()}
# flist = list(factors.values())
# source = GaussianAcousticSource(0., sigma, Fdrive, Adrive)
# pp = ProtocolArray(
#     [f * getPulseTrainProtocol(PDs[k], npulses[k], PRFs[k]) for k, f in factors.items()],
#     minimize_overlap=True)

source = GaussianAcousticSource(0., sigma, Fdrive, 100e3)
pp = PulsedProtocol(1e-3, 10e-3)


if __name__ == '__main__':

    args = getCommandLineArguments()
    figs = {}

    # Bundle model
    bundle = Bundle.get(bundle_contours, length, root=bundle_root, **f_kwargs)
    figs['diams'] = bundle.plotDiameterDistribution()
    figs['cross-section'] = bundle.plotCrossSection()
    figs['offsets'] = bundle.plotLongitudinalOffsets()

    # def simfunc(fiber, pos):
    #     ''' Simulation function to apply to all bundle fibers. '''
    #     source.x0 = - pos[0]
    #     return fiber.simAndSave(
    #         source, pp, outputdir=bundle_root, overwrite=False, full_output=False)

    # # Apply simulation to all fibers
    # fpaths = bundle.forall(simfunc, mpi=args.mpi)
    # figs['raster'] = bundle.rasterPlot(fpaths)

    # f_offsets = {
    #     'f1': [150e-6, 0.],
    #     'f2': [-150e-6, 0.]
    # }
    # f_contours = {k: bundle_contours + np.tile(v, (n, 1)) for k, v in f_offsets.items()}
    # nerve_contours = circleContour(300e-6, n=n)

    # # Nerve model
    # fpath = 'nerve.pkl'
    # if os.path.isfile(fpath):
    #     nerve = Nerve.fromPickle(fpath)
    # else:
    #     nerve = Nerve(nerve_contours, length, f_contours=f_contours,
    #                   f_kwargs={k: f_kwargs for k in f_contours.keys()})
    #     nerve.toPickle(fpath)
    # fig1 = nerve.plotDiameterDistribution()
    # fig2 = nerve.plotCrossSection()

    if args.save:
        saveFigs(figs)
    plt.show()
