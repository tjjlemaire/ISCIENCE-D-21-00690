# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-08 14:56:14
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-22 16:22:32

import os
import numpy as np
from argparse import ArgumentParser
from config import dataroot
from ExSONIC.models import SennFiber, UnmyelinatedFiber


def getSubRoot(subdir):
    subroot = os.path.join(dataroot, subdir)
    if not os.path.exists(subroot):
        os.mkdir(subroot)
    return subroot


def getCommandLineArguments():
    parser = ArgumentParser()
    parser.add_argument(
        '-s', '--save', default=False, action='store_true', help='Save figure')
    parser.add_argument(
        '--mpi', default=False, action='store_true', help='Use multiprocessing')
    parser.add_argument(
        '-d', '--details', default=False, action='store_true', help='Plot detailed figures')
    return parser.parse_args()


def saveFigs(figs):
    figroot = os.path.join(dataroot, 'figs')
    if not os.path.exists(figroot):
        os.mkdir(figroot)
    for k, fig in figs.items():
        fig.savefig(os.path.join(figroot, f'{k}.pdf'), transparent=True)


def getAxesFromGridSpec(fig, gs):
    axes = {}
    for k, v in gs.items():
        if isinstance(v, dict):
            axes[k] = getAxesFromGridSpec(fig, v)
        elif isinstance(v, list):
            axes[k] = [fig.add_subplot(x) for x in v]
        else:
            axes[k] = fig.add_subplot(v)
    return axes


def getNPulses(min_npulses, PRFs):
    # Compute tstim to ensure a minimum number of pulses with the lowest PRF
    tstim = min_npulses / min(PRFs)
    # Compute the corresponding number of pulses with each PRF
    return [int(np.ceil(tstim * PRF)) - 1 for PRF in PRFs]


def getFiber(k, a=32e-9, fs=0.8, fiberL=10e-3):
    ''' Generate fiber model '''
    if k == 'UN':
        return UnmyelinatedFiber(0.8e-6, fiberL=fiberL, a=a, fs=fs)
    elif k == 'MY':
        return SennFiber(10e-6, fiberL=fiberL, a=a, fs=fs)
    raise ValueError(f'invalid fiber key: {k}')
