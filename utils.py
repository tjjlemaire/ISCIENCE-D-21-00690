# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-08 14:56:14
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-19 17:36:23

import os
import pickle
from argparse import ArgumentParser
from PySONIC.utils import logger
from config import dataroot


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


def loadData(fpath):
    logger.info('Loading data from "%s"', os.path.basename(fpath))
    with open(fpath, 'rb') as fh:
        frame = pickle.load(fh)
    return frame['data'], frame['meta']


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
