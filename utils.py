# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-08 14:56:14
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-08 15:17:28

import os
from argparse import ArgumentParser

from config import dataroot


def getBenchmarksRoot():
    benchmarkroot = os.path.join(dataroot, 'benchmarks')
    if not os.path.exists(benchmarkroot):
        os.mkdir(benchmarkroot)
    return benchmarkroot


def getCommandLineArguments():
    parser = ArgumentParser()
    parser.add_argument(
        '-s', '--save', default=False, action='store_true', help='Save figure')
    parser.add_argument(
        '--mpi', default=False, action='store_true', help='Use multiprocessing')
    return parser.parse_args()


def saveFigs(figs):
    figroot = os.path.join(dataroot, 'figs')
    if not os.path.exists(figroot):
        os.mkdir(figroot)
    for k, fig in figs.items():
        fig.savefig(os.path.join(figroot, f'{k}.pdf'))
