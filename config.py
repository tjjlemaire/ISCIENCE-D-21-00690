# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-08 14:42:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-09 00:15:05

import platform

os_name = platform.system()
if os_name == 'Windows':
    dataroot = 'C:\\Users\\lemaire\\Documents\\papers data\\morphoSONIC'
elif os_name == 'Linux':
    dataroot = '../../data/morphoSONIC/'
else:
    dataroot = None
