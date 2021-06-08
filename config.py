# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-06-08 14:42:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-06-08 14:42:03

import platform

os_name = platform.system()
if os_name == 'Windows':
    dataroot = 'C:\\Users\\lemaire\\Documents\\SONIC paper data'
elif os_name == 'Linux':
    dataroot = '../../data/SONIC paper data/'
else:
    dataroot = None
