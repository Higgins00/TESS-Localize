
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))



# Bibtex entry detailing how to cite the package
#include later

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

from .TESS_Localize import *

