#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# for each curve:
# Get threshold voltage from the other file
# Make linear fit for sat region to find early voltage and isat
# Make linear(?) fit for ohmic region
# combine slopes to get intrinsic gain
