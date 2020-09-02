#!/bin/sh

# BLT requires a hard-coded path for sphix, but sometimes
# virtual environments are helpful. This script provides 
# an intermediate so that the sphinx used to generate
# documentation is chosen at build time.


python -m sphinx $@
