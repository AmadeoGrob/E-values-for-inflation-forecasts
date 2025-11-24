# R Package Dependencies for MSc Thesis Project
# This file lists all required R packages for the DRF modeling and analysis components

# Core tidyverse packages
library(readr)     # Reading CSV files
library(dplyr)     # Data manipulation
library(tidyr)     # Data tidying
library(ggplot2)   # Plotting and visualization
library(lubridate) # Date/time manipulation

# Project navigation and file management
library(here)      # Project-relative paths
library(fs)        # File system operations

# Statistical modeling and forecasting
library(drf)       # Distributional Random Forests, load from: https://github.com/lorismichel/drf
library(zoo)       # Time series objects

# Forecast evaluation and scoring
library(scoringRules)  # Proper scoring rules for forecast evaluation
library(epit)          # Statistical testing (e-values), load from: https://github.com/AlexanderHenzi/epit

# Additional analysis packages
library(patchwork)     # Combining ggplot2 plots
library(KernSmooth)    # Kernel density estimation
library(bde)           # Bandwidth selection for density estimation
library(MASS)          # Statistical functions

# Typography and presentation
library(sysfonts)      # Font management
library(showtext)      # Text rendering

# Installation commands for all required packages:
# install.packages(c(
#   "readr", "dplyr", "tidyr", "ggplot2", "lubridate",
#   "here", "fs", "drf", "zoo", "scoringRules", "epit",
#   "patchwork", "KernSmooth", "bde", "MASS",
#   "sysfonts", "showtext"
# ))

# Note: The 'drf' package may require additional system dependencies.
# On Ubuntu/Debian: sudo apt-get install libudunits2-dev libgdal-dev
# On macOS: brew install udunits gdal