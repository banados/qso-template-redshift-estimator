# QSO Template Redshift Estimator

A tool for estimating redshift and magnitudes for high-redshift quasars by comparing observed spectra with template spectra.

This code was used in [Banados+2023 ApJS, 265, 29](https://ui.adsabs.harvard.edu/abs/2023ApJS..265...29B/abstract) and is inspired by earlier work in [Banados+2016, ApJ, 227, 11](https://ui.adsabs.harvard.edu/abs/2016ApJS..227...11B/abstract).

## Features

- Estimates redshift by comparing observed quasar spectra with various templates
- Calculates apparent and absolute magnitudes at specific rest-frame wavelengths (1450Å and 2500Å)
- Supports multiple quasar templates for comparison
- Creates detailed PDF visualizations of results
- Handles spectral masking and smoothing

## Installation

### Dependencies

- numpy
- matplotlib
- astropy
- linetools
- astrotools
- scipy

### Setup

Clone the repository:
```bash
git clone https://github.com/your-username/qso-template-redshift-estimator.git
cd qso-template-redshift-estimator
```

### Usage

## Command line
```
python qso_redshift_magnitude_estimator.py spectrum.fits [options]
```
## Examples
```
# Basic usage
python qso_redshift_magnitude_estimator.py example/P218+28_mods_zp1_Banados2023.spc

# Specify template and redshift range
python qso_redshift_magnitude_estimator.py spectrum.fits --template_spc selsing2016 --zmin 5.5 --zmax 7.0

# Mask specific wavelength ranges
python qso_redshift_magnitude_estimator.py spectrum.fits --mask 9000:9200,9300:9400

# Specify wavelength range for chi-squared calculation
python qso_redshift_magnitude_estimator.py example/P218+28_mods_zp1_Banados2023.spc --zmin 5.7 --zmax 6.1 --wmin 1212 --wmax 1400
```

## As a Module

```
from qso_redshift_magnitude_estimator import QSORedshiftEstimator

# Create an estimator
estimator = QSORedshiftEstimator('your_spectrum.fits', zmin=5.5, zmax=7.0)

# Calculate redshift
results = estimator.calculate_redshift()

# Get best redshift value
best_redshift = results[0]['zbest']

# Create visualization
pdf_file = estimator.create_visualization()
```

## Available Templates

* selsing2016
* yang2021
* weak-lya
* median-lya
* strong-lya
* vandenberk2001

## Command Line Options

```
--smooth INT       Smoothing factor for the observed spectrum
--errors           Use errors in chi-squared calculation and plotting (default)
--no-errors        Ignore errors in chi-squared calculation and plotting
--wmin FLOAT       Min wavelength to use for chi-squared estimation (default: 1240)
--wmax FLOAT       Max wavelength to use for chi-squared estimation (default: 1450)
--zmax FLOAT       Max redshift allowed (default: 6.5)
--zmin FLOAT       Min redshift allowed (default: 5.6)
--mask STR         Regions to mask, format: "x1:x2,x3:x4"
--template_spc STR Template spectrum to use ('all' by default)
--redshift_fixed FLOAT Fixed redshift to evaluate at
```

### Citation
If you use this code in your research, please cite:
```
@ARTICLE{2023ApJS..265...29B,
       author = {{Ba{\~n}ados}, Eduardo and {Mazzucchelli}, Chiara and {Venemans}, Bram P. and et al.},
        title = "{The Pan-STARRS1 Distant z > 5.6 Quasar Survey: Three Years of Observations, Three New z > 6.5 Quasars, and the Slow Evolution at the Highest Redshift}",
      journal = {\apjs},
         year = 2023,
        month = mar,
       volume = {265},
       number = {1},
          eid = {29},
        pages = {29},
          doi = {10.3847/1538-4365/acb59f},
}
```