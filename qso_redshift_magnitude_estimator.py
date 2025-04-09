#!/usr/bin/env python
"""
QSO Redshift Estimator 

Code used to estimate redshift and magnitudes for high-redshift quasars in 
Banados+2023 ApJS, 265 29
https://ui.adsabs.harvard.edu/abs/2023ApJS..265...29B/abstract

Inspired by a rougher code used in Banados+2016, ApJ, 227, 11
https://ui.adsabs.harvard.edu/abs/2016ApJS..227...11B/abstract

This script estimates the redshift and magnitudes at specific rest-frame wavelengths 
for high-redshift quasars by comparing observed spectra with template spectra.

The code loops through different quasar templates, shifting them by redshift, 
and finds the best fit by minimizing chi-squared values.

Usage from command line:
    python qso_redshift_estimator.py spectrum.fits [options]

Examples:
    python qso_redshift_estimator.py spectrum.fits --template_spc selsing2016 --zmin 5.5 --zmax 7.0
    python qso_redshift_estimator.py spectrum.fits --mask 9000:9200,9300:9400
    python qso_redshift_magnitude_estimator.py example/P218+28_mods_zp1_Banados2023.spc --zmin 5.7 --zmax 6.1 --wmin 1212 --wmax 1400

But the code can also be used as a module.

# Import the module
from qso_redshift_estimator import QSORedshiftEstimator

# Create an estimator
estimator = QSORedshiftEstimator('your_spectrum.fits', zmin=5.5, zmax=7.0)

# Calculate redshift
results = estimator.calculate_redshift()

# Get best redshift value
best_redshift = results[0]['zbest']

# Create visualization
pdf_file = estimator.create_visualization()

"""
from __future__ import division, print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import ascii
from linetools.spectra.io import readspec, XSpectrum1D
# from astrotools import get_ind, between
from scipy.signal import medfilt
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM

# Set up cosmology
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)


def get_ind(val, arr):
    '''
    get a value and an array.
    Returns the index of the closest element of the array to the value
    '''
    return min(range(len(arr)), key=lambda i: abs(arr[i] - val))


def between(a, vmin, vmax):
    """ Return a boolean array True where vmin <= a < vmax.
    Notes
    -----
    Careful of floating point issues when dealing with equalities.
    """
    a = np.asarray(a)
    c = a < vmax
    c &= a >= vmin
    return c




class QSORedshiftEstimator:
    """
    A class to estimate redshift and magnitudes for high-redshift quasars
    by comparing observed spectra with template spectra.
    """
    
    def __init__(self, obs_spc_file, template_specs='all', 
                 wmin=1240., wmax=1450., zmin=5.6, zmax=6.5,
                 mask=None, smooth=None, use_errors=True, redshift_fixed=None):
        """
        Initialize the QSO redshift estimator.
        
        Parameters
        ----------
        obs_spc_file : str
            Path to the observed spectrum file
        template_specs : str or list, optional
            Template spectrum or list of template spectra to use
        wmin : float, optional
            Minimum wavelength to use for chi-squared calculation
        wmax : float, optional
            Maximum wavelength to use for chi-squared calculation
        zmin : float, optional
            Minimum redshift to consider
        zmax : float, optional
            Maximum redshift to consider
        mask : str, optional
            Wavelength regions to mask, format: "x1:x2,x3:x4"
        smooth : int, optional
            Smoothing factor for the observed spectrum
        use_errors : bool, optional
            Whether to use errors in chi-squared calculation
        redshift_fixed : float, optional
            Fixed redshift to evaluate at (if not None)
        """
        self.obs_spc_file = obs_spc_file
        self.wmin = wmin
        self.wmax = wmax
        self.zmin = zmin
        self.zmax = zmax
        self.mask = mask
        self.smooth = smooth
        self.use_errors = use_errors
        self.redshift_fixed = redshift_fixed
        
        
        # Set up templates
        self.templates = self._load_templates(template_specs)
        
        # Load observed spectrum
        self.obs_spc = self._load_observed_spectrum()
        
        # Results storage
        self.results = {}
        
        
    def _load_templates(self, template_name):
        """
        Load template spectra for comparison.
        
        Parameters
        ----------
        template_name : str, optional
            Name of the template to use, or 'all' to use all templates
            
        Returns
        -------
        dict
            Dictionary of template spectra information
        """
  

        templates_folder = 'templates/'
        
        templates_dict = {
            'selsing2016': {
                "path": os.path.join(templates_folder, 'selsing2016.txt'),
                "color": "C0",
                "name": 'selsing2016'
            },
            'yang2021': {
                'path': os.path.join(templates_folder, 'yang2021_z6p5qsos.txt'),
                "color": "C1",
                "name": 'yang2021'
            },
            'weak-lya': {
                'path': os.path.join(templates_folder, 'banados2016_smoothed_low_lya_yang2021_wa1300.spc'),
                "color": "C2",
                "name": "weak-lya"
            },
            'median-lya': {
                'path': os.path.join(templates_folder, 'banados2016_smoothed_yang2021_wa1300.spc'),
                "color": "C3",
                "name": "median-lya"
            },
            'strong-lya': {
                'path': os.path.join(templates_folder, 'banados2016_smoothed_high_lya_yang2021_wa1300.spc'),
                "color": "C4",
                "name": "strong-lya"
            },
            'vandenberk2001': {
                "path": os.path.join(templates_folder, 'sdss_qso_vandenberk_fitting.txt'),
                "color": "C5",
                "name": 'vandenberk2001'
            },
        }
        
        if template_name != 'all':
            if template_name in templates_dict:
                return {template_name: templates_dict[template_name]}
            else:
                raise ValueError(f"Template {template_name} not found. Available templates: {list(templates_dict.keys())}")
        
        return templates_dict
        
    def _load_observed_spectrum(self):
        """
        Load and preprocess the observed spectrum.
        
        Returns
        -------
        XSpectrum1D
            Observed spectrum
        """
        print("=" * 60)
        print(f"Processing: {self.obs_spc_file}")
        obs_spc = readspec(self.obs_spc_file)
        
        # Apply mask if specified
        if self.mask is not None:
            print(f"Masking spectrum: {self.mask}")
            mask_list = self.mask.split(",")
            for mask in mask_list:
                wa, wb = np.float_(mask.split(":"))
                print(f"Masking range: {wa} - {wb}")
                m = between(obs_spc.wavelength.value, wa, wb)
                flux_tmp = obs_spc.flux.value
                flux_tmp[m] = np.nan
                obs_spc.flux = flux_tmp
        
        # Apply smoothing if specified
        if self.smooth is not None:
            print(f"Smoothing original spectrum using Gaussian kernel with width: {self.smooth} pixels")
            obs_spc = obs_spc.gauss_smooth(self.smooth)
            
        return obs_spc

    def calculate_redshift(self):
        """
        Calculate redshift by comparing with template spectra.
        
        Returns
        -------
        list
            List of template results sorted by chi-squared
        """
        # Process each template
        for template_name in self.templates.keys():
            print(f"Processing template: {template_name}")
            tmp_spc = ascii.read(self.templates[template_name]['path'])
            
            # Set redshift range
            if self.redshift_fixed is not None:
                print(f"Evaluating at fixed redshift z={self.redshift_fixed}")
                redshifts = [self.redshift_fixed]
            else:
                redshifts = np.arange(self.zmin, self.zmax, 0.01)
            
            # Calculate chi-squared values
            chisquares, scale_values = self._get_chisquare_scale(tmp_spc, redshifts)
            
            # Find best-fit values
            z_best, chimin, scale_best = self._get_best_values(chisquares, redshifts, scale_values)
            
            # Save best-fit values to dictionary
            self.templates[template_name]["zbest"] = z_best
            self.templates[template_name]["chimin"] = chimin
            self.templates[template_name]["scale_best"] = scale_best
            
        # Sort templates by chi-squared
        dictlist = list(self.templates.values())
        self.results['sorted_templates'] = sorted(dictlist, key=lambda d: d['chimin'])
        
        return self.results['sorted_templates']
    
    def _redchisqg(self, ydata, ymod, deg=2, sd=None):
        """
        Calculate the reduced chi-squared error statistic for an arbitrary model.
        
        Parameters
        ----------
        ydata : array-like
            Observed data
        ymod : array-like
            Model prediction
        deg : int, optional
            Number of free parameters in the model
        sd : array-like, optional
            Standard deviations of observed data
            
        Returns
        -------
        float
            Reduced chi-squared value
        """
        if sd is None:
            chisq = np.nansum(((ydata - ymod) / 1e-17) ** 2)
            print("================================")
            print("Not using errors for the chi-square")
            print("================================")
        else:
            chisq = np.nansum(((ydata - ymod) / sd) ** 2)
    
        # Number of degrees of freedom assuming deg free parameters
        ydfinite = np.isfinite(ydata)
        nu = ydfinite.sum() - 1 - deg
    
        return chisq / nu
    
    def _template_spectrum(self, template, obs_wave, obs_flux):
        """
        Scale a template spectrum to the observed spectrum flux.
        
        Parameters
        ----------
        template : dict
            Template spectrum
        obs_wave : array-like
            Wavelength of observed spectrum
        obs_flux : array-like
            Flux of observed spectrum
            
        Returns
        -------
        tuple
            (scaled_template_flux, scale_factor)
        """
        # Regions to use for scaling
        incl_min = [1245, 1285, 1315, 1340, 1425, 1680, 1975, 2150, 5500]
        incl_max = [1260, 1295, 1325, 1375, 1470, 1710, 2050, 2250, 5800]
    
        # Get minimum and maximum indices from the regions that are also in spectrum
        imin = get_ind(val=obs_wave[0], arr=incl_min)
        imax = get_ind(obs_wave[-1], incl_max) + 1
        include_min = incl_min[imin:imax]
        include_max = incl_max[imin:imax]
    
        scales = np.ones_like(include_min, dtype=float)
    
        for i, (a, b) in enumerate(zip(include_min, include_max)):
            i1 = get_ind(a, obs_wave)
            i2 = get_ind(b, obs_wave)
            i11 = get_ind(a, template['wave'])
            i22 = get_ind(b, template['wave'])
            median_observed = np.nanmedian(obs_flux[i1:i2])
            median_template = np.nanmedian(template['flux'][i11:i22])
            scales[i] = median_observed / median_template
    
        scale = np.nanmean(np.ma.masked_array(scales, np.isnan(scales)))
        template_flux = template['flux'] * scale
    
        return template_flux, scale
    
    def _get_chisquare_scale(self, tmp_spc, redshifts):
        """
        Calculate chi-squared values for a range of redshifts.
        
        Parameters
        ----------
        tmp_spc : dict
            Template spectrum
        redshifts : array-like
            Array of redshifts to test
            
        Returns
        -------
        tuple
            (chi_squared_values, scale_values)
        """
        chisquares = []
        scale_values = []
        
        for redshift in redshifts:
            # Convert observed wavelengths to rest frame
            obs_wave = self.obs_spc.wavelength.value / (1. + redshift)
            # Scale flux by (1+z) for fair comparison
            obs_flux = np.ma.masked_equal(self.obs_spc.flux.value, 0) * (1 + redshift)
            obs_flux = medfilt(obs_flux, 5)
    
            if self.use_errors:
                try:
                    obs_error = np.ma.masked_equal(self.obs_spc.sig.value, 0) * (1 + redshift)
                    obs_error = medfilt(obs_error, 5)
                except Exception:
                    obs_error = None
            else:
                obs_error = None
    
            # Scale template to observed spectrum
            tmp_flux, scale = self._template_spectrum(tmp_spc, obs_wave, obs_flux)
    
            # Select wavelength range for chi-squared calculation
            wi1 = get_ind(self.wmin, obs_wave)
            wi2 = get_ind(self.wmax, obs_wave)
    
            wave = obs_wave[wi1:wi2]
            flux = obs_flux[wi1:wi2]
            error = obs_error[wi1:wi2] if obs_error is not None else None
    
            # Interpolate template flux to observed wavelengths
            tmp_flux_interp = np.interp(wave, tmp_spc['wave'], tmp_flux)
            
            # Calculate reduced chi-squared
            chisq = self._redchisqg(flux, tmp_flux_interp, deg=1, sd=error)
    
            chisquares.append(chisq)
            scale_values.append(scale)
    
        return chisquares, scale_values
    
    def _get_best_values(self, chisquares, redshifts, scale_values):
        """
        Find the best-fit redshift and scale.
        
        Parameters
        ----------
        chisquares : array-like
            Chi-squared values
        redshifts : array-like
            Redshift values
        scale_values : array-like
            Scale values
            
        Returns
        -------
        tuple
            (best_redshift, minimum_chi_squared, best_scale)
        """
        imin = np.nanargmin(chisquares)
        chi2 = chisquares[imin]
        z = redshifts[imin]
        scl = scale_values[imin]
        
        print("=" * 30)
        print(f"Minimum chi-squared: {chi2:.4f}")
        print(f"Best-fit redshift: {z:.4f}")
        print(f"Best-fit scale factor: {scl:.4f}")
        print("=" * 30)
    
        return z, chi2, scl
    
    def calculate_magnitudes(self, template_info, ax=None):
        """
        Calculate magnitudes at 1450Å and 2500Å.
        
        Parameters
        ----------
        template_info : dict
            Template information
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
            
        Returns
        -------
        dict
            Dictionary of magnitudes
        """
        tmp_spc = ascii.read(template_info['path'])
        redshift = template_info["zbest"]
        scale = template_info['scale_best']
    
        # Calculate M1450
        range_1450 = between(tmp_spc['wave'], 1430, 1470)
        flambda_1450 = np.nanmedian(tmp_spc['flux'][range_1450]) * scale
        
        fnu1450, m1450, Mabs1450, _ = self._get_fnu_apAbsMags(1450., flambda_1450, redshift)
        
        # Calculate M2500
        range_2500 = between(tmp_spc['wave'], 2480, 2520)
        flambda_2500 = np.nanmedian(tmp_spc['flux'][range_2500]) * scale
        
        fnu2500, m2500, Mabs2500, _ = self._get_fnu_apAbsMags(2500., flambda_2500, redshift)
    
        mag_dict = {
            'm1450': m1450,
            'Mabs1450': Mabs1450,
            'm2500': m2500,
            'Mabs2500': Mabs2500,
        }
        
        if ax is not None:
            plot_dict = {
                'color': template_info['color'],
                'label': template_info['name'],
            }
            
            self._plot_best_template(ax, tmp_spc, redshift, scale, restframe=True, **plot_dict)
            
            ax.plot(tmp_spc['wave'][range_1450], tmp_spc['flux'][range_1450] * scale, color='y')
            ax.plot([1450.], [flambda_1450,], '*', color='k')
            
            ax.plot(tmp_spc['wave'][range_2500], tmp_spc['flux'][range_2500] * scale, color='y')
            ax.plot([2500.], [flambda_2500,], '*', color='k')
            
            self._write_magnitudes(ax, mag_dict, **plot_dict)
    
        return mag_dict
    
    def _get_fnu_apAbsMags(self, wave0, flambda, redshift):
        """
        Calculate flux density and magnitudes.
        
        Parameters
        ----------
        wave0 : float
            Rest wavelength
        flambda : float
            Flux density at wavelength (in wavelength units)
        redshift : float
            Redshift
            
        Returns
        -------
        tuple
            (flux_density, apparent_magnitude, absolute_magnitude, distance_modulus)
        """
        zp1 = 1. + redshift
        lobs = wave0 * zp1
        fnu = flambda * lobs**2 / c.to('Angstrom/s').value / zp1
    
        apparent_magnitude = -2.5 * np.log10(fnu) - 48.60
        dist_modulus = cosmo.distmod(redshift).value - 2.5 * np.log10(zp1)
        absolute_magnitude = apparent_magnitude - dist_modulus
    
        return fnu, apparent_magnitude, absolute_magnitude, dist_modulus
    
    def create_visualization(self, output_file=None):
        """
        Create visualization of the results.
        
        Parameters
        ----------
        output_file : str, optional
            Output PDF file name
            
        Returns
        -------
        str
            Path to the created PDF file
        """
        if not self.results.get('sorted_templates'):
            self.calculate_redshift()
            
        # Create PDF for output
        if output_file is None:
            output_file = f"{os.path.splitext(self.obs_spc_file)[0]}_redshift.pdf"
            
        pdf = PdfPages(output_file)
        
        # Create figure
        fig, axs = self._create_figure()
        
        # Add wavelength range information
        txt = f'Wave range used for χ²: {self.wmin:.1f} - {self.wmax:.1f}'
        if self.mask is not None:
            txt += f" (masked:{self.mask})"
        axs[0].text(0.05, 0.95, txt, transform=axs[0].transAxes,
                    horizontalalignment='left', color='black',
                    fontsize='large',
                    verticalalignment='top')
        
        # Plot observed spectrum
        self._plot_observed_spectrum(axs[2], self.obs_spc)
        
        # Plot chi-squared for each template
        for template_name, template in self.templates.items():
            if self.redshift_fixed is not None:
                redshifts = [self.redshift_fixed]
            else:
                redshifts = np.arange(self.zmin, self.zmax, 0.01)
                
            # We already have the chi-squared values calculated
            z_best = template["zbest"]
            
            plot_dict = {
                'color': template["color"],
                'label': template_name,
            }
            
            # We need to recalculate the chi-squared values for plotting
            chisquares, _ = self._get_chisquare_scale(ascii.read(template['path']), redshifts)
            
            self._plot_chisquare(axs[1], redshifts, chisquares, z_best, **plot_dict)
            self._plot_best_template(
                axs[2], 
                ascii.read(template['path']), 
                z_best, 
                template["scale_best"], 
                **plot_dict
            )
            
        # Set x-limits for observed spectrum
        axs[1].legend(loc='best')
        axs[2].set_xlim(8000, 10000)
        
        # Write information about templates
        self._write_information(axs[0], self.results['sorted_templates'])
        
        # Create rest-frame plot of observed spectrum
        best_z = self.results['sorted_templates'][0]['zbest']
        zp1 = 1. + best_z
        if self.obs_spc.sig is not np.nan:
            spc_rest = XSpectrum1D(self.obs_spc.wavelength / zp1, self.obs_spc.flux * zp1, self.obs_spc.sig * zp1)
        else:
            spc_rest = XSpectrum1D(self.obs_spc.wavelength / zp1, self.obs_spc.flux * zp1)
        
        self._plot_observed_spectrum(axs[3], spc_rest)
        axs[3].set_xlim(1216, 2600)
        
        # Calculate magnitudes
        mag_dict = self.calculate_magnitudes(self.results['sorted_templates'][0], ax=axs[3])
        
        plot_dict = {
            'color': self.results['sorted_templates'][0]['color'],
            'label': self.results['sorted_templates'][0]['name'],
        }
        self._write_magnitudes(axs[0], mag_dict, **plot_dict)
        
        # Save figure and close PDF
        pdf.savefig(fig)
        pdf.close()
        print(f"{output_file} created")
        
        # Try to open PDF
        try:
            os.system(f"open -a Preview {output_file}")
        except Exception as e:
            print(f"Could not open PDF automatically: {e}")
            
        return output_file
    
    def _create_figure(self):
        """
        Create a figure with multiple panels.
        
        Returns
        -------
        tuple
            (figure, tuple_of_axes)
        """
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        ax1 = fig.add_axes([0.1, 0.78, 0.8, 0.17])
        plt.setp(ax1, xticks=[], yticks=[])
        ax2 = fig.add_axes([0.15, 0.58, 0.75, 0.18])
        ax3 = fig.add_axes([0.15, 0.33, 0.75, 0.2])
        ax4 = fig.add_axes([0.15, 0.1, 0.75, 0.15])
    
        return fig, (ax1, ax2, ax3, ax4)
    
    def _plot_chisquare(self, ax, redshifts, chisquares, z_best, **kwargs):
        """
        Plot chi-squared values versus redshift.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        redshifts : array-like
            Redshift values
        chisquares : array-like
            Chi-squared values
        z_best : float
            Best-fit redshift
        **kwargs : dict
            Additional keyword arguments for plot
        """
        ax.plot(redshifts, np.array(chisquares), **kwargs)
        ax.axvline(z_best, color=kwargs['color'])
        ax.set_xlabel('Redshift')
        ax.set_ylabel(r'$\chi^2_{red}$')
    
    def _plot_best_template(self, ax, tmp_spc, redshift, scale, restframe=False, **kwargs):
        """
        Plot the best-fit template.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        tmp_spc : dict
            Template spectrum
        redshift : float
            Redshift
        scale : float
            Scale factor
        restframe : bool, optional
            Whether to plot in rest frame
        **kwargs : dict
            Additional keyword arguments for plot
        """
        if restframe:
            zp1 = 1.0
        else:
            zp1 = redshift + 1.0
        tmp_flux_best = tmp_spc['flux'] * scale
        ax.plot(tmp_spc['wave'] * zp1, tmp_flux_best / zp1, **kwargs)
    
    def _plot_observed_spectrum(self, ax, spc):
        """
        Plot the observed spectrum.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        spc : XSpectrum1D
            Observed spectrum
        """
        ax.plot(spc.wavelength, medfilt(spc.flux, 5),
                lw=2, color='k')
        if spc.sig is not np.nan:
            ax.plot(spc.wavelength, medfilt(spc.sig, 5), color='gray')
    
    def _write_information(self, ax, sorted_list):
        """
        Write information about the best-fit templates.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to write on
        sorted_list : list
            List of templates sorted by chi-squared
        """
        (xt, yt) = (0.02, 0.80)
        for i, template in enumerate(sorted_list):
            plot_dict = {
                'color': template['color'],
                'label': template['name'],
            }
    
            txt = template['name']
            txt += f" zbest={template['zbest']:.2f}"
            txt += f" (minchi={template['chimin']:.2f})"
            
            if i == 0:
                ax.text(xt, yt, txt,
                        transform=ax.transAxes,
                        horizontalalignment='left',
                        fontsize='large', fontweight='extra bold',
                        bbox=dict(facecolor='none', edgecolor=plot_dict['color']),
                        verticalalignment='top', **plot_dict)
            else:
                ax.text(xt, yt, txt,
                        transform=ax.transAxes,
                        horizontalalignment='left',
                        fontsize='large',
                        verticalalignment='top', **plot_dict)
            yt -= 0.13
    
    def _write_magnitudes(self, ax, magdict, **kwargs):
        """
        Write magnitude information.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to write on
        magdict : dict
            Dictionary of magnitude values
        **kwargs : dict
            Additional keyword arguments for text
        """
        (xt, yt) = (0.58, 0.80)
        txt = f" $m_{{1450}}={magdict['m1450']:.2f}$; "
        txt += f"$M_{{1450}}={magdict['Mabs1450']:.2f}$"
        txt += "\n"
        txt += f" $m_{{2500}}={magdict['m2500']:.2f}$; "
        txt += f"$M_{{2500}}={magdict['Mabs2500']:.2f}$"
        
        ax.text(xt, yt, txt,
                transform=ax.transAxes,
                horizontalalignment='left',
                fontsize='large',
                verticalalignment='top', **kwargs)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    description = """
    QSO Redshift Estimator
    
    This script estimates the redshift and magnitudes at specific rest-frame wavelengths 
    for high-redshift quasars by comparing observed spectra with template spectra.
    """
    
    examples = """
    Examples:
    
    1. Basic usage:
       python qso_redshift_estimator.py spectrum.fits
       
    2. Specify template and redshift range:
       python qso_redshift_estimator.py spectrum.fits --template_spc selsing2016 --zmin 5.5 --zmax 7.0
       
    3. Mask specific wavelength ranges:
       python qso_redshift_estimator.py spectrum.fits --mask 9000:9200,9300:9400
    """

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples)

    parser.add_argument('obs_spc', type=str,
                      help='FITS or text file containing the observed spectrum')

    parser.add_argument('--smooth', type=int, required=False,
                      default=None,
                      help='Smoothing factor for the observed spectrum (default: None)')

    parser.add_argument('--errors', dest='errors', action='store_true',
                      help='Use errors in chi-squared calculation and plotting')

    parser.add_argument('--no-errors', dest='errors', action='store_false',
                      help='Ignore errors in chi-squared calculation and plotting')

    parser.set_defaults(errors=True)

    parser.add_argument('--wmin', type=float, required=False,
                      default=1240.,
                      help='Min wavelength to use for chi-squared estimation (default: 1240)')

    parser.add_argument('--wmax', type=float, required=False,
                      default=1450.,
                      help='Max wavelength to use for chi-squared estimation (default: 1450)')

    parser.add_argument('--zmax', type=float, required=False,
                      default=6.5,
                      help='Max redshift allowed (default: 6.5)')

    parser.add_argument('--zmin', type=float, required=False,
                      default=5.6,
                      help='Min redshift allowed (default: 5.6)')

    parser.add_argument('--mask', type=str, required=False,
                      default=None,
                      help='Regions to mask, format: "x1:x2,x3:x4" (default: None)')

    parser.add_argument('--template_spc', type=str, required=False,
                      default='all',
                      help='Template spectrum to use. Options: vandenberk2001, selsing2016, '
                           'weak-lya, strong-lya, median-lya, or "all" (default: all)')
                      
    parser.add_argument('--redshift_fixed', type=float, required=False,
                      default=None,
                      help='Fixed redshift to evaluate at (default: None)')

    return parser.parse_args()


def main():
    """
    Main function to run the QSO redshift estimation.
    """
    args = parse_arguments()
    
    # Initialize and run the QSO redshift estimator
    estimator = QSORedshiftEstimator(
        args.obs_spc,
        template_specs=args.template_spc,
        wmin=args.wmin,
        wmax=args.wmax,
        zmin=args.zmin,
        zmax=args.zmax,
        mask=args.mask,
        smooth=args.smooth,
        use_errors=args.errors,
        redshift_fixed=args.redshift_fixed
    )
    
    # Calculate redshift
    sorted_templates = estimator.calculate_redshift()
    
    # Create visualization
    output_file = estimator.create_visualization()
    
    print(f"Best-fit redshift: {sorted_templates[0]['zbest']:.4f} (template: {sorted_templates[0]['name']})")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()