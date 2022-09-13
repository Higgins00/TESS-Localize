# coding: utf-8

#from __future__ import division, print_function
import warnings
from copy import copy
import numpy as np
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
import lightkurve as lk
from astropy import units as u
import lmfit as lm
from lmfit import Minimizer, Parameters, report_fit
import PRF
import sys
import pygmmis
import pkg_resources



class PCA:
    """Class designed to give users access to analysis functions.
    
    Parameters
    ----------
    targetpixelfile : targetpixelfile object
    frequencies: list
        List of frequencies desired to localize the source location for.
    frequnit: astropy.units.unit
        Units of the frequencies in frequencies list.
    principal_components: int
        Number of components used in PCA for TPF lightcurve.
    aperture: 2D Boolean array, 'auto'
        If not specified user the TPF.pipeline_mask will be used, if a user specified aperture is used it must be the same shape as the TPF.
    Returns
    ----------
    self.tpf
        Targetpixelfile object.
    self.aperture
        Aperture used.
    self.raw_lc
        Lightcurve before PCA.
    self.dm
        Design matrix for PCA
    self.corrected_lc
        Lightcurve after PCA
    self.frequency_list
        List of frequencies used.
    self.autopca
        Automatically determined best number of principal components to remove.
        
    
    
    """
    
    def __init__(self, targetpixelfile,
                 frequencies=[], frequnit=u.uHz, principal_components = 5, 
                 aperture=None):
        
        self.tpf = targetpixelfile
        #Defining an aperture that will be used in plotting and making empty 2-d arrays of the correct size for masks
        self.aperture = aperture
        self.principal_components = principal_components
        self.frequency_list = np.asarray((frequencies*frequnit).to(1/u.d))
            
        if self.aperture is None:
            self.aperture = targetpixelfile.pipeline_mask
            if (targetpixelfile.pipeline_mask.any() == False):
                #will add a flag here if no aperture
                def frequency_aperture(tpf,frequencies,frequnits = 1/u.d):
                    heat = np.empty((tpf.shape[1],tpf.shape[2]))
                    heat[:]=np.nan
                    #Iterating through columns of pixels

                    for i in np.arange(0,tpf.shape[1]):

                        #Iterating through rows of pixels
                        for j in np.arange(0,tpf.shape[2]):


                            #Making an empty 2-d array
                            mask = np.zeros((tpf.shape[1],tpf.shape[2]), dtype=bool)

                            #Iterating to isolate pixel by pixel to get light curves
                            mask[i][j] = True

                            #Getting the light curve for a pixel and excluding any flagged data
                            lightcurve = tpf.to_lightcurve(aperture_mask=mask)
                            lightcurve = lightcurve[np.isfinite(lightcurve['flux']*lightcurve['flux_err'])]
                            lightcurve = lightcurve[np.where(lightcurve[np.isfinite(lightcurve['flux']*lightcurve['flux_err'])].quality==0)]
                            pg = lightcurve.to_periodogram(frequency = frequencies,freq_unit = frequnits,ls_method='slow')

                            heat[i][j] = np.sum(pg.power.value**2)**(1/2)
                    return heat>np.mean(heat)+2*np.std(heat)
                self.aperture = frequency_aperture(tpf=self.tpf,frequencies = frequencies, frequnits = frequnit)
                    
        if aperture =='auto':
            self.aperture = frequency_aperture(tpf=self.tpf,frequencies = frequencies, frequnits = frequnit)
    
        
        # Make a design matrix and pass it to a linear regression corrector
        self.raw_lc1 = self.tpf.to_lightcurve(aperture_mask=self.aperture)
        self.quality_mask = [np.isfinite(self.raw_lc1['flux']*self.raw_lc1['flux_err'])]
        if principal_components !=0:
            self.dm = lk.DesignMatrix(self.tpf.flux[:,~self.tpf.pipeline_mask][np.isfinite(self.raw_lc1['flux']*self.raw_lc1['flux_err']),:][np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)[0],:], name='regressors').pca(principal_components)
            self.raw_lc = self.raw_lc1[self.quality_mask[0]]
            self.raw_lc = self.raw_lc[np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)]
        
            rc = lk.RegressionCorrector(self.raw_lc)
            self.corrected_lc = rc.correct(self.dm.append_constant())
        else:
            self.raw_lc = self.raw_lc1[self.quality_mask[0]]
            self.raw_lc = self.raw_lc[np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)]
            self.corrected_lc = self.raw_lc



        pgs = [lk.LightCurve(time = self.tpf.time.value[self.quality_mask[0]][np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)],flux =self.dm.values.T[i]).to_periodogram(frequency = np.append([0.0001],self.frequency_list),ls_method='slow') for i in np.arange(0,len(self.dm.values.T))]
        pg2 = [lk.LightCurve(time = self.tpf.time.value[self.quality_mask[0]][np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)],flux =self.dm.values.T[i]).to_periodogram() for i in np.arange(0,len(self.dm.values.T))]
        fails = []
        for i in np.arange(0,self.principal_components):
            medians = np.empty(len(self.frequency_list))
            for j in np.arange(0,len(self.frequency_list)):
                mask = np.where((pg2[self.principal_components-1-i].frequency.value>=frequencies[j]-5)&((pg2[self.principal_components-1-i].frequency.value<=frequencies[j]+5)))
                medians[j] = np.nanmedian(pg2[self.principal_components-1-i].power.value[mask])
            if (pgs[self.principal_components-1-i].power.value[1:]> 5*medians).any():
                fails.extend([self.principal_components-i])
        if len(fails) ==0:
            mini = self.principal_components
        else:
            mini = np.min(np.array(fails))-1

        self.autopca = mini









    def plot_pca(self):
        if self.principal_components==0:
            pass
        else:
            pgs = [lk.LightCurve(time = self.tpf.time.value[self.quality_mask[0]][np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)],flux =self.dm.values.T[i]).to_periodogram() for i in np.arange(0,len(self.dm.values.T))]
            fig,ax = plt.subplots(1,2,figsize=(14,5),sharey=True)
            ax[0].plot(self.tpf.time.value[self.quality_mask[0]][np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)], self.dm.values*1 + np.arange(self.principal_components)*0.2)
            ax[0].set_title(r'Principal Components Contributions')
            ax[0].set_xlabel('Offset')
            ax[1].plot(pgs[0].frequency.value,np.array([pgs[i].power.value*10+np.arange(self.principal_components)[i]*0.2 for i in range(len(pgs))]).T)
            ax[1].set_xlim(self.frequency_list.min(),self.frequency_list.max())
            for i in np.arange(0,len(self.frequency_list)):
                ax[1].axvline(x = self.frequency_list[i],color='r',linestyle='--',linewidth=1)
            g2 = self.raw_lc.plot(label='Raw light curve')
            self.corrected_lc.plot(ax=g2, label='Corrected light curve')


class Localize:
    """Class designed to give users access to analysis functions.
    
    Parameters
    ----------
    targetpixelfile : targetpixelfile object
    gaia : Boolean
        True if internet access is available and user wants to display gaia information; User should enter False otherwise.
    magnitude_limit: float
        Lower limit of gaia magnitudes to search for.
    frequencies: list
        List of frequencies desired to localize the source location for.
    frequnit: astropy.units.unit
        Units of the frequencies in frequencies list.
    principal_components: int, str
        Number of components used in PCA for TPF lightcurve, or 'auto' for automatic determination.
    aperture: 2D Boolean array, or 'auto'
        If not specified user the TPF.pipeline_mask will be used, if a user specified aperture is used it must be the same shape as the TPF.
    mask: 2D Boolean array, or None
        Mask of pixels to ignore in the fitting method.

    Returns
    ----------
    self.tpf
        Targetpixelfile object.
    self.aperture
        Aperture used.
    self.raw_lc
        Lightcurve before PCA.
    self.dm
        Design matrix for PCA
    self.corrected_lc
        Lightcurve after PCA
    self.frequency_list
        List of frequencies used.
    self.principal_components
        Number of principal components removed, or 'auto'
    self.initial_phases
        Initial phases fit for the frequencies.
    self.final_phases
        Final phases fit for the frequencies.
    self.heatmap = heats.T
        3D array of amplitude for each frequency at every pixel in the TPF.
    self.heatmap_error
        3D array of amplitude erros for each frequency at every pixel in the TPF.
    self.timeserieslength
    self.gaiadata
    self.location
        Best fit source location in pixels.
    self.location_skycoord
        Best fit source location in RA and DEC.
    self.heatmap
        2D array of composite heatmap for all frequencies.
    self.starfit
        Gaia sources and their distances from the fitted location of the source.
    self.result
        Result parameters of the fit. Use report_fit(self.report) to view.
    self.maxsignal_aperture
        Aperture mask for the pixel with the greatest SNR
    
    
    """
    
    def __init__(self, targetpixelfile, gaia=True, magnitude_limit=18, 
                 frequencies=[], frequnit=u.uHz, principal_components = 'auto', 
                 aperture=None, method = 'PRF', sigma=None, mask=None, **kwargs):
        
        self.tpf = targetpixelfile
        self.method = method
        #Defining an aperture that will be used in plotting and making empty 2-d arrays of the correct size for masks
        self.aperture = aperture
        self.frequency_list = np.asarray((frequencies*frequnit).to(1/u.d))
        self.principal_components = principal_components
        if mask is None:
            self.mask = np.array(self.tpf.pipeline_mask*False,bool)
        else:
            self.mask= np.array(mask,bool)
            for i in range(len(self.tpf.hdu[1].data["FLUX"])):
                self.tpf.hdu[1].data["FLUX"][i][mask] = np.nan
            
            
        def frequency_aperture(tpf,frequencies,frequnits = 1/u.d):
            heat = np.empty((tpf.shape[1],tpf.shape[2]))
            heat[:]=np.nan
            #Iterating through columns of pixels

            for i in np.arange(0,tpf.shape[1]):

                #Iterating through rows of pixels
                for j in np.arange(0,tpf.shape[2]):


                    #Making an empty 2-d array
                    mask = np.zeros((tpf.shape[1],tpf.shape[2]), dtype=bool)

                    #Iterating to isolate pixel by pixel to get light curves
                    mask[i][j] = True

                    #Getting the light curve for a pixel and excluding any flagged data
                    lightcurve = tpf.to_lightcurve(aperture_mask=mask)
                    lightcurve = lightcurve[np.isfinite(lightcurve['flux']*lightcurve['flux_err'])]
                    if len(lightcurve)!=0:
                        lightcurve = lightcurve[np.where(lightcurve[np.isfinite(lightcurve['flux']*lightcurve['flux_err'])].quality==0)]
                        pg = lightcurve.to_periodogram(frequency = frequencies,freq_unit = frequnits,ls_method='slow')

                        heat[i][j] = np.sum(pg.power.value**2)**(1/2)
                    else:
                        heat[i][j]=np.nan
            return heat>np.mean(heat)+2*np.std(heat)
        if self.aperture is None:
            self.aperture = targetpixelfile.pipeline_mask
            if (targetpixelfile.pipeline_mask.any() == False):
                #will add a flag here if no aperture
                self.aperture = frequency_aperture(tpf=self.tpf,frequencies = frequencies, frequnits = frequnit)
                    
        if aperture =='auto':
            self.aperture = frequency_aperture(tpf=self.tpf,frequencies = frequencies, frequnits = frequnit)
            
        if principal_components == 'auto':
            self.principal_components = 5
            self.raw_lc1 = self.tpf.to_lightcurve(aperture_mask=self.aperture)
            self.quality_mask = [np.isfinite(self.raw_lc1['flux']*self.raw_lc1['flux_err'])]
            if self.principal_components !=0:
                self.dm = lk.DesignMatrix(self.tpf.flux[:,~(self.aperture|self.mask)][np.isfinite(self.raw_lc1['flux']*self.raw_lc1['flux_err']),:][np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)[0],:], name='regressors').pca(self.principal_components)
                self.raw_lc = self.raw_lc1[self.quality_mask[0]]
                self.raw_lc = self.raw_lc[np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)]

                rc = lk.RegressionCorrector(self.raw_lc)
                self.corrected_lc = rc.correct(self.dm.append_constant())
            else:
                self.raw_lc = self.raw_lc1[self.quality_mask[0]]
                self.raw_lc = self.raw_lc[np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)]
                self.corrected_lc = self.raw_lc



            pgs = [lk.LightCurve(time = self.tpf.time.value[self.quality_mask[0]][np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)],flux =self.dm.values.T[i]).to_periodogram(frequency = np.append([0.0001],self.frequency_list),ls_method='slow') for i in np.arange(0,len(self.dm.values.T))]
            pg2 = [lk.LightCurve(time = self.tpf.time.value[self.quality_mask[0]][np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)],flux =self.dm.values.T[i]).to_periodogram() for i in np.arange(0,len(self.dm.values.T))]
            fails = []
            for i in np.arange(0,self.principal_components):
                medians = np.empty(len(self.frequency_list))
                for j in np.arange(0,len(self.frequency_list)):
                    mask = np.where((pg2[self.principal_components-1-i].frequency.value>=frequencies[j]-5)&((pg2[self.principal_components-1-i].frequency.value<=frequencies[j]+5)))
                    medians[j] = np.nanmedian(pg2[self.principal_components-1-i].power.value[mask])
                if (pgs[self.principal_components-1-i].power.value[1:]> 5*medians).any():
                    fails.extend([self.principal_components-i])
            if len(fails) ==0:
                mini = self.principal_components
            else:
                mini = np.min(np.array(fails))-1

            self.principal_components = mini
        
        if self.aperture is None:
            self.aperture = targetpixelfile.pipeline_mask
            if targetpixelfile.pipeline_mask.any() == False:
                #will add a flag here if no aperture
                self.aperture = self.tpf.create_threshold_mask()
    
        
        # Make a design matrix and pass it to a linear regression corrector
        self.raw_lc1 = self.tpf.to_lightcurve(aperture_mask=self.aperture)
        self.quality_mask = [np.isfinite(self.raw_lc1['flux']*self.raw_lc1['flux_err'])]
        if self.principal_components !=0:
            self.dm = lk.DesignMatrix(self.tpf.flux[:,~(self.aperture|self.mask)][np.isfinite(self.raw_lc1['flux']*self.raw_lc1['flux_err']),:][np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)[0],:], name='regressors').pca(self.principal_components)
            self.raw_lc = self.raw_lc1[self.quality_mask[0]]
            self.raw_lc = self.raw_lc[np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)]
        
            rc = lk.RegressionCorrector(self.raw_lc)
            self.corrected_lc = rc.correct(self.dm.append_constant())
        else:
            self.raw_lc = self.raw_lc1[self.quality_mask[0]]
            self.raw_lc = self.raw_lc[np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)]
            self.corrected_lc = self.raw_lc
            
        #self.corrected_lc = corrected_lc.remove_outliers()
        
        
        
        def Obtain_Initial_Phase(tpf,corrected_lc,frequency_list):
            flux = corrected_lc.flux.value
            times = corrected_lc.time.value - np.nanmean(corrected_lc.time.value)
            #appending a extra value here to ensure lightkurve doesn't call an error if length of frequencies =1
            pg = corrected_lc.to_periodogram(frequency = np.append([0.0001],frequency_list),ls_method='slow')
            initial_amp= np.asarray(pg.power[1:])#initial_amp

            initial_phase = np.zeros(len(frequency_list))
            mask = np.argsort(initial_amp)[::-1]

            def lc_model(time,amp,freq,phase):
                return amp*np.sin(2*np.pi*freq*time + phase)

            def background_model(time,height):
                return np.ones(len(time))*height

            model = lm.Model(background_model, independent_vars=['time'])
            for j in mask:
                model += lm.Model(lc_model,independent_vars=['time'],prefix='f{0:d}'.format(j))
                for i in range(np.where(mask==j)[0][0]+1):
                    model.set_param_hint('f{0:d}phase'.format(mask[i]), min = -np.pi, max = np.pi ,value= initial_phase[mask][i],vary = False)
                    model.set_param_hint('f{0:d}amp'.format(mask[i]), value = initial_amp[mask][i],vary=False)
                    model.set_param_hint('height', value= np.nanmean(flux),vary=False)
                    model.set_param_hint('f{0:d}freq'.format(mask[i]),value = frequency_list[mask][i], vary = False)

                params = model.make_params()
                #params['f{0:d}amp'.format(j)].set(value = initial_amp[j],vary=False)
                params['f{0:d}phase'.format(j)].set(vary=True)
                params['f{0:d}phase'.format(j)].set(value = initial_phase[j])
                params['f{0:d}phase'.format(j)].set(brute_step=np.pi/10)
                result = model.fit(corrected_lc.flux.value,params,time=times,method = 'brute')
                initial_phase[j]=result.best_values['f{0:d}phase'.format(j)]    

            return initial_phase


#             flux = corrected_lc.flux.value
#             times = corrected_lc.time.value - np.nanmean(corrected_lc.time.value)
#             #appending a extra value here to ensure lightkurve doesn't call an error if length of frequencies =1
#             pg = corrected_lc.to_periodogram(frequency = np.append([0.0001],frequency_list),ls_method='slow')
#             initial_flux= np.asarray(pg.power[1:])#initial_amp

#             initial_phase = np.zeros(len(frequency_list))

#             def lc_model(time,amp,freq,phase):
#                 return amp*np.sin(2*np.pi*freq*time + phase)

#             def background_model(time,height):
#                 return np.ones(len(time))*height
#             for j in np.arange(len(frequency_list)):
#                 for i in np.arange(len(frequency_list)):

#                     if (i == 0):
#                         model = lm.Model(lc_model,independent_vars=['time'],prefix='f{0:d}'.format(i)) 
#                         model += lm.Model(background_model, independent_vars=['time'])
#                     else:
#                         model += lm.Model(lc_model,independent_vars=['time'],prefix='f{0:d}'.format(i))


#                     model.set_param_hint('f{0:d}phase'.format(i), min = -np.pi, max = np.pi ,value= initial_phase[i],vary = False)
#                     model.set_param_hint('f{0:d}amp'.format(i), value = initial_flux[i],vary=False)
#                     model.set_param_hint('height', value= np.nanmean(flux),vary=False)
#                     model.set_param_hint('f{0:d}freq'.format(i),value = frequency_list[i], vary = False)


#                 params = model.make_params()
#                 #params['f{0:d}amp'.format(j)].set(value = initial_flux[j],vary=False)
#                 params['f{0:d}phase'.format(j)].set(vary=True)
#                 params['f{0:d}phase'.format(j)].set(value = initial_phase[j])
#                 params['f{0:d}phase'.format(j)].set(brute_step=np.pi/10)
#                 result = model.fit(corrected_lc.flux.value,params,time=times,method = 'brute')
#                 initial_phase[j]=result.best_values['f{0:d}phase'.format(j)]

            #return initial_phase
        
        self.initial_phases = Obtain_Initial_Phase(self.tpf,self.corrected_lc,self.frequency_list)
        
        def Obtain_Final_Phase(tpf,corrected_lc,frequency_list,initial_phases):

            flux = corrected_lc.flux.value
            times = corrected_lc.time.value - np.nanmean(corrected_lc.time.value)
            pg = corrected_lc.to_periodogram(frequency = np.append([0.0001],frequency_list),ls_method='slow')
            initial_flux= np.asarray(pg.power[1:])


            def lc_model(time,amp,freq,phase):
                return amp*np.sin(2*np.pi*freq*time + phase)

            def background_model(time,height):
                return np.ones(len(time))*height

            for i in np.arange(len(frequency_list)):

                if (i == 0):
                    model = lm.Model(lc_model,independent_vars=['time'],prefix='f{0:d}'.format(i)) 
                    model += lm.Model(background_model, independent_vars=['time'])
                else:
                    model += lm.Model(lc_model,independent_vars=['time'],prefix='f{0:d}'.format(i))


                model.set_param_hint('f{0:d}phase'.format(i), min = -np.pi, max = np.pi ,value= initial_phases[i],vary = True)
                model.set_param_hint('f{0:d}amp'.format(i), value = initial_flux[i],vary=True)
                model.set_param_hint('height', value= np.nanmean(flux),vary=True)
                model.set_param_hint('f{0:d}freq'.format(i),value = frequency_list[i], vary = False)


            params = model.make_params()

            result = model.fit(corrected_lc.flux.value,params,time=times,weights=1/corrected_lc.flux_err.value)
            
            final_phases = [result.best_values['f{0:d}phase'.format(j)] for j in np.arange(len(frequency_list))]

    
            return final_phases

        self.final_phases = Obtain_Final_Phase(self.tpf,self.corrected_lc,self.frequency_list,self.initial_phases)
    
        def Obtain_Final_Fit(tpf,corrected_lc,frequency_list,final_phases):

            flux = corrected_lc.flux.value
            times = corrected_lc.time.value - np.nanmean(corrected_lc.time.value)
            pg = corrected_lc.to_periodogram(frequency = np.append([0.0001],frequency_list),ls_method='slow')
            initial_flux= np.asarray(pg.power[1:])


            def lc_model(time,amp,freq,phase):
                return amp*np.sin(2*np.pi*freq*time + phase)

            def background_model(time,height):
                return np.ones(len(time))*height

            for i in np.arange(len(frequency_list)):

                if (i == 0):
                    model = lm.Model(lc_model,independent_vars=['time'],prefix='f{0:d}'.format(i)) 
                    model += lm.Model(background_model, independent_vars=['time'])
                else:
                    model += lm.Model(lc_model,independent_vars=['time'],prefix='f{0:d}'.format(i))


                model.set_param_hint('f{0:d}phase'.format(i), value= final_phases[i],vary = False)
                model.set_param_hint('f{0:d}amp'.format(i), value = initial_flux[i],vary=True)
                model.set_param_hint('height', value= np.nanmean(flux),vary=True)
                model.set_param_hint('f{0:d}freq'.format(i),value = frequency_list[i], vary = False)


            params = model.make_params()

            result = model.fit(corrected_lc.flux.value,params,time=times,weights=1/corrected_lc.flux_err.value)
            
            return result
        
        
        heats = []
        heats_error =[]
        #Iterating through columns of pixels
        for i in np.arange(0,len(self.aperture)):
            
            #Iterating through rows of pixels
            for j in np.arange(0,len(self.aperture[0])):
                
                
                #Making an empty 2-d array
                mask = np.zeros((len(self.aperture),len(self.aperture[0])), dtype=bool)
                
                #Iterating to isolate pixel by pixel to get light curves
                mask[i][j] = True
                
                #Getting the light curve for a pixel and excluding any flagged data
                lightcurve = self.tpf.to_lightcurve(aperture_mask=mask)
                lightcurve = lightcurve[self.quality_mask[0]]
                lightcurve = lightcurve[np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)]
                if len(lightcurve[np.isfinite(lightcurve['flux']*lightcurve['flux_err'])])!=0:


                    if principal_components !=0:
                        rcc = lk.RegressionCorrector(lightcurve)
                        lc = rcc.correct(self.dm.append_constant())
                    else:
                        lc=lightcurve
                    #lc = lc.remove_outliers()

                    bestfit = Obtain_Final_Fit(self.tpf,lc,self.frequency_list,self.final_phases)
                    heat = np.asarray([bestfit.best_values['f{0:d}amp'.format(n)] for n in np.arange(len(self.frequency_list))])
                    heat_error =  np.asarray([bestfit.params['f{0:d}amp'.format(n)].stderr for n in np.arange(len(self.frequency_list))])

                    #Extending the list of fitting data for each pixel
                    heats.extend([heat])
                    heats_error.extend([heat_error])
                else:
                    heats.extend([np.ones(len(self.frequency_list))*np.nan])
                    heats_error.extend([np.ones(len(self.frequency_list))*np.nan])
                    
        heats = np.asarray(heats)
        heats_error = np.asarray(heats_error)
        self.heats = heats.T
        self.heats_error = heats_error.T
        
        self.timeserieslength = (self.tpf.time.max()-self.tpf.time.min()).value
        self.gaiadata = None
        
        if (gaia == True):
            """Make the Gaia Figure Elements"""
            # Get the positions of the Gaia sources
            c1 = SkyCoord(self.tpf.ra, self.tpf.dec, frame='icrs', unit='deg')
            # Use pixel scale for query size
            pix_scale = 4.0  # arcseconds / pixel for Kepler, default
            if self.tpf.mission == 'TESS':
                pix_scale = 21.0
            # We are querying with a diameter as the radius, overfilling by 2x.
            from astroquery.vizier import Vizier
            Vizier.ROW_LIMIT = -1
            try:
                result = Vizier.query_region(c1, catalog=["I/345/gaia2"],radius=Angle(np.max(self.tpf.shape[1:]) * pix_scale, "arcsec"))
            except:
                result = Vizier.query_region(c1, catalog=["I/345/gaia2"],radius=Angle(np.max(self.tpf.shape[1:]) * pix_scale, "arcsec"), cache=False)
            

            no_targets_found_message = ValueError('Either no sources were found in the query region '
                                                      'or Vizier is unavailable')
            too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
            if result is None:
                raise no_targets_found_message
            elif len(result) == 0:
                raise too_few_found_message
            result = result["I/345/gaia2"].to_pandas()

            result = result[result.Gmag < magnitude_limit]
            if len(result) == 0:
                raise no_targets_found_message

            year = ((self.tpf.time[0].jd - 2457206.375) * u.day).to(u.year)
            pmra = ((np.nan_to_num(np.asarray(result.pmRA)) * u.milliarcsecond/u.year) * year).to(u.deg).value
            pmdec = ((np.nan_to_num(np.asarray(result.pmDE)) * u.milliarcsecond/u.year) * year).to(u.deg).value
            result.RA_ICRS += pmra
            result.DE_ICRS += pmdec
            radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T
            coords = self.tpf.wcs.all_world2pix(radecs, 0) 



            # Size the points by their Gaia magnitude
            sizes = 64.0 / 2**(result['Gmag']/5.0)
            one_over_parallax = 1.0 / (result['Plx']/1000.)
            source = dict(ra=result['RA_ICRS'],
                          dec=result['DE_ICRS'],
                          source=result['Source'].astype(str),
                          Gmag=result['Gmag'],
                          plx=result['Plx'],
                          one_over_plx=one_over_parallax,
                          x=coords[:, 0],
                          y=coords[:, 1],
                          size=sizes)


            self.gaiadata = source
        
        class frequency_heatmap:

            def __init__(self,tpf,heats,heats_error,frequencies,gaia_data,method):
                self.heat_stamp = heats
                self.gaiadata=gaia_data
                self.heatmap_error = heats_error
                self.size = tpf.pipeline_mask.shape
                self.frequencies= frequencies
                self.tpf = tpf
                self.method = method
            def location(self):
                
                if self.method == 'PRF':
                    
                    self.prf = PRF.TESS_PRF(cam = self.tpf.camera, ccd = self.tpf.ccd,
                                        sector = self.tpf.sector, 
                                        colnum = self.tpf.column+self.size[0]/2.,
                                        rownum = self.tpf.row+self.size[1]/2., 
                                        **kwargs)
                    #self.prf = PRF.Gaussian_PRF(sigma)

                    #Residuals to minimize relative to the error bars
                    def residual(params, amp, amperr, prf):

                        x = params['column']
                        y = params['row']

                        res = []
                        localprf = self.prf.locate(x, y, (self.size[0],self.size[1]))
                        for i in np.arange(len(frequencies)):
                            height = params['height{0:d}'.format(i)]
                            model = height*localprf

                            res.extend( [(amp[i].reshape(self.size)-model) / amperr[i].reshape(self.size)])


                        return np.asarray(res)

                    #Set starting values to converge from
                    self.heatmap_error[np.where(self.heatmap_error==None)]=np.nan
                    self.heatmap_error[np.where(self.heatmap_error==0)]=np.nan
                    #reshaping composite heatmap for user to plot
                    composite_heatmap = self.heat_stamp.sum(axis=0).reshape(self.size)# / ((np.nansum(self.heatmap_error**2,axis=0))**(1/2)).reshape(self.size)#checking for binary analysis
                    c = np.where(composite_heatmap==np.nanmax(composite_heatmap))


                    params = Parameters()
                    for i in np.arange(len(frequencies)):#
                        params.add('height{0:d}'.format(i), value=np.nanmax(self.heat_stamp[i]))
                    params.add('column', value=c[1][0])#c[0]) 
                    params.add('row', value=c[0][0])#c[1])
                    #params.add('sigma', value=1)
                else:
                    if sigma is None:
                        self.sigma=0.8
                    else:
                        self.sigma = sigma
                    self.prf = PRF.Gaussian_PRF(self.sigma)

                    #Residuals to minimize relative to the error bars
                    def residual(params, amp, amperr, prf):

                        x = params['column']
                        y = params['row']
                        
                        res = []
                        for i in np.arange(len(frequencies)):
                            height = params['height{0:d}'.format(i)]
                            model = height*self.prf.locate(x, y, (self.size[0],self.size[1]))

                            res.extend( [(amp[i].reshape(self.size)-model) / amperr[i].reshape(self.size)])


                        return np.asarray(res)

                    #Set starting values to converge from
                    self.heatmap_error[np.where(self.heatmap_error==None)]=np.nan
                    self.heatmap_error[np.where(self.heatmap_error==0)]=np.nan
                    #reshaping composite heatmap for user to plot
                    composite_heatmap = self.heat_stamp.sum(axis=0).reshape(self.size) / ((np.nansum(self.heatmap_error**2,axis=0))**(1/2)).reshape(self.size)#issue with numpy using sqrt?
                    c = np.where(composite_heatmap==np.nanmax(composite_heatmap))


                    params = Parameters()
                    for i in np.arange(len(frequencies)):#
                        params.add('height{0:d}'.format(i), value=np.nanmax(self.heat_stamp[i]))
                    params.add('column', value=c[1][0])
                    params.add('row', value=c[0][0])
                    
                    
                #Do the fit
                minner = Minimizer(residual, params, fcn_args=(self.heat_stamp, self.heatmap_error, self.prf),nan_policy='omit')
                result = minner.minimize()
                #include mean pointing offset from DVA
                mean_pos_corr1 = np.nanmean(self.tpf.hdu[1].data['POS_CORR1']) #Mean pixel motion
                mean_pos_corr2 = np.nanmean(self.tpf.hdu[1].data['POS_CORR2'])
                result.params['column'].value += mean_pos_corr1
                result.params['row'].value += mean_pos_corr2
                self.result = result
                fit = self.result.params.valuesdict()
                self.x = fit['column']
                self.y = fit['row']
                #Read in extrinsic error model from fitting thousands of binary systems
                _error_fname = pkg_resources.resource_filename(__name__, "error_model.npz")
                error_ext = pygmmis.GMM(K=2, D=2)
                error_ext.load(_error_fname)
                #Combined error model
                self.error_model = copy(error_ext) #extrinsic error
                self.error_model.covar += self.result.covar[-2:,-2:] #intrinsic error
                self.error_model.mean += [self.x,self.y] #Locate relative to best-fit position
                self.logL = self.error_model.logL
                
            def star_list(self):
                gaia_data = self.gaiadata
                no_gaia_data_message = ValueError('No gaia data initialized in PixelMapPeriodogram class')
                if gaia_data ==None :
                    starlist = None

                else:
                    distances = np.square(self.x-gaia_data['x'])+np.square(self.y-gaia_data['y'])
                    #closest_star_mask = np.where(np.square(self.x-gaia_data['x'])+np.square(self.y-gaia_data['y'])==(np.square(self.x-gaia_data['x'])+np.square(self.y-gaia_data['y'])).min())
                    stars = dict(ra = np.asarray(gaia_data['ra']),
                                 dec = np.asarray(gaia_data['dec']),
                                 source = np.asarray(gaia_data['source']),
                                 x = np.asarray(gaia_data['x']),
                                 y = np.asarray(gaia_data['y']),
                                 Gmag = np.asarray(gaia_data['Gmag']),
                                 distance = distances)
                    #compute likelihoods of gaia sources
                    L = 10**self.logL(np.vstack((gaia_data["x"],gaia_data["y"])).T) #likelihoods
                    L /= np.sum(L) #normalized
                    stars["likelihood"] = L
                    starlist = pd.DataFrame.from_dict(stars)
                    self.stars = starlist.sort_values(by=[r'likelihood'],ascending = False)
                    
        
        fh = frequency_heatmap(self.tpf,self.heats,self.heats_error,self.frequency_list,self.gaiadata,self.method) 
        fh.location()
        
        self.location = [fh.x,fh.y]
        self.location_skycoord = self.tpf.wcs.all_pix2world([self.location], 0)[0]
        self.heatmap = self.heats.sum(axis=0).reshape(self.aperture.shape[0],self.aperture.shape[1]) / np.sqrt((self.heats_error**2).sum(axis=0)).reshape(self.aperture.shape[0],self.aperture.shape[1])
        self.maxsignal_aperture = self.heatmap == np.nanmax(self.heatmap)
        self.result = fh.result
        
        

        
        if (self.gaiadata is not None):
            fh.star_list()
            self.starfit= fh.stars.reset_index()

    
    def info(self):
        plt.imshow(self.heatmap,origin='lower')
        plt.title('SNR')
        #plot the location
        if (self.gaiadata != None):
            plt.scatter(self.gaiadata['x'],self.gaiadata['y'],s=self.gaiadata['size']*5,c='white',alpha=.6)
            plt.scatter(self.location[0],self.location[1],marker='X',c='black',s=70)
            print(self.starfit)
        plt.xlim(-.5,self.aperture.shape[1]-1+.5)
        plt.ylim(-.5,self.aperture.shape[0]-1+.5)
        report_fit(self.result)
        if (np.asarray([self.result.params['height{0:d}'.format(j)].stderr for j in np.arange(len(self.frequency_list))]) / np.asarray([self.result.params['height{0:d}'.format(j)].value for j in np.arange(len(self.frequency_list))])>.2).any():
            #possibly reword this
            warnings.warn('Frequencies used may not all belong to the same source and provided fit could be unreliable')
        if ((self.location[0]<0) and (self.location[0]>self.tpf.shape[1])) or ((self.location[1]<0) and (self.location[1]>self.tpf.shape[2])):
            warnings.warn('Source fit to a location outside the TPF, refitting using a TPF centered around source is recommended')
    
    def pca(self):
        if self.principal_components==0:
            pass
        else:
            pgs = [lk.LightCurve(time = self.tpf.time.value[self.quality_mask[0]][np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)],flux =self.dm.values.T[i]).to_periodogram() for i in np.arange(0,len(self.dm.values.T))]
            fig,ax = plt.subplots(1,2,figsize=(14,5),sharey=True)
            ax[0].plot(self.tpf.time.value[self.quality_mask[0]][np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)], self.dm.values*1 + np.arange(self.principal_components)*0.2)
            ax[0].set_title(r'Principal Components Contributions')
            ax[0].set_xlabel('Offset')
            ax[1].plot(pgs[0].frequency.value,np.array([pgs[i].power.value*10+np.arange(self.principal_components)[i]*0.2 for i in range(len(pgs))]).T)
            ax[1].set_xlim(self.frequency_list.min(),self.frequency_list.max())
            for i in np.arange(0,len(self.frequency_list)):
                ax[1].axvline(x = self.frequency_list[i],color='r',linestyle='--',linewidth=1)
            g2 = self.raw_lc.plot(label='Raw light curve')
            self.corrected_lc.plot(ax=g2, label='Corrected light curve')
            
    def plot_lc(self,lightcurve_aperture=None,save = None,figuresize = (10,5)):
        """Plot the amplitude heatmap, snr, errors, or the fit model.
        Parameters
        ----------
        lightcurve_aperture: 2D Boolean array, or None
        If not specified user the self.maxsignal_aperture will be used, if a user specified aperture is used it must be the same shape as the TPF.
        save: str, or None
            'filename.png' if you want to save the png of the plot
        figuresize: size of plot
        
        """
        plt.figure(figsize = (figuresize))
        if (lightcurve_aperture is None):
            lightcurve_aperture = self.maxsignal_aperture
        lightcurve = self.tpf.to_lightcurve(aperture_mask=lightcurve_aperture)
        lightcurve = lightcurve[self.quality_mask[0]]
        lightcurve = lightcurve[np.where(self.raw_lc1[self.quality_mask[0]].quality ==0)]


        if self.principal_components !=0:
            rcc = lk.RegressionCorrector(lightcurve)
            lc = rcc.correct(self.dm.append_constant())
        else:
            lc=lightcurve
        flux = lc.flux.value
        times = lc.time.value - np.nanmean(lc.time.value)
        freq = self.frequency_list
        phase = self.final_phases
        fit = 0
        for i in range(len(self.frequency_list)):
            fit += self.result.params[self.result.var_names[:-2][i]].value*np.sin(2*np.pi*freq[i]*times + phase[i])
        
        plt.scatter(times,flux,s=.5,label='Lightcurve')
        plt.plot(times,fit+np.mean(flux),c='r',linestyle='-',lw=1,alpha=.7,label = 'Lightcurve Fit')

        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.legend()
        

            

        if save != None:
            plt.savefig(save)
            
    def plot(self,frequencylist_index = 0,method = 'amp',save = None,figuresize = (10,10)):
        """Plot the amplitude heatmap, snr, errors, or the fit model.
        Parameters
        ----------
        frequencylist_index: int
        method: 'amp', 'snr', 'errors', 'model'
        save: str, or None
            'filename.png' if you want to save the png of the plot
        figuresize: size of plot
        
        """
        plt.figure(figsize = (figuresize))
        if (method=='amp'):
            plt.imshow(self.heats[frequencylist_index].reshape(self.tpf.shape[1:]),origin='lower')
        elif (method=='snr'):
            plt.imshow(self.heats[frequencylist_index].reshape(self.tpf.shape[1:])/self.heats_error[frequencylist_index].reshape(self.tpf.shape[1:]),origin='lower')
        elif (method=='errors'):
            plt.imshow(self.heats_error[frequencylist_index].reshape(self.tpf.shape[1:]),origin='lower')
        elif (method=='model'):
            prf = PRF.TESS_PRF(cam = self.tpf.camera, ccd = self.tpf.ccd,
                               sector = self.tpf.sector, colnum = self.tpf.column+self.tpf.pipeline_mask.shape[0]/2.,
                               rownum = self.tpf.row+self.tpf.pipeline_mask.shape[1]/2.)
            model = prf.locate(self.location[0],self.location[1], self.tpf.shape[1:])
            plt.imshow(model,origin='lower')
        
        if (self.gaiadata != None):
            plt.scatter(self.gaiadata['x'],self.gaiadata['y'],s=self.gaiadata['size']*5,c='white',alpha=.6)
            plt.scatter(self.location[0],self.location[1],marker='X',c ='black',s=70)
            
        plt.xlim(-.5,self.aperture.shape[1]-1+.5)
        plt.ylim(-.5,self.aperture.shape[0]-1+.5)
        if save != None:
            plt.savefig(save)
