# coding: utf-8

from __future__ import division, print_function
import logging
import warnings
import numpy as np
import os
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
import lightkurve as lk
from astropy import units as u
import lmfit as lm
from lmfit import Minimizer, Parameters, report_fit
import PRF


class PixelMapFit:
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
    principal_components: int
        Number of components used in PCA for TPF lightcurve.
    aperture: 2D Boolean array
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
    self.principal_components
        Number of principal components removed.
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
        Best fit source location.
    self.heatmap
        2D array of composite heatmap for all frequencies.
    self.starfit
        Gaia sources and their distances from the fitted location of the source.
    self.result
        Result paramters of the fit. Use report_fit(self.report) to view.
    
    
    """
    
    def __init__(self, targetpixelfile, gaia=True, magnitude_limit=18, 
                 frequencies=[], frequnit=u.uHz, principal_components = 5, 
                 aperture=None, **kwargs):
        
        self.tpf = targetpixelfile
        #Defining an aperture that will be used in plotting and making empty 2-d arrays of the correct size for masks
        self.aperture = aperture
        if self.aperture is None:
            self.aperture = targetpixelfile.pipeline_mask
            if targetpixelfile.pipeline_mask.any() == False:
                #will add a flag here if no aperture
                self.aperture = self.tpf.create_threshold_mask()
    
        
        # Make a design matrix and pass it to a linear regression corrector
        self.raw_lc = self.tpf.to_lightcurve(aperture_mask=self.aperture)
        self.dm = lk.DesignMatrix(tpf.flux[:,~tpf.pipeline_mask][~np.isnan(raw_lc['flux']),:], name='regressors').pca(principal_components)
        
        #fixing bug where PCA cant occur if there is a nan in flux_err
        self.raw_lc['flux_err'][np.isnan(self.raw_lc['flux_err'].value)]=np.nanmean(self.raw_lc['flux_err'])
        
        rc = lk.RegressionCorrector(self.raw_lc.remove_nans())
        corrected_lc = rc.correct(self.dm.append_constant())
        corrected_lc[np.where(corrected_lc.quality == 0)]
        self.corrected_lc = corrected_lc.remove_outliers()
        self.frequency_list = np.asarray((frequencies*frequnit).to(1/u.d))
        self.principal_components = principal_components
        
        def Obtain_Initial_Phase(tpf,corrected_lc,frequency_list):

            flux = corrected_lc.flux.value
            times = corrected_lc.time.value - np.nanmean(corrected_lc.time.value)
            #appending a extra value here to ensure lightkurve doesn't call an error if length of frequencies =1
            pg = corrected_lc.to_periodogram(frequency = np.append([0.0001],frequency_list),ls_method='slow')
            initial_flux= np.asarray(pg.power[1:])

            initial_phase = np.zeros(len(frequency_list))

            def lc_model(time,amp,freq,phase):
                return amp*np.sin(2*np.pi*freq*time + phase)

            def background_model(time,height):
                return np.ones(len(time))*height
            for j in np.arange(len(frequency_list)):
                for i in np.arange(len(frequency_list)):

                    if (i == 0):
                        model = lm.Model(lc_model,independent_vars=['time'],prefix='f{0:d}'.format(i)) 
                        model += lm.Model(background_model, independent_vars=['time'])
                    else:
                        model += lm.Model(lc_model,independent_vars=['time'],prefix='f{0:d}'.format(i))


                    model.set_param_hint('f{0:d}phase'.format(i), min = -np.pi, max = np.pi ,value= initial_phase[i],vary = False)
                    model.set_param_hint('f{0:d}amp'.format(i), value = initial_flux[i],vary=False)
                    model.set_param_hint('height', value= np.nanmean(flux),vary=False)
                    model.set_param_hint('f{0:d}freq'.format(i),value = frequency_list[i], vary = False)


                params = model.make_params()
                params['f{0:d}phase'.format(j)].set(vary=True)
                params['f{0:d}phase'.format(j)].set(value = initial_phase[j])
                params['f{0:d}phase'.format(j)].set(brute_step=np.pi/10)
                result = model.fit(corrected_lc.flux.value,params,time=times,method = 'brute')
                initial_phase[j]=result.best_values['f{0:d}phase'.format(j)]

            return initial_phase
        
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

            result = model.fit(corrected_lc.flux.value,params,time=times,weights=corrected_lc.flux_err.value)
            
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

            result = model.fit(corrected_lc.flux.value,params,time=times,weights=corrected_lc.flux_err.value)
            
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
                
                #fixing bug where PCA cant occur if there is a nan in flux_err
                lightcurve['flux_err'][np.isnan(lightcurve['flux_err'].value)]=np.nanmean(lightcurve['flux_err'])
                
                rcc = lk.RegressionCorrector(lightcurve.remove_nans())
                lc = rcc.correct(self.dm.append_constant())
                              
                bestfit = Obtain_Final_Fit(self.tpf,lc,self.frequency_list,self.final_phases)
                heat = np.asarray([bestfit.best_values['f{0:d}amp'.format(n)] for n in np.arange(len(self.frequency_list))])
                heat_error =  np.asarray([bestfit.params['f{0:d}amp'.format(n)].stderr for n in np.arange(len(self.frequency_list))])
                
                #Extending the list of fitting data for each pixel
                heats.extend([heat])
                heats_error.extend([heat_error])
                    
        heats = np.asarray(heats)
        heats_error = np.asarray(heats_error)
        self.heatmap = heats.T
        self.heatmap_error = heats_error.T
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
            result = Vizier.query_region(c1, catalog=["I/345/gaia2"],radius=Angle(np.max(self.tpf.shape[1:]) * pix_scale, "arcsec"))

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

            def __init__(self,tpf,heats,heats_error,frequencies,gaia_data):
                self.heat_stamp = heats
                self.gaiadata=gaia_data
                self.heatmap_error = heats_error
                self.size = tpf.pipeline_mask.shape
                self.frequencies= frequencies
                self.tpf = tpf
            def location(self):
                self.prf = PRF.TESS_PRF(cam = self.tpf.camera, ccd = self.tpf.ccd,
                                    sector = self.tpf.sector, colnum = self.tpf.column,
                                    rownum = self.tpf.row, **kwargs)
                
                #Residuals to minimize relative to the error bars
                def residual(params, amp, amperr, prf):

                    x = params['x']
                    y = params['y']
                    
                    res = []
                    for i in np.arange(len(frequencies)):
                        height = params['height{0:d}'.format(i)]
                        #prf = PRF.Gaussian_PRF(sigma)
                        model = height*prf.locate(x+.5, y+.5, (self.size[0],self.size[1]))

                        res.extend( [(amp[i].reshape(self.size)-model) / amperr[i].reshape(self.size)])


                    return np.asarray(res)

                #Set starting values to converge from
                self.heatmap_error[np.where(self.heatmap_error==None)]=np.nan
                self.heatmap_error[np.where(self.heatmap_error==0)]=np.nan
                #reshaping composite heatmap for user to plot
                composite_heatmap = self.heat_stamp.sum(axis=0).reshape(self.size) / ((np.nansum(self.heatmap_error**2,axis=0))**(1/2)).reshape(self.size)#issue with numpy using sqrt?
                c = np.where(composite_heatmap==composite_heatmap.max())


                params = Parameters()
                for i in np.arange(len(frequencies)):#
                    params.add('height{0:d}'.format(i), value=np.max(self.heat_stamp[i]))
                params.add('x', value=c[1][0])#c[0]) 
                params.add('y', value=c[0][0])#c[1])
                #params.add('sigma', value=1)

                
                #Do the fit
                minner = Minimizer(residual, params, fcn_args=(self.heat_stamp, self.heatmap_error, self.prf))
                self.result = minner.minimize()
                fit = self.result.params.valuesdict()
                self.x = fit['x']
                self.y = fit['y']

                
            def star_list(self):
                gaia_data = self.gaiadata
                no_gaia_data_message = ValueError('No gaia data initialized in PixelMapPeriodogram class')
                if gaia_data ==None :
                    starlist = None

                else:
                    distances = np.square(self.x-gaia_data['x'])+np.square(self.y-gaia_data['y'])
                    closest_star_mask = np.where(np.square(self.x-gaia_data['x'])+np.square(self.y-gaia_data['y'])==(np.square(self.x-gaia_data['x'])+np.square(self.y-gaia_data['y'])).min())
                    stars = dict(ra = np.asarray(gaia_data['ra']),
                                 dec = np.asarray(gaia_data['dec']),
                                 source = np.asarray(gaia_data['source']),
                                 x = np.asarray(gaia_data['x']),
                                 y = np.asarray(gaia_data['y']),
                                 distance = distances)
                                 #include a probability metric here eventually 
                        
                    starlist = pd.DataFrame.from_dict(stars)
                    self.stars = starlist.sort_values(by=[r'distance'])
                    
        fh = frequency_heatmap(self.tpf,self.heatmap,self.heatmap_error,self.frequency_list,self.gaiadata) 
        fh.location()
        self.location = [fh.x,fh.y]
        self.heatmap = self.heatmap.sum(axis=0).reshape(self.aperture.shape[0],self.aperture.shape[1]) / np.sqrt((self.heatmap_error**2).sum(axis=0)).reshape(self.aperture.shape[0],self.aperture.shape[1])
        self.result = fh.result
        if (self.gaiadata !=None):
            fh.star_list()
            self.starfit= fh.stars.reset_index()

    
    def info(self):
        plt.imshow(self.heatmap,origin='lower')
        #plot the location
        if (self.gaiadata != None):
            plt.scatter(self.gaiadata['x'],self.gaiadata['y'],s=self.gaiadata['size']*5,c='white',alpha=.6)
            plt.scatter(self.location[0],self.location[1],marker='X',c='black',s=70)
            print(self.starfit)
        plt.xlim(-.5,self.aperture.shape[1]-1+.5)
        plt.ylim(-.5,self.aperture.shape[0]-1+.5)
        report_fit(self.result)
        if (np.asarray([self.result.params['height{0:d}'.format(j)].stderr for j in np.arange(len(self.frequency_list))]) / np.asarray([self.result.params['height{0:d}'.format(j)].value for j in np.arange(len(self.frequency_list))])).any() >.2:
            Warning('Frequencies used may not all belong to the same source and provided fit could be unreliable')
    
    def pca(self):
        plt.figure(figsize=(12,5))
        plt.plot(self.tpf.time.value, self.dm.values + np.arange(self.principal_components)*0.2)
        plt.title('Principal Components Contributions')
        plt.xlabel('Offset')
        g2 = self.raw_lc.plot(label='Raw light curve')
        self.corrected_lc.plot(ax=g2, label='Corrected light curve')

        plt.show()

    def plot(self,save = None,figuresize = (10,10)):
        plt.figure(figsize = (figuresize))
        plt.imshow(self.heatmap,origin='lower')
        
        if (self.gaiadata != None):
            plt.scatter(self.gaiadata['x'],self.gaiadata['y'],s=self.gaiadata['size']*5,c='white',alpha=.6)
            plt.scatter(self.location[0],self.location[1],marker='X',c ='black',s=70)
            
        plt.xlim(-.5,self.aperture.shape[1]-1+.5)
        plt.ylim(-.5,self.aperture.shape[0]-1+.5)
        if save != None:
            plt.savefig(save)
