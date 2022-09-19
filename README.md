# TESS-Localize
Code for localizing variable star signatures in TESS Photometry.

The primary use of this package is to identify the location on the TPF where sources of variablity found in the aperture originate. The user only needs to provide a list of frequencies found in the aperture that belong to the same source and the number of principal components needed to be removed from the light curve to ensure it is free of systematic trends. A more detailed usage example can be found in the examples folder.

See the paper describing the methodology behind this software [on ADS](https://ui.adsabs.harvard.edu/abs/2022arXiv220406020H/abstract).

## Installation 

To install use 

```pip install TESS_Localize```

Or clone this repository and run 
```bash
cd TESS_Localize
pip install .
```

## Usage

```python
import TESS_Localize as tl
import lightkurve as lk
import astropy.units as u

frequency_list = [9.51112996, 19.02225993, 28.53338989, 38.04451986, 47.55564982, 57.06677979, 66.57790975, 76.08903972]

search_result = lk.search_targetpixelfile('TIC117070953')
tpf = search_result.download(quality_bitmask='default')

tutorial_example = tl.Localize(targetpixelfile=tpf, gaia=True, magnitude_limit=18,
                                  frequencies=frequency_list, frequnit=u.uHz, principal_components = 3)
```
### Useful Functions:
```python
tutorial_example.pca()
```
Result: Plots of the individual prinicpal components and plots of the lightcurve before and after removing them.

![pca](https://github.com/Higgins00/TESS-Localizer/blob/main/pca.png)

```python
tutorial_example.info()
```
Result: Gaia sources filtered by their distance from the fitted location of the source, and the fit statistics.

![result1](https://github.com/Higgins00/TESS-Localizer/blob/main/result.png)

```python
tutorial_example.plot(frequencylist_index=2,method='snr')
```
Result: An image of the signal to noise heatmap created from the amplitude of the most significant frequency at every pixel. The white dots are Gaia sources, and the black 'X' is the best fit location of the source.

![Plot](https://github.com/Higgins00/TESS-Localizer/blob/main/SNR.png)
