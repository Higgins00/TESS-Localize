# TESS-Localizer
Code for localizing variable star signatures in TESS Photometry.

The primary use of this package is to identify the location on the TPF where sources of variablity found in the aperture originate. The user must provide a list of frequencies found in the aperture that belong to the same source and the number of principal components to remove from the light curve to ensure it is free of systematic trends.

```python
import Disentangler_Draft as dd
import lightkurve as lk
import astropy.units as u

frequency_list = [9.51112996, 19.02225993, 28.53338989, 38.04451986, 47.55564982, 57.06677979, 66.57790975, 76.08903972]

search_result = lk.search_targetpixelfile('TIC117070953')
tpf = search_result.download(quality_bitmask='default')

tutorial_example = dd.PixelMapFit(targetpixelfile=tpf, gaia=True, magnitude_limit=18,
                                  frequencies=frequency_list, frequnit=u.uHz, principal_components = 3)
```

![pca](https://github.com/Higgins00/TESS-Localizer/blob/main/pca.png)
