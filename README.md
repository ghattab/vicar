# MarkerFreeAlignment

A data driven and marker-free image alignment approach to correct spatial shift in bacterial time series. 
This work was funded by the German-Canadian DFG International Research Training Group GRK 1906/1 and the “Phenotypic Heterogeneity and Sociobiology of Bacterial Populations” DFG SPP1617.

<p align="center">
<img 
src="https://cloud.githubusercontent.com/assets/13886161/10096344/06099b76-6371-11e5-9066-e0451aa0aae9.gif" 
loop=infinite 
alt="Example result"
width="600">
</p>

## Data

The employed datasets are available under The Open Data Commons Attribution License (ODC-By) v1.0.

Schlueter, J. - P., McIntosh, M., Hattab, G., Nattkemper, T. W., and Becker, A. (2015). Phase Contrast and Fluorescence Bacterial Time-Lapse Microscopy Image Data. Bielefeld University. [doi:10.4119/unibi/2777409](http://doi.org/10.4119/unibi/2777409).

## Dependencies

For better reproducibility the versions that were used for development are mentioned in parentheses.

* Python (2.7.10)
* OpenCV (2.4.9)
* Image Processing SciKit (skimage 0.11dev)
* Python Imaging Library (PIL 1.1.7)

## Usage

```bash
# Set file permissions
$ chmod +x main.py 

# Run alignment on a folder containing all image files 
# Formatted by channel : luminance, red, green, blue as c1, c2, c3, c4 respectively for every time point
$./main.py -i ../data/img_folder/ -p
#  -h --help                 Prints this
#  -i --input                Supplies a directory containing the data
#  -p --param                Flag for treating images
```

## License
```
The MIT License (MIT)

Copyright (c) 2016 Georges Hattab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
```
