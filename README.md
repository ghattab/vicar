# MarkerFreeAlignment

A data driven and marker-free image alignment approach to correct spatial shift in bacterial time series. 


## Dependencies

For better reproducibility the versions that were used for development are mentioned in parentheses.

1. Python (2.7.10)
2. OpenCV (2.4.9)
3. Image Processing SciKit (skimage 0.11dev)

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

Copyright (c) 2015 Georges Hattab

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
SOFTWARE.```
