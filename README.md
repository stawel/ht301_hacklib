# ht301_hacklib
Thermal camera opencv python lib.

Supported thermal cameras:
- Hti HT-301
- Xtherm T3S, thanks to Angel-1024!
- Xtherm T2L, T2S+, thanks to Christoph Mair

It's a very simple hacked together lib, might be useful for somebody,  
uses `matplotlib` which is a little bit on the slow side,  
or pure `opencv` - much faster but without most features

Tested on ubuntu 20.04 and windows 11:

```
$ ./pyplot.py
keys:
    'h'      - help
    'q'      - quit
    ' '      - pause, resume
    'd'      - set diff
    'x','c'  - enable/disable diff, enable/disable annotation diff
    'f'      - full screen
    'u'      - calibrate
    't'      - draw min, max, center temperature
    'e'      - remove user temperature annotations
    'w'      - save to file date.png
    'r'      - save raw data to file date.npy
    ',', '.' - change color map
    'a', 'z' - auto exposure on/off, auto exposure type
    'k', 'l' - set the thermal range to normal/high (only for the TS2+)
    'o'      - change the camera orientation (rotate 90 degs CW). Only in OpenCV
    
    left, right, up, down - set exposure limits

mouse:
    left  button - add Region Of Interest (ROI)
    right button - add user temperature annotation
```
![pyplot output](docs/pyplot-output1.png)
![pyplot output](docs/pyplot-output2.png)

View saved ('r' key) raw data file:
```
$ ./pyplot.py 2022-09-11_18-49-07.npy
```

Opencv version:
```
$ ./opencv.py
```
![opencv output](docs/opencv-output.png)

<br>

## Related projects

- https://gitlab.com/netman69/inficam
- https://github.com/MCMH2000/OpenHD_HT301_Driver
- https://github.com/sumpster/ht301_viewer
- https://github.com/cmair/ht301_hacklib
- https://github.com/mcguire-steve/ht301_ircam

## Related materials
- https://www.mdpi.com/1424-8220/17/8/1718

