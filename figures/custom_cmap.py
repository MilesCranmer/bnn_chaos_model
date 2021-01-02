import numpy as np
from matplotlib import colors as mcolors

colorstr = """*** Primary color:

   shade 0 = #A0457E = rgb(160, 69,126) = rgba(160, 69,126,1) = rgb0(0.627,0.271,0.494)
   shade 1 = #CD9CBB = rgb(205,156,187) = rgba(205,156,187,1) = rgb0(0.804,0.612,0.733)
   shade 2 = #BC74A1 = rgb(188,116,161) = rgba(188,116,161,1) = rgb0(0.737,0.455,0.631)
   shade 3 = #892665 = rgb(137, 38,101) = rgba(137, 38,101,1) = rgb0(0.537,0.149,0.396)
   shade 4 = #74104F = rgb(116, 16, 79) = rgba(116, 16, 79,1) = rgb0(0.455,0.063,0.31)

*** Secondary color (1):

   shade 0 = #CDA459 = rgb(205,164, 89) = rgba(205,164, 89,1) = rgb0(0.804,0.643,0.349)
   shade 1 = #FFE9C2 = rgb(255,233,194) = rgba(255,233,194,1) = rgb0(1,0.914,0.761)
   shade 2 = #F1D195 = rgb(241,209,149) = rgba(241,209,149,1) = rgb0(0.945,0.82,0.584)
   shade 3 = #B08431 = rgb(176,132, 49) = rgba(176,132, 49,1) = rgb0(0.69,0.518,0.192)
   shade 4 = #956814 = rgb(149,104, 20) = rgba(149,104, 20,1) = rgb0(0.584,0.408,0.078)

*** Secondary color (2):

   shade 0 = #425B89 = rgb( 66, 91,137) = rgba( 66, 91,137,1) = rgb0(0.259,0.357,0.537)
   shade 1 = #8C9AB3 = rgb(140,154,179) = rgba(140,154,179,1) = rgb0(0.549,0.604,0.702)
   shade 2 = #697DA0 = rgb(105,125,160) = rgba(105,125,160,1) = rgb0(0.412,0.49,0.627)
   shade 3 = #294475 = rgb( 41, 68,117) = rgba( 41, 68,117,1) = rgb0(0.161,0.267,0.459)
   shade 4 = #163163 = rgb( 22, 49, 99) = rgba( 22, 49, 99,1) = rgb0(0.086,0.192,0.388)

*** Complement color:

   shade 0 = #A0C153 = rgb(160,193, 83) = rgba(160,193, 83,1) = rgb0(0.627,0.757,0.325)
   shade 1 = #E0F2B7 = rgb(224,242,183) = rgba(224,242,183,1) = rgb0(0.878,0.949,0.718)
   shade 2 = #C9E38C = rgb(201,227,140) = rgba(201,227,140,1) = rgb0(0.788,0.89,0.549)
   shade 3 = #82A62E = rgb(130,166, 46) = rgba(130,166, 46,1) = rgb0(0.51,0.651,0.18)
   shade 4 = #688C13 = rgb(104,140, 19) = rgba(104,140, 19,1) = rgb0(0.408,0.549,0.075)"""

colors = []
shade = 0
for l in colorstr.replace(' ', '').split('\n'):
    elem = l.split('=')
    if len(elem) != 5: continue
    if shade == 0:
        new_color = []
    rgb = lambda x, y, z: np.array([x, y, z]).astype(np.float32)
    
    new_color.append(eval(elem[2]))
    
    shade += 1
    if shade == 5:
        colors.append(np.array(new_color))
        shade = 0
colors = np.array(colors)/255.0

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

# custom_cmap = make_colormap([
    # colors[2][4],
    # colors[1][1]
# ][::-1])
custom_cmap = make_colormap([
    colors[2][4],
    np.array([1., 1., 1.])
][::-1])

