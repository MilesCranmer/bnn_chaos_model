from matplotlib import pyplot as plt
import rebound
import numpy as np
from rebound.plotting import OrbitPlotOneSlice
from celluloid import Camera
from scipy import interpolate
import numpy as onp
import matplotlib as mpl

def plotat(sim, t, ax):

    sim = rebound.Simulation()
    sim.add(m=1.0)
    sim.add(m=1.0e-2, a=1.0)
    sim.add(m=1.3e-2, a=1.3, e=0.01, f=56)
    sim.add(m=0.5e-2, a=2, e=0.3, f=180)
    sim.integrate(t, exact_finish_time=True)
    fig = OrbitPlotOneSlice(sim, ax, trails=True)
    ax = plt.gca()
    for key in ax.spines:
        ax.spines[key].set_visible(False)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelbottom=False,
                   labeltop=False,
                   labelleft=False,
                   labelright=False,
                   bottom=False,
                   left=False,
                    axis='both',
                  which='both')
    return sim



sim = rebound.Simulation()
sim.add(m=1.0)
sim.add(m=1.0e-3, a=1.0)
sim.add(m=1.3e-3, a=1.3, e=0.01, f=56)
sim.add(m=0.5e-3, a=2, e=0.3, f=180)
ps = []
for t in np.arange(2150):
    sim.integrate(t)
    sp = sim.calculate_com()
    shift = (sp.x,sp.y,sp.vx,sp.vy)
    ps.append([
        [sim.particles[i].x-shift[0], sim.particles[i].y-shift[1]] for i in range(4)
    ])
ps = np.array(ps)

mpl.cm.__dir__()

colors = np.array([
    [116, 16, 79],#primary1
    [149, 104, 20],#primary2
    [22, 49, 99],#comp2
    [104, 140, 19]#comp1
])

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
    rgb = lambda x, y, z: np.array([x, y, z])
    
    new_color.append(eval(elem[2]))
    
    shade += 1
    if shade == 5:
        colors.append(np.array(new_color))
        shade = 0
        
colors = np.array(colors)

def make_transparent_color(ntimes, fraction, minalpha=0.0):
    rgba = onp.ones((ntimes, 4))
    alpha = onp.linspace(minalpha, 1, ntimes)[:, np.newaxis]
    shade = 2
    rgba[:, 0] = colors[fraction, shade, 0]
    rgba[:, 1] = colors[fraction, shade, 1]
    rgba[:, 2] = colors[fraction, shade, 2]
    rgba[:, 3] = 255
    rgba /= 255.0
    rgba[:, :] = 1*(1-alpha) + rgba*alpha
    rgba[:, 3] = alpha[:, 0]
    return rgba

plt.style.use('science')

mult = 2
fig, ax = plt.subplots(1, 1, dpi=600, figsize=(20/mult, 4/mult))
spread = 100
for i in range(4):
    tck, u = interpolate.splprep(ps[:, i, [0, 1]].T, s=0)
    unew = np.linspace(0, 1, num=100000)
    out = interpolate.splev(unew, tck)
    label = 'Planet %d'%(i,) if i > 0 else None
    dt = 100000//8
    allsub = np.arange(0, 100000, dt)
    for sub in allsub:
        if sub == allsub[1]: continue
        color = make_transparent_color(dt, (i-1)%4)
        sizes = np.ones(dt)*0.01/mult
        sizes[-1] *= 2000
        plt.scatter(
            (out[0]+(sub/dt/8.0)*spread)[sub:sub+dt],
            (-out[1])[sub:sub+dt],
            color=color,
            s=sizes,
            label=(label if sub == allsub[-1] else None))
        
    
prim = sim.particles[0]
for i in range(8):
    if i == 1: continue
    ax.scatter(spread*i/8.0, 0.0, marker="*",
           s=35/mult, facecolor="white",
           edgecolor='black', zorder=3)

ax.axis('equal')

for key in ax.spines:
    ax.spines[key].set_visible(False)

ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(labelbottom=False,
               labeltop=False,
               labelleft=False,
               labelright=False,
               bottom=False,
               left=False,
               top=False,
               right=False,
               axis='both',
               which='both')

plt.savefig('orbits.png', dpi=600)
