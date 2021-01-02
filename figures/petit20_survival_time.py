import numpy as np
import pandas as pd

from math import gcd
from numpy import pi
import scipy as sc


def Tsurv(nu12,nu23,masses,m0=1.,fudge=1,res=False):
    """
    Main result from the paper. Return the survival time estimate as a function of the initial period ratios and masses (eq. 81).
    In units of of the innermost orbit
    Returns np.inf if separation wide enough that 3-body MMRs don't overlap.
    
    nu12, nu23 : Initial period ratios
    masses : planet masses
    m0 : star mass
    fudge : fudge factor to adjust the number of resonances taken into account for more than three planets. Usually 1.
    res : Boolean. True means that the exact distance to the closest resonance is used. However, since we do not take into account the actual shape of the two planet MMR the results are less good than by assuming a constant distance $\nu/2$ to the MMR.
    """
    plsepov = get_plsep_ov(nu12,nu23,masses,m0)*fudge**.25
    al12 = nu12**(2/3)
    al23 = nu23**(2/3)
    eta = nu12*(1-nu23)/(1-nu12*nu23)
    plsep = (1-al12)*(1-al23)/(2-al12-al23)
    

    try:
        Tnorm = 2**1.5/9*(plsep/plsepov)**6/(1-(plsep/plsepov)**4)*10**(-np.log(1-(plsep/plsepov)**4))
    except:
        return np.inf
        
    A = np.sqrt(38/pi)
    Mfac = get_Mfac(nu12,nu23,masses,m0)
    PrefacD = Mfac*nu12*A*np.sqrt(eta*(1-eta))*fudge**-2
    if res:
        Deta,u0=distance_eta_diffusion_direction(nu12,nu23,masses)
        Detanorm = np.maximum(Deta,np.sum(masses)**(2/3)/plsep**(1/3))/plsep # the Deta is in unit of plsep since plsep already in Tnorm
        # Minimum distance is comparable to resonance size
        u0 = np.minimum(0.7,np.maximum(0.3,u0)) #Bounds to somewhat take into account the time spent inside the MMR
        Tsurv = ((Detanorm)**2/(PrefacD)*Tnorm*3/2*u0**2*(1-u0)**2)
    else:
        Tsurv = (3/2)**2/PrefacD*Tnorm*3/32 #Deta=3/2 in units of plsep
    return np.nan_to_num(Tsurv,nan=np.inf)
    
def Tnorm(nu12,nu23,masses,fudge=1,exact_F=True):
    "In unit of P_1"
    plsepov = get_plsep_ov(nu12,nu23,masses)*fudge
    al12 = nu12**(2/3)
    al23 = nu23**(2/3)
    eta = nu12*(1-nu23)/(1-nu12*nu23)
    plsep = (1-al12)*(1-al23)/(2-al12-al23)
    
    if exact_F:
        xiov = exact_xiov(nu12,nu23,masses,fudge=fudge)
        F = (1-sc.special.dawsn((xiov/2)**.5)/(xiov/2)**.5)**2
        Tnorm = xiov*(xiov+1)/(1-(plsep/plsepov)**4)*F
    else:
        Tnorm = 2**1.5/9*(plsep/plsepov)**6/(1-(plsep/plsepov)**4)*10**(-np.log(1-(plsep/plsepov)**4))
    return Tnorm


def _exact_xiov(nu12,nu23,masses,fudge=1):
    plsep_ov = get_plsep_ov(nu12,nu23,masses)*fudge**.25
    al12 = nu12**(2/3)
    al23 = nu23**(2/3)
    plsep = (1-al12)*(1-al23)/(2-al12-al23)
    return -1-sc.special.lambertw((-1+(plsep/plsep_ov)**4)/np.exp(1),-1).real

# Random walk distribution
def distribution_exittimes(t,u0,du=1.,D=1.,N=100):
    """Formula for the exit time probability distribution 
    
    t is the time 
    u0 is the distance to the boundary
    du and D are units that are set to 1 in  general 
    """
    res = 0
    for k in np.arange(-N-1,N+1):
        uk = du-u0+k*du
        res+=uk/(4*pi*D*t**3)**.5*(-1.)**k*np.exp(-uk**2/(4*D*t))
    return res


#Auxiliary functions

def get_Mfac(nu12,nu23,masses,m0=1):
    m1,m2,m3 = masses
    eta = nu12*(1-nu23)/(1-nu12*nu23)
    return (m1*m3/m0**2*(eta**2/nu12**(4/3)*m2/m1+1+(1-eta)**2*nu23**(4/3)*m2/m3))**.5

def get_plsep_ov(nu12,nu23,masses,m0=1.):
    m1,m2,m3 = masses
    eta = nu12*(1-nu23)/(1-nu12*nu23)
    plsep = 1/(1/(1-nu12**(2/3))+1/(1-nu23**(2/3)))
    Mfac = get_Mfac(nu12,nu23,masses)
    Ares = 4*2**.5*np.sqrt(38/pi)/3
    return (Mfac*Ares*(eta*(1-eta))**1.5)**.25

#Change of variables in period ratio space
def nus_to_eta(nu12,nu23):
    return nu12*(1-nu23)/(1-nu12*nu23)

def nus_to_nu_eta(nu12,nu23):
    return 1/(1/(1/nu12-1)+1/(1-nu23)),nu12*(1-nu23)/(1-nu12*nu23)
def nu_eta_to_nus(nu,eta):
    return eta/(eta+nu),(1-eta-nu)/(1-eta)


#Diffusion direction
#Not using it extensively so please ask for documentation if interested
#Exact curve but not pratical
import scipy.integrate as integrate

def diffusion_direction_curve(nu12,nu23,masses=np.ones(3)):
    'From an initial position and masses, computes the diffusion curve by solving the differential equation'
    solfor = integrate.solve_ivp(lambda t,y: diffusion_slope(t,y,masses=masses),
                                 (nu12,1),np.array([nu23]),events=_eventnu23,max_step=0.01)
    solback = integrate.solve_ivp(lambda t,y: diffusion_slope(t,y,masses=masses),
                                  (nu12,0),np.array([nu23]),events=_eventnu23,max_step=0.01)
    nu12s = np.concatenate((solback.t[:0:-1],solfor.t))
    nu23s = np.concatenate((solback.y[0][:0:-1],solfor.y[0]))
    return nu12s,nu23s
def _eventnu23(nu12,nu23):
    return 1-nu23
_eventnu23.terminal=True
    
def diffusion_slope(nu12,nu23,masses=np.ones(3)):
    return -nu23/nu12*(1-nu12*nu23+(1-nu12)*masses[0]/masses[1]*nu12**(-1/3))/(1-nu12*nu23+nu12*(1-nu23)*masses[2]/masses[1]*nu23**(1/3))

def integrate_gradeta(nu12,nu23,masses=np.ones(3)):
    solfor = integrate.solve_ivp(lambda t,y: gradeta_slope(t,y,masses=masses),
                                 (nu12,1),np.array([nu23]),events=eventnu23,max_step=0.01)
    solback = integrate.solve_ivp(lambda t,y: gradeta_slope(t,y,masses=masses),
                                  (nu12,0),np.array([nu23]),events=eventnu23,max_step=0.01)
    nu12s = np.concatenate((solback.t[:0:-1],solfor.t))
    nu23s = np.concatenate((solback.y[0][:0:-1],solfor.y[0]))
    return nu12s,nu23s

def gradeta_slope(nu12,nu23,masses=np.ones(3)):
    eta = nu12*(1-nu23)/(1-nu12*nu23)
    return -eta/(1-eta)*nu12**-2

#Good even if approx
def distance_eta_diffusion_direction(nu12,nu23,masses=np.ones(3)):
    P = np.ceil(nu12/(1-nu12))
    Q = np.ceil(nu23/(1-nu23))
    eta0 = nus_to_eta(nu12,nu23)

    s0 = diffusion_slope(nu12,nu23,masses)

    # res up
    nu12up = (Q/(Q+1)-nu23)/s0+nu12
    # res down
    nu12do = (1-1/Q-nu23)/s0+nu12
    #res right
    nu23ri = nu23+s0*(P/(P+1)-nu12)
    #res left
    nu23le = nu23+s0*(1-1/P-nu12)

    nu12sma = np.maximum(nu12up,1-1/P)
    nu23sma = np.minimum(nu23le,Q/(Q+1))
    etasma = nus_to_eta(nu12sma,nu23sma)

    nu12lar = np.minimum(nu12do,(P)/(P+1))
    nu23lar = np.maximum(nu23ri,1-1/Q)
    etalar = nus_to_eta(nu12lar,nu23lar)

    deta = etalar-etasma

    return deta,(eta0-etasma)/(etalar-etasma)
