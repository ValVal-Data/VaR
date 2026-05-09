#Define custom disitrbution for GARCH

import scipy.special as sp
import numpy as np
import scipy.integrate as inte
import scipy.optimize as op

#Skewed t-distribution according to Hansen 1994
#Constant calculation
def t_skewed_constants(nu:float,lam:float)->list:
    """
    Skewed t-distribution according to Hansen 1994 constant calculations

    Parameters:
        nu (float): nu parameter
        lam (float): lambda parameter

    Returns:
        a,b,c (list): Parameters for the distribution
    """
    c=np.exp(sp.gammaln((nu+1)/2)-sp.gammaln(nu/2))/np.sqrt(np.pi*(nu-2))
    a=4*lam*c*(nu-2)/(nu-1)
    b=np.sqrt(1+3*nu**2-a**2)
    return a,b,c
#Calculation of the pdf
def t_skewed_pdf(z:float,nu:float,lam:float)->float:
    """
    PDF calculation at z for Skewed t-distribution according to Hansen 1994 constant calculations

    Parameters:
        z (float): Where to calculate the PDF
        nu (float): nu parameter
        lam (float): lambda parameter

    Returns:
        PDF (float)

    Notes:
        - Can handle both single z/(nu,lam) or list of values for each of them
    """
    a,b,c=t_skewed_constants(nu,lam)
    z0=-a/b
    if np.isscalar(z0):
        z0=np.asarray([z0])
        a=np.asarray([a])
        b=np.asarray([b])
        c=np.asarray([c])
    else:
        z0=np.asarray(z0)
        a=np.asarray(a)
        b=np.asarray(b)
        c=np.asarray(c)
    if np.isscalar(z):
        zz=np.asarray([z])
    else:
        zz=np.asarray(z)
    ll=[]
    for x in range(len(z0)):
        lll=[]
        for y in zz:
            if y<z0[x]:
                lam2=(1-lam)
            else:
                lam2=(1+lam)
            lll.append((b[x]*c[x]/lam2)*(1+(((b[x]*y+a[x])/lam2)**2)/(nu-2))**(-(nu+1)/2))
        ll.append(lll)
    if len(ll)==1 and len(ll[0])==1:
        return ll[0][0]
    elif len(ll)==1 :
        return np.asarray(ll[0])
    else:
        return np.asarray(ll)

def t_skewed_cdf(z:float,nu:float,lam:float)->float:
    """
    CDF calculation at z for Skewed t-distribution according to Hansen 1994 constant calculations

    Parameters:
        z (float): Where to calculate the PDF
        nu (float): nu parameter
        lam (float): lambda parameter

    Returns:
        PDF (float)

    Notes:
        - Use quad, so this function is quite slow
    """
    val, _=inte.quad(t_skewed_pdf,-np.inf,z,args=(nu,lam))
    return val

def t_skewed_ppf(u:float,nu:float,lam:float)->float:
    """
    PPF calculation at z for Skewed t-distribution according to Hansen 1994 constant calculations

    Parameters:
        z (float): Where to calculate the PDF
        nu (float): nu parameter
        lam (float): lambda parameter

    Returns:
        PPF (float)

    Notes:
        - Use brentq, so this function is quite slow
    """
    return op.brentq(lambda x: t_skewed_cdf(x,nu,lam)-u,-5,5)

def t_skewed_cdf(z:float,nu:float,lam:float)->float:
    """
    CDF calculation at z for Skewed t-distribution according to Hansen 1994 constant calculations

    Parameters:
        z (float): Where to calculate the PDF
        nu (float): nu parameter
        lam (float): lambda parameter

    Returns:
        CDF (float)

    Notes:
        - Use quad, so this function is quite slow
    """
    xs=np.linspace(-10,10,20000)
    pdf_vals=t_skewed_pdf(xs,nu,lam)
    val, _=inte.quad(t_skewed_pdf,-np.inf,z,args=(nu,lam))
    return val

