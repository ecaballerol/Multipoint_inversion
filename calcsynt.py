import numpy as np


def sdr2MT(strike,dip,rake,M0=1e28):
    '''
    Fill self.MT from strike, dip, rake and M0 (optional,default M0=1e28 dyne-cm)
    Args:
    * strike, dip, rake angles in deg
    * M0 in dyne-cm (optional)
    '''
    # deg to rad conversion
    Phi    = strike * np.pi/180.
    Delta  = dip    * np.pi/180.
    Lambda = rake   * np.pi/180.
    
    # Sin/Cos 
    sinP = np.sin(Phi)
    cosP = np.cos(Phi)
    sin2P = np.sin(2.*Phi)
    cos2P = np.cos(2.*Phi)
    sinD = np.sin(Delta)
    cosD = np.cos(Delta)
    sin2D = np.sin(2.*Delta)
    cos2D = np.cos(2.*Delta)
    sinL = np.sin(Lambda)
    cosL = np.cos(Lambda)
    
    # MT
    MT = np.zeros((6,))
    MT[0] = +M0 * sin2D * sinL
    MT[1] = -M0 * (sinD * cosL * sin2P + sin2D * sinL * sinP*sinP)
    MT[2] = +M0 * (sinD * cosL * sin2P - sin2D * sinL * cosP*cosP)
    MT[3] = -M0 * (cosD * cosL * cosP  + cos2D * sinL * sinP)
    MT[4] = +M0 * (cosD * cosL * sinP  - cos2D * sinL * cosP)
    MT[5] = -M0 * (sinD * cosL * cos2P + 0.5 * sin2D * sinL * sin2P)

    # All done
    return MT

def synt(times,strikes,dips,rakes,Green,data,npts,M0s=None):

    # Number of sources
    nsrc = len(times)

    # Convert times to sample index assuming delta=1
    Gi0 = np.around(times,decimals=0).astype('int')
    
    # Compute synthetics for unitary moment tensors
    G = np.zeros((data.size,len(times)),dtype='float64')
    for n in range(nsrc):
        # Moment Tensor
        MT = sdr2MT(strikes[n],dips[n],rakes[n],M0=1.)
        # Green's functions
        pred = Green[n].dot(MT)
        # Synthetics
        i0 = 0
        for sta in range(len(npts)):
            G[i0+Gi0[n]:i0+npts[sta],n] = pred[i0:i0+npts[sta]-Gi0[n]]
            i0 += npts[sta]                 

    # Invert seismic moment and compute final predictions
    if M0s is None:
        M0s = np.linalg.inv(G.T.dot(G)).dot(G.T.dot(data))
    pred = G.dot(M0s)

    # All done
    return pred, M0s
