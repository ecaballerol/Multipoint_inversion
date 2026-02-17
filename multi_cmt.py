'''
Class for multi point source cmt problems

'''

# Ext modules

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import h5py
from glob import glob
from scipy import signal
import shutil
import pickle

#Personals
import sacpy
import cmt
from metropolis import metropolis

from Arguments import *

# -------------------------------------------
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

def verify(M,prior_bounds):
    '''

    Verify the model parameters are in bounds
    '''

    # Check stuff
    for i in range(M.size):

        if M[i]<prior_bounds[i,0] or M[i]>prior_bounds[i,1]:

            return False
    # All done
    return True

# -------------------------------------------


def calcGFdabase(N_src=None,i_cmt_file=None,GF_DIR=None):
    '''
    Calculate the  GFs database for each source location 
    It is based on Wphase scripts.
    You need to have N_src CMT solutions
    Args:
        N_src:  Number of source to calculate
        i_cmt_file
        GF_DIR: Generic name of the GFs


    '''
    for i in np.arange(N_src):
        shutil.copy('i_master_bk','i_master')
        tmp_cmtfile = i_cmt_file + "_%02d" % (i)
        GF_tmp = GF_DIR + "_%02d" % (i)
        cmd_cmt = "sed -i \'\'  \'s|%s|%s|\' i_master" %(i_cmt_file,tmp_cmtfile)
        os.system(cmd_cmt + ' > GFs.out')
        cmd = 'prep_kernels inv_sac_file_lst l -gfdir %s' %GF_tmp
        os.system(cmd + ' > GFs.out')

    return


class multicmt(object):
    '''
    Classes implementing multi-point functions
    Args:
        active_src: List of active source indices
        cmtp: CMT object with data defined

    '''

    def __init__(self,active_src=None,cmtp=None):

        self.cmtp = cmtp
        self.active_src = active_src
        
        self.Cd = None
        self.iCd = None
        self.bigD = None
        self.bigG = None
        self.map = None

        return

    def prepare_src_kernels(self,GF_names=None, **kwargs):
        '''
        Prepare kernels for the multipoint
        Args:
            cmtp: Individual cmtp problem
            GF_names: List of GF database  
        '''
        assert type(GF_names) is list, "GF_names is not a list"

        self.Green = []

        for i in self.active_src:
            cmtevent = self.cmtp.copy()
            cmtevent.preparekernels(GF_names[i],**kwargs)
            cmtevent.buildG()
            self.Green.append(cmtevent.G.copy())
            
        return 

    def synth(self,times,strikes,dips,rakes,data,npts,M0s=None):
        '''
        Compute synthetics
        '''
        # Convert times to sample index assuming delta=1
        Gi0 = np.around(times,decimals=0).astype('int')
        
        # Compute synthetics for unitary moment tensors
        G = np.zeros((data.size,len(self.active_src)),dtype='float64')
        for n in range(len(self.active_src)):
            # Moment Tensor
            MT = sdr2MT(strikes[n],dips[n],rakes[n],M0=1.)
            # Green's functions
            pred = self.Green[n].dot(MT)
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

    def buildCdfromRes(self,exp_cor_len,npts,PriorTime=None,PriorStrikes=None,\
                       PriorDips=None,PriorRakes=None,relative_error=0.2):
        '''
        Calculate a Cd from an initial solution
        Args:
            exp_cor_len
            Times
            Strikes
            Dips
            Rakes
            npts
        '''
        if PriorTime is None:
            PriorTime   = self.iniTimes
            PriorStrikes = self.iniStrikes
            PriorDips = self.iniDips
            PriorRakes = self.iniRakes
            
        # Calculate residuals for the initial model
        pred, M0s = self.synth(PriorTime,PriorStrikes,PriorDips,PriorRakes,self.cmtp.D,npts)

        R = pred - self.cmtp.D
        sd_all = R.std()

        # Autocorrelation of the residuals
        tcor = (np.arange(2*len(R)-1)-len(R)+1).astype('float64')
        corE = np.exp(-np.abs(tcor)/(exp_cor_len))
        cor = signal.correlate(R,R)
        cor /= cor.max()


        i0 = 0            
        Nc = len(R)    
        Cd = np.zeros((len(self.cmtp.D),len(self.cmtp.D)))
        for i in range(len(npts)):
            sd = np.abs(self.cmtp.D[i0:i0+npts[i]]).max()*relative_error

            if sd<sd_all:
                sd = sd_all

            # Prepare Cd for station
            C = np.zeros((npts[i],npts[i]),dtype='float64')

            for k1 in range(npts[i]):
                for k2 in range(npts[i]):
                    dk = k1-k2
                    C[k1,k2] = corE[Nc+dk-1]*sd*sd
            Cd[i0:i0+npts[i],i0:i0+npts[i]] = C.copy()
            i0 += npts[i]

        # Inverse Cd for station and save it
        iCd = np.linalg.inv(Cd)
        self.Cd = Cd
        self.iCd = iCd
        
        return

    def buildG(self,npts,preweight=True):
        '''
        Preweight the corresponding matrix to
        Args:
            preweight

        '''
        if preweight:
            assert self.Cd is not None, 'Cd is not defined, predefined'

        iCd = self.iCd
        dobs = self.cmtp.D

        G    = []
        for i in range(len(self.active_src)):
            G.append(np.zeros(self.Green[i].shape))

        i0 = 0
        Data = np.zeros(dobs.shape)
        for i in range(len(npts)):
            if preweight:
                L   = np.linalg.cholesky(iCd[i0:i0+npts[i],i0:i0+npts[i]])
                Data[i0:i0+npts[i]] = L.T.dot(dobs[i0:i0+npts[i]])
            else:
                Data[i0:i0+npts[i]] = dobs[i0:i0+npts[i]]

            for n in range(len(self.active_src)):
                if preweight:
                    G[n][i0:i0+npts[i],:] = L.T.dot(self.Green[n][i0:i0+npts[i],:])
                else:
                    G[n][i0:i0+npts[i],:] = self.Green[n][i0:i0+npts[i],:]

            i0 += npts[i]

        self.bigD = Data
        self.bigG = G
        return

    def SetParamap(self):
        '''
        Function to map the parameters index of the active sources
        '''
        iT = np.arange(len(self.active_src))
        iS = np.arange(len(self.active_src)) + len(self.active_src)
        iD = np.arange(len(self.active_src)) + len(self.active_src) * 2
        iR = np.arange(len(self.active_src)) + len(self.active_src) * 3

        self.map={'iT': iT, 'iS':iS,
        'iD':iD, 'iR':iR
        }
        return

    def calcLLK(self,theta,data_dict):
        '''
        Compute Log likelihood
        Args:
            * theta: Model
            * data_dict: Input data
        '''

        # Parse input

        Data  = data_dict['data']
        Green = data_dict['green']
        sig   = data_dict['sigma']
        iT    = data_dict['iT']
        iS    = data_dict['iS']
        iD    = data_dict['iD']
        iR    = data_dict['iR']
        npts = data_dict['npts']

        # Get relevant parameters

        Times   = theta[iT] # Times
        Strikes = theta[iS] # Strike
        Dips    = theta[iD] # Dip
        Rakes   = theta[iR] # Rake

        # Convert times to sample index assuming delta=1
        Gi0 = np.around(Times,decimals=0).astype('int')

        # Compute synthetics for unitary moment tensors

        G = np.zeros((Data.size,len(self.active_src)),dtype='float64')

        for n in range(len(self.active_src)):

            # Moment Tensor
            MT = sdr2MT(Strikes[n],Dips[n],Rakes[n],M0=1.)

            # Green's functions
            pred = Green[n].dot(MT)

            # Synthetics
            i0 = 0

            for sta in range(len(npts)):
                G[i0+Gi0[n]:i0+npts[sta],n] = pred[i0:i0+npts[sta]-Gi0[n]]

                i0 += npts[sta]                 

        # Invert seismic moment and compute final predictions

        M0s = np.linalg.inv(G.T.dot(G)).dot(G.T.dot(Data))
        res = (G.dot(M0s) - Data)/sig

        # Log-Likelihood

        Norm = -(np.log(sig)+0.5*np.log(2*np.pi))*Data.size    
        logLLK = Norm-0.5*(res*res).sum() # log of model likelihood

        # All done

        del G,pred,res  

        return logLLK
    
    def DefineInitMod(self,i_cmt_file,Times=None,Strikes=None,Dips=None,\
                      Rakes=None,M0s=None,priorDict=None):
        '''
        Function to definite initial model for the sampler
        Args:
            Times: Description
            Strikes: Description
            Dips: Description
            Rakes: Description
            M0s: Description
            priorDict: Description
        '''
        
        if priorDict is not None:
            Times = []
            Strikes = []
            Dips = []
            Rakes = []
            with open(priorDict, 'rb') as file:
                data = pickle.load(file)
                for i in self.active_src:
                    tmp_cmtfile = i_cmt_file + "_%02d" % (i)
                    cmttmp = cmt.cmt()
                    cmttmp.read(tmp_cmtfile)
                    FaultName = cmttmp.pdeline.strip().split(',')[0].split(' ')[-1]
                    print(FaultName)
                    Times.append(np.mean(data[FaultName]['time']))
                    Strikes.append(data[FaultName]['sd'][0])
                    Dips.append(data[FaultName]['sd'][1])
                    Rakes.append(np.mean(data[FaultName]['rake']))
                   
        else:
            assert all(x is not None for x in \
            [Times, Strikes, Dips, Rakes]), \
            "Initial Model not completedly defined, check input"

        self.iniTimes   = Times
        self.iniStrikes = Strikes
        self.iniDips    = Dips
        self.iniRakes   = Rakes
        self.iniM0s     = M0s

        return


    def DefinePrior(self,i_cmt_file,prior_bounds=None,priorDict=None):
        '''
        Function to define priori bounds from array or Dictionary
        Args:
            i_cmt_file: CMT file name (used if priorDict is used to get the priori associaited to each fault)
            prior_bounds: List of bounds for each parameter
            Dictionary: Map of parameters index (optional, default self.map)
            
        '''

        # assert prior_bounds is not None and priorDict is not None, 'prior_bounds or Dictionary not defined'
        assert any(x is not None for x in [prior_bounds, priorDict]), "prior_bounds and/or Dictionary not defined"

        self.prior_bounds = []
        if prior_bounds is not None:
            self.prior_bounds = prior_bounds

        elif priorDict is not None:
            prior_bounds  = []
            Param = ['time', 'strike', 'dip', 'rake']
            with open(priorDict, 'rb') as file:
                data = pickle.load(file)
                for iParam in Param:
                    for i in self.active_src:
                        tmp_cmtfile = i_cmt_file + "_%02d" % (i)
                        cmttmp = cmt.cmt()
                        cmttmp.read(tmp_cmtfile)
                        FaultName = cmttmp.pdeline.strip().split(',')[0].split(' ')[-1]
                        # priortmp.append(data[FaultName][iParam])
                        prior_bounds.append(data[FaultName][iParam])
            self.prior_bounds = np.array(prior_bounds)
            
        return

    def MultiSrcInv(self,n_samples,prop_cov,npts,init_Model=None,WriRes=True):
        '''
        Function to inverse multi point sources
        Args:
            n_samples:  Total number of samples
            n_burn:     Number of samples in-burn

        '''
        assert self.bigD is not None, 'Data not defined, buildG first'
        assert self.bigG is not None, 'big G not defined, build G first'
        assert self.map is not None, 'Map of parameters not defined, defined first'
        assert self.prior_bounds is not None, 'Prior bounds not defined, defined first'

        data_dict = {'data':self.bigD, 'green': self.bigG, 'sigma':1.}
        data_dict.update(self.map)
        data_dict['npts']= npts
        prior_bounds = self.prior_bounds

        if init_Model is None:
            init_Model = np.append(np.append(\
            np.append(self.iniTimes,self.iniStrikes),\
                self.iniDips),self.iniRakes)

        #Sample the slip distribution
        print('Metropolis')
        start_time = time.time()

        Samples,LLK,accepted = metropolis(n_samples,self.calcLLK,verify,\
                data_dict,init_Model,prior_bounds,prop_cov,verbose=True)

        run_time = time.time() - start_time

        print('Run time: %.2f s'%(run_time))

        if WriRes:
            h5f=h5py.File('results.h5','w')
            h5f.create_dataset('Samples', data=Samples)
            h5f.create_dataset('LLK', data=LLK)
            h5f.create_dataset('accepted', data=accepted)
            h5f.close()

        return Samples, LLK, accepted
    
    def ReadResults(self,hfile='results.h5'):
        '''
        Read results from h5 file
        Args:
            hfile: h5 file name (default 'results.h5')
        '''
        h5f=h5py.File(hfile,'r')

        Samples  = h5f['Samples'].value
        LLK      = h5f['LLK'].value
        accepted = h5f['accepted'].value
        h5f.close()

        return Samples, LLK, accepted
    
    def getBIC(self,act_sources,llk=None,theta=None,npts=None,AIC=False):
        '''
        Compute BIC Bayesian Information Criterion
        Args:
            llk: Log-likelihood of the model
            act_sources: number of point sources
            theta: model parameters (optional, only if llk is not provided)
        '''
        if llk is None:
            assert theta is not None, 'theta should be provided if llk is not provided'
            data_dict = {'data':self.bigD, 'green': self.bigG, 'sigma':1.}
            data_dict.update(self.map)
            data_dict['npts']= npts
            llk = self.calcLLK(theta,data_dict)

        N = self.cmtp.D.shape #N number of data
        M = act_sources * 4 # Number of parameters (time, strike, dip, rake for each source)
        if AIC:
            AIC = 2*M - 2*llk # !! BIC = -p(D) of Bishop, 2006)
            return AIC
        else:
            BIC = M*np.log(N) - 2*llk # !! BIC = -2*p(D) of Bishop, 2006)
            return BIC
        
    def calcMean(self,Samples,n_burn=200):
        '''
        Compute mean model from samples
        Args:
            Samples: MCMC samples
            n_burn: Number of burn-in samples to discard (default 200)
        '''
        meanModel ={}
        for iparm in self.map:
            meanModel[iparm] = Samples[n_burn:,self.map[iparm]].mean(axis=0)

        mTimes = meanModel['iT']
        mStrikes = meanModel['iS']
        mDips = meanModel['iD']
        mRakes = meanModel['iR']
        return mTimes, mStrikes, mDips, mRakes



# for n in range(nsrc):

#     c=cmt.cmt(filename='CMTSOLUTION.%d'%(n+1))

#     c.ts=mTimes[n]

#     c.dep = depths[n]

#     c.sdr2MT(mStrikes[n],mDips[n],mRakes[n],M0s[n]*1.0e28)

#     c.wcmtfile('o_M_CMTSOLUTION.%d'%(n+1))

# plt.plot(dobs)    

# plt.plot(predM)

# plt.plot(pred)

# res = (predM-dobs)*1e3

# RMS = np.sqrt((res*res).sum()/float(res.size))

# print('RMS misfit is %f'%(RMS))


# # Write Strike,Dip,Rakes for each source
# print('Write strike-dip rakes for each source')

# for i in range(nsrc):

#     f = open('SDR.%d'%(i+1),'wt')

#     for j in range(n_burn,n_samples,100):        

#         f.write('%8.3f %8.3f %8.3f\n'%tuple(Samples[j,[iS[i],iD[i],iR[i]]]))

#     f.close()


# # Equilibrium

# print('Equilibrium')

# plt.figure()

# for i in range(Samples.shape[1]):

#     ax = plt.subplot(4,3,i+1)

#     ax.plot(Samples[:,i])

#     plt.xlabel('Sample number')

#     plt.ylabel('Sample value')

# # Autocorrelation

# print('Autocorrelation functions')

# import scipy.stats

# lags = np.arange(1, 100)

# plt.figure(facecolor='w')

# for i in range(Samples.shape[1]):

#     ax = plt.subplot(4,3,i+1)

#     ax.plot(lags, [scipy.stats.pearsonr(Samples[n_burn:-l,i],Samples[n_burn+l:,i])[0] for l in lags])

#     plt.ylim([-0.01,1.])

#     plt.ylabel('autocorrelation param %d'%(i+1))

# plt.xlabel('Sample lag')    

# # Posteriors
# print('Posteriors')
# plt.figure()
# for i in range(Samples.shape[1]):

#     ax = plt.subplot(4,3,i+1)

#     ax.hist(Samples[n_burn:,i],normed=True)

#     plt.ylabel('Model')

# plt.show()

