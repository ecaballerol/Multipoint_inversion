# Ext modules
import numpy as np
import os
import time
import h5py
from glob import glob
from scipy import signal
import shutil

# Int modules
import sacpy
import cmt
#from Sampler import simpleSampler

from Arguments import *
from multi_cmt import *

# Prepare data (data is should already be filtered)
cmtp = cmt.cmtproblem()
cmtp.preparedata(i_sac_lst,wpwin=wpwin,swwin=swwin)
cmtp.buildD()
npts = []
for chan_id in cmtp.chan_ids:
    npts.append(cmtp.data[chan_id].npts)
npts = np.array(npts)
dobs = cmtp.D

# Build Green GF_names dictionary for each sub-source
if calc_GFs:
    calcGFdabase(N_src,i_cmt_file,GF_DIR)
    
GF_names = []
s = sacpy.Sac()
# GF_names = {}
for i in range(N_src):
    GF_names.append({})
    for dkey in cmtp.data:
        GF_names[-1][dkey] = {}
        data = cmtp.data[dkey]
        sac_file = '%s.%s.HN%s.--.SAC.sac.bp'%(data.kstnm,data.knetwk,data.kcmpnm[-1])
        for j in range(6):
            MTnm = cmtp.cmt.MTnm[j]
            dir_name = os.path.join(GF_DIR +"_%02d" % (i),'gf_%s'%(MTnm))
            GF_names[-1][dkey][MTnm] = os.path.join(dir_name,sac_file)

# Compute Greens
multi = multicmt(N_src,cmtp)
options = {'derivate':True}
multi.prepare_src_kernels(GF_names,**options)
# multi.prepare_src_kernels(GF_names)
multi.SetParamap()

multi.buildCdfromRes(exp_cor_len,Times,Strikes,Dips,Rakes,npts,)
multi.buildG(npts)


# cmtp.cmt.rcmtfile(i_cmt_file)
# cmtp.cmt.ts = time_shift

# # Invert
# cmtp.cmtinv()
# cmtp.cmt.wcmtfile(o_cmt_file,scale=GF_M0)

# # Predict
# cmtp.calcsynt()

