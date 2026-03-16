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
# cmtp.preparedata(i_sac_lst)
cmtp.preparedata(i_sac_lst)
# cmtp.preparedata(i_sac_lst,wpwin=wpwin,swwin=swwin)
cmtp.buildD()
npts = []
for chan_id in cmtp.chan_ids:
    npts.append(cmtp.data[chan_id].npts)
npts = np.array(npts)

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
        sac_file = '%s.%s.HN%s.--.SAC.sac'%(data.kstnm,data.knetwk,data.kcmpnm[-1])
        # sac_file = '%s.%s.HN%s.--.SAC.sac'%(data.kstnm,data.knetwk,data.kcmpnm[-1])
        for j in range(6):
            MTnm = cmtp.cmt.MTnm[j]
            dir_name = os.path.join(GF_DIR +"_%02d" % (i),'gf_%s'%(MTnm))
            GF_names[-1][dkey][MTnm] = os.path.join(dir_name,sac_file)

# ---- INITIAL MODEL (start with best single source instead of forcing 0) ----
BIC_Comparison = 1e8
print("Evaluating single-source models...")

for i in range(N_src):
    #Selection of source
    active_tmp = [i]

    prop_cov = (2.38*2.38/float(4)) * \
               np.eye(4,4)

    multi = multicmt(active_tmp, cmtp)
    options = {'derivate':True,'scale':GF_M0,'filter_freq':BP}
    multi.prepare_src_kernels(GF_names, **options)
    multi.SetParamap()
    multi.DefinePrior(i_cmt_file, prior_bounds=None, priorDict='apriori_dict.pkl')
    multi.DefineInitMod(i_cmt_file, 
                        M0s=M0s[0],
                        priorDict='apriori_dict.pkl')
    
    # multi.buildDiagCd(sigma_file=Cd_sac_lst)
    multi.buildCdfromRes(exp_cor_len,saveCd=False)
    multi.buildG()

    Samples, LLK, accepted = multi.MultiSrcInv(n_samples, prop_cov)

    BICtmp = multi.getBIC(len(active_tmp),
                    llk=np.mean(LLK[-500:]),AIC=False)
    print(f"Source {i} → LLK = {LLK[-1]}")

    if BICtmp < BIC_Comparison:
        best_idx = i
        BIC_Comparison = BICtmp
        print("Find a better first point source")
        print(f"Source {i} BIC = {BICtmp}")
        mTimes,mStrikes,mDips,mRakes = multi.calcMean(Samples,n_burn=n_burn)
        
    
# Choose best single source
active_src = [best_idx]
BICsource = BIC_Comparison
remaining = list(set(range(N_src)) - set(active_src))

print(f"\nInitial best source: {best_idx}")
print(f"Initial BIC: {BICsource}")
print("\nGenerating initial Cd from residuals of best single source...")

multi = multicmt(active_src, cmtp)
options = {'derivate':True,'scale':GF_M0,'filter_freq':BP}
# multi.prepare_src_kernels(GF_names, **options)
# Cdtmp = multi.buildCdfromRes(exp_cor_len,PriorTime=mTimes,PriorStrikes=mStrikes,\
#             PriorDips=mDips,PriorRakes=mRakes,relative_error=0.2,saveCd=True)


# ---- FORWARD COMPETITIVE ADDITION ----

while True:

    print("\nTesting additions...")
    BIC_candidates = []

    for i in remaining:
        if i == 15 or i ==18: # Skip source 4 for now (known to be bad)
            continue

        test_src = active_src + [i]

        prop_cov = (2.38*2.38/float(len(test_src*4))) * \
                   np.eye(len(test_src*4),len(test_src*4))

        multi = multicmt(test_src, cmtp)
        options = {'derivate':True,'scale':GF_M0,'filter_freq':BP}
        multi.prepare_src_kernels(GF_names, **options)
        multi.SetParamap()
        multi.DefinePrior(i_cmt_file,
                          prior_bounds=None,
                          priorDict='apriori_dict.pkl')
        multi.DefineInitMod(i_cmt_file,
                            M0s=M0s[:len(test_src)],
                            priorDict='apriori_dict.pkl')
        
        multi.buildDiagCd(sigma_file=Cd_sac_lst)
        # multi.setCd(Cdtmp) # Use Cd from initial best single source
        # multi.buildCdfromRes(exp_cor_len, npts)
        multi.buildG()

        Samples, LLK, accepted = multi.MultiSrcInv(n_samples,
                                                   prop_cov)

        BICtmp = multi.getBIC(len(test_src),
                              llk=np.mean(LLK[-500:]),AIC=False)

        BIC_candidates.append((i, BICtmp))

        print(f"Test add {i} → BIC = {BICtmp}")


    # Find best candidate among ALL remaining
    best_i, best_BIC = min(BIC_candidates, key=lambda x: x[1])

    if best_BIC < BICsource:
        print(f"Accepting source {best_i}")
        active_src.append(best_i)
        remaining.remove(best_i)
        BICsource = best_BIC
    else:
        print("No improvement. Stopping selection.")
        break


print("\nFinal selected sources:", active_src)
print("Final BIC:", BICsource)


prop_cov  = (2.38*2.38/float(len(active_src*4))) * np.eye(len(active_src*4),len(active_src*4))
# Compute Greens
multi = multicmt(active_src,cmtp)
# options = {'derivate':True,'filter_freq':BP}
options = {'derivate':True,'scale':GF_M0,'filter_freq':BP}
# options = {'derivate':True,'scale':GF_M0}
multi.prepare_src_kernels(GF_names,**options)
# multi.prepare_src_kernels(GF_names)
multi.SetParamap()

multi.DefinePrior(i_cmt_file,prior_bounds=None,priorDict='apriori_dict.pkl')
multi.DefineInitMod(i_cmt_file,M0s=M0s[:len(active_src)],priorDict='apriori_dict.pkl')

multi.buildCdfromRes(exp_cor_len)
multi.buildG()

Samples,LLK,accepted = multi.MultiSrcInv(n_samples,prop_cov)

BICtmp =multi.getBIC(len(active_src),llk=np.mean(LLK[-500:]),AIC=False)

# mTimes,mStrikes,mDips,mRakes = multi.calcMean(Samples,n_burn=n_burn)
# predI , Mpost = multi.synth( mTimes,mStrikes,mDips,mRakes,multi.cmtp.D,npts)
#  predpost , Mj = multi.synth( mTimes,mStrikes,mDips,mRakes,multi.cmtp.D,npts,Mpost)

# cmtp.cmt.rcmtfile(i_cmt_file)
# cmtp.cmt.ts = time_shift

# # Invert
# cmtp.cmtinv()
# cmtp.cmt.wcmtfile(o_cmt_file,scale=GF_M0)

# # Predict
# cmtp.calcsynt()

