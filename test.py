import tomorectv3d
import dxchange
import numpy as np
import signal
import os
import sys
import ctypes

def getp(a):
    return a.__array_interface__['data'][0]

if __name__ == "__main__":        
    N = 256
    Ntheta = N*3//2    
    Nz = 32
    center = N/2+16

    Nzp = 8 # number of slices for simultaneous processing by 1 gpu
    ngpus = 1 # number of gpus to process the data (index 0,1,2,.. are taken)
    lambda0 = 6e-8 # regularization parameter
    niter = 32 # number of iterations in the Chambolle-Pock algorithm

    # read object
    f = -dxchange.read_tiff('chip/delta-chip-256.tiff')[45:45+Nz].copy()   

    # create class solver on GPU
    cl = tomorectv3d.tomorectv3d(N, Ntheta, Nz, Nzp, ngpus, center, lambda0)
    # generate and set angles
    theta = np.array(np.linspace(0,np.pi,Ntheta).astype('float32'),order='C')
    cl.settheta(getp(theta))

    # (technical part) Free gpu memory after SIGINT, SIGSTSTP
    def signal_handler(sig, frame):
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)
    
    # generate data
    data = np.zeros([Nz,Ntheta,N],dtype='float32',order='C')
    cl.radon_wrap(getp(data),getp(f))    
    #add noise
    #sdata += np.random.normal(0,np.max(data)/50,data.shape)
    dxchange.write_tiff_stack(data,'data/data.tiff',overwrite=True)
    # reconstruction with 3d tv
    res = np.zeros([Nz,N,N],dtype='float32',order='C')
    cl.itertvR_wrap(getp(res), getp(data), niter)
    dxchange.write_tiff_stack(res,'res/res.tiff',overwrite=True)
    

    # error
    print('L2 error:', np.linalg.norm(f-res)/np.linalg.norm(f))    











    
    
    # gres = np.zeros([Ntheta,Nz,N],dtype='float32',order='C')
    # ggres = np.zeros([Ntheta,Nz,N],dtype='float32',order='C')
    # fres = np.zeros([Nz,N,N],dtype='float32',order='C')


    # cl.radon_wrap(getp(gres),getp(f))    
    # cl.radonadj_wrap(getp(fres),getp(gres))
    # cl.radon_wrap(getp(ggres),getp(fres))

    # res = np.zeros([N*Nz*N],dtype='float32')
    # cl.itertvR_wrap(getp(res), getp(gres), niter)
    # res = np.reshape(res,[Nz,N,N]).astype('float32')

    # print(np.linalg.norm(f))
    # print(np.linalg.norm(gres))
    # print(np.linalg.norm(fres))
    # print(np.sum(f*fres))
    # print(np.sum(gres*gres))

    # print(np.sum(ggres*gres)/np.sum(ggres*ggres))

    
    # dxchange.write_tiff_stack(res,'res/res.tiff',overwrite=True)
    # dxchange.write_tiff_stack(gres,'gres/gres.tiff',overwrite=True)

    # print(np.linalg.norm(f-res)/np.linalg.norm(f))


