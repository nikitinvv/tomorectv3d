import tomorectv3d
import dxchange
import numpy as np
import signal
import os
import sys


if __name__ == "__main__":
    #igpu = np.int(sys.argv[1])
    #print("gpu id:",igpu)

    niter = 128
    N = 256
    Ntheta = N*3//2
    Nz = 32
    Nzp = 16
    ngpus = 2
    lambda0 = 2e-8
    f = -dxchange.read_tiff('data/delta-chip-256.tiff')[50:50+Nz].copy()    
    
    print(f.shape)

    theta = np.linspace(0,np.pi,Ntheta).astype('float32')
    cl = tomorectv3d.tomorectv3d(N, Ntheta, Nz, Nzp, ngpus,lambda0)
    cl.settheta(theta)
    # Free gpu memory after SIGINT, SIGSTSTP
    def signal_handler(sig, frame):
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    gres = np.zeros([Ntheta*Nz*N],dtype='float32')
    ggres = np.zeros([Ntheta*Nz*N],dtype='float32')
    fres = np.zeros([N*Nz*N],dtype='float32')
    ff = np.ndarray.flatten(f)

    cl.radon_wrap(gres,ff)
    cl.radonadj_wrap(fres,gres)
    cl.radon_wrap(ggres,fres)

    res = np.zeros([N*Nz*N],dtype='float32')
    cl.itertvR_wrap(res, gres, niter)
    res = np.reshape(res,[Nz,N,N]).astype('float32')

    gres = np.reshape(gres,[Nz,Ntheta,N]).astype('float32')
    ggres = np.reshape(ggres,[Nz,Ntheta,N]).astype('float32')
    fres = np.reshape(fres,[Nz,N,N]).astype('float32')
    
    print(gres.shape)
    print(fres.shape)
    #print(gres)
    print(np.linalg.norm(f))
    print(np.linalg.norm(gres))
    print(np.linalg.norm(fres))
    print(np.sum(f*fres))
    print(np.sum(gres*gres))

    print(np.sum(ggres*gres)/np.sum(ggres*ggres))

    

    
    dxchange.write_tiff_stack(res,'res/res.tiff',overwrite=True)
    dxchange.write_tiff_stack(f,'f/f.tiff',overwrite=True)

    print(np.linalg.norm(f-res)/np.linalg.norm(f))


