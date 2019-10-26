#include <stdio.h>
#include <omp.h>
#include "tomorectv3d.cuh"
#include "kernels.cuh"

tomorectv3d::tomorectv3d(size_t N_, size_t Ntheta_, size_t Nz_, size_t Nzp_, size_t method_, size_t ngpus_, float center_, float lambda_)
{
	N = N_;
	Ntheta = Ntheta_;	
	Nz = Nz_;
	Nzp = Nzp_;	
	lambda = lambda_;	
	center = center_;
	ngpus = min(ngpus_,(size_t)(Nz/Nzp));
	tau=1/sqrt(2+(Nz!=0)); 
	method = method_;
	omp_set_num_threads(ngpus);	
	//Managed memory on GPU	
	cudaMallocManaged((void**)&f,N*N*Nz*sizeof(float));
	cudaMallocManaged((void**)&fn,N*N*Nz*sizeof(float));
	cudaMallocManaged((void**)&ft,N*N*Nz*sizeof(float));
	cudaMallocManaged((void**)&ftn,N*N*Nz*sizeof(float));	
	cudaMallocManaged((void**)&g,N*Ntheta*Nz*sizeof(float));
	cudaMallocManaged((void**)&h1,N*Ntheta*Nz*sizeof(float));
	cudaMallocManaged((void**)&h2,(N+1)*(N+1)*(Nzp+1)*Nz/Nzp*sizeof(float3));	        

	//Class for applying Radon transform
	rad = new radonusfft*[ngpus];
	//tmp arrays
	ftmp = new float*[ngpus]; 
	gtmp = new float*[ngpus]; 
	ftmps = new float*[ngpus]; 
	gtmps = new float*[ngpus]; 	
	theta = new float*[ngpus]; 


	dim3 BS1d(1024);
	dim3 GS1d0(ceil(Ntheta/(float)BS1d.x));    

	for (int igpu=0;igpu<ngpus;igpu++)
	{
		cudaSetDevice(igpu);
		rad[igpu] = new radonusfft(N,Ntheta,Nzp,center);
		cudaMalloc((void**)&ftmp[igpu],2*(N+2)*(N+2)*(Nzp+2)*sizeof(float));
		cudaMalloc((void**)&gtmp[igpu],2*N*Ntheta*Nzp*sizeof(float));    
		cudaMalloc((void**)&ftmps[igpu],2*N*N*Nzp*sizeof(float));
		cudaMalloc((void**)&gtmps[igpu],2*N*Ntheta*Nzp*sizeof(float));    		
		cudaMalloc((void**)&theta[igpu],Ntheta*sizeof(float));    
		
	}
	cudaDeviceSynchronize();
}

tomorectv3d::~tomorectv3d()
{
	cudaFree(f);
	cudaFree(fn);
	cudaFree(ft);
	cudaFree(ftn);	
	cudaFree(g);
	cudaFree(h1);
	cudaFree(h2);	
	for (int igpu=0;igpu<ngpus;igpu++)
	{
		cudaSetDevice(igpu);
		delete rad[igpu];        
		cudaFree(ftmp[igpu]);
		cudaFree(gtmp[igpu]);        
		cudaFree(ftmps[igpu]);        
		cudaFree(gtmps[igpu]);        		
		cudaFree(theta[igpu]);
		cudaDeviceReset();
	}	    
}



void tomorectv3d::radon(float *g, float* f, int igpu, cudaStream_t s)
{
	//tmp arrays on gpus
	float2* ftmp0=(float2*)ftmp[igpu];
	float2* gtmp0=(float2*)gtmp[igpu];
	float* theta0=(float*)theta[igpu];

	cudaMemsetAsync(ftmp0,0,2*N*N*Nzp*sizeof(float),s);    
	cudaMemsetAsync(gtmp0,0,2*N*Ntheta*Nzp*sizeof(float),s);
	
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(N/(float)BS3d.x),ceil(N/(float)BS3d.y),ceil(Nzp/(float)BS3d.z));    
	dim3 GS3d2(ceil(N/(float)BS3d.x),ceil(Ntheta/(float)BS3d.y),ceil(Nzp/(float)BS3d.z));    
	//switch to complex numbers
	makecomplexf<<<GS3d0,BS3d,0,s>>>(ftmp0,f,N,Nzp);	
	rad[igpu]->fwdR(gtmp0,ftmp0,theta0,s);
	mulc<<<GS3d2,BS3d,0,s>>>(gtmp0,sqrt(PI),N,Ntheta,Nzp);
	addg<<<GS3d2,BS3d,0,s>>>(g,gtmp0,tau,N,Ntheta,Nzp);        	
}


void tomorectv3d::radonadj(float *f, float* g, int igpu, cudaStream_t s)
{
	//tmp arrays on gpus
	float2* ftmp0=(float2*)ftmp[igpu];
	float2* ftmps0=(float2*)ftmps[igpu];
	float2* gtmp0=(float2*)gtmp[igpu];
	float2* gtmps0=(float2*)gtmps[igpu];	
	float* theta0=(float*)theta[igpu];

	cudaMemsetAsync(ftmp0,0,2*N*N*Nzp*sizeof(float),s);    
	cudaMemsetAsync(ftmps0,0,2*N*N*Nzp*sizeof(float),s);    
	cudaMemsetAsync(gtmp0,0,2*N*Ntheta*Nzp*sizeof(float),s);
	cudaMemsetAsync(gtmps0,0,2*N*Ntheta*Nzp*sizeof(float),s);

	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(N/(float)BS3d.x),ceil(N/(float)BS3d.y),ceil(Nzp/(float)BS3d.z));    
	dim3 GS3d1(ceil(N/(float)BS3d.x),ceil(N/(float)BS3d.y),ceil(Nzp/(float)BS3d.z));    
	dim3 GS3d2(ceil(N/(float)BS3d.x),ceil(Ntheta/(float)BS3d.y),ceil(Nzp/(float)BS3d.z));    
	
	//switch to complex numbers	
	makecomplexR<<<GS3d2,BS3d,0,s>>>(gtmp0,g,N,Ntheta,Nzp);
	//constant for normalization and shift
	mulc<<<GS3d2,BS3d,0,s>>>(gtmp0,sqrt(PI),N,Ntheta,Nzp);
	//gather Radon data over all angles
	cudaMemsetAsync(gtmps0,0,2*N*Ntheta*Nzp*sizeof(float),s);
	adds<<<GS3d2,BS3d,0,s>>>(gtmps0,gtmp0,0,N,Ntheta,Ntheta,Nzp);
	//adjoint Radon tranform for [0,pi) interval            
	rad[igpu]->adjR(ftmps0,gtmps0,theta0,0,s);                        
	     
	addf<<<GS3d0,BS3d,0,s>>>(f,ftmps0,tau,N,Nzp);
}

void tomorectv3d::gradient(float3* h2, float* ft, int iz, int igpu, cudaStream_t s)
{
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil((N+2)/(float)BS3d.x),ceil((N+2)/(float)BS3d.y),ceil((Nzp+2)/(float)BS3d.z));    
	float* ftmp0 = ftmp[igpu];     
	//repeat border values      
	extendf<<<GS3d0,BS3d,0,s>>>(ftmp0, ft,iz!=0,iz!=Nz/Nzp-1,N+2,Nzp+2);
	grad<<<GS3d0,BS3d,0,s>>>(h2,ftmp0,tau,N+1,Nzp+1);	            
}

void tomorectv3d::divergent(float* fn, float* f, float3* h2, int igpu, cudaStream_t s)
{
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(N/(float)BS3d.x),ceil(N/(float)BS3d.y),ceil(Nzp/(float)BS3d.z));    
	div<<<GS3d0,BS3d,0,s>>>(fn,f,h2,tau,N,Nzp);	   
}

void tomorectv3d::prox(float* h1, float3* h2, float* g, int method, int igpu, cudaStream_t s)
{
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil((N+1)/(float)BS3d.x),ceil((N+1)/(float)BS3d.y),ceil((Nzp+1)/(float)BS3d.z));    
	dim3 GS3d1(ceil(N/(float)BS3d.x),ceil(Ntheta/(float)BS3d.y),ceil(Nzp/(float)BS3d.z));
	
	if (method==0)
		prox1tv<<<GS3d1,BS3d,0,s>>>(h1,g,tau,N,Ntheta,Nzp);
	if (method==1)
		prox1tve<<<GS3d1,BS3d,0,s>>>(h1,g,tau,N,Ntheta,Nzp);
	if (method==2)
		prox1tvl1<<<GS3d1,BS3d,0,s>>>(h1,g,tau,N,Ntheta,Nzp);		
	prox2<<<GS3d0,BS3d,0,s>>>(h2,lambda,N+1,Nzp+1);
}

void tomorectv3d::updateft(float* ftn, float* fn, float* f, int igpu, cudaStream_t s)
{
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(N/(float)BS3d.x),ceil(N/(float)BS3d.y),ceil(Nzp/(float)BS3d.z));    	
	updateft_ker<<<GS3d0,BS3d,0,s>>>(ftn,fn,f,N,Nzp);	
}

void tomorectv3d::itertvR(float *fres, float *g_, size_t niter)
{
	// compatibility with tomopy
	// for (int i=0;i<Nz;i++)
	// 	for (int j=0;j<Ntheta;j++)
	// 		cudaMemcpy(&g[(i*Ntheta+j)*N],&g_[(j*Nz+i)*N],N*sizeof(float),cudaMemcpyHostToHost);	    	
	cudaMemcpy(g,g_,Nz*Ntheta*N*sizeof(float),cudaMemcpyHostToHost);	    	
	cudaMemcpy(f,fres,N*N*Nz*sizeof(float),cudaMemcpyHostToHost);
	cudaMemcpy(ft,f,N*N*Nz*sizeof(float),cudaMemcpyHostToHost);
	cudaMemcpy(fn,f,N*N*Nz*sizeof(float),cudaMemcpyHostToHost);
	cudaMemcpy(ftn,f,N*N*Nz*sizeof(float),cudaMemcpyHostToHost);	
	memset(h1,0,N*Ntheta*Nz*sizeof(float));
	memset(h2,0,(N+1)*(N+1)*(Nzp+1)*Nz/Nzp*sizeof(float3));
	float start = omp_get_wtime();
#pragma omp parallel
	{
		int igpu = omp_get_thread_num();
		cudaSetDevice(igpu);
		cudaStream_t s1,s2,s3,st;
		cudaEvent_t e1,e2,et;
		cudaStreamCreate(&s1);
		cudaStreamCreate(&s2);
		cudaStreamCreate(&s3);
		cudaEventCreate(&e1);
		cudaEventCreate(&e2);
		for(int iter=0;iter<niter;iter++)
		{    	
			//parts in z
			int iz=igpu*Nz/Nzp/ngpus;
			float* f0 = &f[N*N*iz*Nzp];
			float* fn0 = &fn[N*N*iz*Nzp];
			float* ft0 = &ft[N*N*iz*Nzp];
			float* ftn0 = &ftn[N*N*iz*Nzp];
			float* h10 = &h1[N*Ntheta*iz*Nzp];
			float3* h20 = &h2[(N+1)*(N+1)*iz*(Nzp+1)];
			float* g0 = &g[N*Ntheta*iz*Nzp];
			cudaMemPrefetchAsync(f0, N*N*Nzp*sizeof(float), igpu, s2);
			cudaMemPrefetchAsync(fn0, N*N*Nzp*sizeof(float), igpu, s2);
			cudaMemPrefetchAsync(&ft0[-(iz!=0)*N*N], N*N*(Nzp+2-(iz==0)-(iz==Nz/Nzp-1))*sizeof(float), igpu, s2);
			cudaMemPrefetchAsync(ftn0, N*N*Nzp*sizeof(float), igpu, s2);
			cudaMemPrefetchAsync(h10, N*Ntheta*Nzp*sizeof(float), igpu, s2);
			cudaMemPrefetchAsync(h20, (N+1)*(N+1)*(Nzp+1)*sizeof(float3), igpu, s2);
			cudaMemPrefetchAsync(g0, N*Ntheta*Nzp*sizeof(float), igpu, s2);

			cudaEventRecord(e1, s2); 
			float* f0s=f0;
			float* fn0s=fn0;
			float* ft0s=ft0;
			float* ftn0s=ftn0;
			float* h10s=h10;
			float3* h20s=h20;
			float* g0s=g0;
#pragma omp for    
			for(int iz=0;iz<Nz/Nzp;iz++)
			{     	
				cudaEventSynchronize(e1);  
				cudaEventSynchronize(e2);

				//forward step				
				gradient(h20,ft0,iz,igpu,s1);//iz for border control
				radon(h10,ft0,igpu,s1);
				//proximal
				prox(h10,h20,g0,method,igpu,s1);
				//backward step
				divergent(fn0,f0,h20,igpu,s1);
				radonadj(fn0,h10,igpu,s1);                     
				//update ft
				updateft(ftn0,fn0,f0,igpu,s1);
				cudaEventRecord(e1, s1); 

				if (iz < (igpu+1)*Nz/Nzp/ngpus-1) 
				{
					// make sure the stream is idle to force non-deferred HtoD prefetches first 
					cudaStreamSynchronize(s2);       
					//parts in z
					f0s = &f[N*N*(iz+1)*Nzp];
					fn0s = &fn[N*N*(iz+1)*Nzp];
					ft0s = &ft[N*N*(iz+1)*Nzp];
					ftn0s = &ftn[N*N*(iz+1)*Nzp];
					h10s = &h1[N*Ntheta*(iz+1)*Nzp];
					h20s = &h2[(N+1)*(N+1)*(iz+1)*(Nzp+1)];
					g0s = &g[N*Ntheta*(iz+1)*Nzp];
					cudaMemPrefetchAsync(f0s, N*N*Nzp*sizeof(float), igpu, s2);
					cudaMemPrefetchAsync(fn0s, N*N*Nzp*sizeof(float), igpu, s2);
					cudaMemPrefetchAsync(&ft0s[N*N], N*N*(Nzp-(iz+1==Nz/Nzp-1))*sizeof(float), igpu, s2);
					cudaMemPrefetchAsync(ftn0s, N*N*Nzp*sizeof(float), igpu, s2);
					cudaMemPrefetchAsync(h10s, N*Ntheta*Nzp*sizeof(float), igpu, s2);
					cudaMemPrefetchAsync(h20s, (N+1)*(N+1)*(Nzp+1)*sizeof(float3), igpu, s2);
					cudaMemPrefetchAsync(g0s, N*Ntheta*Nzp*sizeof(float), igpu, s2);
				
					cudaEventRecord(e2, s2); 
				} 								

				cudaMemPrefetchAsync(f0, N*N*Nzp*sizeof(float), cudaCpuDeviceId, s1);
				cudaMemPrefetchAsync(fn0, N*N*Nzp*sizeof(float), cudaCpuDeviceId, s1);
				cudaMemPrefetchAsync(&ft0[-(iz!=0)*N*N], N*N*(Nzp-(iz==0)-(iz==Nz/Nzp-1)+2*(iz==(igpu+1)*Nz/Nzp/ngpus-1))*sizeof(float), cudaCpuDeviceId, s1);

				cudaMemPrefetchAsync(ftn0, N*N*Nzp*sizeof(float), cudaCpuDeviceId, s1);
				cudaMemPrefetchAsync(h10, N*Ntheta*Nzp*sizeof(float), cudaCpuDeviceId, s1);
				cudaMemPrefetchAsync(h20, (N+1)*(N+1)*(Nzp+1)*sizeof(float3), cudaCpuDeviceId, s1);
				cudaMemPrefetchAsync(g0, N*Ntheta*Nzp*sizeof(float), cudaCpuDeviceId, s1);

		
				f0=f0s;
				fn0=fn0s;
				ft0=ft0s;
				ftn0=ftn0s;
				h10=h10s;
				h20=h20s;
				g0=g0s;
				// rotate streams and swap events 
				st = s1; s1 = s2; s2 = st; 
				st = s2; s2 = s3; s3 = st; 
				et = e1; e1 = e2; e2 = et; 
			}		

			cudaEventSynchronize(e1);  
			cudaEventSynchronize(e2);
			cudaDeviceSynchronize();
#pragma omp barrier
#pragma omp single
			{
				float* tmp=0;
				tmp=ft;ft=ftn;ftn=tmp;
				tmp=f;f=fn;fn=tmp;
				fprintf(stderr,"iterations (%d/%d) \n",iter, niter); fflush(stdout);

			}
		}
	cudaDeviceSynchronize();
#pragma omp barrier
	}
	float end = omp_get_wtime();
	printf("Elapsed time: %fs.\n", end-start);
	cudaMemPrefetchAsync(ft, N*N*Nz*sizeof(float), cudaCpuDeviceId,0);	
	for(int i=0;i<N*N*Nz;i++)
		ft[i]*=sqrt(PI);
	cudaMemcpy(fres,ft,N*N*Nz*sizeof(float),cudaMemcpyDefault);	
}


void tomorectv3d::radonmany(float *gres_, float *f_)
{
	cudaMemcpy(f,f_,N*N*Nz*sizeof(float),cudaMemcpyHostToHost);	    
	cudaMemcpy(ft,f,N*N*Nz*sizeof(float),cudaMemcpyHostToHost);
	cudaMemcpy(fn,f,N*N*Nz*sizeof(float),cudaMemcpyHostToHost);
	cudaMemcpy(ftn,f,N*N*Nz*sizeof(float),cudaMemcpyHostToHost);	
	memset(h1,0,N*Ntheta*Nz*sizeof(float));
	memset(h2,0,(N+1)*(N+1)*(Nzp+1)*Nz/Nzp*sizeof(float3));
	
#pragma omp parallel
	{
		int igpu = omp_get_thread_num();
		cudaSetDevice(igpu);
		cudaStream_t s1,s2,s3,st;
		cudaEvent_t e1,e2,et;
		cudaStreamCreate(&s1);
		cudaStreamCreate(&s2);
		cudaStreamCreate(&s3);
		cudaEventCreate(&e1);
		cudaEventCreate(&e2);


    //parts in z
        int iz=igpu*Nz/Nzp/ngpus;
        float* f0 = &f[N*N*iz*Nzp];
        float* fn0 = &fn[N*N*iz*Nzp];
        float* ft0 = &ft[N*N*iz*Nzp];
        float* ftn0 = &ftn[N*N*iz*Nzp];
        float* h10 = &h1[N*Ntheta*iz*Nzp];
        float3* h20 = &h2[(N+1)*(N+1)*iz*(Nzp+1)];
        float* g0 = &g[N*Ntheta*iz*Nzp];
        cudaMemPrefetchAsync(f0, N*N*Nzp*sizeof(float), igpu, s2);
        cudaMemPrefetchAsync(fn0, N*N*Nzp*sizeof(float), igpu, s2);
        cudaMemPrefetchAsync(&ft0[-(iz!=0)*N*N], N*N*(Nzp+2-(iz==0)-(iz==Nz/Nzp-1))*sizeof(float), igpu, s2);
        cudaMemPrefetchAsync(ftn0, N*N*Nzp*sizeof(float), igpu, s2);
        cudaMemPrefetchAsync(h10, N*Ntheta*Nzp*sizeof(float), igpu, s2);
        cudaMemPrefetchAsync(g0, N*Ntheta*Nzp*sizeof(float), igpu, s2);

        cudaEventRecord(e1, s2); 
        float* f0s=f0;
        float* fn0s=fn0;
        float* ft0s=ft0;
        float* ftn0s=ftn0;
        float* h10s=h10;
        float3* h20s=h20;
        float* g0s=g0;
#pragma omp for    
        for(int iz=0;iz<Nz/Nzp;iz++)
        {     	
            cudaEventSynchronize(e1);  
            cudaEventSynchronize(e2);

			//forward step			
			radon(h10,ft0,igpu,s1);
            cudaEventRecord(e1, s1); 

            if (iz < (igpu+1)*Nz/Nzp/ngpus-1) 
            {
                // make sure the stream is idle to force non-deferred HtoD prefetches first 
                cudaStreamSynchronize(s2);       
                //parts in z
                f0s = &f[N*N*(iz+1)*Nzp];
                fn0s = &fn[N*N*(iz+1)*Nzp];
                ft0s = &ft[N*N*(iz+1)*Nzp];
                ftn0s = &ftn[N*N*(iz+1)*Nzp];
                h10s = &h1[N*Ntheta*(iz+1)*Nzp];
                g0s = &g[N*Ntheta*(iz+1)*Nzp];
                cudaMemPrefetchAsync(f0s, N*N*Nzp*sizeof(float), igpu, s2);
                cudaMemPrefetchAsync(fn0s, N*N*Nzp*sizeof(float), igpu, s2);
                cudaMemPrefetchAsync(&ft0s[N*N], N*N*(Nzp-(iz+1==Nz/Nzp-1))*sizeof(float), igpu, s2);
                cudaMemPrefetchAsync(ftn0s, N*N*Nzp*sizeof(float), igpu, s2);
                cudaMemPrefetchAsync(h10s, N*Ntheta*Nzp*sizeof(float), igpu, s2);
                cudaMemPrefetchAsync(h20s, (N+1)*(N+1)*(Nzp+1)*sizeof(float3), igpu, s2);
                cudaMemPrefetchAsync(g0s, N*Ntheta*Nzp*sizeof(float), igpu, s2);
            
                cudaEventRecord(e2, s2); 
            } 								

            cudaMemPrefetchAsync(f0, N*N*Nzp*sizeof(float), cudaCpuDeviceId, s1);
            cudaMemPrefetchAsync(fn0, N*N*Nzp*sizeof(float), cudaCpuDeviceId, s1);
            cudaMemPrefetchAsync(&ft0[-(iz!=0)*N*N], N*N*(Nzp-(iz==0)-(iz==Nz/Nzp-1)+2*(iz==(igpu+1)*Nz/Nzp/ngpus-1))*sizeof(float), cudaCpuDeviceId, s1);

            cudaMemPrefetchAsync(ftn0, N*N*Nzp*sizeof(float), cudaCpuDeviceId, s1);
            cudaMemPrefetchAsync(h10, N*Ntheta*Nzp*sizeof(float), cudaCpuDeviceId, s1);
            cudaMemPrefetchAsync(h20, (N+1)*(N+1)*(Nzp+1)*sizeof(float3), cudaCpuDeviceId, s1);
            cudaMemPrefetchAsync(g0, N*Ntheta*Nzp*sizeof(float), cudaCpuDeviceId, s1);

    
            f0=f0s;
            fn0=fn0s;
            ft0=ft0s;
            ftn0=ftn0s;
            h10=h10s;
            h20=h20s;
            g0=g0s;
            // rotate streams and swap events 
            st = s1; s1 = s2; s2 = st; 
            st = s2; s2 = s3; s3 = st; 
            et = e1; e1 = e2; e2 = et; 
        }		

        cudaEventSynchronize(e1);  
        cudaEventSynchronize(e2);
        cudaDeviceSynchronize();
#pragma omp barrier
#pragma omp single
        {
            float* tmp=0;
            tmp=ft;ft=ftn;ftn=tmp;
            tmp=f;f=fn;fn=tmp;            
        }    
	cudaDeviceSynchronize();
#pragma omp barrier
	}
	cudaMemPrefetchAsync(h1, N*Ntheta*Nz*sizeof(float), cudaCpuDeviceId,0);	
	cudaMemcpy(gres_,h1,N*Ntheta*Nz*sizeof(float),cudaMemcpyDefault);	
}


void tomorectv3d::radonmanyadj(float *fres_, float *g_)
{
	cudaMemcpy(g,g_,N*Ntheta*Nz*sizeof(float),cudaMemcpyHostToHost);	    
	memset(f,0,N*N*Nz*sizeof(float));
	cudaMemcpy(ft,f,N*N*Nz*sizeof(float),cudaMemcpyHostToHost);
	cudaMemcpy(fn,f,N*N*Nz*sizeof(float),cudaMemcpyHostToHost);
	cudaMemcpy(ftn,f,N*N*Nz*sizeof(float),cudaMemcpyHostToHost);	
	
	memset(h1,0,N*Ntheta*Nz*sizeof(float));
	memset(h2,0,(N+1)*(N+1)*(Nzp+1)*Nz/Nzp*sizeof(float3));
	
	cudaMemcpy(h1,g,N*Ntheta*Nz*sizeof(float),cudaMemcpyHostToHost);	   
#pragma omp parallel
	{
		int igpu = omp_get_thread_num();
		cudaSetDevice(igpu);
		cudaStream_t s1,s2,s3,st;
		cudaEvent_t e1,e2,et;
		cudaStreamCreate(&s1);
		cudaStreamCreate(&s2);
		cudaStreamCreate(&s3);
		cudaEventCreate(&e1);
		cudaEventCreate(&e2);
		
		//parts in z
		int iz=igpu*Nz/Nzp/ngpus;
		float* f0 = &f[N*N*iz*Nzp];
		float* fn0 = &fn[N*N*iz*Nzp];
		float* ft0 = &ft[N*N*iz*Nzp];
		float* ftn0 = &ftn[N*N*iz*Nzp];
		float* h10 = &h1[N*Ntheta*iz*Nzp];
		float3* h20 = &h2[(N+1)*(N+1)*iz*(Nzp+1)];
		float* g0 = &g[N*Ntheta*iz*Nzp];
		cudaMemPrefetchAsync(f0, N*N*Nzp*sizeof(float), igpu, s2);
		cudaMemPrefetchAsync(fn0, N*N*Nzp*sizeof(float), igpu, s2);
		cudaMemPrefetchAsync(&ft0[-(iz!=0)*N*N], N*N*(Nzp+2-(iz==0)-(iz==Nz/Nzp-1))*sizeof(float), igpu, s2);
		cudaMemPrefetchAsync(ftn0, N*N*Nzp*sizeof(float), igpu, s2);
		cudaMemPrefetchAsync(h10, N*Ntheta*Nzp*sizeof(float), igpu, s2);
		cudaMemPrefetchAsync(h20, (N+1)*(N+1)*(Nzp+1)*sizeof(float3), igpu, s2);
		cudaMemPrefetchAsync(g0, N*Ntheta*Nzp*sizeof(float), igpu, s2);

		cudaEventRecord(e1, s2); 
		float* f0s=f0;
		float* fn0s=fn0;
		float* ft0s=ft0;
		float* ftn0s=ftn0;
		float* h10s=h10;
		float3* h20s=h20;
		float* g0s=g0;
#pragma omp for    
		for(int iz=0;iz<Nz/Nzp;iz++)
		{     	
			cudaEventSynchronize(e1);  
			cudaEventSynchronize(e2);

			radonadj(fn0,h10,igpu,s1);                  
			
			cudaEventRecord(e1, s1); 

			if (iz < (igpu+1)*Nz/Nzp/ngpus-1) 
			{
				// make sure the stream is idle to force non-deferred HtoD prefetches first 
				cudaStreamSynchronize(s2);       
				//parts in z
				f0s = &f[N*N*(iz+1)*Nzp];
				fn0s = &fn[N*N*(iz+1)*Nzp];
				ft0s = &ft[N*N*(iz+1)*Nzp];
				ftn0s = &ftn[N*N*(iz+1)*Nzp];
				h10s = &h1[N*Ntheta*(iz+1)*Nzp];
				h20s = &h2[(N+1)*(N+1)*(iz+1)*(Nzp+1)];
				g0s = &g[N*Ntheta*(iz+1)*Nzp];
				cudaMemPrefetchAsync(f0s, N*N*Nzp*sizeof(float), igpu, s2);
				cudaMemPrefetchAsync(fn0s, N*N*Nzp*sizeof(float), igpu, s2);
				cudaMemPrefetchAsync(&ft0s[N*N], N*N*(Nzp-(iz+1==Nz/Nzp-1))*sizeof(float), igpu, s2);
				cudaMemPrefetchAsync(ftn0s, N*N*Nzp*sizeof(float), igpu, s2);
				cudaMemPrefetchAsync(h10s, N*Ntheta*Nzp*sizeof(float), igpu, s2);
				cudaMemPrefetchAsync(h20s, (N+1)*(N+1)*(Nzp+1)*sizeof(float3), igpu, s2);
				cudaMemPrefetchAsync(g0s, N*Ntheta*Nzp*sizeof(float), igpu, s2);
			
				cudaEventRecord(e2, s2); 
			} 								

			cudaMemPrefetchAsync(f0, N*N*Nzp*sizeof(float), cudaCpuDeviceId, s1);
			cudaMemPrefetchAsync(fn0, N*N*Nzp*sizeof(float), cudaCpuDeviceId, s1);
			cudaMemPrefetchAsync(&ft0[-(iz!=0)*N*N], N*N*(Nzp-(iz==0)-(iz==Nz/Nzp-1)+2*(iz==(igpu+1)*Nz/Nzp/ngpus-1))*sizeof(float), cudaCpuDeviceId, s1);

			cudaMemPrefetchAsync(ftn0, N*N*Nzp*sizeof(float), cudaCpuDeviceId, s1);
			cudaMemPrefetchAsync(h10, N*Ntheta*Nzp*sizeof(float), cudaCpuDeviceId, s1);
			cudaMemPrefetchAsync(h20, (N+1)*(N+1)*(Nzp+1)*sizeof(float3), cudaCpuDeviceId, s1);
			cudaMemPrefetchAsync(g0, N*Ntheta*Nzp*sizeof(float), cudaCpuDeviceId, s1);

	
			f0=f0s;
			fn0=fn0s;
			ft0=ft0s;
			ftn0=ftn0s;
			h10=h10s;
			h20=h20s;
			g0=g0s;
			// rotate streams and swap events 
			st = s1; s1 = s2; s2 = st; 
			st = s2; s2 = s3; s3 = st; 
			et = e1; e1 = e2; e2 = et; 
		}		

		cudaEventSynchronize(e1);  
		cudaEventSynchronize(e2);
		cudaDeviceSynchronize();
#pragma omp barrier
#pragma omp single
		{
			float* tmp=0;
			tmp=ft;ft=ftn;ftn=tmp;
			tmp=f;f=fn;fn=tmp;			

		}		
	cudaDeviceSynchronize();
#pragma omp barrier
	}

	cudaMemPrefetchAsync(f, N*N*Nz*sizeof(float), cudaCpuDeviceId,0);
	cudaMemcpy(fres_,f,N*N*Nz*sizeof(float),cudaMemcpyDefault);	
}


void tomorectv3d::settheta(size_t theta_)
{
	for (int igpu=0;igpu<ngpus;igpu++)
	{
		cudaSetDevice(igpu);
		cudaMemcpy(theta[igpu],(float*)theta_,Ntheta*sizeof(float),cudaMemcpyDefault);
	}
}

void tomorectv3d::itertvR_wrap(size_t fres, size_t g, size_t niter)
{
	itertvR((float*)fres,(float*)g,niter);
}

void tomorectv3d::radon_wrap(size_t gres, size_t f)
{
	radonmany((float*)gres,(float*)f);
}

void tomorectv3d::radonadj_wrap(size_t fres, size_t g)
{
	radonmanyadj((float*)fres,(float*)g);
}
