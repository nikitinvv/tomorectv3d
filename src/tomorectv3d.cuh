#include <cuda_runtime.h>
#include "radonusfft.cuh"

class tomorectv3d
{
	//parameters
	size_t N;
	size_t Ntheta;	
	size_t Nz;
	size_t Nzp;	
	size_t method;
	float tau;
	float lambda;
	float center;
	
	//number of gpus
	size_t ngpus;

	//class for applying Radon transform
	radonusfft** rad;

	//vars
	float *f;
	float *fn;
	float *ft;
	float *ftn;
	float *g;
	float *h1;
	float3 *h2;
	float **theta;
	

	//temporary arrays on gpus
	float **ftmp;
	float **gtmp;
	float **ftmps;
	float **gtmps;

	void radon(float *g, float* f, int igpu, cudaStream_t s);
	void radonadj(float *f, float* g, int igpu, cudaStream_t s);
	void gradient(float3 *g, float* f, int iz, int igpu, cudaStream_t s);
	void divergent(float *fn, float* f, float3* g, int igpu, cudaStream_t s);	
	void prox(float *h1, float3* h2, float *g, int method, int igpu, cudaStream_t s);	
	void updateft(float* ftn, float* fn, float* f, int igpu, cudaStream_t s);	
    
  public:
	tomorectv3d(size_t N, size_t Ntheta, size_t Nz, size_t Nzp, size_t method_,
		  size_t ngpus, float center, float lambda);
	~tomorectv3d();
	void itertvR(float *fres,float *g, size_t niter);
	void radonmany(float *gres_, float *f_);
	void radonmanyadj(float* fres, float *g_);
	
	void settheta(size_t theta);
    void itertvR_wrap(size_t fres, size_t g, size_t niter);
	void radon_wrap(size_t gres, size_t f);
	void radonadj_wrap(size_t fres, size_t g);
};