/*interface*/
%module tomorectv

%{
#define SWIG_FILE_WITH_INIT
#include "tomorectv.cuh"
%}


class tomorectv
{
	public:
		%immutable;
		size_t N;
		size_t Ntheta;
		size_t Nz;
		size_t Nzp;	
		size_t method;
		float tau;
		float lambda;
		float center;
		size_t ngpus;

	
  	%mutable;
  	tomorectv(size_t N, size_t Ntheta, size_t Nz, size_t Nzp, int method,
		  size_t ngpus,float center, float lambda);
	~tomorectv();
	void itertvR(float *fres,float *g, size_t niter);
    void radonmany(float *gres, float *f);
	void radonmanyadj(float *gres, float *f);

	void settheta(size_t theta);
	void itertvR_wrap(size_t fres, size_t g, size_t niter);

    void radon_wrap(size_t gres, size_t f);
    void radonadj_wrap(size_t fres, size_t g);
	void radon(float *g, float* f, int igpu, cudaStream_t s);
	void radonadj(float *f, float* g, int igpu, cudaStream_t s);
	void gradient(float3 *g, float* f, int iz, int igpu, cudaStream_t s);
	void divergent(float *fn, float* f, float3* g, int igpu, cudaStream_t s);	
	void prox(float *h1, float3* h2, float *g, int method, int igpu, cudaStream_t s);	
	void updateft(float* ftn, float* fn, float* f, int igpu, cudaStream_t s);	

 	void free();
};