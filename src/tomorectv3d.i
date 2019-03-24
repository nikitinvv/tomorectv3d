/*interface*/
%module tomorectv3d

%{
#define SWIG_FILE_WITH_INIT
#include "tomorectv3d.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}


class tomorectv3d
{
	//parameters
	size_t N;
	size_t Ntheta;
	size_t Nz;
	size_t Nzp;	
	float tau;
	float lambda;

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
	void prox(float *h1, float3* h2, float *g, int igpu, cudaStream_t s);	
	void updateft(float* ftn, float* fn, float* f, int igpu, cudaStream_t s);	

  public:
	tomorectv3d(size_t N, size_t Ntheta, size_t Nz, size_t Nzp,
		  size_t ngpus, float lambda);
	~tomorectv3d();
	void itertvR(float *fres,float *g, size_t niter);
    void radonmany(float *gres_, float *f_);
	void radonmanyadj(float *gres_, float *f_);
%apply (float* IN_ARRAY1, int DIM1) {(float* theta_, int N1)};
	void settheta(float* theta_, int N1);
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* fres, int N0)};
%apply (float* IN_ARRAY1, int DIM1) {(float* g_, int N1)};
	void itertvR_wrap(float *fres, int N0, float *g_, int N1, size_t niter);

%apply (float* INPLACE_ARRAY1, int DIM1) {(float* gres, int N2)};
%apply (float* IN_ARRAY1, int DIM1) {(float* f, int N3)};
    void radon_wrap(float *gres, int N2, float *f, int N3);
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* fres, int N2)};
%apply (float* IN_ARRAY1, int DIM1) {(float* g, int N3)};
    void radonadj_wrap(float *fres, int N2, float *g, int N3);

};