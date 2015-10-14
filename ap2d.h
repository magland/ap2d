#ifndef AP2D_H
#define AP2D_H

struct AP2D_OPTIONS {
	double *initial_re; //(N1xN2)
	double *initial_im; //(N1xN2)
    double *initial_stdevs; //(N1xN2)
    double tolerance;
    int max_iterations;
	double *mask;
	double alpha1,alpha2,beta;
};

void ap2d(int N1,int N2,double *f_out,double *u_in,const AP2D_OPTIONS &opts);
void fft_real2real(int N1,int N2,double *out,double *in);
double compute_resid(int N1,int N2,double *f,double *u,AP2D_OPTIONS opts);
double register_and_compute_error(int N1, int N2, double *f1, double *f2, double *u);

// MCWRAP [ f_out[N1,N2] ]=ap2d(u_in[N1,N2],COMPLEX initial[N1,N2],initial_stdevs[N1,N2],tolerance,max_iterations,mask[N1,N2],alpha1,alpha2,beta)
// SET_INPUT N1=size(u_in,1)
// SET_INPUT N2=size(u_in,2)
// SOURCES ap2d.o
// MEXARGS -largeArrayDims -lm -lgomp -lfftw3
// IMPORTANT: In order to get openmp to work properly you must generate blocknufft3d.o separately using g++
//  > g++ -fopenmp -c ap2d.cpp -fPIC -O3

void ap2d(int N1,int N2,double *f_out,double *u_in,double *initial,double *initial_stdevs,double tolerance,int max_iterations,double *mask,double alpha1,double alpha2,double beta);

#endif // AP2D_H

