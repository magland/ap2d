#include "ap2d.h"
#include <stdlib.h>
#include "fftw3.h"
#include <math.h>
#include "memory.h"
#include "omp.h"

double *malloc_double(int N) {
    return (double *)malloc(sizeof(double)*N);
}

void ifft_real2real(int N1,int N2,double *out,double *in) {
    int NN=N1*N2;
    int M1=(N1+1)/2;
    int M2=(N2+1)/2;
    fftw_complex *in2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);
    fftw_complex *out2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);
    for (int i2=0; i2<N2; i2++) {
        int j2=(i2-M2+N2)%N2;
        for (int i1=0; i1<N1; i1++) {
            int j1=(i1-M1+N1)%N1;
            in2[j1+N1*j2][0]=in[i1+N1*i2];
            in2[j1+N1*j2][1]=0;
        }
    }

    fftw_plan p;
    #pragma omp critical
    p=fftw_plan_dft_2d(N2,N1,in2,out2,FFTW_BACKWARD,FFTW_ESTIMATE);

    fftw_execute(p);
    for (int i2=0; i2<N2; i2++) {
        int j2=(i2-M2+N2)%N2;
        for (int i1=0; i1<N1; i1++) {
            int j1=(i1-M1+N1)%N1;
            out[i1+N1*i2]=out2[j1+N1*j2][0]/NN;
        }
    }
    fftw_free(in2);
    fftw_free(out2);
    #pragma omp critical
    fftw_destroy_plan(p);
}

void fft_real2real(int N1,int N2,double *out,double *in) {
    int M1=(N1+1)/2;
    int M2=(N2+1)/2;
    fftw_complex *in2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);
    fftw_complex *out2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);
    for (int i2=0; i2<N2; i2++) {
        int j2=(i2-M2+N2)%N2;
        for (int i1=0; i1<N1; i1++) {
            int j1=(i1-M1+N1)%N1;
            in2[j1+N1*j2][0]=in[i1+N1*i2];
            in2[j1+N1*j2][1]=0;
        }
    }

    fftw_plan p;
    #pragma omp critical
    p=fftw_plan_dft_2d(N2,N1,in2,out2,FFTW_FORWARD,FFTW_ESTIMATE);

    fftw_execute(p);
    for (int i2=0; i2<N2; i2++) {
        int j2=(i2-M2+N2)%N2;
        for (int i1=0; i1<N1; i1++) {
            int j1=(i1-M1+N1)%N1;
            out[i1+N1*i2]=out2[j1+N1*j2][0];
        }
    }
    fftw_free(in2);
    fftw_free(out2);
    #pragma omp critical
    fftw_destroy_plan(p);
}

void ifft_complex2real(int N1,int N2,double *out,double *in,fftw_plan p,fftw_complex *out2,fftw_complex *in2) {
    int NN=N1*N2;
    int M1=(N1+1)/2;
    int M2=(N2+1)/2;

    for (int i2=0; i2<N2; i2++) {
        int j2=(i2-M2+N2)%N2;
        for (int i1=0; i1<N1; i1++) {
            int j1=(i1-M1+N1)%N1;
            in2[j1+N1*j2][0]=in[(i1+N1*i2)*2];
            in2[j1+N1*j2][1]=in[(i1+N1*i2)*2+1];
        }
    }

    fftw_execute(p);
    for (int i2=0; i2<N2; i2++) {
        int j2=(i2-M2+N2)%N2;
        for (int i1=0; i1<N1; i1++) {
            int j1=(i1-M1+N1)%N1;
            out[i1+N1*i2]=out2[j1+N1*j2][0]/NN;
        }
    }
}

void ifft_complex2real(int N1,int N2,double *out,double *in) {

    fftw_complex *in2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);
    fftw_complex *out2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);

    fftw_plan p;
    #pragma omp critical
    p=fftw_plan_dft_2d(N2,N1,in2,out2,FFTW_BACKWARD,FFTW_ESTIMATE);

    ifft_complex2real(N1,N2,out,in,p,out2,in2);

    fftw_free(in2);
    fftw_free(out2);
    #pragma omp critical
    fftw_destroy_plan(p);
}

void ifft_complex2complex(int N1,int N2,double *out,double *in,fftw_plan p,fftw_complex *out2,fftw_complex *in2) {
    int NN=N1*N2;
    int M1=(N1+1)/2;
    int M2=(N2+1)/2;

    for (int i2=0; i2<N2; i2++) {
        int j2=(i2-M2+N2)%N2;
        for (int i1=0; i1<N1; i1++) {
            int j1=(i1-M1+N1)%N1;
            in2[j1+N1*j2][0]=in[(i1+N1*i2)*2];
            in2[j1+N1*j2][1]=in[(i1+N1*i2)*2+1];
        }
    }

    fftw_execute(p);
    for (int i2=0; i2<N2; i2++) {
        int j2=(i2-M2+N2)%N2;
        for (int i1=0; i1<N1; i1++) {
            int j1=(i1-M1+N1)%N1;
            out[(i1+N1*i2)*2]=out2[j1+N1*j2][0]/NN;
            out[(i1+N1*i2)*2+1]=out2[j1+N1*j2][1]/NN;
        }
    }
}

void ifft_complex2complex(int N1,int N2,double *out,double *in) {

    fftw_complex *in2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);
    fftw_complex *out2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);

    fftw_plan p;
    #pragma omp critical
    p=fftw_plan_dft_2d(N2,N1,in2,out2,FFTW_BACKWARD,FFTW_ESTIMATE);

    ifft_complex2complex(N1,N2,out,in,p,out2,in2);

    fftw_free(in2);
    fftw_free(out2);
    #pragma omp critical
    fftw_destroy_plan(p);
}

void fft_real2complex(int N1,int N2,double *out,double *in,fftw_plan p,fftw_complex *out2,fftw_complex *in2) {
    int M1=(N1+1)/2;
    int M2=(N2+1)/2;

    for (int i2=0; i2<N2; i2++) {
        int j2=(i2-M2+N2)%N2;
        for (int i1=0; i1<N1; i1++) {
            int j1=(i1-M1+N1)%N1;
            in2[j1+N1*j2][0]=in[i1+N1*i2];
            in2[j1+N1*j2][1]=0;
        }
    }

    fftw_execute(p);

    for (int i2=0; i2<N2; i2++) {
        int j2=(i2-M2+N2)%N2;
        for (int i1=0; i1<N1; i1++) {
            int j1=(i1-M1+N1)%N1;
            out[(i1+N1*i2)*2]=out2[j1+N1*j2][0];
            out[(i1+N1*i2)*2+1]=out2[j1+N1*j2][1];
        }
    }
}

void fft_real2complex(int N1,int N2,double *out,double *in) {
    fftw_complex *in2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);
    fftw_complex *out2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);
    fftw_plan p;
    #pragma omp critical
    p=fftw_plan_dft_2d(N2,N1,in2,out2,FFTW_FORWARD,FFTW_ESTIMATE);

    fft_real2complex(N1,N2,out,in,p,out2,in2);

    fftw_free(in2);
    fftw_free(out2);
    #pragma omp critical
    fftw_destroy_plan(p);
}

double compute_norm(int N,double *v) {
    double ret=0;
    for (int i=0; i<N; i++) {
        ret+=v[i]*v[i];
    }
    return sqrt(ret);
}

double randu() {
    double ret=((double) rand() / (RAND_MAX));
    return ret;
}

double randn() {
    double v1=randu();
    double v2=randu();
    return sqrt(-2*log(v1))*cos(2*M_PI*v2); //box-muller - https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
}

void pi_1(int N1,int N2,double *out,double *in,double alpha1,double *mask) {
    int ii=0;
    for (int y=0; y<N2; y++) {
        for (int x=0; x<N1; x++) {
			if ((in[ii]>=0)&&(mask[ii])) {
            //if ((xmin<=x)&&(x<=xmax)&&(ymin<=y)&&(y<=ymax)) {
                out[ii]=in[ii];
            }
            else {
                out[ii]=in[ii]*(1-alpha1);
            }
            ii++;
        }
    }
}

void pi_2(int N1,int N2,double *out,double *in,double *u,double alpha2) {
    for (int ii=0; ii<N1*N2; ii++) {
        double re=in[ii*2],re2;
        double im=in[ii*2+1],im2;
        double u0=u[ii];
        double norm0=sqrt(re*re+im*im);
        if (norm0) {
            re2=re/norm0*u0;
            im2=im/norm0*u0;
        }
        else {
            re2=u0;
            im2=0;
        }
        out[ii*2]=re*(1-alpha2)+re2*alpha2;
        out[ii*2+1]=im*(1-alpha2)+im2*alpha2;
    }
}

void copy_double(int N,double *out,double *in) {
    for (int i=0; i<N; i++) {
        out[i]=in[i];
    }
}

void ap2d(int N1,int N2,double *f_out,double *u_in,const AP2D_OPTIONS &opts) {
	double alpha1=opts.alpha1;
	double alpha2=opts.alpha2;
	double beta=opts.beta;
    double mu1=alpha1*beta;
    double mu2=alpha2*mu1/(alpha1-alpha1*alpha2);

    double *u_ifft=malloc_double(N1*N2);
    double *f=malloc_double(N1*N2);
    double *f2=malloc_double(N1*N2);
    double *f2hat=malloc_double(N1*N2*2); //complex
    double *f3hat=malloc_double(N1*N2*2); //complex
    double *f3=malloc_double(N1*N2);

    ifft_real2real(N1,N2,u_ifft,u_in);
    double u_norm=compute_norm(N1*N2,u_ifft);

	if (opts.initial_re) {
        for (int ii=0; ii<N1*N2; ii++) {
			f3hat[ii*2]=opts.initial_re[ii]+randn()*opts.initial_stdevs[ii];
			f3hat[ii*2+1]=opts.initial_im[ii]+randn()*opts.initial_stdevs[ii];
        }
    }
    else {
        for (int ii=0; ii<N1*N2; ii++) {
            f3hat[ii*2]=randn();
            f3hat[ii*2+1]=randn();
        }
    }
    ifft_complex2real(N1,N2,f,f3hat);

    fftw_complex *real2complex_in=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);
    fftw_complex *real2complex_out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);
    fftw_plan real2complex_plan;
    #pragma omp critical
    real2complex_plan=fftw_plan_dft_2d(N2,N1,real2complex_in,real2complex_out,FFTW_FORWARD,FFTW_ESTIMATE);

    fftw_complex *complex2real_in=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);
    fftw_complex *complex2real_out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1*N2);
    fftw_plan complex2real_plan;
    #pragma omp critical
    complex2real_plan=fftw_plan_dft_2d(N2,N1,complex2real_in,complex2real_out,FFTW_BACKWARD,FFTW_ESTIMATE);

    int it=1;
    bool done=false;
    double last_resid=0;
    int num_steps_within_tolerance=0;
    while (!done) {
        /////////////////////////////
		pi_1(N1,N2,f2,f,alpha1,opts.mask);

        /////////////////////////////
        fft_real2complex(N1,N2,f2hat,f2,real2complex_plan,real2complex_out,real2complex_in);

        /////////////////////////////
        pi_2(N1,N2,f3hat,f2hat,u_in,alpha2);

        /////////////////////////////
        ifft_complex2real(N1,N2,f3,f3hat,complex2real_plan,complex2real_out,complex2real_in);

        double resid0=0;
        for (int ii=0; ii<N1*N2; ii++) {
            resid0+=mu1*(f[ii]-f2[ii])*(f[ii]-f2[ii])/(u_norm*u_norm);
            resid0+=mu2*(f[ii]-f3[ii])*(f[ii]-f3[ii])/(u_norm*u_norm);
        }
        if (it>1) {
            double resid_diff=fabs(resid0-last_resid);
            if (resid_diff<opts.tolerance) num_steps_within_tolerance++;
            else num_steps_within_tolerance=0;
            if (num_steps_within_tolerance>10) {
                done=true;
            }
        }
        copy_double(N1*N2,f,f3);
        if (it>=opts.max_iterations) {
            done=true;
        }
        last_resid=resid0;
        it++;
    }

    copy_double(N1*N2,f_out,f);

    fftw_free(real2complex_in);
    fftw_free(real2complex_out);
    #pragma omp critical
    fftw_destroy_plan(real2complex_plan);

    free(u_ifft);
    free(f);
    free(f2);
    free(f2hat);
    free(f3hat);
    free(f3);
}


double compute_resid(int N1, int N2, double *f, double *u,AP2D_OPTIONS opts)
{
    double *u_ifft=malloc_double(N1*N2);
    double *f2=malloc_double(N1*N2);

    ifft_real2real(N1,N2,u_ifft,u);
    double u_norm=compute_norm(N1*N2,u_ifft);
	pi_1(N1,N2,f2,f,1,opts.mask);
    double val=0;
    for (int ii=0; ii<N1*N2; ii++) {
        val+=(f[ii]-f2[ii])*(f[ii]-f2[ii]);
    }
    double ret=sqrt(val)/u_norm;

    free(u_ifft);
    free(f2);

    return ret;
}

void compute_center_of_mass(int N1, int N2, double *f,double &cx,double &cy)
{
    double numerx=0,numery=0,denomx=0,denomy=0;
    int jj=0;
    for (int y=0; y<N2; y++) {
        for (int x=0; x<N1; x++) {
            numerx+=x*f[jj];
            denomx+=f[jj];
            numery+=y*f[jj];
            denomy+=f[jj];
            jj++;
        }
    }
    cx=numerx/denomx;
    cy=numery/denomy;
}

double register_and_compute_error_2(int N1, int N2, double *f1, double *f2, double *u)
{
    int M1=(N1+1)/2;
    int M2=(N2+1)/2;

    double *u_ifft=malloc_double(N1*N2);
    ifft_real2real(N1,N2,u_ifft,u);
    double u_norm=compute_norm(N1*N2,u_ifft);
    free(u_ifft);

    double *f1hat=malloc_double(N1*N2*2);
    double *f2hat=malloc_double(N1*N2*2);
    double *ghat=malloc_double(N1*N2*2);
    double *g=malloc_double(N1*N2*2);

    fft_real2complex(N1,N2,f1hat,f1);
    fft_real2complex(N1,N2,f2hat,f2);
    for (int ii=0; ii<N1*N2; ii++) {
        double re1=f1hat[ii*2];
        double im1=f1hat[ii*2+1];
        double re2=f2hat[ii*2];
        double im2=f2hat[ii*2+1]; im2=-im2; //complex conjugate
        double re3=re1*re2-im1*im2;
        double im3=re1*im2+re2*im1;
        ghat[ii*2]=re3;
        ghat[ii*2+1]=im3;
    }
    ifft_complex2complex(N1,N2,g,ghat);
    double maxval=0;
    int bestx=0;
    int besty=0;
    int jj=0;
    for (int y=0; y<N2; y++) {
        for (int x=0; x<N1; x++) {
            double val = g[jj*2]*g[jj*2] + g[jj*2+1]*g[jj*2+1];
            if (val>maxval) {
                maxval=val;
                bestx=x;
                besty=y;
            }
            jj++;
        }
    }

    double shift_x,shift_y;
    shift_x=bestx-M1; //right direction? -- tweaked this out carefully so I recommend letting it be
    shift_y=besty-M2;
    for (int y=0; y<N2; y++) {
        for (int x=0; x<N1; x++) {
            double theta=2*M_PI*((shift_x*(x-M1)/N1)+(shift_y*(y-M2)/N2));
            double re0=cos(theta);
            double im0=sin(theta);
            double re1=f1hat[(x+N1*y)*2]*re0-f1hat[(x+N1*y)*2+1]*im0;
            double im1=f1hat[(x+N1*y)*2]*im0+f1hat[(x+N1*y)*2+1]*re0;
            f1hat[(x+N1*y)*2]=re1;
            f1hat[(x+N1*y)*2+1]=im1;
        }
    }
    ifft_complex2real(N1,N2,f1,f1hat);

    double cx1,cy1,cx2,cy2;
    compute_center_of_mass(N1,N2,f1,cx1,cy1);
    compute_center_of_mass(N1,N2,f2,cx2,cy2);
    shift_x=cx2-cx1; //right direction?
    shift_y=cy2-cy1;
    for (int y=0; y<N2; y++) {
        for (int x=0; x<N1; x++) {
            double theta=-2*M_PI*((shift_x*(x-M1)/N1)+(shift_y*(y-M2)/N2)); //notice the negative sign -- tweaked this out carefully, so I recommend letting it be
            double re0=cos(theta);
            double im0=sin(theta);
            double re1=f1hat[(x+N1*y)*2]*re0-f1hat[(x+N1*y)*2+1]*im0;
            double im1=f1hat[(x+N1*y)*2]*im0+f1hat[(x+N1*y)*2+1]*re0;
            f1hat[(x+N1*y)*2]=re1;
            f1hat[(x+N1*y)*2+1]=im1;
        }
    }
    ifft_complex2real(N1,N2,f1,f1hat);

    free(f1hat);
    free(f2hat);
    free(ghat);
    free(g);

    double val=0;
    for (int ii=0; ii<N1*N2; ii++) {
        val+=(f1[ii]-f2[ii])*(f1[ii]-f2[ii]);
    }
    double ret=sqrt(val)/u_norm;

    return ret;
}

double register_and_compute_error(int N1, int N2, double *f1, double *f2, double *u) {
    double *f1_reversed=malloc_double(N1*N2);
    double *f1hat=malloc_double(N1*N2*2);
    fft_real2complex(N1,N2,f1hat,f1);
    for (int ii=0; ii<N1*N2; ii++) {
        f1hat[ii*2+1]=-f1hat[ii*2+1]; //take the complex conjugate!
    }
    ifft_complex2real(N1,N2,f1_reversed,f1hat);

    double err1=register_and_compute_error_2(N1,N2,f1,f2,u);
    double err2=register_and_compute_error_2(N1,N2,f1_reversed,f2,u);

    double err=err1;
    if (err2<err1) {
        memcpy(f1,f1_reversed,N1*N2*sizeof(double));
        err=err2;
    }

    free(f1_reversed);
    free(f1hat);

    return err;
}

void ap2d(int N1,int N2,double *f_out,double *u_in,double *initial,double *initial_stdevs,double tolerance,int max_iterations,double *mask,double alpha1,double alpha2,double beta) {
	AP2D_OPTIONS opts;
	opts.initial_re=(double *)malloc(sizeof(double)*N1*N2);
	opts.initial_im=(double *)malloc(sizeof(double)*N1*N2);
	for (int ii=0; ii<N1*N2; ii++) {
		opts.initial_re[ii]=initial[ii*2];
		opts.initial_im[ii]=initial[ii*2+1];
	}
	opts.initial_stdevs=initial_stdevs;
	opts.tolerance=tolerance;
	opts.max_iterations=max_iterations;
	opts.mask=mask;
	opts.alpha1=alpha1;
	opts.alpha2=alpha2;
	opts.beta=beta;
    
	ap2d(N1,N2,f_out,u_in,opts);

	free(opts.initial_re);
	free(opts.initial_im);
}