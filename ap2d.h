#ifndef AP2D_H
#define AP2D_H

struct AP2D_OPTIONS {
    double *initial_means_re; //(N1xN2)
    double *initial_means_im; //(N1xN2)
    double *initial_stdevs; //(N1xN2)
    double tolerance;
    int max_iterations;
    double oversamp;
};

void ap2d(int N1,int N2,double *f_out,double *u_in,const AP2D_OPTIONS &opts);
void fft_real2real(int N1,int N2,double *out,double *in);
double compute_resid(int N1,int N2,double *f,double *u,AP2D_OPTIONS opts);
double register_and_compute_error(int N1, int N2, double *f1, double *f2, double *u);

#endif // AP2D_H

