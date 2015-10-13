#include "fftw3.h"
#include <math.h>
#include "ap2d.h"
#include "mda.h"
#include "parse_command_line_params.h"
#include "omp.h"

void usage() {
	printf("usage: ap2d input.mda --out-recon=output.mda --out-resid-err=residerr.mda --tol=1e-8 --maxit=50000 --count=10 --ref=reference.mda --num-threads=2 --oversamp=1 --init-re=imr.mda --init-im=imi.mda --init-stdevs=is.mda --alpha1=0.9 --alpha2=0.95 --beta=1.5\n");
}

unsigned long long rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long long)hi << 32) | lo;
}

int main(int argc, char *argv[])
{
    //srand (time(NULL));
    srand (rdtsc());

	QStringList required; required << "tol" << "maxit" << "count" << "mask";
	QStringList optional; optional << "ref" << "out-recon" << "out-resid-err" << "num-threads" << "init-re" << "init-im" << "init-stdevs" << "alpha1" << "alpha2" << "beta";
    CLParams CLP=parse_command_line_params(argc,argv,required,optional);

    if ((!CLP.success)||(CLP.unnamed_parameters.count()!=1)) {
        usage();
        return 0;
    }

    double tolerance=CLP.named_parameters.value("tol").toDouble();
    int max_iterations=CLP.named_parameters.value("maxit").toInt();
    int count=CLP.named_parameters.value("count").toInt();
    QString reference_path=CLP.named_parameters.value("ref");
    int num_threads=CLP.named_parameters.value("num-threads","1").toInt();
	QString mask_path=CLP.named_parameters.value("mask");
	double alpha1=CLP.named_parameters.value("alpha1","0.9").toDouble();
	double alpha2=CLP.named_parameters.value("alpha2","0.95").toDouble();
	double beta=CLP.named_parameters.value("beta","1.5").toDouble();

    QString input_path=CLP.unnamed_parameters.value(0);
    QString out_recon_path=CLP.named_parameters.value("out-recon");
    QString out_resid_err_path=CLP.named_parameters.value("out-resid-err");

    Mda U; U.read(input_path);
    int N1=U.N1();
    int N2=U.N2();

    /*
    Mda X; X.read(input_path);
    int N1=X.N1();
    int N2=X.N2();

    Mda U;
    U.allocate(N1,N2);
    fft_real2real(N1,N2,U.dataPtr(),X.dataPtr());
    for (int ii=0; ii<N1*N2; ii++) {
        U.dataPtr()[ii]=fabs(U.dataPtr()[ii]);
    }
    */

    Mda Y;
    if (!out_recon_path.isEmpty()) {
        Y.allocate(N1,N2,count);
    }
    Mda resid_err;
    resid_err.allocate(count,2);

    Mda R;
    if (!reference_path.isEmpty()) {
        R.read(reference_path);
        if ((R.N1()!=N1)||(R.N2()!=N2)) {
            printf("Inconsistent dimensions for reference.\n");
            return 0;
        }
    }

    AP2D_OPTIONS opts;
	opts.initial_re=0;
	opts.initial_im=0;
    opts.initial_stdevs=0;
    opts.max_iterations=max_iterations;
    opts.tolerance=tolerance;
	opts.alpha1=alpha1;
	opts.alpha2=alpha2;
	opts.beta=beta;

	Mda init_re;
	Mda init_im;
    Mda init_stdevs;
	if (!CLP.named_parameters.value("init-re").isEmpty()) {
		init_re.read(CLP.named_parameters["init-re"]);
		init_im.read(CLP.named_parameters["init-im"]);
        init_stdevs.read(CLP.named_parameters["init-stdevs"]);

		if ((init_re.N1()!=N1)||(init_re.N2()!=N2)) {
			printf("Inconsistent dimensions for init-re %d,%d <> %d,%d\n",init_re.N1(),init_re.N2(),N1,N2);
            return 0;
        }
		if ((init_im.N1()!=N1)||(init_im.N2()!=N2)) {
			printf("Inconsistent dimensions for init-im\n");
            return 0;
        }
        if ((init_stdevs.N1()!=N1)||(init_stdevs.N2()!=N2)) {
            printf("Inconsistent dimensions for init-stdevs\n");
            return 0;
        }
		opts.initial_re=init_re.dataPtr();
		opts.initial_im=init_im.dataPtr();
        opts.initial_stdevs=init_stdevs.dataPtr();
    }

	Mda mask;
	mask.read(mask_path);
	if ((mask.N1()!=N1)||(mask.N2()!=N2)) {
		printf("Inconsistent dimensions for mask %d,%d <> %d,%d\n",mask.N1(),mask.N2(),N1,N2);
		return 0;
	}
	opts.mask=mask.dataPtr();

    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        printf("Using %d threads.\n",omp_get_num_threads());
        Mda Y0;
        Y0.allocate(N1,N2);
        Mda U0; U0=U;

        #pragma omp for
        for (int cc=0; cc<count; cc++) {
            ap2d(N1,N2,Y0.dataPtr(),U0.dataPtr(),opts);

            double resid0=compute_resid(N1,N2,Y0.dataPtr(),U.dataPtr(),opts);

            resid_err.setValue(resid0,cc,0);

            if (R.N1()>1) {
                //do the following *after* computing the residual!!
                double error0=register_and_compute_error(N1,N2,Y0.dataPtr(),R.dataPtr(),U.dataPtr());
                resid_err.setValue(error0,cc,1);
                printf("Run %d/%d, resid=%g error=%g\n",cc+1,count,resid0,error0);
            }
            else {
                printf("Run %d/%d, resid=%g\n",cc+1,count,resid0);
            }

            if (!out_recon_path.isEmpty()) {
                memcpy(&Y.dataPtr()[N1*N2*cc],Y0.dataPtr(),N1*N2*sizeof(double));
            }
        }
    }

    if (!out_recon_path.isEmpty()) {
        Y.write(out_recon_path);
    }
    if (!out_resid_err_path.isEmpty()) {
        resid_err.write(out_resid_err_path);
    }

    return 0;
}
