
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <stdio.h>
#include<stdlib.h>


#define MaxIT 200000 

#define BLOCK_SIZE 384
#define BATCH 1
#define MaxLoop 50 


#define ArrangeStep 10

#define NRANK 3
/*----------------------------------------------------------------------------------------------*/
#define  UNDRFLOW  	1.0e-75
#define  OVRFLOW	1.0e+75

/*----------------------------------------------------------------------------------------------*/
//-----------Gloable variables of the system---------------------------------------------------
#define Nx 1
#define Ny 240
#define Nz 240
__constant__ int Nx_cu=Nx;
__constant__ int Ny_cu=Ny;
__constant__ int Nz_cu=Nz;
#define Pi 3.141592653589

int NsA, dNsB, Narm;
double kx[Nx],kz[Nz],ky[Ny],*kxyzdz,dx,dy,dz,*wdz,*wdz_cu,*kxyzdz_cu;
double lx, ly, lz;
double hAB, fA, fB, dfB, ds0, ds2;
long NxNyNz, NxNyNz1;
long Nxh1;
char FEname[50], phname[50];

cufftHandle p_forward, p_backward;
cufftDoubleReal *in;
cufftDoubleComplex *out;
__device__ double sum[BLOCK_SIZE*BATCH];


//-------------subroutine of cuda-------------------------------------------------
cudaError_t addWithCuda(int *c,  int *a, int *b, unsigned int size);// test for add routine
void Init_g_w(int *c, int *a, const int*b);
__global__ void addKernel(int *c, const int *a, const int *b);

__global__ void pripri(double *q1,double *q2,double *q3,double *q4);

//--------------SCFT cuda sub routine-----------------------------------------------------------------------------------
__global__ void qInt_init(double *qInt);
__global__ void g_initInverse(double *g,double *qInt,int ns1,int ns);
__global__ void g_init(double *g,double *qInt,int ns1);
__global__ void w_to_wdz(double *wdz,double *w,double ds2);
__global__ void gw_to_in(cufftDoubleReal *in,double *wdz_cu,double *g,int ns1,int iz);
__global__ void sufaceField(cufftDoubleComplex *out,double *kxyzdz,int Nxh1);
__global__ void in_to_g(double *g,double *wdz_cu,cufftDoubleReal *in,long NxNyNz,int ns1,int iz);
__global__ void gw_to_inInverse(cufftDoubleReal *in,double *wdz_cu,double *g,int ns1,int iz);
__global__ void qa_to_qInt(double *qInt,double *qA,int NsA);
__global__ void qa_to_qInt2(double *qInt,double *qcB,int dNsB);
__global__ void w_to_phi(double *phlA, double *phlB,double *qA_cu,double *qcA_cu,double *qB_cu,double *qcB_cu,int NsA,int dNsB,double ffl);
__global__ void cal_ql(double *ql_cu,double *qB_cu,int dNsB);
__global__ void wa_average(double *wa0_cu,double *wA_cu);
__global__ void wa_translate(double *wA_cu,double *wa0_cu);
__global__ void phi_w(double *wA_cu,double *wB_cu,double *phA_cu,double *phB_cu, double hAB);
__global__ void phi_w_constrained(double *wA_cu,double *wB_cu,double *phA_cu,double *phB_cu, double *PhA_cu,double hAB,double lambda);
__global__ void phi_w_constrainedEx(double *wA_cu,double *wB_cu,double *phA_cu,double *phB_cu, double *PhA_cu,double hAB);
__global__ void init_out(cufftDoubleComplex *wA);

//------------------------------SCFT main subroutine-------------------------------------------------------------
void write_ph(double *phA,double *phB,double *wA,double *wB);
void sovDifFft(double *g,double *w,double *qInt,int ns,int sign);
double getConc(double *phlA_cu,double *phlB_cu,double *wA_cu,double *wB_cu,double *ql);
double freeE(double *wA,double *wB, double *phA,double *phB);
double freeE_constrained(double *wA,double *wB, double *phA,double *phB,double *PhiA,double lambda);
double freeE_constrainedEx(double *wA,double *wB, double *phA,double *phB,double *PhiA);

//------------------------------String main subroutine-------------------------------------------------------------
void swap(double *PhiA,int i,int j);
double distance_field(double *PhiA,int i,int j);
void soft(double *PhiA);
int translation(double *PhiA,int ith,int direction,int dis);
void translation_good(double *PhiA);
double distance_field_afterum(double *PhiA,double *phA,int i);

/*----------------------------------------------------------------------------------------------*/
/* spline.c
   Cubic interpolating spline. 
   from http://www.mech.uq.edu.au/staff/jacobs/nm_lib/cmathsrc
*/

/************************************************/
/*                                              */
/*  CMATH.  Copyright (c) 1989 Design Software  */
/*                                              */
/************************************************/

/*----------------------------------------------------------------------------------------------*/
/* spline.c
   Cubic interpolating spline.  from http://www.mech.uq.edu.au/staff/jacobs/nm_lib/cmathsrc
*/   
/* Cubic spline coefficients */
int 	spline (int n, int e1, int e2, double s1, double s2, double x[], double y[], double b[], double c[], double d[], int *flag);
/* spline evaluation */
double 	seval (int n, double xx, double x[], double y[], double b[], double c[], double d[], int *last);
/* derivative evaluation */
double 	deriv (int n, double xx, double x[], double b[], double c[], double d[], int *last);
/* integral of spline */
double 	sinteg (int n, double u, double x[], double y[], double b[], double c[], double d[], int *last);
double splineValue (int n,double x0,
            double x[],double y[],
            double b[], double c[], double d[],double *y0
            );
int spline_straightline (int n, 
            double x[], double y[],
            double b[], double c[], double d[],
            int *iflag);
double splineValue_straightline (int n,double x0,
            double x[],double y[],
            double b[], double c[], double d[],double *y0
            );
/*----------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------*/


//-------------------------------main function-------------------------------------------

int main()
{
	//int arraysize = 16000;
	//int *a,*b,*c;
	NxNyNz=Nx*Ny*Nz;

//-----------------scft variable-------------------------------------------------
	double *wA,*wB,*phA,*phB;
	double *wA_cu,*wB_cu,*phA_cu,*phB_cu;
	double e1,e2,ksq;
	double *PhiA;
	int i,j,k,intag,iseed=-3; //local_x_starti;
	long ijk;
	char comment[201],density_name[201];
	double temp;
	int n[NRANK] = {Nz, Ny, Nx};
	
	FILE *fp;
	time_t ts;

	Nxh1=Nx/2+1;
	NxNyNz=Nx*Ny*Nz;
	NxNyNz1=Nx*Ny*Nxh1;

	iseed=time(&ts);
	srand(iseed);
	double lambda;
	double *qInt;
	int loop;
	double X[BATCH],Y[BATCH],X_Inv[BATCH],dis;
	double x[1000],y[1000],d[1000],b[1000],c[1000],b1[1000],c1[1000],d1[1000];
	int iflag;
	double x0,y0;
	double string_length[MaxLoop];
	FILE *dp;
//-----------------init cuda-------------------------
 	cudaError_t cudaStatus;
	int computer;
	


	cudaStatus = cudaSetDevice(3);	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaGetDevice(&computer);
	printf("woring on GPU %d\n");

	if(cudaStatus!=cudaSuccess){
		printf("cuda init on device failed, check GPU");
	}
	/*if (cufftPlan3d(&p_forward, Nz, Ny, Nx, CUFFT_D2Z) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed! for p_forward\n");
		exit(0);	
	}*/	
	if (cufftPlanMany(&p_forward, NRANK, n, 
				  NULL, 1, Nx*Ny*Nz, // *inembed, istride, idist 
				  NULL, 1, (Nx/2+1)*Ny*Nz, // *onembed, ostride, odist
				  CUFFT_D2Z, BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;	
	}	

	/*if (cufftPlan3d(&p_backward, Nz, Ny, Nx, CUFFT_Z2D) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed! for p_backward\n");
		exit(0);
	}*/
	
	if (cufftPlanMany(&p_backward, NRANK, n, 
				  NULL, 1, (Nx/2+1)*Ny*Nz, // *inembed, istride, idist 
				  NULL, 1, NxNyNz, // *onembed, ostride, odist
				  CUFFT_Z2D, BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;	
	}		
	cudaStatus = cudaMalloc((void**)&in, NxNyNz * sizeof(cufftDoubleReal)*BATCH);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(0);
 	}
	cudaStatus = cudaMalloc((void**)&out, (Nx/2+1)*Ny*Nz * sizeof(cufftDoubleComplex)*BATCH);
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(0);
 	}

//---------------init variable------------------
	wA=(double *)malloc(sizeof(double)*NxNyNz*BATCH);
	wB=(double *)malloc(sizeof(double)*NxNyNz*BATCH);
	phA=(double *)malloc(sizeof(double)*NxNyNz*BATCH);
	phB=(double *)malloc(sizeof(double)*NxNyNz*BATCH);
	PhiA=(double *)malloc(sizeof(double)*NxNyNz*BATCH);
	kxyzdz=(double *)malloc(sizeof(double)*NxNyNz);
	wdz=(double *)malloc(sizeof(double)*NxNyNz*BATCH);
	//lambda=50.0;

	
	cudaMalloc((void**)&kxyzdz_cu, NxNyNz* sizeof(double));
	cudaMalloc((void**)&wdz_cu, NxNyNz* sizeof(double)*BATCH);
	
	fp=fopen("ab.txt","r");
	
	fscanf(fp,"%d",&intag);		//in=1: inputing configuration is given;
	fscanf(fp,"%lf",&hAB);
	fscanf(fp, "%lf", &fA);
	fscanf(fp,"%lf, %lf, %lf",&lx, &ly, &lz);
	fscanf(fp,"%s",FEname);				//output file name for parameters;
	fscanf(fp,"%s",phname);				//output file name for configuration;
	fscanf(fp, "%lf", &ds0);
	fscanf(fp, "%d", &Narm);
	fclose(fp);
	printf("%d\n",intag);
	ds2=ds0/2;

	fB=1.0-fA;

	dx=lx/Nx;
	if(intag==220||intag==2){ly=sqrt(3.0)*lx;lz=lx;}
	if(intag==8888||intag==440||intag==4||intag==1||intag==110){ly=lx;lz=lx;}
	if(intag==3332||intag==3333||intag==990||intag==9||intag==660||intag==6||intag==330||intag==3)
	{
		//ly=lx; lz=lx;	/* gyroid, fcc, bcc */
	}
	if(intag==2255||intag==22||intag==1155||intag==11) /* perforated lam */
	{
		ly=sqrt(3.0)*lx;
	}
	
	dy=ly/Ny;
	dz=lz/Nz;

	printf("nx=%d ny=%d nz=%d,lx=%lf ly=%lf lz=%lf\n",Nx,Ny,Nz,lx,ly,lz);
	printf("%lf\n", hAB);
	printf("%lf\n", fA);		
    	fp=fopen(FEname,"w");
	fprintf(fp,"Nx=%d, Nz=%d\n",Nx,Nz);
   	fprintf(fp,"dx=%.6lf, dz=%.6lf\n",dx,dz);	
	fclose(fp);

	dfB=fB/Narm;
	NsA = ((int)(fA/ds0+1.0e-8));
	dNsB = ((int)(dfB/ds0+1.0e-8));

	printf("NsA = %d, dNsB = %d\n", NsA, dNsB);
	
	//**************************definition of surface field and confinement***********************


	for(i=0;i<=Nx/2-1;i++)kx[i]=2*Pi*i*1.0/Nx/dx;
	for(i=Nx/2;i<Nx;i++)kx[i]=2*Pi*(i-Nx)*1.0/dx/Nx;
	for(i=0;i<Nx;i++)kx[i]*=kx[i];

	for(i=0;i<=Ny/2-1;i++)ky[i]=2*Pi*i*1.0/Ny/dy;
	for(i=Ny/2;i<Ny;i++)ky[i]=2*Pi*(i-Ny)*1.0/dy/Ny;
	for(i=0;i<Ny;i++)ky[i]*=ky[i];

    	for(i=0;i<=Nz/2-1;i++)kz[i]=2*Pi*i*1.0/Nz/dz;
    	for(i=Nz/2;i<Nz;i++)kz[i]=2*Pi*(i-Nz)*1.0/dz/Nz;
    	for(i=0;i<Nz;i++)kz[i]*=kz[i];	

	for(k=0;k<Nz;k++)for(j=0;j<Ny;j++)for(i=0;i<Nx;i++)
	{
		ijk=(long)((k*Ny+j)*Nx+i);
		ksq=kx[i]+ky[j]+kz[k];
		kxyzdz[ijk]=exp(-ds0*ksq);
	}
	if (cudaMemcpy(kxyzdz_cu,  kxyzdz,sizeof(double)*NxNyNz,cudaMemcpyHostToDevice)!= cudaSuccess){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;	
	}
	
	/***************Initialize wA, wB******************/
	
	if(intag==0);//initW(wA,wB);
	
	
	else if(intag==1024)
	{
		printf("intag==24\n");
		fp=fopen("pha_eq.dat","r");
		fgets(comment,200,fp);       
		fgets(comment,200,fp);
		
                for(ijk=0;ijk<NxNyNz;ijk++)
                {

                        
			fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",&temp,&temp,&temp,&e1,&e2,&temp,&temp,&temp,&temp,&temp,&temp);
                        wA[ijk]=hAB*e1;
                        wB[ijk]=hAB*e2;

                }
                fclose(fp);
		//initW(wA,wB);
	}
	else if(intag==1026){
	/*
			sprintf(density_name,"phi_29.dat",i);

			

				fp=fopen(density_name,"r");

				fgets(comment,200,fp);       

				fgets(comment,200,fp);



               		for(ijk=0;ijk<NxNyNz;ijk++){

				fscanf(fp,"%lf %lf %lf %lf\n",&PhiA[ijk],&temp,&wA[ijk],&wB[ijk]);

						

				

				}

				fclose(fp);
		*/
			
			
			for(i=1;i<=BATCH;i++){	
				sprintf(density_name,"phi_%d.dat",i);
			
				fp=fopen(density_name,"r");
				fgets(comment,200,fp);       
				fgets(comment,200,fp);

               		for(ijk=0;ijk<NxNyNz;ijk++){
				fscanf(fp,"%lf %lf %lf %lf\n",&PhiA[ijk+NxNyNz*(i-1)],&temp,&wA[ijk+NxNyNz*(i-1)],&wB[ijk+NxNyNz*(i-1)]);
						
				
				}
				fclose(fp);
			}
            
			
	}
//----------------------delete after Test-----------------------
	//freeE_constrainedEx(wA,wB,phA,phB,PhiA);
	
	//freeE_constrained(wA,wB,phA,phB,PhiA,lambda);
	

	freeE(wA,wB,phA,phB);
	/*
	cudaEvent_t stop,start;
	cudaError_t error;
	float  msecTotal,msec;
	error=cudaEventCreate(&start);
	error=cudaEventCreate(&stop);
	error=cudaEventCreate(&start);
	error=cudaEventCreate(&stop);
	for(loop=0;loop<MaxLoop;loop++){
		
		translation_good(PhiA);
		//soft(PhiA);
		
		//freeE_constrained(wA,wB,phA,phB,PhiA,lambda);
		error=cudaEventRecord(start,NULL);
		
		
		
		
			
			

		
		lambda=40;
		freeE_constrained(wA,wB,phA,phB,PhiA,lambda);
		error=cudaDeviceSynchronize();
		if(error!=cudaSuccess) printf("something wrong!before %d\n",loop);
		error=cudaEventRecord(stop,NULL);	
		cudaEventSynchronize(stop);	
			
		error=cudaEventElapsedTime(&msec,start,stop);
		printf("loop=%d %0.10f\n",loop,msec);
		for(i=1;i<=BATCH;i++){
			printf("%g ",PhiA[(i-1)*NxNyNz+10]);
			
		}
		printf("\n");
		printf("------------------------------------------------------\n");
		for(i=1;i<=BATCH;i++){
			printf("%g ",phA[(i-1)*NxNyNz+10]);
			
		}
		printf("\n");
		printf("----------------------------------------------\n");
		dis=0.0;
		X[0]=0;
		for(i=1;i<BATCH;i++){
			X_Inv[i-1]=distance_field(PhiA,i,i+1);
			dis+=X_Inv[i-1];
			X[i]=dis;
			
		}
		
		for(i=0;i<BATCH;i++){
			printf("%g ",X[i]);
			
		}
		printf("\n");
		printf("----------------------------------------------\n");
		string_length[loop]=dis;
		for(i=1;i<=BATCH;i++){
			X[i-1]=X[i-1]/dis;
		}
		for(ijk=0;ijk<NxNyNz;ijk++){
			
				for(i=1;i<=BATCH;i++){
					Y[i-1]=phA[ijk+(i-1)*NxNyNz];
					//printf("%g %g\n",X[i-1],Y[i-1]);
				
				}
				
				
				//spline_straightline(BATCH,X,Y,b1,c1,d1,&iflag);
				
				spline(BATCH,0,0,0.0,0.0,X,Y,b,c,d,&iflag);
				for(i=0;i<BATCH;i++){
					//printf("%d %g %g %g %g %g\n",iflag,X[i],Y[i],b[i],c[i],d[i]);
				}
				PhiA[ijk]=phA[ijk];
				for(i=2;i<BATCH;i++){
					x0=(double)(i-1)/(BATCH-1);
					
					//splineValue_straightline(BATCH,x0,X,Y,b1,c1,d1,&y0);
					//splineValue(BATCH,x0,X,Y,b,c,d,&y0);
					//PhiA[ijk+(i-1)*NxNyNz]=phA[ijk+(i-1)*NxNyNz];
					y0=seval (BATCH, x0,X,Y,b,  c,  d,&iflag);
					//printf("%g %g\n",x0,y0);
					PhiA[ijk+(i-1)*NxNyNz]= y0;
					//if(i==5||i==6){
						//printf("%g %g %g %g %g\n",X[i],Y[i],b[i],c[i],d[i]);
					//}
				}
				//printf("\n");
				PhiA[ijk+(BATCH-1)*NxNyNz]=phA[ijk+(BATCH-1)*NxNyNz];
			
		}
		for(i=1;i<=BATCH;i++){
			printf("%g ",PhiA[(i-1)*NxNyNz+10]);
			
		}
		printf("\n");
		printf("----------------------------------------------\n");
		
		//freeE_constrainedEx(wA,wB,phA,phB,PhiA);
		write_ph(phA,phB,wA,wB);
		
	}
	dp=fopen("length.dat","w");
	for(i=0;i<MaxLoop;i++){
		fprintf(dp,"%d %g\n",i,string_length[i]);
	}
	fclose(dp);
	*/

//---------------------	free memery----------------------------
	cufftDestroy(p_forward);
	cufftDestroy(p_backward);
	cudaFree(in);
    	cudaFree(out);
	cudaFree(kxyzdz_cu);
	cudaFree(wdz_cu);
	cudaFree(sum);
	//free(wA);
	//free(wB);
	//free(phA);
	//free(phB);
	
    	return 0;
}


//--------------------------------------------------------------------------------------
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c,int *a,  int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;
	dim3 grid(40,40,1);
	dim3 grid_d(10,1,1);
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	// cudaStatus = cudaMalloc((void**)&in, size * sizeof(double));
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
   // addKernel<<<grid, 10>>>(dev_c, dev_a, dev_b);
	Init_g_w(dev_c, dev_a, dev_b);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
//********************Output configuration******************************

void write_ph(double *phA,double *phB,double *wA,double *wB)
{
	int i,j,k;
	long ijk;
	

      
	for(i=1;i<=BATCH;i++){
		sprintf(phname,"pha_%d.dat",i);
		FILE *fp=fopen(phname,"w");
		fprintf(fp, "Nx=%d, Ny=%d, Nz=%d\n", Nx, Ny, Nz);
        	fprintf(fp, "dx=%lf, dy=%lf, dz=%lf\n", dx, dy, dz);
		for(ijk=0;ijk<NxNyNz;ijk++)
		{
			fprintf(fp,"%lf %lf %lf %lf\n",phA[ijk+(i-1)*NxNyNz],phB[ijk+(i-1)*NxNyNz],wA[ijk+(i-1)*NxNyNz],wB[ijk+(i-1)*NxNyNz]);
		}
		fclose(fp);
	}
	
}
//--------------------------------------------------------------
__global__ void pri(double *q1,double *q2,double *q3,double *q4){
int i,j;
long NxNyNz=Nx*Ny*Nz;
for(i=0;i<NxNyNz;i++){
	j=i*51+20;
printf("%d %g %g %g %g\n",i,q1[j],q2[j],q3[j],q4[j]);
	j=j+NxNyNz*51;
printf("%d %g %g %g %g\n",i,q1[j],q2[j],q3[j],q4[j]);

}

//printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);

__syncthreads();
}
__global__ void addKernel(int *c, const int *a, const int *b,int ns1)
{
	
	
	int i = blockIdx.x+blockIdx.y*BLOCK_SIZE+threadIdx.x*1600;
	
		c[i] = a[i] + b[i]+ns1;
}

void Init_g_w(int *c, int *a, const int*b)
{
	dim3 grid(40,40,1);
	int ns1;
	printf("all ok\n");
	//cudaMalloc((void**)&ns1, sizeof(int));
	ns1=50;
	 addKernel<<<grid, 10>>>(c, a, b,ns1);

}

__global__ void qInt_init(double *qInt){
	//int i=blockIdx.x*BLOCK_SIZE;
	//qInt[i]=1.0;
	long i,j;
	long NxNyNz=Nx*Ny*Nz;
	long k=blockIdx.x*NxNyNz;
	i=threadIdx.x*(NxNyNz/BLOCK_SIZE);
	for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++)
		qInt[i+j+k]=1.0;
__syncthreads();

}
__global__ void g_init(double *g,double *qInt,int ns1){
	long i=blockIdx.x+blockIdx.y*Nx+blockIdx.z*Nx*Ny;
	long NxNyNz=Nx*Ny*Nz;
	long offset=threadIdx.x*NxNyNz*ns1;
	g[i*ns1+offset]=qInt[i+offset/ns1];
	for(int j=1;j<(ns1);j++)
		g[i*ns1+j+offset]=0.0;
__syncthreads();

}
__global__ void g_initInverse(double *g,double *qInt,int ns1,int ns){
	long i=blockIdx.x+blockIdx.y*Nx+blockIdx.z*Nx*Ny;
	long offset=threadIdx.x*Nx*Ny*Nz*ns1;
	g[i*ns1+ns+offset]=qInt[i+offset/ns1];
	
	for(int j=0;j<(ns);j++)
		g[i*ns1+j+offset]=0.0;
	for(int j=ns+1;j<(ns1);j++)
		g[i*ns1+j+offset]=0.0;
__syncthreads();

}
__global__ void w_to_wdz(double *wdz_cu,double *w,double ds2){
	long NxNyNz=Nx*Ny*Nz;
	long i=blockIdx.x+blockIdx.y*Nx+blockIdx.z*Nx*Ny;
	long j=threadIdx.x*(NxNyNz);
	
	wdz_cu[i+j]=exp(-w[i+j]*ds2);
__syncthreads();
}
__global__ void gw_to_in(cufftDoubleReal *in,double *wdz_cu,double *g,int ns1,int iz){
	//int i=blockIdx.x+blockIdx.y*Nx+blockIdx.z*Nx*Ny;
	//in[i]=wdz_cu[i]*g[i*ns1+iz-1];
	long i,j;
	long NxNyNz=Nx*Ny*Nz;
	long offset=blockIdx.x*NxNyNz;
	
	long offset_q=blockIdx.x*NxNyNz*ns1;
	i=threadIdx.x*(NxNyNz/BLOCK_SIZE);
	
	for(long j=0;j<(NxNyNz/BLOCK_SIZE);j++){
			//in[i+j+offset]=i+j+offset;
			in[i+j+offset]=wdz_cu[i+j+offset]*g[(i+j)*ns1+iz-1+offset_q];
			
	}
__syncthreads();
}
__global__ void gw_to_inInverse(cufftDoubleReal *in,double *wdz_cu,double *g,int ns1,int iz){
	//int i=blockIdx.x+blockIdx.y*Nx+blockIdx.z*Nx*Ny;
	//in[i]=wdz_cu[i]*g[i*ns1+iz+1];
	long i,j;
	long NxNyNz=Nx*Ny*Nz;
	long offset=blockIdx.x*Nx*Ny*Nz*ns1;
	long offset_in=blockIdx.x*(Nx)*Ny*Nz;
	i=threadIdx.x*(NxNyNz/BLOCK_SIZE);
	
	for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++){
			in[i+j+offset_in]=wdz_cu[i+j+offset/ns1]*g[(i+j)*ns1+iz+1+offset];
			
	}
__syncthreads();
}
__global__ void sufaceField(cufftDoubleComplex *out,double *kxyzdz,int Nxh1){

long ijk,ijkr;

long NxNyNz=Nx*Ny*Nz;
long Nxh1NyNz=(Nx/2+1)*Ny*Nz;
long offset=Nxh1NyNz*threadIdx.x;

ijk=blockIdx.x+blockIdx.y*Nxh1+blockIdx.z*Nxh1*Ny+offset;

ijkr=blockIdx.x+blockIdx.y*Nx+blockIdx.z*Nx*Ny;

out[ijk].x*=kxyzdz[ijkr];
out[ijk].y*=kxyzdz[ijkr];

__syncthreads();
}
__global__ void in_to_g(double *g,double *wdz_cu,cufftDoubleReal *in,long NxNyNz,int ns1,int iz){
	//int i=blockIdx.x+blockIdx.y*Nx+blockIdx.z*Nx*Ny;
	//g[i*ns1+iz]=in[i]*wdz_cu[i]/NxNyNz;
	long i,j;
	long offset=NxNyNz*blockIdx.x;
	i=threadIdx.x*(NxNyNz/BLOCK_SIZE);
	
	for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++)
		g[(i+j)*ns1+iz+offset*ns1]=in[(i+j)+offset]*wdz_cu[i+j+offset]/NxNyNz;
__syncthreads();
}
__global__ void qa_to_qInt(double *qInt,double *qA,int NsA){
	long NxNyNz=Nx*Ny*Nz;
	long i=blockIdx.x+blockIdx.y*Nx+blockIdx.z*Nx*Ny;
	long offset=threadIdx.x*NxNyNz*(NsA+1);


	qInt[i+offset/(NsA+1)]=qA[i*(NsA+1)+NsA+offset];
__syncthreads();
}
__global__ void qa_to_qInt2(double *qInt,double *qcB,int dNsB){
	int NxNyNz=Nx*Ny*Nz;
	long i=blockIdx.x+blockIdx.y*Nx+blockIdx.z*Nx*Ny;
	int offset=threadIdx.x*NxNyNz*(dNsB+1);
	
	qInt[i+offset/(dNsB+1)]=qcB[i*(dNsB+1)+offset];
__syncthreads();
}
__global__ void w_to_phi(double *phlA, double *phlB,double *qA_cu,double *qcA_cu,double *qB_cu,double *qcB_cu,int NsA,int dNsB,double *ffl){
	long ijkiz;
	long NxNyNz=Nx*Ny*Nz;
	long offset=blockIdx.x*NxNyNz;
	long offset_qA=blockIdx.x*NxNyNz*(NsA+1);
	long offset_qB=blockIdx.x*NxNyNz*(dNsB+1);
	long i=threadIdx.x*NxNyNz/BLOCK_SIZE;
	int iz;
	for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++){
		phlA[i+j+offset]=0.0;
		phlB[i+j+offset]=0.0;
	}
	
	
	
	for(iz=0;iz<=NsA;iz++){
		for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++){
			ijkiz=(i+j)*(NsA+1)+iz+offset_qA;
			if(iz==0||(iz==NsA))phlA[i+j+offset]+=(0.50*qA_cu[ijkiz]*qcA_cu[ijkiz]);
			else phlA[i+j+offset]+=qA_cu[ijkiz]*qcA_cu[ijkiz];
			//if(threadIdx.x==0&&blockIdx.x==0)
			//printf("%d %g %g %g\n",ijkiz,qA_cu[ijkiz],qcA_cu[ijkiz],phlA[i+j+offset] );
		}
	}
	
	//if(threadIdx.x==0&&blockIdx.x==0) printf("GPU= %g %g %d\n",phlA[0],phlB[0],i);
	
	//printf("GPU %d %d %d\n",offset_qA,threadIdx.x,blockIdx.x);
	
	for(iz=0;iz<=dNsB;iz++){
		
		for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++){
			ijkiz=(i+j)*(dNsB+1)+iz+offset_qB;
			if(iz==0||(iz==dNsB))phlB[i+j+offset]+=(0.50*qB_cu[ijkiz]*qcB_cu[ijkiz]);
			else phlB[i+j+offset]+=qB_cu[ijkiz]*qcB_cu[ijkiz];
		}
	}
	
//	if(threadIdx.x==0&&blockIdx.x==0) printf("GPU= %g %g %d\n",phlA[i+0+offset],phlB[i+0+offset],i);
	
	//if(threadIdx.x==0&&blockIdx.x==0) printf("0 GPU= %g %g %d\n",phlA[i+0],phlB[i+0],i);
	for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++){
		phlA[i+j+offset]*=ffl[blockIdx.x];
		phlB[i+j+offset]*=ffl[blockIdx.x];
	}
//	printf("%g\n",phlA[0]);
__syncthreads();
	

}
__global__ void cal_ql(double *ql_cu,double *qB_cu,int dNsB){
	long i,j;
	
	long NxNyNz=Nx*Ny*Nz;
	i=(NxNyNz/BLOCK_SIZE)*threadIdx.x;

	long offset_sum=BLOCK_SIZE*blockIdx.x;
	long offset_qB=NxNyNz*(dNsB+1)*blockIdx.x;
		sum[threadIdx.x+offset_sum]=0.0;
	
	for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++) {
		sum[threadIdx.x+offset_sum]+=qB_cu[(i+j)*(dNsB+1)+dNsB+offset_qB];//qB_cu[(i+j)*(dNsB+1)+dNsB];
		
	
	}

	
	__syncthreads();
if(threadIdx.x==0){

	for(j=0;j<BLOCK_SIZE;j++)ql_cu[blockIdx.x]+=sum[j+offset_sum];

}


	

//printf("Id=%d i=%d %g %g\n",threadIdx.x,i,sum[BLOCK_SIZE-1],ss);
	
	//ql_cu[blockIdx.x]=1;
	/*
	*///
__syncthreads();
}

__global__ void wa_average(double *wa0_cu,double *wA_cu){
	long i,j;
	long NxNyNz=Nx*Ny*Nz;
	i=BLOCK_SIZE*threadIdx.x;
	long offset=NxNyNz*blockIdx.x;
	int offset_sum=BLOCK_SIZE*blockIdx.x;
	sum[threadIdx.x+offset_sum]=0;
	i= threadIdx.x*(NxNyNz/BLOCK_SIZE)+offset;
	for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++)
		sum[threadIdx.x+offset_sum]+=wA_cu[i+j];
	__syncthreads();
	if(threadIdx.x==0){
		for(j=0;j<BLOCK_SIZE;j++)
			wa0_cu[blockIdx.x]+=sum[j+offset_sum];
		wa0_cu[blockIdx.x]/=NxNyNz;
	}

}
__global__ void wa_translate(double *wA_cu,double *wa0_cu){
	long i,j;
	long NxNyNz=Nx*Ny*Nz;
	i=threadIdx.x*(NxNyNz/BLOCK_SIZE);
	long offset=NxNyNz*blockIdx.x;
	for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++)
		wA_cu[i+j+offset]-=wa0_cu[blockIdx.x];
__syncthreads();
}

__global__ void phi_w(double *wA_cu,double *wB_cu,double *phA_cu,double *phB_cu, double hAB){
	long i,j;
	long NxNyNz=Nx*Ny*Nz;
	i=threadIdx.x*(NxNyNz/BLOCK_SIZE);
	double eta;
	double psum;
	double waDiff,wbDiff;
	double wcmp,wopt;
	
		wopt=0.050;
		wcmp=0.10;
	
	long offset=blockIdx.x*NxNyNz;
	//printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
	//printf("%g %g\n",wA_cu[i],wB_cu[i]);
	
	for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++){
		eta=(wA_cu[i+j+offset]+wB_cu[i+j+offset]-hAB)/2;
		psum=1.0-phA_cu[i+j+offset]-phB_cu[i+j+offset];
		
		
		waDiff=hAB*phB_cu[i+j+offset]+eta-wA_cu[i+j+offset];
		wbDiff=hAB*phA_cu[i+j+offset]+eta-wB_cu[i+j+offset];
		waDiff-=wcmp*psum;
		wbDiff-=wcmp*psum;
		wA_cu[i+j+offset]+=wopt*waDiff;
		wB_cu[i+j+offset]+=wopt*wbDiff;
	}
	//printf("%g %g %d\n",phA_cu[i],phB_cu[i],NxNyNz/BLOCK_SIZE);	
__syncthreads();
}

__global__ void phi_w_constrained(double *wA_cu,double *wB_cu,double *phA_cu,double *phB_cu, double *PhA_cu,double hAB,double lambda){
	long i,j;
	long NxNyNz=Nx*Ny*Nz;
	i=threadIdx.x*(NxNyNz/BLOCK_SIZE);
	double eta;
	double psum;
	double waDiff,wbDiff;
	double wcmp,wopt;
	
		wopt=0.050;
		wcmp=0.10;
	
	long offset=blockIdx.x*NxNyNz;
	
	
	for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++){
		eta=(wA_cu[i+j+offset]+wB_cu[i+j+offset]-hAB)/2;
		psum=1.0-phA_cu[i+j+offset]-phB_cu[i+j+offset];
		
		
		waDiff=hAB*phB_cu[i+j+offset]+eta-wA_cu[i+j+offset]+lambda*(phA_cu[i+j+offset]-PhA_cu[i+j+offset]);
		wbDiff=hAB*phA_cu[i+j+offset]+eta-wB_cu[i+j+offset]+lambda*(phB_cu[i+j+offset]-(1-PhA_cu[i+j+offset]));
		waDiff-=wcmp*psum;
		wbDiff-=wcmp*psum;
		wA_cu[i+j+offset]+=wopt*waDiff;
		wB_cu[i+j+offset]+=wopt*wbDiff;
	}
	//printf("%g %g %d\n",phA_cu[i],phB_cu[i],NxNyNz/BLOCK_SIZE);	
__syncthreads();
}

__global__ void phi_w_constrainedEx(double *wA_cu,double *wB_cu,double *phA_cu,double *phB_cu, double *PhA_cu,double hAB){
	long i,j;
	long NxNyNz=Nx*Ny*Nz;
	i=threadIdx.x*(NxNyNz/BLOCK_SIZE);
	double eta,wum;
	double psum,psum1;
	double waDiff,wbDiff;
	double wcmp,wopt;
	
		wopt=1.60;
		wcmp=0.50;
	
	long offset=blockIdx.x*NxNyNz;
	
	
	for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++){
		wum=-hAB*(PhA_cu[i+j+offset]*2-1)-(wA_cu[i+j+offset]-wB_cu[i+j+offset]);
		eta=(wA_cu[i+j+offset]+wB_cu[i+j+offset]-hAB+wum)/2;
		psum=1.0-phA_cu[i+j+offset]-phB_cu[i+j+offset];
		psum1=phA_cu[i+j+offset]-PhA_cu[i+j+offset];
		waDiff=hAB*(1-PhA_cu[i+j+offset])+eta-wA_cu[i+j+offset]-wum;
		wbDiff=hAB*PhA_cu[i+j+offset]+eta-wB_cu[i+j+offset];
		waDiff-=2.0*wcmp*psum-2.0*wcmp*psum1;//+wcmp*fpsum1*5;
		wbDiff-=2*wcmp*psum1;
		wA_cu[i+j+offset]+=wopt*waDiff;
		wB_cu[i+j+offset]+=wopt*wbDiff;
//-------------------------------------------------------------------------------------
/*
		eta=(wA_cu[i+j+offset]+wB_cu[i+j+offset]-hAB)/2;
		psum=1.0-phA_cu[i+j+offset]-phB_cu[i+j+offset];
		
		
		waDiff=hAB*phB_cu[i+j+offset]+eta-wA_cu[i+j+offset];
		wbDiff=hAB*phA_cu[i+j+offset]+eta-wB_cu[i+j+offset];
		waDiff-=wcmp*psum;
		wbDiff-=wcmp*psum;
		wA_cu[i+j+offset]+=wopt*waDiff;
		wB_cu[i+j+offset]+=wopt*wbDiff;
*/
	}
	//printf("%g %g %d\n",phA_cu[i],phB_cu[i],NxNyNz/BLOCK_SIZE);	
__syncthreads();
}


__global__ void init_out(cufftDoubleComplex *wA){
	long i,j;
	long NxNyNz=Nx*Ny*Nz;
	i=threadIdx.x*(NxNyNz/BLOCK_SIZE);
	
	for(int j=0;j<(NxNyNz/BLOCK_SIZE);j++){
	wA[i+j].x=0.0;
	wA[i+j].y=0.0;
	}
		
__syncthreads();
}
//-------------subroutine of sovle diffusion function-------------------------------------------------
void sovDifFft(double *g,double *w,double *qInt,int ns,int sign){
	cudaError_t cudaStatus;
	cufftResult error1;
	
	dim3 grid(Nx,Ny,Nz);
	dim3 grid2(Nxh1,Ny,Nz);
	int ns1;
	
	ns1=ns+1;
	cufftResult state;
	cudaEvent_t stop,start;
	cudaError_t error;
	float  msecTotal,msec;
	
	//error=cudaEventCreate(&start);
	//error=cudaEventCreate(&stop);
	//error=cudaEventCreate(&start);
	//error=cudaEventCreate(&stop);
		
	w_to_wdz<<<grid,BATCH>>>(wdz_cu,w,ds2);

	if(sign==1){
		g_init<<<grid, BATCH>>>(g,qInt,ns1);
		
		for(int iz=1;iz<=ns;iz++){
			
			gw_to_in<<<BATCH, BLOCK_SIZE>>>(in,wdz_cu,g,ns1,iz);
			error=cudaDeviceSynchronize();
			if(error!=cudaSuccess) printf("something wrong!before \n");	
			error1=cufftExecD2Z(p_forward, in, out);
			//error1=cufftExecD2Z(p_forward, in, out);
			if (error1!= CUFFT_SUCCESS){
					
					switch (error1)
    					{
        					case CUFFT_SUCCESS:
          						printf( "CUFFT_SUCCESS\n"); break;

       						case CUFFT_INVALID_PLAN:
            						printf( "CUFFT_INVALID_PLAN\n"); break;

       						case CUFFT_ALLOC_FAILED:
            						printf("CUFFT_ALLOC_FAILED\n"); break;

       						case CUFFT_INVALID_TYPE:
           				 		printf("CUFFT_INVALID_TYPE\n"); break;

        					case CUFFT_INVALID_VALUE:
            						printf( "CUFFT_INVALID_VALUE\n"); break;

        					case CUFFT_INTERNAL_ERROR:
            						printf("CUFFT_INTERNAL_ERROR\n"); break;

      			  			case CUFFT_EXEC_FAILED:
            						printf("CUFFT_EXEC_FAILED\n"); break;

        					case CUFFT_SETUP_FAILED:
            						printf("CUFFT_SETUP_FAILED\n"); break;

        					case CUFFT_INVALID_SIZE:
            						printf("CUFFT_INVALID_SIZE\n"); break;

        					case CUFFT_UNALIGNED_DATA:
            						printf("CUFFT_UNALIGNED_DATA\n"); break;
   		 			}

					fprintf(stderr, "CUFFT error: Plan Exec forward failed...\n");
					exit(0);	
			}	
			
			sufaceField<<<grid2,BATCH>>>(out,kxyzdz_cu, Nxh1);
			
		
			state=cufftExecZ2D(p_backward, out, in);
			
		
			if (state!=CUFFT_SUCCESS){
				if(state==CUFFT_INVALID_PLAN) printf("CUFFT_INVALID_PLAN\n");
				
				fprintf(stderr, "CUFFT error: Plan Exec backward failed!!!!!\n");
				exit(0);	
			}
			
			
			
		
		
			in_to_g<<<BATCH, BLOCK_SIZE>>>(g,wdz_cu,in,NxNyNz,ns1,iz);
			/*
			
			*/
			
		}
			
	
	
	}
		
	else{
		
		g_initInverse<<<grid, BATCH>>>(g,qInt,ns1,ns);
		
		for(int iz=ns-1;iz>=0;iz--){
			
			gw_to_inInverse<<<BATCH, BLOCK_SIZE>>>(in,wdz_cu,g,ns1,iz);

			if (cufftExecD2Z(p_forward, in, out)!= CUFFT_SUCCESS){
					fprintf(stderr, "CUFFT error: Plan Exec forward failed\n");
					exit(0);	
			}
			
			
			sufaceField<<<grid2,BATCH>>>(out,kxyzdz_cu, Nxh1);	
			
			state=cufftExecZ2D(p_backward, out, in);
				
			if (state!=CUFFT_SUCCESS){
				fprintf(stderr, "CUFFT error: Plan Exec backward failed!!!!!\n");
				exit(0);	
			}
			in_to_g<<<BATCH, BLOCK_SIZE>>>(g,wdz_cu,in,NxNyNz,ns1,iz);
			
				
		}
			
			
			

	}
	
    	
}
double getConc(double *phlA_cu,double *phlB_cu,double *wA_cu,double *wB_cu,double *ql){

	double *qA_cu,*qcA_cu,*qB_cu,*qcB_cu;
	int i,j,k;
	double *qInt_cu;
	double *ffl,*ffl_cu,*ql_cu;
	dim3 grid(Nx,Ny,Nz);
	cudaError_t error;
	
	cudaMalloc((void**)&qA_cu, sizeof(double)* NxNyNz*(NsA+1)*BATCH);
	cudaMalloc((void**)&qcA_cu, sizeof(double)* NxNyNz*(NsA+1)*BATCH);
	cudaMalloc((void**)&qB_cu, sizeof(double)* NxNyNz*(dNsB+1)*BATCH);
	cudaMalloc((void**)&qcB_cu, sizeof(double)* NxNyNz*(dNsB+1)*BATCH);
	cudaMalloc((void**)&qInt_cu, sizeof(double)* NxNyNz*BATCH);
	cudaMalloc((void**)&ql_cu, sizeof(double)*BATCH);
	cudaMalloc((void**)&ffl_cu, sizeof(double)*BATCH);
	ffl=(double *)malloc(sizeof(double)*BATCH);
	

	qInt_init<<<BATCH, BLOCK_SIZE>>>(qInt_cu);
	
			
	
			
	sovDifFft(qA_cu,wA_cu,qInt_cu,NsA,1);
	
	

	sovDifFft(qcB_cu,wB_cu,qInt_cu,dNsB,-1);
	qa_to_qInt<<<grid, BATCH>>>(qInt_cu,qA_cu,NsA);
	
	
	
	sovDifFft(qB_cu,wB_cu,qInt_cu,dNsB,1);
	
	
	qa_to_qInt2<<<grid, BATCH>>>(qInt_cu,qcB_cu,dNsB);
	
	sovDifFft(qcA_cu,wA_cu,qInt_cu,NsA,-1);
	
	for(i=0;i<BATCH;i++){
		ql[i]=0.0;
	}
	
	cudaMemcpy(ql_cu,  ql,sizeof(double)*BATCH,cudaMemcpyHostToDevice);
	
	cal_ql<<<BATCH,BLOCK_SIZE>>>(ql_cu,qB_cu,dNsB);
	
	
	cudaMemcpy(ql,  ql_cu,sizeof(double)*BATCH,cudaMemcpyDeviceToHost);
	for(i=0;i<BATCH;i++){
		ql[i]/=NxNyNz;
		ffl[i]=ds0/ql[i];
	}
	
	//for(i=0;i<BATCH;i++)
	//printf("%g %g\n",ql[i],ffl[i]);
	

		
	error=cudaDeviceSynchronize();
	if(error!=cudaSuccess) printf("here problem!!!! \n");
	
	cudaMemcpy(ffl_cu,  ffl,sizeof(double)*BATCH,cudaMemcpyHostToDevice);
	
	w_to_phi<<<BATCH,BLOCK_SIZE>>>(phlA_cu,phlB_cu,qA_cu,qcA_cu,qB_cu,qcB_cu,NsA,dNsB,ffl_cu);
	

	error=cudaDeviceSynchronize();
	if(error!=cudaSuccess) printf("here problem! \n");
	//exit(0);

	
	
	cudaFree(qA_cu);
	cudaFree(qcA_cu);
	cudaFree(qB_cu);
	cudaFree(qcB_cu);
	cudaFree(qInt_cu);
	cudaFree(ql_cu);
	cudaFree(ffl_cu);
	free(ffl);
	return 0;

/*
	
	if (cudaMemcpy(qInt_cu,  qInt,sizeof(double)*NxNyNz,cudaMemcpyHostToDevice)!= cudaSuccess){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;	
	}
	if (cudaMemcpy(wA_cu,  wA,sizeof(double)*NxNyNz,cudaMemcpyHostToDevice)!= cudaSuccess){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;	
	}
	

	sovDifFft(qA_cu,wA_cu,qInt_cu,NsA,-1);
	
	

	if (cudaMemcpy(qA,  qA_cu,sizeof(double)*(NsA+1)*64,cudaMemcpyDeviceToHost)!= cudaSuccess){
		fprintf(stderr, "CUFFT error: coppy failed!!!!!\n");
		return;	
	}
	for(i=0;i<NxNyNz*(1);i++){//NsA+1
		printf("%d %g\n",i,qA[i]);
	}
*/

}

double freeE(double *wA,double *wB, double *phA,double *phB){
	int i,j,k,iter,maxIter;
	long ijk;
	double freeEnergy[BATCH],freeOld[BATCH],*qCab,Sm1,Sm2;
	double freeW[BATCH],freeAB[BATCH],freeS[BATCH],freeDiff[BATCH],freeWsurf[BATCH];
	double *wa0,*wb0;
	double *wa0_cu,*wb0_cu;
	double inCompMax[BATCH];
	FILE *fp;
	dim3 grid(Nx,Ny,Nz);
	double *wA_cu,*wB_cu,*phA_cu,*phB_cu;
	double psum[BATCH],fpsum[BATCH];
	maxIter=MaxIT;
	
	Sm1=0.2e-7;
	Sm2=0.1e-10;
	iter=0;	
	
	wa0=(double *)malloc(sizeof(double)*BATCH);
	wb0=(double *)malloc(sizeof(double)*BATCH);
	
	cudaMalloc((void**)&wa0_cu,  sizeof(double)*BATCH);
	cudaMalloc((void**)&wb0_cu,  sizeof(double)*BATCH);
	cudaMalloc((void**)&wA_cu,  sizeof(double)*NxNyNz*BATCH);
	cudaMalloc((void**)&wB_cu,  sizeof(double)*NxNyNz*BATCH);
	cudaMalloc((void**)&phA_cu,  sizeof(double)*NxNyNz*BATCH);
	cudaMalloc((void**)&phB_cu,  sizeof(double)*NxNyNz*BATCH);
	
	qCab=(double*)malloc(sizeof(double)*BATCH);
	cudaMemcpy(wA_cu,  wA,sizeof(double)*NxNyNz*BATCH,cudaMemcpyHostToDevice);
	cudaMemcpy(wB_cu,  wB,sizeof(double)*NxNyNz*BATCH,cudaMemcpyHostToDevice);



	cudaEvent_t stop,start;
	cudaError_t error;
	float  msecTotal,msec;
	error=cudaEventCreate(&start);
	error=cudaEventCreate(&stop);
	error=cudaEventCreate(&start);
	error=cudaEventCreate(&stop);
	
	
	
	
	do{
		iter=iter+1;
		for(i=0;i<BATCH;i++){
			wa0[i]=0.0;
			wb0[i]=0.0;
			inCompMax[i]=10.0;
			freeDiff[i]=10.0;
			freeEnergy[i]=0.0;
		}
		
		cudaMemcpy(wa0_cu,  wa0,sizeof(double)*BATCH,cudaMemcpyHostToDevice);
		cudaMemcpy(wb0_cu,  wb0,sizeof(double)*BATCH,cudaMemcpyHostToDevice);
		wa_average<<<BATCH,BLOCK_SIZE>>>(wa0_cu,wA_cu);
		wa_average<<<BATCH,BLOCK_SIZE>>>(wb0_cu,wB_cu);
		//cudaMemcpy(&wa0,  wa0_cu,sizeof(double),cudaMemcpyDeviceToHost);
		//cudaMemcpy(&wb0,  wb0_cu,sizeof(double),cudaMemcpyDeviceToHost);
		
		//printf("%g %g\n",wa0,wb0);
		wa_translate<<<BATCH,BLOCK_SIZE>>>(wA_cu,wa0_cu);
		wa_translate<<<BATCH,BLOCK_SIZE>>>(wB_cu,wb0_cu);

		
		
		error=cudaEventRecord(start,NULL);
		
		getConc(phA_cu,phB_cu,wA_cu,wB_cu,qCab);//---------------------------- core function------------------------------

		

		
		error=cudaDeviceSynchronize();
		if(error!=cudaSuccess) printf("something wrong!before %d\n",iter);
		error=cudaEventRecord(stop,NULL);	
		cudaEventSynchronize(stop);	
			
		error=cudaEventElapsedTime(&msec,start,stop);
		printf("time=%g\n",msec);	

		error=cudaDeviceSynchronize();
		if(error!=cudaSuccess) printf("something wrong!before i\n",iter);
		
		phi_w<<<BATCH,BLOCK_SIZE>>>(wA_cu,wB_cu,phA_cu,phB_cu, hAB);
		
			
			
			
		error=cudaDeviceSynchronize();
		
		if(error!=cudaSuccess) printf("something wrong!before i\n",iter);
		if(iter%ArrangeStep==0){
			
			cudaMemcpy(wA,  wA_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			cudaMemcpy(wB,  wB_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(phA,  phA_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(phB,  phB_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			if(error!=cudaSuccess) printf("something wrong!before \n");
			for(i=0;i<BATCH;i++){	
				freeW[i]=0.0;
				freeAB[i]=0.0;
				freeS[i]=0.0;
				freeWsurf[i]=0.0;
				inCompMax[i]=0.0;
			}
			
			for(i=0;i<BATCH;i++){
				for(ijk=0;ijk<NxNyNz;ijk++){
					psum[i]=1-phA[ijk+i*NxNyNz]-phB[ijk+i*NxNyNz];
					fpsum[i]=fabs(psum[i]);
					if(fpsum[i]>inCompMax[i]) inCompMax[i]=fpsum[i];
					freeAB[i]=freeAB[i]+hAB*phA[ijk+i*NxNyNz]*phB[ijk+i*NxNyNz];
					freeW[i]=freeW[i]-(wA[ijk+i*NxNyNz]*phA[ijk+i*NxNyNz]+wB[ijk+i*NxNyNz]*phB[ijk+i*NxNyNz]);
				
				}
				freeAB[i]/=NxNyNz;
				//printf("freeW=%0.10f\n",freeW);
				freeW[i]/=NxNyNz;
				freeWsurf[i]/=NxNyNz;
				
				freeS[i]=-log(qCab[i]);
				//printf("%d %.10f %.10f %.10f %.10f\n",i,qCab[0],qCab[1],freeS[i],-log(qCab[1]));
				freeOld[i]=freeEnergy[i];
				freeEnergy[i]=freeAB[i]+freeW[i]+freeS[i];
				printf(" %5d : %.8e, %.8e, %.8e,%.8e, %.8e\n", iter, freeEnergy[i],freeAB[i],freeW[i], freeS[i],inCompMax[i]);
			}
			write_ph(phA,phB,wA,wB);	
			
		}
			cudaMemcpy(wA,  wA_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			cudaMemcpy(wB,  wB_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(phA,  phA_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(phB,  phB_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			if(error!=cudaSuccess) printf("something wrong!before \n");
				
		error=cudaDeviceSynchronize();
		if(error!=cudaSuccess) printf("something wrong!before i\n",iter);
	}while(iter<maxIter);
		
	write_ph(phA,phB,wA,wB);
	cudaFree(wa0_cu);
	cudaFree(wb0_cu);
	cudaFree(phA_cu);
	cudaFree(phB_cu);
	cudaFree(wA_cu);
	cudaFree(wB_cu);

	free(wa0);
	free(wb0);
return 0;
}

double freeE_constrained(double *wA,double *wB, double *phA,double *phB,double *PhiA,double lambda){
	int i,j,k,iter,maxIter;
	long ijk;
	double freeEnergy[BATCH],freeOld[BATCH],*qCab,Sm1,Sm2;
	double freeW[BATCH],freeAB[BATCH],freeS[BATCH],freeDiff[BATCH],freeWsurf[BATCH];
	double *wa0,*wb0;
	double *wa0_cu,*wb0_cu;
	double inCompMax[BATCH];
	FILE *fp;
	dim3 grid(Nx,Ny,Nz);
	double *wA_cu,*wB_cu,*phA_cu,*phB_cu,*PhA_cu;
	double psum[BATCH],fpsum[BATCH];
	maxIter=MaxIT;
	
	Sm1=0.2e-7;
	Sm2=0.1e-10;
	iter=0;	
	
	wa0=(double *)malloc(sizeof(double)*BATCH);
	wb0=(double *)malloc(sizeof(double)*BATCH);
	
	cudaMalloc((void**)&wa0_cu,  sizeof(double)*BATCH);
	cudaMalloc((void**)&wb0_cu,  sizeof(double)*BATCH);
	cudaMalloc((void**)&wA_cu,  sizeof(double)*NxNyNz*BATCH);
	cudaMalloc((void**)&wB_cu,  sizeof(double)*NxNyNz*BATCH);
	cudaMalloc((void**)&phA_cu,  sizeof(double)*NxNyNz*BATCH);
	cudaMalloc((void**)&phB_cu,  sizeof(double)*NxNyNz*BATCH);
	cudaMalloc((void**)&PhA_cu,  sizeof(double)*NxNyNz*BATCH);
	
	qCab=(double*)malloc(sizeof(double)*BATCH);
	
	cudaMemcpy(wA_cu,  wA,sizeof(double)*NxNyNz*BATCH,cudaMemcpyHostToDevice);
	cudaMemcpy(wB_cu,  wB,sizeof(double)*NxNyNz*BATCH,cudaMemcpyHostToDevice);
	cudaMemcpy(PhA_cu,  PhiA,sizeof(double)*NxNyNz*BATCH,cudaMemcpyHostToDevice);


	cudaEvent_t stop,start;
	cudaError_t error;
	float  msecTotal,msec;
	error=cudaEventCreate(&start);
	error=cudaEventCreate(&stop);
	error=cudaEventCreate(&start);
	error=cudaEventCreate(&stop);
		
	
	
	
	do{
		iter=iter+1;
		for(i=0;i<BATCH;i++){
			wa0[i]=0.0;
			wb0[i]=0.0;
			inCompMax[i]=10.0;
			freeDiff[i]=10.0;
			freeEnergy[i]=0.0;
		}
		
		cudaMemcpy(wa0_cu,  wa0,sizeof(double)*BATCH,cudaMemcpyHostToDevice);
		cudaMemcpy(wb0_cu,  wb0,sizeof(double)*BATCH,cudaMemcpyHostToDevice);
		wa_average<<<BATCH,BLOCK_SIZE>>>(wa0_cu,wA_cu);
		wa_average<<<BATCH,BLOCK_SIZE>>>(wb0_cu,wB_cu);
		//cudaMemcpy(&wa0,  wa0_cu,sizeof(double),cudaMemcpyDeviceToHost);
		//cudaMemcpy(&wb0,  wb0_cu,sizeof(double),cudaMemcpyDeviceToHost);
		
		//printf("%g %g\n",wa0,wb0);
		wa_translate<<<BATCH,BLOCK_SIZE>>>(wA_cu,wa0_cu);
		wa_translate<<<BATCH,BLOCK_SIZE>>>(wB_cu,wb0_cu);
		
		
		
		
		error=cudaEventRecord(start,NULL);
		
		getConc(phA_cu,phB_cu,wA_cu,wB_cu,qCab);//---------------------------- core function------------------------------

		
		
		error=cudaDeviceSynchronize();
		if(error!=cudaSuccess) printf("something wrong!before %d\n",iter);
		error=cudaEventRecord(stop,NULL);	
			cudaEventSynchronize(stop);	
			
			error=cudaEventElapsedTime(&msec,start,stop);
			
			//printf("time=%0.10f\n",msec);
			
		//printf("%d %.10f\n",iter,qCab[0]);
		
		
		error=cudaDeviceSynchronize();
		if(error!=cudaSuccess) printf("something wrong!before i\n",iter);
		
		phi_w_constrained<<<BATCH,BLOCK_SIZE>>>(wA_cu,wB_cu,phA_cu,phB_cu,PhA_cu, hAB,lambda);
		
			
			
			
		error=cudaDeviceSynchronize();
		
		if(error!=cudaSuccess) printf("something wrong!before i\n",iter);
		if(iter%ArrangeStep==0){
			
			cudaMemcpy(wA,  wA_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			cudaMemcpy(wB,  wB_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(phA,  phA_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(phB,  phB_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			if(error!=cudaSuccess) printf("something wrong!before \n");
			for(i=0;i<BATCH;i++){	
				freeW[i]=0.0;
				freeAB[i]=0.0;
				freeS[i]=0.0;
				freeWsurf[i]=0.0;
				inCompMax[i]=0.0;
			}
			
			for(i=0;i<BATCH;i++){
				for(ijk=0;ijk<NxNyNz;ijk++){
					psum[i]=1-phA[ijk+i*NxNyNz]-phB[ijk+i*NxNyNz];
					fpsum[i]=fabs(psum[i]);
					if(fpsum[i]>inCompMax[i]) inCompMax[i]=fpsum[i];
					freeAB[i]=freeAB[i]+hAB*phA[ijk+i*NxNyNz]*phB[ijk+i*NxNyNz];
					freeW[i]=freeW[i]-(wA[ijk+i*NxNyNz]*phA[ijk+i*NxNyNz]+wB[ijk+i*NxNyNz]*phB[ijk+i*NxNyNz]);
				
				}
				freeAB[i]/=NxNyNz;
				//printf("freeW=%0.10f\n",freeW);
				freeW[i]/=NxNyNz;
				freeWsurf[i]/=NxNyNz;
				
				freeS[i]=-log(qCab[i]);
				//printf("%d %.10f %.10f %.10f %.10f\n",i,qCab[0],qCab[1],freeS[i],-log(qCab[1]));
				freeOld[i]=freeEnergy[i];
				freeEnergy[i]=freeAB[i]+freeW[i]+freeS[i];
				printf(" %5d : %.8e, %.8e, %.8e,%.8e, %.8e\n", iter, freeEnergy[i],freeAB[i],freeW[i], freeS[i],inCompMax[i]);
			}
			//write_ph(phA,phB,wA,wB);	
			
		}
			cudaMemcpy(wA,  wA_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			cudaMemcpy(wB,  wB_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(phA,  phA_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(phB,  phB_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			if(error!=cudaSuccess) printf("something wrong!before \n");
				
		error=cudaDeviceSynchronize();
		if(error!=cudaSuccess) printf("something wrong!before i\n",iter);
	}while(iter<maxIter);
		
	//write_ph(phA,phB,wA,wB);
	cudaFree(wa0_cu);
	cudaFree(wb0_cu);
	cudaFree(phA_cu);
	cudaFree(phB_cu);
	cudaFree(wA_cu);
	cudaFree(wB_cu);
	cudaFree(PhA_cu);
	free(wa0);
	free(wb0);
return 0;
}
double freeE_constrainedEx(double *wA,double *wB, double *phA,double *phB,double *PhiA){
	int i,j,k,iter,maxIter;
	long ijk;
	double freeEnergy[BATCH],freeOld[BATCH],*qCab,Sm1,Sm2;
	double freeW[BATCH],freeAB[BATCH],freeS[BATCH],freeDiff[BATCH],freeWsurf[BATCH];
	double *wa0,*wb0;
	double *wa0_cu,*wb0_cu;
	double inCompMax[BATCH],inCompMax1[BATCH];
	FILE *fp,*fp_out;
	dim3 grid(Nx,Ny,Nz);
	double *wA_cu,*wB_cu,*phA_cu,*phB_cu,*PhA_cu;
	double psum[BATCH],fpsum[BATCH],psum1[BATCH],fpsum1[BATCH];
	maxIter=MaxIT;
	
	Sm1=0.2e-7;
	Sm2=0.1e-10;
	iter=0;	
	
	wa0=(double *)malloc(sizeof(double)*BATCH);
	wb0=(double *)malloc(sizeof(double)*BATCH);
	
	cudaMalloc((void**)&wa0_cu,  sizeof(double)*BATCH);
	cudaMalloc((void**)&wb0_cu,  sizeof(double)*BATCH);
	cudaMalloc((void**)&wA_cu,  sizeof(double)*NxNyNz*BATCH);
	cudaMalloc((void**)&wB_cu,  sizeof(double)*NxNyNz*BATCH);
	cudaMalloc((void**)&phA_cu,  sizeof(double)*NxNyNz*BATCH);
	cudaMalloc((void**)&phB_cu,  sizeof(double)*NxNyNz*BATCH);
	cudaMalloc((void**)&PhA_cu,  sizeof(double)*NxNyNz*BATCH);
	
	qCab=(double*)malloc(sizeof(double)*BATCH);
	cudaMemcpy(wA_cu,  wA,sizeof(double)*NxNyNz*BATCH,cudaMemcpyHostToDevice);
	cudaMemcpy(wB_cu,  wB,sizeof(double)*NxNyNz*BATCH,cudaMemcpyHostToDevice);
	cudaMemcpy(PhA_cu,  PhiA,sizeof(double)*NxNyNz*BATCH,cudaMemcpyHostToDevice);


	cudaEvent_t stop,start;
	cudaError_t error;
	float  msecTotal,msec;
	error=cudaEventCreate(&start);
	error=cudaEventCreate(&stop);
	error=cudaEventCreate(&start);
	error=cudaEventCreate(&stop);
	
	
	
	
	do{
		iter=iter+1;
		for(i=0;i<BATCH;i++){
			wa0[i]=0.0;
			wb0[i]=0.0;
			inCompMax[i]=10.0;
			freeDiff[i]=10.0;
			freeEnergy[i]=0.0;
		}
		
		cudaMemcpy(wa0_cu,  wa0,sizeof(double)*BATCH,cudaMemcpyHostToDevice);
		cudaMemcpy(wb0_cu,  wb0,sizeof(double)*BATCH,cudaMemcpyHostToDevice);
		wa_average<<<BATCH,BLOCK_SIZE>>>(wa0_cu,wA_cu);
		wa_average<<<BATCH,BLOCK_SIZE>>>(wb0_cu,wB_cu);
		//cudaMemcpy(&wa0,  wa0_cu,sizeof(double),cudaMemcpyDeviceToHost);
		//cudaMemcpy(&wb0,  wb0_cu,sizeof(double),cudaMemcpyDeviceToHost);
		
		//printf("%g %g\n",wa0,wb0);
		wa_translate<<<BATCH,BLOCK_SIZE>>>(wA_cu,wa0_cu);
		wa_translate<<<BATCH,BLOCK_SIZE>>>(wB_cu,wb0_cu);

		
		
		error=cudaEventRecord(start,NULL);
		
		getConc(phA_cu,phB_cu,wA_cu,wB_cu,qCab);//---------------------------- core function------------------------------

		

		
		error=cudaDeviceSynchronize();
		if(error!=cudaSuccess) printf("something wrong!before %d\n",iter);
		error=cudaEventRecord(stop,NULL);	
			cudaEventSynchronize(stop);	
			
			error=cudaEventElapsedTime(&msec,start,stop);
			
			//printf("time=%0.10f\n",msec);
			
		//printf("%d %.10f %.10f\n",iter,qCab[0],qCab[1]);
		
		
		error=cudaDeviceSynchronize();
		if(error!=cudaSuccess) printf("something wrong!before i\n",iter);
		
		phi_w_constrainedEx<<<BATCH,BLOCK_SIZE>>>(wA_cu,wB_cu,phA_cu,phB_cu,PhA_cu, hAB);
		
			
			
			
		error=cudaDeviceSynchronize();
		
		if(error!=cudaSuccess) printf("something wrong!before i\n",iter);
		if(iter%ArrangeStep==0){
			
			cudaMemcpy(wA,  wA_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);

			cudaMemcpy(wB,  wB_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(phA,  phA_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(phB,  phB_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			if(error!=cudaSuccess) printf("something wrong!before \n");
			for(i=0;i<BATCH;i++){	
				freeW[i]=0.0;
				freeAB[i]=0.0;
				freeS[i]=0.0;
				freeWsurf[i]=0.0;
				inCompMax[i]=0.0;
				inCompMax1[i]=0.0;
			}
			
			for(i=0;i<BATCH;i++){
				for(ijk=0;ijk<NxNyNz;ijk++){
					psum[i]=1-phA[ijk+i*NxNyNz]-phB[ijk+i*NxNyNz];
					psum1[i]=phA[ijk+i*NxNyNz]-PhiA[ijk+i*NxNyNz];
					fpsum[i]=fabs(psum[i]);
					fpsum1[i]=fabs(psum1[i]);
					if(fpsum[i]>inCompMax[i]) inCompMax[i]=fpsum[i];
					if(fpsum1[i]>inCompMax1[i]) inCompMax1[i]=fpsum1[i];
					freeAB[i]=freeAB[i]+hAB*phA[ijk+i*NxNyNz]*phB[ijk+i*NxNyNz];
					freeW[i]=freeW[i]-(wA[ijk+i*NxNyNz]*phA[ijk+i*NxNyNz]+wB[ijk+i*NxNyNz]*phB[ijk+i*NxNyNz]);
				
				}
				freeAB[i]/=NxNyNz;
				//printf("freeW=%0.10f\n",freeW);
				freeW[i]/=NxNyNz;
				freeWsurf[i]/=NxNyNz;
				
				freeS[i]=-log(qCab[i]);
				//printf("%d %.10f %.10f %.10f %.10f\n",i,qCab[0],qCab[1],freeS[i],-log(qCab[1]));
				freeOld[i]=freeEnergy[i];
				freeEnergy[i]=freeAB[i]+freeW[i]+freeS[i];
				printf(" %5d : %.8e, %.8e, %.8e,%.8e, %.8e %0.8e\n", iter, freeEnergy[i],freeAB[i],freeW[i], freeS[i],inCompMax[i],inCompMax1[i]);
				
			}
			
			write_ph(phA,phB,wA,wB);	
			
		}
			cudaMemcpy(wA,  wA_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			cudaMemcpy(wB,  wB_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(phA,  phA_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			error=cudaMemcpy(phB,  phB_cu,sizeof(double)*NxNyNz*BATCH,cudaMemcpyDeviceToHost);
			if(error!=cudaSuccess) printf("something wrong!before \n");
				
		error=cudaDeviceSynchronize();
		if(error!=cudaSuccess) printf("something wrong!before i\n",iter);
	}while(iter<maxIter);
	fp_out=fopen("free_energy.dat","w");
	for(i=0;i<BATCH;i++){
		fprintf(fp_out,"%d %.10f\n",i,freeEnergy[i]);
	}
	
	write_ph(phA,phB,wA,wB);
	cudaFree(wa0_cu);
	cudaFree(wb0_cu);
	cudaFree(phA_cu);
	cudaFree(phB_cu);
	cudaFree(wA_cu);
	cudaFree(wB_cu);
	cudaFree(PhA_cu);
	free(qCab);
	free(wa0);
	free(wb0);
	fclose(fp_out);
return 0;
}
//--------------------Swap ith field with jth field---------------------------i,j>=1--------------------------------------
void swap(double *PhiA,int i,int j){
double *p;
p=(double *)malloc(sizeof(double)*NxNyNz);
int ijk;
	for(ijk=0;ijk<NxNyNz;ijk++){
		p[ijk]=PhiA[ijk+(i-1)*NxNyNz];
		PhiA[ijk+(i-1)*NxNyNz]=PhiA[ijk+(j-1)*NxNyNz];
		PhiA[ijk+(j-1)*NxNyNz]=p[ijk];
	}
free(p);

}
//-----------------distance field--------------------------------------------------------
double distance_field(double *PhiA,int i,int j){
	int ijk;
	double distance=0;
	for(ijk=0;ijk<NxNyNz;ijk++){
		distance+=(PhiA[ijk+(i-1)*NxNyNz]-PhiA[ijk+(j-1)*NxNyNz])*(PhiA[ijk+(i-1)*NxNyNz]-PhiA[ijk+(j-1)*NxNyNz])/NxNyNz;

	}
	distance=sqrt(distance);
return distance;

}
//-----------------distance field--------------------------------------------------------
double distance_field_afterum(double *PhiA,double *phA,int i){
	int ijk;
	double distance=0;
	for(ijk=0;ijk<NxNyNz;ijk++){
		distance+=(PhiA[ijk+(i-1)*NxNyNz]-phA[ijk+(i-1)*NxNyNz])*(PhiA[ijk+(i-1)*NxNyNz]-phA[ijk+(i-1)*NxNyNz])/NxNyNz;

	}
	distance=sqrt(distance);
return distance;

}
//------------- soft field------------------------------------------------------------
void soft(double *PhiA){

	int i,j;
	long ijk;
	double dis_i[BATCH],temp1,temp2;
	
	for(i=1;i<BATCH;i++){
		for(j=i+2;j<BATCH;j++){
			temp1=distance_field(PhiA,i,j);
			temp2=distance_field(PhiA,i,i+1);
			if(temp1<temp2){
				swap(PhiA,i+1,j);
			}
		}

	}
	for(ijk=0;ijk<NxNyNz;ijk++){
		PhiA[ijk+NxNyNz*(BATCH-2)]=(PhiA[ijk+NxNyNz*(BATCH-1)]+PhiA[ijk+NxNyNz*(BATCH-3)])/2;
	}
}
//-----------------------translation-------------------------------------------------------------------------------
//direction=1 y direction move forward
//direction=2 y direction move backward
//direction=3 z direction move forward
//direction=4 z direction move backward
int translation(double *PhiA,int ith,int direction,int dis){
	int i,j,k;
	long ijk;
	double *temp;
	temp=(double *)malloc(sizeof(double)*NxNyNz);
	if(direction==1){

		for(k=0;k<Nz;k++)
		for(j=0;j<Ny;j++)
		for(i=0;i<Nx;i++){
			temp[i+Nx*j+k*Nx*Ny]=PhiA[i+Nx*((j+dis+Ny)%Ny)+k*Nx*Ny+(ith-1)*NxNyNz];
		}

	}
	else if(direction==2){

		for(k=0;k<Nz;k++)
		for(j=0;j<Ny;j++)
		for(i=0;i<Nx;i++){
			temp[i+Nx*j+k*Nx*Ny]=PhiA[i+Nx*((j-dis+Ny)%Ny)+k*Nx*Ny+(ith-1)*NxNyNz];
		}

	}
	else if(direction==3){

		for(k=0;k<Nz;k++)
		for(j=0;j<Ny;j++)
		for(i=0;i<Nx;i++){
			temp[i+Nx*j+k*Nx*Ny]=PhiA[i+Nx*j+((k+dis+Nz)%Nz)*Nx*Ny+(ith-1)*NxNyNz];
		}

	}
	else if(direction==4){

		for(k=0;k<Nz;k++)
		for(j=0;j<Ny;j++)
		for(i=0;i<Nx;i++){
			temp[i+Nx*j+k*Nx*Ny]=PhiA[i+Nx*j+((k-dis+Nz)%Nz)*Nx*Ny+(ith-1)*NxNyNz];
		}

	}
	for(ijk=0;ijk<NxNyNz;ijk++){
		PhiA[ijk+(ith-1)*NxNyNz]=temp[ijk];

	}
	free(temp);
	return 1;
}
void translation_good(double *PhiA){
	int i,j,k;
	int ith;
	int tag,direction;
	double para_temp,para;

	for(ith=2;ith<=BATCH;ith++){
		tag=0;
		para=distance_field(PhiA,1,ith);
		while(tag==0){
			direction=1;
			translation(PhiA,ith,direction,1);
			para_temp=distance_field(PhiA,1,ith);
			
			if(para_temp>=para){
				direction=2;
				translation(PhiA,ith,direction,1);
				
				tag=1;
			}
			else{
				para=para_temp;

			}
		}//end direction 1
		tag=0;
		
		while(tag==0){
			direction=2;
			translation(PhiA,ith,direction,1);
			para_temp=distance_field(PhiA,1,ith);
			
			if(para_temp>=para){
				direction=1;
				translation(PhiA,ith,direction,1);
				
				tag=1;
			}
			else{
				para=para_temp;

			}
		}//end direction 2
		while(tag==0){
			direction=3;
			translation(PhiA,ith,direction,1);
			para_temp=distance_field(PhiA,1,ith);
			
			if(para_temp>=para){
				direction=4;
				translation(PhiA,ith,direction,1);
				
				tag=1;
			}
			else{
				para=para_temp;

			}
		}//end direction 3
		while(tag==0){
			direction=4;
			translation(PhiA,ith,direction,1);
			para_temp=distance_field(PhiA,1,ith);
			
			if(para_temp>=para){
				direction=3;
				translation(PhiA,ith,direction,1);
				
				tag=1;
			}
			else{
				para=para_temp;

			}
		}//end direction 4
	}
	

}
/*----------------------------------------------------------------------------------------------*/
/* spline.c
   Cubic interpolating spline. 
   from http://www.mech.uq.edu.au/staff/jacobs/nm_lib/cmathsrc
*/

/************************************************/
/*                                              */
/*  CMATH.  Copyright (c) 1989 Design Software  */
/*                                              */
/************************************************/

int spline (int n, int end1, int end2,
            double slope1, double slope2,
            double x[], double y[],
            double b[], double c[], double d[],
            int *iflag)

/* Purpose ...
   -------
   Evaluate the coefficients b[i], c[i], d[i], i = 0, 1, .. n-1 for
   a cubic interpolating spline

   S(xx) = Y[i] + b[i] * w + c[i] * w**2 + d[i] * w**3
   where w = xx - x[i]
   and   x[i] <= xx <= x[i+1]

   The n supplied data points are x[i], y[i], i = 0 ... n-1.

   Input :
   -------
   n       : The number of data points or knots (n >= 2)
   end1,
   end2    : = 1 to specify the slopes at the end points
             = 0 to obtain the default conditions
   slope1,
   slope2  : the slopes at the end points x[0] and x[n-1]
             respectively
   x[]     : the abscissas of the knots in strictly
             increasing order
   y[]     : the ordinates of the knots

   Output :
   --------
   b, c, d : arrays of spline coefficients as defined above
             (See note 2 for a definition.)
   iflag   : status flag
            = 0 normal return
            = 1 less than two data points; cannot interpolate
            = 2 x[] are not in ascending order

   This C code written by ...  Peter & Nigel,
   ----------------------      Design Software,
                               42 Gubberley St,
                               Kenmore, 4069,
                               Australia.

   Version ... 1.1, 30 September 1987
   -------     2.0, 6 April 1989    (start with zero subscript)
                                     remove ndim from parameter list
               2.1, 28 April 1989   (check on x[])
               2.2, 10 Oct   1989   change number order of matrix

   Notes ...
   -----
   (1) The accompanying function seval() may be used to evaluate the
       spline while deriv will provide the first derivative.
   (2) Using p to denote differentiation
       y[i] = S(X[i])
       b[i] = Sp(X[i])
       c[i] = Spp(X[i])/2
       d[i] = Sppp(X[i])/6  ( Derivative from the right )
   (3) Since the zero elements of the arrays ARE NOW used here,
       all arrays to be passed from the main program should be
       dimensioned at least [n].  These routines will use elements
       [0 .. n-1].
   (4) Adapted from the text
       Forsythe, G.E., Malcolm, M.A. and Moler, C.B. (1977)
       "Computer Methods for Mathematical Computations"
       Prentice Hall
   (5) Note that although there are only n-1 polynomial segments,
       n elements are requird in b, c, d.  The elements b[n-1],
       c[n-1] and d[n-1] are set to continue the last segment
       past x[n-1].
*/

/*----------------------------------------------------------------*/

{  /* begin procedure spline() */

int    nm1, ib, i;
double t;
int    ascend;

nm1    = n - 1;
*iflag = 0;

if (n < 2)
  {  /* no possible interpolation */
  *iflag = 1;
  goto LeaveSpline;
  }

ascend = 1;
for (i = 1; i < n; ++i) if (x[i] <= x[i-1]) ascend = 0;
if (!ascend)
   {
   *iflag = 2;
   goto LeaveSpline;
   }

if (n >= 3)
   {    /* ---- At least quadratic ---- */

   /* ---- Set up the symmetric tri-diagonal system
           b = diagonal
           d = offdiagonal
           c = right-hand-side  */
   d[0] = x[1] - x[0];
   c[1] = (y[1] - y[0]) / d[0];
   for (i = 1; i < nm1; ++i)
      {
      d[i]   = x[i+1] - x[i];
      b[i]   = 2.0 * (d[i-1] + d[i]);
      c[i+1] = (y[i+1] - y[i]) / d[i];
      c[i]   = c[i+1] - c[i];
      }

   /* ---- Default End conditions
           Third derivatives at x[0] and x[n-1] obtained
           from divided differences  */
   b[0]   = -d[0];
   b[nm1] = -d[n-2];
   c[0]   = 0.0;
   c[nm1] = 0.0;
   if (n != 3)
      {
      c[0]   = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0]);
      c[nm1] = c[n-2] / (x[nm1] - x[n-3]) - c[n-3] / (x[n-2] - x[n-4]);
      c[0]   = c[0] * d[0] * d[0] / (x[3] - x[0]);
      c[nm1] = -c[nm1] * d[n-2] * d[n-2] / (x[nm1] - x[n-4]);
      }

   /* Alternative end conditions -- known slopes */
   if (end1 == 1)
      {
      b[0] = 2.0 * (x[1] - x[0]);
      c[0] = (y[1] - y[0]) / (x[1] - x[0]) - slope1;
      }
   if (end2 == 1)
      {
      b[nm1] = 2.0 * (x[nm1] - x[n-2]);
      c[nm1] = slope2 - (y[nm1] - y[n-2]) / (x[nm1] - x[n-2]);
      }

   /* Forward elimination */
   for (i = 1; i < n; ++i)
     {
     t    = d[i-1] / b[i-1];
     b[i] = b[i] - t * d[i-1];
     c[i] = c[i] - t * c[i-1];
     }

   /* Back substitution */
   c[nm1] = c[nm1] / b[nm1];
   for (ib = 0; ib < nm1; ++ib)
      {
      i    = n - ib - 2;
      c[i] = (c[i] - d[i] * c[i+1]) / b[i];
      }

   /* c[i] is now the sigma[i] of the text */

   /* Compute the polynomial coefficients */
   b[nm1] = (y[nm1] - y[n-2]) / d[n-2] + d[n-2] * (c[n-2] + 2.0 * c[nm1]);
   for (i = 0; i < nm1; ++i)
      {
      b[i] = (y[i+1] - y[i]) / d[i] - d[i] * (c[i+1] + 2.0 * c[i]);
      d[i] = (c[i+1] - c[i]) / d[i];
      c[i] = 3.0 * c[i];
      }
   c[nm1] = 3.0 * c[nm1];
   d[nm1] = d[n-2];

   }  /* at least quadratic */

else  /* if n >= 3 */
   {  /* linear segment only  */
   b[0] = (y[1] - y[0]) / (x[1] - x[0]);
   c[0] = 0.0;
   d[0] = 0.0;
   b[1] = b[0];
   c[1] = 0.0;
   d[1] = 0.0;
   }

LeaveSpline:
return 0;
}  /* end of spline() */
/*-------------------------------------------------------------------*/

double seval (int n, double u,
              double x[], double y[],
              double b[], double c[], double d[],
              int *last)

/*Purpose ...
  -------
  Evaluate the cubic spline function

  S(xx) = y[i] + b[i] * w + c[i] * w**2 + d[i] * w**3
  where w = u - x[i]
  and   x[i] <= u <= x[i+1]
  Note that Horner's rule is used.
  If u < x[0]   then i = 0 is used.
  If u > x[n-1] then i = n-1 is used.

  Input :
  -------
  n       : The number of data points or knots (n >= 2)
  u       : the abscissa at which the spline is to be evaluated
  Last    : the segment that was last used to evaluate U
  x[]     : the abscissas of the knots in strictly increasing order
  y[]     : the ordinates of the knots
  b, c, d : arrays of spline coefficients computed by spline().

  Output :
  --------
  seval   : the value of the spline function at u
  Last    : the segment in which u lies

  Notes ...
  -----
  (1) If u is not in the same interval as the previous call then a
      binary search is performed to determine the proper interval.

*/
/*-------------------------------------------------------------------*/

{  /* begin function seval() */

int    i, j, k;
double w;

i = *last;
if (i >= n-1) i = 0;
if (i < 0)  i = 0;

if ((x[i] > u) || (x[i+1] < u))
  {  /* ---- perform a binary search ---- */
  i = 0;
  j = n;
  do
    {
    k = (i + j) / 2;         /* split the domain to search */
    if (u < x[k])  j = k;    /* move the upper bound */
    if (u >= x[k]) i = k;    /* move the lower bound */
    }                        /* there are no more segments to search */
  while (j > i+1);
  }
*last = i;

/* ---- Evaluate the spline ---- */
w = u - x[i];
w = y[i] + w * (b[i] + w * (c[i] + w * d[i]));
return (w);
}
/*-------------------------------------------------------------------*/

double deriv (int n, double u,
              double x[],
              double b[], double c[], double d[],
              int *last)

/* Purpose ...
   -------
   Evaluate the derivative of the cubic spline function

   S(x) = B[i] + 2.0 * C[i] * w + 3.0 * D[i] * w**2
   where w = u - X[i]
   and   X[i] <= u <= X[i+1]
   Note that Horner's rule is used.
   If U < X[0] then i = 0 is used.
   If U > X[n-1] then i = n-1 is used.

   Input :
   -------
   n       : The number of data points or knots (n >= 2)
   u       : the abscissa at which the derivative is to be evaluated
   last    : the segment that was last used
   x       : the abscissas of the knots in strictly increasing order
   b, c, d : arrays of spline coefficients computed by spline()

   Output :
   --------
   deriv : the value of the derivative of the spline
           function at u
   last  : the segment in which u lies

   Notes ...
   -----
   (1) If u is not in the same interval as the previous call then a
       binary search is performed to determine the proper interval.

*/
/*-------------------------------------------------------------------*/

{  /* begin function deriv() */

int    i, j, k;
double w;

i = *last;
if (i >= n-1) i = 0;
if (i < 0) i = 0;

if ((x[i] > u) || (x[i+1] < u))
  {  /* ---- perform a binary search ---- */
  i = 0;
  j = n;
  do
    {
    k = (i + j) / 2;          /* split the domain to search */
    if (u < x[k])  j = k;     /* move the upper bound */
    if (u >= x[k]) i = k;     /* move the lower bound */
    }                         /* there are no more segments to search */
  while (j > i+1);
  }
*last = i;

/* ---- Evaluate the derivative ---- */
w = u - x[i];
w = b[i] + w * (2.0 * c[i] + w * 3.0 * d[i]);
return (w);

} /* end of deriv() */

/*-------------------------------------------------------------------*/

double sinteg (int n, double u,
              double x[], double y[],
              double b[], double c[], double d[],
              int *last)

/*Purpose ...
  -------
  Integrate the cubic spline function

  S(xx) = y[i] + b[i] * w + c[i] * w**2 + d[i] * w**3
  where w = u - x[i]
  and   x[i] <= u <= x[i+1]

  The integral is zero at u = x[0].

  If u < x[0]   then i = 0 segment is extrapolated.
  If u > x[n-1] then i = n-1 segment is extrapolated.

  Input :
  -------
  n       : The number of data points or knots (n >= 2)
  u       : the abscissa at which the spline is to be evaluated
  Last    : the segment that was last used to evaluate U
  x[]     : the abscissas of the knots in strictly increasing order
  y[]     : the ordinates of the knots
  b, c, d : arrays of spline coefficients computed by spline().

  Output :
  --------
  sinteg  : the value of the spline function at u
  Last    : the segment in which u lies

  Notes ...
  -----
  (1) If u is not in the same interval as the previous call then a
      binary search is performed to determine the proper interval.

*/
/*-------------------------------------------------------------------*/

{  /* begin function sinteg() */

int    i, j, k;
double sum, dx;

i = *last;
if (i >= n-1) i = 0;
if (i < 0)  i = 0;

if ((x[i] > u) || (x[i+1] < u))
  {  /* ---- perform a binary search ---- */
  i = 0;
  j = n;
  do
    {
    k = (i + j) / 2;         /* split the domain to search */
    if (u < x[k])  j = k;    /* move the upper bound */
    if (u >= x[k]) i = k;    /* move the lower bound */
    }                        /* there are no more segments to search */
  while (j > i+1);
  }
*last = i;

sum = 0.0;
/* ---- Evaluate the integral for segments x < u ---- */
for (j = 0; j < i; ++j)
   {
   dx = x[j+1] - x[j];
   sum += dx *
          (y[j] + dx *
          (0.5 * b[j] + dx *
          (c[j] / 3.0 + dx * 0.25 * d[j])));
   }

/* ---- Evaluate the integral fot this segment ---- */
dx = u - x[i];
sum += dx *
       (y[i] + dx *
       (0.5 * b[i] + dx *
       (c[i] / 3.0 + dx * 0.25 * d[i])));

return (sum);
}
/*-------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------*/
double splineValue (int n,double x0,
            double x[],double y[],
            double b[], double c[], double d[],double *y0
            )
{
double re,w;
int i,j,k;
int num;
num=0;
for(i=1;i<n;i++){
	if((x0>=x[i-1])&&(x0<x[i])){
		num=i-1;
		break; 
	}
	else if((x0=x[i-1])&&(x0==x[i])){
		
	}
}

w=x0-x[num];

re=y[num] + b[num]*w+d[num] * w*w*w+ c[num]*w*w;// 

*y0=re;
return re;

}
int spline_straightline (int n, 
            double x[], double y[],
            double b[], double c[], double d[],
            int *iflag){
int i;
for(i=0;i<n-1;i++){
	b[i]=(y[i+1]-y[i])/(x[i+1]-x[i]);
	c[i]=0.0;
	d[i]=0.0;

}
return 0;

}
double splineValue_straightline (int n,double x0,
            double x[],double y[],
            double b[], double c[], double d[],double *y0
            )
{
double re,w;
	int i,j,k;
	int num;
	double dis;
	double Inv[BATCH-1];
	double L,l;
	num=0;

	dis=0.0;
	for(i=0;i<BATCH-1;i++){
		dis+=sqrt((y[i+1]-y[i])*(y[i+1]-y[i])+(x[i+1]-x[i])*(x[i+1]-x[i]));
		Inv[i]=dis;

	}
	L=x0*dis;
	for(i=1;i<n;i++){
		if((L>=Inv[i-1])&&(L<Inv[i])){
			num=i-1;
			break; 
		}

	}
	
	l=L-Inv[num];

	
	if(b[num]>0) re=y[num]+l/(1+(1/b[num])*(1/b[num]));
	else if(b[num]<0) re=y[num]+l/(1+(1/b[num])*(1/b[num]));
	else if(b[num]==0) re=y[num];
	
*y0=re;
return re;

}



//-----------------distance field--------------------------------------------------------

//					fprintf(stderr, "CUFFT error: Plan Exec forward failed\n");
//					if(state==CUFFT_INVALID_PLAN) printf("CUFFT_INVALID_PLAN\n");
//					else if(state==CUFFT_INVALID_VALUE)printf("CUFFT_INVALID_VALUE\n");
//					else if(state==CUFFT_INTERNAL_ERROR)printf("CUFFT_INTERNAL_ERROR\n");
//					else if(state==CUFFT_EXEC_FAILED)printf("CUFFT_EXEC_FAILED\n");

//					else if(state==CUFFT_SETUP_FAILED)printf("CUFFT_SETUP_FAILED\n");
/*
cufftDoubleReal *AA;
			AA=(cufftDoubleReal*)malloc(sizeof(cufftDoubleReal)*16);
			if (cudaMemcpy(AA,  in,sizeof(cufftDoubleReal)*16,cudaMemcpyDeviceToHost)!= cudaSuccess){
				fprintf(stderr, "CUFFT error: coppy failed here..!!!!!\n");
			}
			for(int i=0;i<16;i++){
				if(iz==1)
				printf("%g\n",AA[i]);
			}
			*/
/* record time
	cudaEvent_t stop;
	cudaEvent_t start;
	cudaError_t error;
	error=cudaEventCreate(&start);
	error=cudaEventCreate(&stop);
	error=cudaEventRecord(start,NULL);
	printf("start\n");
printf("finish\n");
	error=cudaEventRecord(stop,NULL);
	
	float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);
	printf("%.3f\n",msecTotal);


cufftDoubleReal *AA;
			cufftDoubleComplex *BB;
			AA=(cufftDoubleReal*)malloc(sizeof(cufftDoubleReal)*NxNyNz);
			BB=(cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*NxNyNz);
			
			if (cudaMemcpy(BB,  out,sizeof(cufftDoubleComplex)*NxNyNz,cudaMemcpyDeviceToHost)!= cudaSuccess){
				fprintf(stderr, "CUFFT error: coppy failed here..!!!!!\n");
			}
			for(int i=0;i<NxNyNz/16;i++){
				if(iz==1)
				printf("%d %.10f %.10f \n",i,BB[i].x,BB[i].y);
				//printf("%d, %.10f\n",i,AA[i]);
			}
cufftDoubleReal *AA;
			
			AA=(cufftDoubleReal*)malloc(sizeof(cufftDoubleReal)*NxNyNz*(51));
			
			if (cudaMemcpy(AA,qcA_cu,sizeof(cufftDoubleReal)*NxNyNz*51,cudaMemcpyDeviceToHost)!= cudaSuccess){
				fprintf(stderr, "CUFFT error: coppy failed here..!!!!!\n");
			}
		
			for(int i=0;i<NxNyNz*51;i++){
			
				printf("%d %0.10f\n",i,AA[i]);
			}
error=cudaDeviceSynchronize();
		if(error!=cudaSuccess) printf("something wrong!before \n");	
	*/
