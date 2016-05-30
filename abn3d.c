#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
//#include "mpi.h"
#include</usr/include/fftw3.h>
//#include</home/weihua.li/lwh/scft/fftw3/include/fftw3.h>
//#include</export/home/wli/fftw3/include/fftw3.h>
//#include</export/fftw3/include/fftw3.h>
//#include</opt/sharcnet/fftw/2.1.5/pathscale/include/drfftw_mpi.h>
#define MaxIT 200000           //Maximum iteration steps
#define Nx 1 
#define Ny 240
#define Nz 240			//grid size

long NxNyNz, NxNyNz1;
int Nxh1;

/*#define NxNyNz (Nx*Ny*Nz)

#define Nxh1 (Nx/2+1)
#define NxNyNz1 (Nz*Ny*Nxh1)*/

#define Pi 3.141592653589

void initWcyl(double *wA,double *wB);
void initIWP(double *wA, double *wB, double *phA, double *phB);
void initWc4(double *wA,double *wB);
void initW_PLrev(double *wA, double *wB, double *phA, double *phB);
void initW_PL(double *wA, double *wB, double *phA, double *phB);
void initW1d(double *wA,double *wB);
void initW(double *wA,double *wB);
double freeE(double *wA,double *wB,double *phA,double *phB);
double getConc(double *phlA,double *phlB,double *wA,
		double *wB);
void sovDifFft(double *g,double *w,double *qInt,
	        double z,int ns,int sign);
void write_ph(double *phA,double *phB,double *wA,double *wB);

int NsA, dNsB, Narm;
double kx[Nx],kz[Nz],ky[Ny],*kxyzdz,dx,dy,dz,*wdz;
double lx, ly, lz;
double hAB, fA, fB, dfB, ds0, ds2;
double *in;
fftw_complex *out;
fftw_plan p_forward, p_backward;
double temp;
char FEname[50], phname[50];

int main(int argc, char **argv)
{
	double *wA,*wB,*phA,*phB,w_a,w_b;
	double e1,e2,e3,e4,ksq;
	double rjk,yj,zk,phat,phbt;
	int i,j,k,intag,iseed=-3,ntyp; //local_x_starti;
	long ijk;
	char comment[201];
	char density_name[20];
	//MPI_Status status;
	FILE *fp;
	time_t ts;
	
	Nxh1=Nx/2+1;
	NxNyNz=Nx*Ny*Nz;
	NxNyNz1=Nx*Ny*Nxh1;

	iseed=time(&ts);
	srand48(iseed);

	wA=(double *)malloc(sizeof(double)*NxNyNz);
	wB=(double *)malloc(sizeof(double)*NxNyNz);
	phA=(double *)malloc(sizeof(double)*NxNyNz);
	phB=(double *)malloc(sizeof(double)*NxNyNz);
	kxyzdz=(double *)malloc(sizeof(double)*NxNyNz);
	wdz=(double *)malloc(sizeof(double)*NxNyNz);

	in=(double *)malloc(sizeof(double)*NxNyNz); /* for fftw3 */
	out=(fftw_complex *)malloc(sizeof(fftw_complex)*NxNyNz);

        p_forward = fftw_plan_dft_r2c_3d(Nz, Ny, Nx, in, out, FFTW_ESTIMATE);
        p_backward = fftw_plan_dft_c2r_3d(Nz, Ny, Nx, out, in, FFTW_ESTIMATE);

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
	if(intag==1024){
		
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

	/***************Initialize wA, wB******************/
	if(intag==0)initW(wA,wB);
	else if(intag==220)initWcyl(wA,wB);
	else if(intag==440)initWc4(wA,wB);
	else if(intag==110)initW1d(wA,wB);
	else if(intag==1155)initW_PLrev(wA,wB,phA,phB);
        else if(intag==2255)initW_PL(wA,wB,phA,phB);
	else if(intag==8888)initIWP(wA,wB,phA,phB);
	else if(intag==2222)
	{
                fp=fopen("in-w.d","r");
		for(j=0;j<Ny;j++)for(i=0;i<Nx;i++)
                {

                        fscanf(fp,"%lf %lf %lf %lf",&e1,&e1,&e2,&e3);
			for(k=0;k<Nz;k++)
			{
				ijk=(long)((k*Ny+j)*Nx+i);

                        	wA[ijk]=e2;
                        	wB[ijk]=e3;
			}
                }
                fclose(fp);

	}
	else if(intag==3333)  /* read single diamond */
	{
                fp=fopen("in-w.d","r");
                for(ijk=0;ijk<NxNyNz;ijk++)
                {
                        fscanf(fp,"%d",&ntyp);

			phat=0.10; phbt=0.90;
			if(ntyp==3){phat=1.0; phbt=0.0;}
                        wA[ijk]=hAB*phbt;
                        wB[ijk]=hAB*phat;
                }
                fclose(fp);
	}
        else if(intag==3332) /** read from old data format **/
        {
                fp=fopen("in-w.d","r");
                for(i=0;i<Nx*2;i++)for(j=0;j<Ny*2;j++)for(k=0;k<Nz*2;k++)
                {
                        fscanf(fp,"%lf %lf %lf %lf",&e1,&e2,&e3,&e3);

                        ijk=(long)(((k/2)*Ny+(j/2))*Nx+(i/2));
                        wA[ijk]=hAB*e2;
                        wB[ijk]=hAB*e1;
                }
                fclose(fp);
        }
	else if(intag==1||intag==2||intag==3||intag==4||intag==6||intag==11||intag==22){
		fp=fopen("in-w.d","r");
                fgets(comment,200,fp);
                printf("%s\n", comment);
                fgets(comment,200,fp);
                printf("%s\n", comment);
		for(ijk=0;ijk<NxNyNz;ijk++)
		{

			fscanf(fp,"%lf %lf %lf %lf",&e1,&e1,&e2,&e3);

			wA[ijk]=e2;
			wB[ijk]=e3;

		}
		fclose(fp);
	}
	else if(intag==330)
	{
                fp=fopen("in-w.d","r");
                fgets(comment,200,fp);
                printf("%s\n", comment);
                fgets(comment,200,fp);
                printf("%s\n", comment);
                for(ijk=0;ijk<NxNyNz;ijk++)
                {

                        fscanf(fp,"%lf %lf %lf %lf",&e1,&e2,&e3,&e3);

                        wA[ijk]=hAB*e1;
                        wB[ijk]=hAB*e2;

                }
                fclose(fp);
	}
	else if(intag==1024)
	{
		fp=fopen("AverDens_8.4e+06","r");
		 fgets(comment,200,fp);       
		 fgets(comment,200,fp);
		
                for(ijk=0;ijk<NxNyNz;ijk++)
                {

                        
			fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",&temp,&temp,&temp,&e1,&e2,&temp,&temp,&temp,&temp,&temp,&temp);
                        wA[ijk]=hAB*e2;
                        wB[ijk]=hAB*e1;

                }
                fclose(fp);
		//initW(wA,wB);
	}
	else if(intag==1028)
	{
			sprintf(density_name,"pha.dat");
			fp=fopen(density_name,"r");
			fgets(comment,200,fp);       
			fgets(comment,200,fp);
			
               		 for(ijk=0;ijk<NxNyNz;ijk++){
				fscanf(fp,"%lf %lf %lf %lf\n",&temp,&temp,&wA[ijk],&wB[ijk]);
				
			}
                	fclose(fp);
	}
	else if(intag==1026)
	{
		sprintf(density_name,"phi.dat");
			fp=fopen(density_name,"r");
			fgets(comment,200,fp);       
			fgets(comment,200,fp);
			
               		 for(ijk=0;ijk<NxNyNz;ijk++){
				fscanf(fp,"%lf %lf %lf %lf\n",&temp,&temp,&wA[ijk],&wB[ijk]);
				
			}
                	fclose(fp);
	}
	else if(intag==1030)
	{
			sprintf(density_name,"phi.dat");
			fp=fopen(density_name,"r");
			fgets(comment,200,fp);       
			fgets(comment,200,fp);
			for(k=0;k<Nz/2;k++)
			for(j=0;j<Ny/2;j++)
               		for(i=0;i<Nx/2;i++){
						
						fscanf(fp,"%lf %lf %lf %lf\n",&temp,&temp,&w_a,&w_b);
						wA[2*i+2*j*Nx+2*k*Nx*Ny]=w_a;
						wA[2*i+1+2*j*Nx+2*k*Nx*Ny]=w_a;						
						wA[2*i+(2*j+1)*Nx+2*k*Nx*Ny]=w_a;
						wA[2*i+2*j*Nx+(2*k+1)*Nx*Ny]=w_a;
						wA[2*i+1+(2*j+1)*Nx+2*k*Nx*Ny]=w_a;	
						wA[2*i+1+(2*j)*Nx+(2*k+1)*Nx*Ny]=w_a;
						wA[2*i+(2*j+1)*Nx+(2*k+1)*Nx*Ny]=w_a;
						wA[2*i+1+(2*j+1)*Nx+(2*k+1)*Nx*Ny]=w_a;
						wB[2*i+2*j*Nx+2*k*Nx*Ny]=w_b;
						wB[2*i+1+2*j*Nx+2*k*Nx*Ny]=w_b;
						wB[2*i+(2*j+1)*Nx+2*k*Nx*Ny]=w_b;
						wB[2*i+2*j*Nx+(2*k+1)*Nx*Ny]=w_b;
						wB[2*i+1+(2*j+1)*Nx+2*k*Nx*Ny]=w_b;	
						wB[2*i+1+(2*j)*Nx+(2*k+1)*Nx*Ny]=w_b;
						wB[2*i+(2*j+1)*Nx+(2*k+1)*Nx*Ny]=w_b;
						wB[2*i+1+(2*j+1)*Nx+(2*k+1)*Nx*Ny]=w_b;
						//wA[i+j*Nx+k*Nx*Ny]=w_a;
						//wB[i+j*Nx+k*Nx*Ny]=w_b;
						
				
			}
			write_ph(phA,phB,wA,wB);
			
                	fclose(fp);
	}
	else if(intag==1032){
		sprintf(density_name,"phi.dat");
		fp=fopen(density_name,"r");
		fgets(comment,200,fp);       
		fgets(comment,200,fp);
			for(k=0;k<Nz/2;k++)
			for(j=0;j<Ny/2;j++)
               		for(i=0;i<12;i++){
				if(i==0){
					fscanf(fp,"%lf %lf %lf %lf\n",&temp,&temp,&w_a,&w_b);
					wA[2*i+2*j*Nx+2*k*Nx*Ny]=w_a;
					wA[2*i+(2*j+1)*Nx+2*k*Nx*Ny]=w_a;
					wA[2*i+2*j*Nx+(2*k+1)*Nx*Ny]=w_a;
					wA[2*i+(2*j+1)*Nx+(2*k+1)*Nx*Ny]=w_a;
					wB[2*i+2*j*Nx+2*k*Nx*Ny]=w_b;
					wB[2*i+(2*j+1)*Nx+2*k*Nx*Ny]=w_b;
					wB[2*i+2*j*Nx+(2*k+1)*Nx*Ny]=w_b;
					wB[2*i+(2*j+1)*Nx+(2*k+1)*Nx*Ny]=w_b;
				}
				else if(i!=0)
					fscanf(fp,"%lf %lf %lf %lf\n",&temp,&temp,&temp,&temp);		
				
			}
			write_ph(phA,phB,wA,wB);
			
                	fclose(fp);

	}
	else if(intag==1034){
		sprintf(density_name,"phi.dat");
		fp=fopen(density_name,"r");
		fgets(comment,200,fp);       
		fgets(comment,200,fp);
			for(k=0;k<Nz/2;k++)
			for(j=0;j<Ny/2;j++)
               		for(i=0;i<1;i++){
				
					fscanf(fp,"%lf %lf %lf %lf\n",&temp,&temp,&w_a,&w_b);
					wA[i+j*Nx+k*Nx*Ny]=w_a;
					wA[(j+Ny/2)*Nx+k*Nx*Ny]=w_a;
					wA[j*Nx+(k+Nz/2)*Nx*Ny]=w_a;
					wA[(j+Ny/2)*Nx+(k+Nz/2)*Nx*Ny]=w_a;
					wB[i+j*Nx+k*Nx*Ny]=w_b;
					wB[(j+Ny/2)*Nx+k*Nx*Ny]=w_b;
					wB[j*Nx+(k+Nz/2)*Nx*Ny]=w_b;
					wB[(j+Ny/2)*Nx+(k+Nz/2)*Nx*Ny]=w_b;
				
				
			}
			write_ph(phA,phB,wA,wB);
			
                	fclose(fp);
			

	}
	e1=freeE(wA,wB,phA,phB);

        fftw_destroy_plan ( p_forward );
        fftw_destroy_plan ( p_backward );

	free(wA);
	free(wB);
	
	free(phA);
	free(phB);
	
	free(kxyzdz);
	free(wdz);

	free(in);
	free(out);

	return 1;
}

//********************Output configuration******************************

void write_ph(double *phA,double *phB,double *wA,double *wB)
{
	int i,j,k;
	long ijk;
	FILE *fp=fopen(phname,"w");

        fprintf(fp, "Nx=%d, Ny=%d, Nz=%d\n", Nx, Ny, Nz);
        fprintf(fp, "dx=%lf, dy=%lf, dz=%lf\n", dx, dy, dz);

	for(ijk=0;ijk<NxNyNz;ijk++)
	{
		fprintf(fp,"%lf %lf %lf %lf\n",phA[ijk],phB[ijk],wA[ijk],wB[ijk]);
	}

	fclose(fp);
}

void initW(double *wA,double *wB)
{
	int i,j,k;
	long ijk;
	
	for(i=0;i<Nx;i++)for(j=0;j<Ny;j++)
	{
		for(k=0;k<Nz;k++)
		{
			ijk=(long)((k*Ny+j)*Nx+i);
			wA[ijk]=hAB*fB+0.040*(drand48()-0.5);
			wB[ijk]=hAB*fA+0.040*(drand48()-0.5);
		}		

	}
}

void initWc4(double *wA,double *wB)
{
        int i,j,k,n,tag;
        long ijk;
        double xc[4], yc[4], xi, yi, rij, rcsq;
        double phat, phbt;

        xc[0]=0.0; yc[0]=0.0;
        xc[1]=0.0; yc[1]=lz;
        xc[2]=lx;  yc[2]=0.0;
        xc[3]=lx;  yc[3]=lz;

        rcsq = lx*lz*fA/Pi;

	for(k=0;k<Nz;k++)
        for(i=0;i<Nx;i++)
        {
                xi = i*dx;
                for(j=0;j<Ny;j++)
                {
                        yi = j*dy;

                        phat = 0.10; phbt = 0.90;
                        tag = 0;
                        for(n=0; n<4; n++)
                        {
                                rij = (xi-xc[n])*(xi-xc[n]);
                                rij += (yi-yc[n])*(yi-yc[n]);
                                if(rij<rcsq)tag=1;
                        }

                        if(tag){ phat = 0.90; phbt = 0.10;}

                        ijk=(long)((k*Ny+j)*Nx+i);
                        wA[ijk]=hAB*phbt+0.040*(drand48()-0.5);
                        wB[ijk]=hAB*phat+0.040*(drand48()-0.5);
                }

        }
}

void initIWP(double *wA, double *wB,double *phA,double *phB)
{
        int i,j,k,n,tag;
        long ijk;
        double xc[17], yc[17], zc[17], xi, yi, zi, rij, rcsq;
        double phat, phbt, r0;

	xc[0]=0.0; yc[0]=0.0; zc[0]=0.0;
	xc[1]=0.0; yc[1]=0.0; zc[1]=lz;
	xc[2]=0.0; yc[2]=ly;  zc[2]=0.0;
	xc[3]=lx;  yc[3]=0.0; zc[3]=0.0;
	xc[4]=lx;  yc[4]=ly;  zc[4]=0.0;
	xc[5]=lx;  yc[5]=0.0; zc[5]=lz;
	xc[6]=0.0; yc[6]=ly;  zc[6]=lz;
	xc[7]=lx;  yc[7]=ly;  zc[7]=lz;
	xc[8]=lx/2;yc[8]=ly/2;zc[8]=lz/2;

	for(n=0;n<=7;n++)
	{
		xc[n+9]=(xc[n]+xc[8])/2;
		yc[n+9]=(yc[n]+yc[8])/2;
		zc[n+9]=(zc[n]+zc[8])/2;
	}

	r0=pow(3*fA*lx*ly*lz/17/4/Pi, 1.0/3);

        for(k=0;k<Nz;k++)
        {
                zi=k*dz;

                for(j=0;j<Ny;j++)
                {
                        yi=j*dy;
                        for(i=0;i<Nx;i++)
                        {

                                xi=i*dx;

                                ijk=(long)((k*Ny+j)*Nx+i);
                                phat=0.10; phbt=0.90;

				tag=0;
				for(n=0;n<17;n++)
				{
					rij=(xi-xc[n])*(xi-xc[n]);
					rij+=(yi-yc[n])*(yi-yc[n]);
					rij+=(zi-zc[n])*(zi-zc[n]);
					if(rij<r0*r0)tag=1;
				}

				if(tag==1){phat=0.90; phbt=0.10;}

                                wA[ijk]=hAB*phbt+0.040*(drand48()-0.5);
                                wB[ijk]=hAB*phat+0.040*(drand48()-0.5);
                                phA[ijk]=phat;
                                phB[ijk]=phbt;
                        }
                }
        }

        write_ph(phA,phB,wA,wB);
}

void initW_PL(double *wA, double *wB,double *phA,double *phB)
{
        int i,j,k,n,tag;
        long ijk;
        double xc[9], yc[9], xi, yi, zi, rij, rcsq;
        double phat, phbt, r0, w0;

        xc[0] = 0.0;  yc[0] = 0.0;
        xc[1] = 0.0;  yc[1] = ly;
        xc[2] = lx/2; yc[2] = ly/2;
        xc[3] = lx;   yc[3] = 0.0;
        xc[4] = lx;   yc[4] = ly;

        xc[5] = 0.0;  yc[5] = ly/3;
        xc[6] = lx/2; yc[6] = 5*ly/6;
        xc[7] = lx;   yc[7] = ly/3;

        w0 = 0.20*lz;
        r0 = 0.30*lz;

        for(k=0;k<Nz;k++)
        {
                zi=k*dz;

                for(j=0;j<Ny;j++)
                {
                        yi=j*dy;
                        for(i=0;i<Nx;i++)
                        {

                                xi=i*dx;

                                ijk=(long)((k*Ny+j)*Nx+i);
                                phat=0.10; phbt=0.90;

                                if(zi<w0/2||zi>=(lz-w0/2)||(zi>=(lz/2-w0/2)&&zi<=(lz/2+w0/2)))
                                {
                                        phat=0.90; phbt=0.10;
                                }
                                else
                                {
                                        if(zi>=w0/2&&zi<(lz/2-w0/2))
                                        {
                                                tag=0;
                                                for(n=0;n<5;n++)
                                                {
                                                        rij=(xi-xc[n])*(xi-xc[n])+(yi-yc[n])*(yi-yc[n]);
                                                        if(rij<r0*r0)tag=1;
                                                }
                                                if(tag){phat=0.90; phbt=0.10;}
                                        }
                                        else if(zi>=(lz/2+w0/2)&&zi<(lz-w0/2))
                                        {
                                                tag=0;
                                                for(n=5;n<=7;n++)
                                                {
                                                        rij=(xi-xc[n])*(xi-xc[n])+(yi-yc[n])*(yi-yc[n]);
                                                        if(rij<r0*r0)tag=1;
                                                }
                                                if(tag){phat=0.90; phbt=0.10;}

                                        }
                                }
                                wA[ijk]=hAB*phat+0.040*(drand48()-0.5);	/* switch a and b */
                                wB[ijk]=hAB*phbt+0.040*(drand48()-0.5);
                                phA[ijk]=phbt;
                                phB[ijk]=phat;
                        }
                }
        }

        write_ph(phA,phB,wA,wB);
}

void initW_PLrev(double *wA, double *wB,double *phA,double *phB)
{
	int i,j,k,n,tag;
	long ijk;
	double xc[9], yc[9], xi, yi, zi, rij, rcsq;
	double phat, phbt, r0, w0;

	xc[0] = 0.0;  yc[0] = 0.0;
	xc[1] = 0.0;  yc[1] = ly;
	xc[2] = lx/2; yc[2] = ly/2;
	xc[3] = lx;   yc[3] = 0.0;
	xc[4] = lx;   yc[4] = ly;
	
	xc[5] = 0.0;  yc[5] = ly/3;
	xc[6] = lx/2; yc[6] = 5*ly/6;
	xc[7] = lx;   yc[7] = ly/3;

	w0 = 0.150*lz;
	r0 = 0.10*lz;

	for(k=0;k<Nz;k++)
	{
		zi=k*dz;

		for(j=0;j<Ny;j++)
		{
			yi=j*dy;
			for(i=0;i<Nx;i++)
			{
				xi=i*dx;

				ijk=(long)((k*Ny+j)*Nx+i);
				phat=0.10; phbt=0.90;

				if(zi<w0/2||zi>=(lz-w0/2)||(zi>=(lz/2-w0/2)&&zi<=(lz/2+w0/2)))
				{
					phat=0.90; phbt=0.10;
				}
				else
				{
					if(zi>=w0/2&&zi<(lz/2-w0/2))
					{
						tag=0;
						for(n=0;n<5;n++)
						{
							rij=(xi-xc[n])*(xi-xc[n])+(yi-yc[n])*(yi-yc[n]);
							if(rij<r0*r0)tag=1;
						}
						if(tag){phat=0.90; phbt=0.10;}
					}
					else if(zi>=(lz/2+w0/2)&&zi<(lz-w0/2))
                                        {
                                                tag=0;
                                                for(n=5;n<=7;n++)
                                                {
                                                        rij=(xi-xc[n])*(xi-xc[n])+(yi-yc[n])*(yi-yc[n]);
                                                        if(rij<r0*r0)tag=1;
                                                }
                                                if(tag){phat=0.90; phbt=0.10;}

					}
				}
                        	wA[ijk]=hAB*phbt+0.040*(drand48()-0.5);
                        	wB[ijk]=hAB*phat+0.040*(drand48()-0.5);
				phA[ijk]=phat;
				phB[ijk]=phbt;
			}
		}
	}

	write_ph(phA,phB,wA,wB);
}

void initWcyl(double *wA,double *wB)
{
        int i,j,k,n,tag;
        long ijk;
	double xc[5], yc[5], xi, yi, rij, rcsq;
	double phat, phbt;

	xc[0]=0.0; yc[0]=0.0;
	xc[1]=0.0; yc[1]=lz;
	xc[2]=lx/2;yc[2]=lz/2;
	xc[3]=lx;  yc[3]=0.0;
	xc[4]=lx;  yc[4]=lz;

	rcsq = lx*lz*fA/Pi/2; 

	for(k=0;k<Nz;k++)
        for(i=0;i<Nx;i++)
        {
		xi = i*dx;
                for(j=0;j<Ny;j++)
                {
			yi = j*dy;

			phat = 0.10; phbt = 0.90;
			tag = 0;
			for(n=0; n<5; n++)
			{
				rij = (xi-xc[n])*(xi-xc[n]);
				rij += (yi-yc[n])*(yi-yc[n]);
				if(rij<rcsq)tag=1;
			}

			if(tag){ phat = 0.90; phbt = 0.10;}
			
                        ijk=(long)((k*Ny+j)*Nx+i);
                        wA[ijk]=hAB*phbt+0.040*(drand48()-0.5);
                        wB[ijk]=hAB*phat+0.040*(drand48()-0.5);
                }

        }
}

void initW1d(double *wA,double *wB)
{
        int i,j,k;
        long ijk;
	double ran1, ran2;

        for(i=0;i<Nx;i++)
        {
		ran1 = drand48()-0.50;
		ran2 = drand48()-0.50;
                for(j=0;j<Ny;j++)for(k=0;k<Nz;k++)
                {
                        ijk=(long)((k*Ny+j)*Nx+i);
                        wA[ijk]=hAB*fB+4.0*ran1;
                        wB[ijk]=hAB*fA+4.0*ran2;
                }

        }
}

//**********************main loop*************************

double freeE(double *wA,double *wB,double *phA,double *phB)
{
	int i,j,k,iter,maxIter;
	long ijk;
	double freeEnergy,freeOld,qCab,eta;
	double freeW,freeAB,freeS,freeDiff,freeWsurf;
	double Sm1,Sm2,wopt,wcmp,beta,psum,fpsum;
	double waDiff,wbDiff,inCompMax,wa0, wb0, wc0;
	FILE *fp;

	Sm1=0.2e-7;
	Sm2=0.1e-10;
	maxIter=MaxIT;
	wopt=0.050;
	wcmp=0.10;
	beta=1.0;
	iter=0;	

	freeEnergy=0.0;
	
	do
	{
		iter=iter+1;

		wa0 = 0.0;
		wb0 = 0.0;

		for(ijk=0; ijk<NxNyNz;ijk++)
		{
			wa0 += wA[ijk];
			wb0 += wB[ijk];
		}

		wa0/=NxNyNz;
		wb0/=NxNyNz;
		
		for(ijk=0; ijk<NxNyNz;ijk++)
		{
			wA[ijk]-=wa0;
			wB[ijk]-=wb0;
		}
		
		qCab=getConc(phA,phB,wA,wB);
		
		freeW=0.0;
		freeAB=0.0;
		freeS=0.0;
		freeWsurf=0.0;
		inCompMax=0.0;
				
		for(ijk=0; ijk<NxNyNz; ijk++)
		{
			eta=(wA[ijk]+wB[ijk]-hAB)/2;

			psum=1.0-phA[ijk]-phB[ijk];
			fpsum=fabs(psum);
			if(fpsum>inCompMax)inCompMax=fpsum;
			waDiff=hAB*phB[ijk]+eta-wA[ijk];
			wbDiff=hAB*phA[ijk]+eta-wB[ijk];
			waDiff-=wcmp*psum;
			wbDiff-=wcmp*psum;
			
			freeAB=freeAB+hAB*phA[ijk]*phB[ijk];
			freeW=freeW-(wA[ijk]*phA[ijk]+wB[ijk]*phB[ijk]);
			
			wA[ijk]+=wopt*waDiff;
			wB[ijk]+=wopt*wbDiff;
		}											
			
		freeAB/=NxNyNz;
		freeW/=NxNyNz;
		freeWsurf/=NxNyNz;
		freeS=-log(qCab);
			
		freeOld=freeEnergy;
		freeEnergy=freeAB+freeW+freeS;

		//**** print out the free energy and error results ****
			
		if(iter%5==0||iter>=maxIter)
        	{
			fp=fopen("printout.txt","a");
			fprintf(fp,"%d\n",iter);
			fprintf(fp,"%10.8e, %10.8e, %10.8e, %10.8e, %10.8e, %e\n",
				freeEnergy,freeAB,freeW,freeS,freeWsurf,inCompMax);
			fclose(fp);
		}
		if(iter%10==0)printf(" %5d : %.8e, %.8e\n", iter, freeEnergy, inCompMax);
		freeDiff=fabs(freeEnergy-freeOld);
        
		if(iter%50==0)write_ph(phA,phB,wA,wB);
	}while(iter<maxIter&&(inCompMax>Sm1||freeDiff>Sm2));

	fp=fopen("fe_end.dat","w");
        fprintf(fp,"%d\n",iter);
	fprintf(fp,"%10.8e, %10.8e, %10.8e, %10.8e, %10.8e, %e\n",freeEnergy,freeAB,freeW,freeS,freeWsurf,inCompMax);
	fclose(fp);

	write_ph(phA,phB,wA,wB);
	
	return freeDiff;
}

double getConc(double *phlA,double *phlB,double *wA,double *wB)
{
	int i,j,k,iz,m;
	long ijk,ijkiz;
	double *qA,*qcA,*qB,*qcB;
	double ql,ffl,*qInt,qtmp;
	//MPI_Status status;
	
	qA=(double *)malloc(sizeof(double)*NxNyNz*(NsA+1));
	qcA=(double *)malloc(sizeof(double)*NxNyNz*(NsA+1));
        qB=(double *)malloc(sizeof(double)*NxNyNz*(dNsB+1));
        qcB=(double *)malloc(sizeof(double)*NxNyNz*(dNsB+1));
	qInt=(double *)malloc(sizeof(double)*NxNyNz);

	for(ijk=0;ijk<NxNyNz;ijk++)
	{
		qInt[ijk]=1.0;
	}
	
	sovDifFft(qA,wA,qInt,fA,NsA,1);  /* A(n-1)+A_star_B */
	sovDifFft(qcB,wB,qInt,dfB,dNsB,-1);

        for(ijk=0;ijk<NxNyNz;ijk++)
        {
                qInt[ijk]=qA[ijk*(NsA+1)+NsA];
		qtmp=qcB[ijk*(dNsB+1)];
		for(m=1;m<Narm;m++)qInt[ijk]*=qtmp;
        }

	sovDifFft(qB,wB,qInt,dfB,dNsB,1);      //fa to 0 for qcA

	for(ijk=0;ijk<NxNyNz;ijk++)
	{
		qInt[ijk]=qcB[ijk*(dNsB+1)];
		qtmp=qcB[ijk*(dNsB+1)];
		for(m=1;m<Narm;m++)qInt[ijk]*=qtmp;
	}

	sovDifFft(qcA,wA,qInt,fA,NsA,-1);      //fb to 1 for qB

	ql=0.0;
	for(ijk=0; ijk<NxNyNz; ijk++)
	{
		ql+=qB[ijk*(dNsB+1)+dNsB];
	}

	ql/=NxNyNz;
	ffl=ds0/ql;

	for(ijk=0; ijk<NxNyNz; ijk++)
	{
		phlA[ijk]=0.0;
		phlB[ijk]=0.0;

		for(iz=0;iz<=NsA;iz++)
		{
			ijkiz=ijk*(NsA+1)+iz;
			if(iz==0||iz==NsA)phlA[ijk]+=(0.50*qA[ijkiz]*qcA[ijkiz]);
			else phlA[ijk]+=(qA[ijkiz]*qcA[ijkiz]);
		}


		for(iz=0;iz<=dNsB;iz++)
		{
			ijkiz=ijk*(dNsB+1)+iz;
			if(iz==0||iz==dNsB)phlB[ijk]+=(0.50*qB[ijkiz]*qcB[ijkiz]);
			else phlB[ijk]+=(qB[ijkiz]*qcB[ijkiz]);
		}

		phlA[ijk]*=ffl;
		phlB[ijk]*=(Narm*ffl);
	}
	free(qA);
	free(qcA);
	free(qB);
	free(qcB);

	free(qInt);

	return ql;
}

void sovDifFft(double *g,double *w,double *qInt,
	double z,int ns,int sign)
{
    
	int i,j,k,iz,ns1;
	long ijk,ijkr;

	ns1=ns+1;	

	for(ijk=0;ijk<NxNyNz;ijk++)
    	{
        	wdz[ijk]=exp(-w[ijk]*ds2);
    	}
	
	if(sign==1)
	{
		for(ijk=0;ijk<NxNyNz;ijk++)
		{
			g[ijk*ns1]=qInt[ijk];
		}

		for(iz=1;iz<=ns;iz++)
		{
			for(ijk=0;ijk<NxNyNz;ijk++)
			{
				in[ijk]=g[ijk*ns1+iz-1]*wdz[ijk];
			}

			fftw_execute (p_forward);

        
			for(k=0;k<Nz;k++)for(j=0;j<Ny;j++)for(i=0;i<Nxh1;i++)
			{
				ijk=(long)((k*Ny+j)*Nxh1+i);
				ijkr=(long)((k*Ny+j)*Nx+i);
                
				out[ijk][0]*=kxyzdz[ijkr];	//out[].re or .im for fftw2
				out[ijk][1]*=kxyzdz[ijkr];	//out[][0] or [1] for fftw3
			}

			fftw_execute(p_backward);
			
			for(ijk=0;ijk<NxNyNz;ijk++)
				g[ijk*ns1+iz]=in[ijk]*wdz[ijk]/NxNyNz;
		}
	}
	else 
	{

		for(ijk=0;ijk<NxNyNz;ijk++)
		{
			g[ijk*ns1+ns] = qInt[ijk];
		}
		
		for(iz=ns-1;iz>=0;iz--)
		{
			for(ijk=0;ijk<NxNyNz;ijk++)
			{
				in[ijk]=g[ijk*ns1+iz+1]*wdz[ijk];
			}

			fftw_execute(p_forward);
			
			for(k=0;k<Nz;k++)for(j=0;j<Ny;j++)for(i=0;i<Nxh1;i++)
			{
				ijk=(long)((k*Ny+j)*Nxh1+i);
				ijkr=(long)((k*Ny+j)*Nx+i);
                
				out[ijk][0]*=kxyzdz[ijkr];
				out[ijk][1]*=kxyzdz[ijkr];
			}

			fftw_execute(p_backward);

			for(ijk=0;ijk<NxNyNz;ijk++)
				g[ijk*ns1+iz]=in[ijk]*wdz[ijk]/NxNyNz;
		}
	}
}


