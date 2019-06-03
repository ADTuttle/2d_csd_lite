#include "constants.h"
#include "functions.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void mclin(struct FluxData *flux,PetscInt index,PetscReal pc,PetscInt zi,PetscReal ci,PetscReal ce,PetscReal phim,PetscInt ADD)
{
    //Returns the flux value by ref.
    // pc is the permeativity, zi is valence, ci/e intra/extra concentration
    //phim is membrane voltage, index is the index in the flux struct
    //compute value and derivatives of the function:
    //mflux=pc.*(log(ci./ce)+z*phim)
    //for trans-membrane flux of an ion obeying a linear current-voltage equation
    if(ADD) { //If add, we accumulate the result
        flux->mflux[index] += pc*(log(ci/ce)+zi*phim);
        flux->dfdci[index] += pc/ci;
        flux->dfdce[index] += -pc/ce;
        flux->dfdphim[index] += zi*pc;
    } else{ //If not add we reninitialize
        flux->mflux[index] = pc*(log(ci/ce)+zi*phim);
        flux->dfdci[index] = pc/ci;
        flux->dfdce[index] = -pc/ce;
        flux->dfdphim[index] = zi*pc;
    }
}
void mcGoldman(struct FluxData *flux,PetscInt index,PetscReal pc,PetscInt zi,PetscReal ci,PetscReal ce,PetscReal phim,PetscInt ADD)
{
    //compute value and derivatives of the function:
    //mflux=p.*(z*phim).*(ci.*exp(z*phim)-ce)./(exp(z*phim)-1)
    //for trans-membrane flux of an ion obeying the GHK equations
    PetscReal xi = zi*phim/2;
    PetscReal r = exp(xi);

    //compute s=x/sinh(x)
    //Watch out for division by zero
    PetscReal s;
    if(xi!=0) {
        s = xi/sinh(xi);
    } else {
        s = 1;
    }
    //compute dfdci,dfdce,mflux
    PetscReal dfdci = pc*s*r;
    PetscReal dfdce = -pc*s/r;
    PetscReal mi = ci*dfdci;
    PetscReal me = ce*dfdce;
    PetscReal mflux = mi+me;
    //compute w=(sinh(x)/x-cosh(x))/x
    PetscReal w;
    if(fabs(xi)<0.2) //use Taylor expan. for small values
    {
        //w0 = -x0.^9/3991680-x0.^7/45360-x0.^5/840-x0.^3/30-x0/3
        w = -pow(xi,9)/3991680-pow(xi,7)/45360-pow(xi,5)/840-pow(xi,3)/30-xi/3;
    }
    else //use formula for larger values
    {
        w = (sinh(xi)/xi-cosh(xi))/xi;
    }
    //compute dfdphim
    PetscReal dfdphim = (PetscReal)zi/2*(mflux*s*w+mi-me);
    if(ADD) { //If ADD, we accumulate this result
        flux->mflux[index] += mflux;
        flux->dfdce[index] += dfdce;
        flux->dfdci[index] += dfdci;
        flux->dfdphim[index] += dfdphim;
    } else { // If not ADD, we reset the values
        flux->mflux[index] = mflux;
        flux->dfdce[index] = dfdce;
        flux->dfdci[index] = dfdci;
        flux->dfdphim[index] = dfdphim;
    }
}

void glutamate_flux(struct FluxData *flux,PetscInt x,PetscInt y,struct SimState *state_vars,struct SimState *state_vars_past,
                    PetscInt Nx,PetscReal Glut_Excite)
{
    PetscReal Glu_n,Glu_g,Glu_np,Glu_gp,vn,vg,NaGlu,ce;

    Glu_n = state_vars->c[c_index(x,y,0,3,Nx)];
    Glu_g = state_vars->c[c_index(x,y,1,3,Nx)];
    Glu_np = state_vars_past->c[c_index(x,y,0,3,Nx)];
    Glu_gp= state_vars_past->c[c_index(x,y,1,3,Nx)];
    ce= state_vars_past->c[c_index(x,y,Nc-1,3,Nx)];
    vn = state_vars->phi[phi_index(x,y,0,Nx)]-state_vars->phi[phi_index(x,y,Nc-1,Nx)];
    vg = state_vars->phi[phi_index(x,y,1,Nx)]-state_vars->phi[phi_index(x,y,Nc-1,Nx)];
// /*
    PetscReal frac = 1.0/(Glu_n+glut_eps);//1.0/(pow(cn,1.19)+glut_eps);//
    PetscReal expo = 0.76e-3*exp(-0.0044*pow(vn*RTFC-8.66,2));

    //Neuronal portion
    flux->mflux[c_index(x,y,0,3,Nx)] = -(-glut_A*Glu_n*frac*expo+glut_gamma*glut_Bn*(ce-glut_Re*glut_Rg*Glu_np)+ //(ce-glut_Re*Glu_gp)+
                                            glut_Bg*(Glu_gp-glut_Rg*Glu_np)-Glut_Excite);
    flux->dfdci[c_index(x,y,0,3,Nx)] = -(-glut_A*expo*glut_eps*pow(frac,2));
    flux->dfdce[c_index(x,y,0,3,Nx)] = 0;
    flux->dfdphim[c_index(x,y,0,3,Nx)] = -(RTFC*0.0088*(vn*RTFC-8.66)*expo*glut_A*Glu_n*frac);


    //Glial Portion
    flux->mflux[c_index(x,y,1,3,Nx)] = -((1-glut_gamma)*glut_Bn*(ce-glut_Re*Glu_gp)-glut_Bg*(Glu_gp-glut_Rg*Glu_np));
    flux->dfdci[c_index(x,y,1,3,Nx)] = 0;
    flux->dfdce[c_index(x,y,1,3,Nx)] = 0;//-((1-glut_gamma)*glut_Bn);
    flux->dfdphim[c_index(x,y,1,3,Nx)] = 0;

    //Extracell portion is = -Neuron-Glia. Which is implemented in the solvers.

    // For uniformity scale these up by ell
    //Neuronal portion
    flux->mflux[c_index(x,y,0,3,Nx)] *= ell;
    flux->dfdci[c_index(x,y,0,3,Nx)] *= ell;
    flux->dfdce[c_index(x,y,0,3,Nx)] *= ell;
    flux->dfdphim[c_index(x,y,0,3,Nx)] *= ell;

    //Glial Portion
    flux->mflux[c_index(x,y,1,3,Nx)] *= ell;
    flux->dfdci[c_index(x,y,1,3,Nx)] *= ell;
    flux->dfdce[c_index(x,y,1,3,Nx)] *= ell;
    flux->dfdphim[c_index(x,y,1,3,Nx)] *= ell;
//*/
/*
    //Extracellular conc.
    PetscReal Nae = state_vars_past->c[c_index(x,y,Nc-1,0,Nx)];
    PetscReal Ke = state_vars_past->c[c_index(x,y,Nc-1,1,Nx)];
    PetscReal Glu_e = state_vars->c[c_index(x,y,Nc-1,3,Nx)];

    //Neuron part
    PetscReal Na = state_vars_past->c[c_index(x,y,0,0,Nx)];
    PetscReal K = state_vars_past->c[c_index(x,y,0,1,Nx)];
    // Add the membrane currents
//    NaGlu = pNaGl_n*(vn-0.5*log(pow(Nae/Na,3)*(K/Ke)*(Glu_e/Glu_n)*pHratio));
    NaGlu = pNaGl_n*(0.5*log(pow(Na/Nae,3)*(Ke/K)*(Glu_n/Glu_e)*pHratio)-vn);
    flux->mflux[c_index(x,y,0,0,Nx)]+=3*NaGlu; //Sodium
    flux->mflux[c_index(x,y,0,1,Nx)]-=NaGlu; //Potassium
    flux->mflux[c_index(x,y,0,3,Nx)]+=NaGlu; //Glu
    //Derivs
    flux->dfdci[c_index(x,y,0,3,Nx)]+=pNaGl_n/(2*Glu_n); //Just glutamate is implicit
    flux->dfdce[c_index(x,y,0,3,Nx)]-=pNaGl_n/(2*Glu_e);
    flux->dfdphim[c_index(x,y,0,3,Nx)]+=pNaGl_n;
    flux->dfdphim[c_index(x,y,0,0,Nx)]+=3*pNaGl_n;
    flux->dfdphim[c_index(x,y,0,1,Nx)]-=pNaGl_n;

    //Glia part
    Na = state_vars_past->c[c_index(x,y,1,0,Nx)];
    K = state_vars_past->c[c_index(x,y,1,1,Nx)];
    // Add the membrane currents
//    NaGlu = pNaGl_g*(vg-0.5*log(pow(Nae/Na,3)*(K/Ke)*(Glu_e/Glu_g)*pHratio));
    NaGlu = pNaGl_g*(0.5*log(pow(Na/Nae,3)*(Ke/K)*(Glu_g/Glu_e)*pHratio)-vg);
    flux->mflux[c_index(x,y,1,0,Nx)]+=3*NaGlu; //Sodium
    flux->mflux[c_index(x,y,1,1,Nx)]-=NaGlu; //Potassium
    flux->mflux[c_index(x,y,1,3,Nx)]+=NaGlu; //Glu
    //Derivs
    flux->dfdci[c_index(x,y,1,3,Nx)]+=pNaGl_g/(2*Glu_g); //Just glutamate is implicit
    flux->dfdce[c_index(x,y,1,3,Nx)]-=pNaGl_g/(2*Glu_e);
    flux->dfdphim[c_index(x,y,1,3,Nx)]+=pNaGl_g;
    flux->dfdphim[c_index(x,y,1,0,Nx)]+=3*pNaGl_n;
    flux->dfdphim[c_index(x,y,1,1,Nx)]-=pNaGl_n;
    */

}
PetscReal xoverexpminusone(PetscReal v,PetscReal aa,PetscReal bb,PetscReal cc,PetscInt dd)
{
    //computes aa*(v+bb)/(exp(cc*(v+bb))-1) if dd==0
    //computes aa*(v+bb)/(1-exp(-cc*(v+bb)) otherwise
    //for computing gating variables
    v+=bb;
    if(v==0) {
        return aa/cc;
    }
    if(dd==0) {
        return aa*v/(2*sinh(cc/2*v))*exp(-cc/2*v);
    }
    else {
        return aa*v/(2*sinh(cc/2*v))*exp(cc/2*v);
    }
}
PetscReal inwardrect(PetscReal ci,PetscReal ce,PetscReal phim)
{
    //inwardrect determines the effective conductance for the inward-rectifying
    //potassium channel - formula and constants from Steinberg et al 2005
    PetscReal Enernst = RTFC*log(ce/ci);
    PetscReal EKdef = -85.2; //#-85.2 mV is base reversal potential
    PetscReal cKo = .003; //#3 mM is base extracellular potassium concentration
    return sqrt(ce/cKo)*(1+exp(18.5/42.5))/(1+exp((RTFC*phim-Enernst+18.5)/42.5))*(1+exp((-118.6+EKdef)/44.1))/(1+exp((-118.6+RTFC*phim)/44.1));
}
PetscReal cz(const PetscReal *cmat,const PetscInt *zvals,PetscInt x,PetscInt y,PetscInt Nx,PetscInt comp,struct AppCtx *user)
{
    //function to compute sum over i of c_i*z_i
    PetscReal accumulate=0;
    for(PetscInt ion=0;ion<Ni;ion++) {
        accumulate += zvals[ion]*cmat[c_index(x,y,comp,ion,Nx)];
    }
    return accumulate;
}
void diff_coef(PetscReal *Dc,const PetscReal *alp,PetscReal scale,struct AppCtx* user)
{
    //diffusion coefficients at all points, for all ions, in all compartments, in both x and y directions
    PetscReal tortuosity=1.6;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    struct ConstVars *con_vars = user->con_vars;
    PetscReal alNcL,alNcR,alNcU;
    for(PetscInt x=0;x<Nx;x++) {
        for(PetscInt y=0;y<Ny;y++) {
            alNcL=1-alp[al_index(x,y,0,Nx)]-alp[al_index(x,y,1,Nx)]; //Left extracell
            alNcR = 0;
            if(x<Nx-1) {
                alNcR = 1 - alp[al_index(x + 1, y, 0,Nx)] - alp[al_index(x + 1, y, 1,Nx)]; //Right extracell
            }
            alNcU = 0;
            if(y<Ny-1) {
                alNcU = 1 - alp[al_index(x, y + 1, 0,Nx)] - alp[al_index(x, y + 1, 1,Nx)];
            }
            for(PetscInt ion = 0; ion<Ni;ion++) {
                //diffusion coefficients in x direction
                if(x==(Nx-1)) {
                    //Boundary is zero
                    Dc[c_index(x,y,Nc-1,ion,Nx)*2] = con_vars->DExtracellScale[xy_index(x,y,Nx)*2]*scale*D[ion]*(alNcL)/(tortuosity*tortuosity);
                }
                else {
                    //diffusion coefficients in the extracellular space proportional to volume fraction
                    Dc[c_index(x,y,Nc-1,ion,Nx)*2] = con_vars->DExtracellScale[xy_index(x,y,Nx)*2]*scale*D[ion]/2*(alNcL+alNcR)/(tortuosity*tortuosity);
                }
                //diffusion coefficients in neuronal compartment equal to 0
                Dc[c_index(x,y,0,ion,Nx)*2] = con_vars->DNeuronScale[xy_index(x,y,Nx)*2]*scale*D[ion]*alphao[al_index(0,0,0,Nx)]/pow(tortuosity,2);
                //diffusion coefficients in glial compartment proportional to default volume fraction
                Dc[c_index(x,y,1,ion,Nx)*2] = con_vars->DGliaScale[xy_index(x,y,Nx)*2]*scale*D[ion]*alphao[al_index(0,0,1,Nx)]/pow(tortuosity,2);
                //diffusion coefficients in y direction
                if(y==(Ny-1)) {
                    //Boundary is zero
                    Dc[c_index(x,y,Nc-1,ion,Nx)*2+1] = con_vars->DExtracellScale[xy_index(x,y,Nx)*2+1]*scale*D[ion]*(alNcL)/pow(tortuosity,2);
                }
                else {
                    //diffusion coefficients in the extracellular space proportional to volume fraction
                    Dc[c_index(x,y,Nc-1,ion,Nx)*2+1] = con_vars->DExtracellScale[xy_index(x,y,Nx)*2+1]*scale*D[ion]/2*(alNcL+alNcU)/pow(tortuosity,2);

                }
                //diffusion coefficients in neuronal compartment equal to 0
                Dc[c_index(x,y,0,ion,Nx)*2+1] =  con_vars->DNeuronScale[xy_index(x,y,Nx)*2+1]*scale*D[ion]*alphao[al_index(0,0,0,Nx)]/pow(tortuosity,2);
                //diffusion coefficients in glial compartment proportional to default volume fraction
                Dc[c_index(x,y,1,ion,Nx)*2+1] = con_vars->DGliaScale[xy_index(x,y,Nx)*2+1]*scale*D[ion]*alphao[al_index(0,0,1,Nx)]/pow(tortuosity,2);

            }
        }
    }
}

void gatevars_update(struct GateType *gate_vars,struct GateType *gate_vars_past, struct SimState *state_vars,PetscReal dtms,struct AppCtx *user,PetscInt firstpass)
{
    if(Profiling_on){
        PetscLogEventBegin(event[4],0,0,0,0);
    }
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;

    PetscReal v,alpha,beta,Gphi;

    int Num_NMDA_States = 4;
    PetscReal a11,a12,a21,a22,r6,detA;
    PetscReal k1,k2,k3,k4,k5,k6,Fglu,npow,K_r,Rstar,D1,D2,S;
    if(firstpass){
        for(PetscInt x = 0; x < Nx; x++){
            for(PetscInt y = 0; y < Ny; y++){

                //membrane potential in mV
                v = (state_vars->phi[phi_index(x,y,0,Nx)]-state_vars->phi[phi_index(x,y,Nc-1,Nx)])*RTFC;
                //Iniitialize the point gating variables

                //compute current NaT
                //gating variables mNaT
                alpha = xoverexpminusone(v,0.32,51.9,0.25,1); //0.32*(Vm+51.9)./(1-exp(-0.25*(Vm+51.9)))
                beta = xoverexpminusone(v,0.28,24.89,0.2,0); //0.28*(Vm+24.89)./(exp(0.2*(Vm+24.89))-1)
                gate_vars->mNaT[xy_index(x,y,Nx)] = alpha/(alpha+beta);


                //gating variable hNaT
                alpha = 0.128*exp(-(0.056*v+2.94));
                beta = 4/(exp(-(0.2*v+6))+1);
                gate_vars->hNaT[xy_index(x,y,Nx)] = alpha/(alpha+beta);

                gate_vars->gNaT[xy_index(x,y,Nx)] =
                        pow(gate_vars->mNaT[xy_index(x,y,Nx)],3)*gate_vars->hNaT[xy_index(x,y,Nx)];

                //compute current NaP
                //gating variable mNaP
                alpha = 1/(1+exp(-(0.143*v+5.67)))/6;
                beta = 1.0/6-alpha; //1./(1+exp(0.143*Vm+5.67))/6
                gate_vars->mNaP[xy_index(x,y,Nx)] = alpha/(alpha+beta);

                //gating variable hNaP
                alpha = 5.12e-6*exp(-(0.056*v+2.94));
                beta = 1.6e-4/(1+exp(-(0.2*v+8)));
                gate_vars->hNaP[xy_index(x,y,Nx)] = alpha/(alpha+beta);

                gate_vars->gNaP[xy_index(x,y,Nx)] =
                        pow(gate_vars->mNaP[xy_index(x,y,Nx)],2)*gate_vars->hNaP[xy_index(x,y,Nx)];
                //compute KDR current
                //gating variable mKDR
                alpha = xoverexpminusone(v,0.016,34.9,0.2,1); //0.016*(Vm+34.9)./(1-exp(-0.2*(Vm+34.9)))
                beta = 0.25*exp(-(0.025*v+1.25));
                gate_vars->mKDR[xy_index(x,y,Nx)] = alpha/(alpha+beta);

                gate_vars->gKDR[xy_index(x,y,Nx)] = pow(gate_vars->mKDR[xy_index(x,y,Nx)],2);

                //compute KA current
                //gating variable mKA
                alpha = xoverexpminusone(v,0.02,56.9,0.1,1); //0.02*(Vm+56.9)./(1-exp(-0.1*(Vm+56.9)))
                beta = xoverexpminusone(v,0.0175,29.9,0.1,0); //0.0175*(Vm+29.9)./(exp(0.1*(Vm+29.9))-1)
                gate_vars->mKA[xy_index(x,y,Nx)] = alpha/(alpha+beta);

                //gating variable hKA
                alpha = 0.016*exp(-(0.056*v+4.61));
                beta = 0.5/(exp(-(0.2*v+11.98))+1);
                gate_vars->hKA[xy_index(x,y,Nx)] = alpha/(alpha+beta);

                gate_vars->gKA[xy_index(x,y,Nx)] =
                        pow(gate_vars->mKA[xy_index(x,y,Nx)],2)*gate_vars->hKA[xy_index(x,y,Nx)];

                //gating variable NMDA
                if(Ni > 3){
                    if(Num_NMDA_States==2){
                        alpha = 72*state_vars->c[c_index(x,y,Nc-1,3,Nx)];
                        beta = 6.6e-3; //just 6.6
                        gate_vars->yNMDA[xy_index(x,y,Nx)] = alpha/(alpha+beta);

//                    Gphi = 1/(1+0.28*exp(-0.062*v)); //Other gating "variable" given by just this.
                        Gphi = 1/(1+0.56*exp(-0.062*v)); //From Rossi/Atwell

                        gate_vars->gNMDA[xy_index(x,y,Nx)] = gate_vars->yNMDA[xy_index(x,y,Nx)]*Gphi;
                    }

                    if(Num_NMDA_States==3){
                        // 3 Stage Solve
//                    r=[0,6.9e-3,0,160e-3,4.7e-3,190e-3]
                        r6 = 190.0;//190e-3;
                        a11 = 6.9e-3;
                        a12 = -160e-3;
                        a21 = (r6*state_vars->c[c_index(x,y,Nc-1,3,Nx)]);
                        a22 = (r6*state_vars->c[c_index(x,y,Nc-1,3,Nx)]+160e-3+4.7e-3);

                        detA = a11*a22-a12*a21;
                        gate_vars->yNMDA[xy_index(x,y,Nx)] = (a22*0-a12*(r6*state_vars->c[c_index(x,y,Nc-1,3,Nx)]));
                        gate_vars->zNMDA[xy_index(x,y,Nx)] = (-a21*0+a11*(r6*state_vars->c[c_index(x,y,Nc-1,3,Nx)]));

                        gate_vars->yNMDA[xy_index(x,y,Nx)] /= detA;
                        gate_vars->zNMDA[xy_index(x,y,Nx)] /= detA;
//                    Gphi = 1/(1+0.28*exp(-0.062*v));
                        Gphi = 1/(1+0.56*exp(-0.062*v));
                        gate_vars->gNMDA[xy_index(x,y,Nx)] = gate_vars->yNMDA[xy_index(x,y,Nx)]*Gphi;

                    }
                    if(Num_NMDA_States==4){
                    // 4 Stage Solve
                    K_r = 2.3e-6;//34.9e-6;
                    npow = 1.5;//1.4;
                    Fglu = pow(state_vars->c[c_index(x,y,Nc-1,3,Nx)],npow)
                           /(pow(state_vars->c[c_index(x,y,Nc-1,3,Nx)],npow)+pow(K_r,npow));
                    k1 = 3.94e-3*Fglu;
                    k2 = 1.94e-3;
                    k3 = 0.0213e-3;
                    k4 = 0.00277e-3;

                    gate_vars->yNMDA[xy_index(x,y,Nx)] = (k2*k4)/(k1*k3+k1*k4+k2*k4);
                    gate_vars->zNMDA[xy_index(x,y,Nx)] = (k1*k4)/(k1*k3+k1*k4+k2*k4);
                    gate_vars->dNMDA[xy_index(x,y,Nx)] = (k1*k3)/(k1*k3+k1*k4+k2*k4);

                    Gphi = 1/(1+0.56*exp(-0.062*v));
                    gate_vars->gNMDA[xy_index(x,y,Nx)] = (gate_vars->yNMDA[xy_index(x,y,Nx)]*Fglu
                            +Desensitize[0]*gate_vars->zNMDA[xy_index(x,y,Nx)]
                            +Desensitize[1]*gate_vars->dNMDA[xy_index(x,y,Nx)])*Gphi;
                   }
                }
            }
        }
    }else{ //if it's not the firstpass, then we actually have values in v.
        for(PetscInt x = 0; x < Nx; x++){
            for(PetscInt y = 0; y < Ny; y++){
                //membrane potential in mV
                v = (state_vars->phi[phi_index(x,y,0,Nx)]-state_vars->phi[phi_index(x,y,Nc-1,Nx)])*RTFC;

                //compute current NaT
                //gating variables mNaT
                alpha = xoverexpminusone(v,0.32,51.9,0.25,1); //0.32*(Vm+51.9)./(1-exp(-0.25*(Vm+51.9)))
                beta = xoverexpminusone(v,0.28,24.89,0.2,0); //0.28*(Vm+24.89)./(exp(0.2*(Vm+24.89))-1)
                gate_vars->mNaT[xy_index(x,y,Nx)] =
                        (gate_vars_past->mNaT[xy_index(x,y,Nx)]+alpha*dtms)/(1+(alpha+beta)*dtms);

                //gating variable hNaT
                alpha = 0.128*exp(-(0.056*v+2.94));
                beta = 4/(exp(-(0.2*v+6))+1);
                gate_vars->hNaT[xy_index(x,y,Nx)] = (gate_vars_past->hNaT[xy_index(x,y,Nx)] + alpha*dtms)/(1+(alpha+beta)*dtms);

                gate_vars->gNaT[xy_index(x,y,Nx)] =
                        pow(gate_vars->mNaT[xy_index(x,y,Nx)],3)*gate_vars->hNaT[xy_index(x,y,Nx)];
                //compute current NaP
                //gating variable mNaP
                alpha = 1/(1+exp(-(0.143*v+5.67)))/6;
                beta = 1.0/6.0-alpha; //1./(1+exp(0.143*Vm+5.67))/6
                gate_vars->mNaP[xy_index(x,y,Nx)] =
                        (gate_vars_past->mNaP[xy_index(x,y,Nx)]+alpha*dtms)/(1+(alpha+beta)*dtms);

                //gating variable hNaP
                alpha = 5.12e-6*exp(-(0.056*v+2.94));
                beta = 1.6e-4/(1+exp(-(0.2*v+8)));
                gate_vars->hNaP[xy_index(x,y,Nx)] =
                        (gate_vars_past->hNaP[xy_index(x,y,Nx)]+alpha*dtms)/(1+(alpha+beta)*dtms);

                gate_vars->gNaP[xy_index(x,y,Nx)] =
                        pow(gate_vars->mNaP[xy_index(x,y,Nx)],2)*gate_vars->hNaP[xy_index(x,y,Nx)];

                //compute KDR current
                //gating variable mKDR
                alpha = xoverexpminusone(v,0.016,34.9,0.2,1); //0.016*(Vm+34.9)./(1-exp(-0.2*(Vm+34.9)))
                beta = 0.25*exp(-(0.025*v+1.25));
                gate_vars->mKDR[xy_index(x,y,Nx)] =
                        (gate_vars_past->mKDR[xy_index(x,y,Nx)]+alpha*dtms)/(1+(alpha+beta)*dtms);

                gate_vars->gKDR[xy_index(x,y,Nx)] = pow(gate_vars->mKDR[xy_index(x,y,Nx)],2);

                //compute KA current
                //gating variable mKA
                alpha = xoverexpminusone(v,0.02,56.9,0.1,1); //0.02*(Vm+56.9)./(1-exp(-0.1*(Vm+56.9)))
                beta = xoverexpminusone(v,0.0175,29.9,0.1,0); //0.0175*(Vm+29.9)./(exp(0.1*(Vm+29.9))-1)
                gate_vars->mKA[xy_index(x,y,Nx)] =
                        (gate_vars_past->mKA[xy_index(x,y,Nx)]+alpha*dtms)/(1+(alpha+beta)*dtms);

                //gating variable hKA
                alpha = 0.016*exp(-(0.056*v+4.61));
                beta = 0.5/(exp(-(0.2*v+11.98))+1);
                gate_vars->hKA[xy_index(x,y,Nx)] =
                        (gate_vars_past->hKA[xy_index(x,y,Nx)]+alpha*dtms)/(1+(alpha+beta)*dtms);

                gate_vars->gKA[xy_index(x,y,Nx)] =
                        pow(gate_vars->mKA[xy_index(x,y,Nx)],2)*gate_vars->hKA[xy_index(x,y,Nx)];

                //gating variable NMDA
                if(Ni > 3){
                    if(Num_NMDA_States==2){
                        //72 mM/sec->72 1e-3mM/l *1e-3 1/msec

                        alpha = 72*state_vars->c[c_index(x,y,Nc-1,3,Nx)];
                        beta = 6.6e-3; // 6.6 (sec)^-1->6.6e-3 msec^-1
                        gate_vars->yNMDA[xy_index(x,y,Nx)] =
                                (gate_vars_past->yNMDA[xy_index(x,y,Nx)]+alpha*dtms)/(1+(alpha+beta)*dtms);

//                Gphi = 1/(1+0.28*exp(-0.062*v)); //Other gating "variable" given by just this.
                        Gphi = 1/(1+0.56*exp(-0.062*v)); //From Rossi/Atwell

                        gate_vars->gNMDA[xy_index(x,y,Nx)] = gate_vars->yNMDA[xy_index(x,y,Nx)]*Gphi;
                    }
                    if(Num_NMDA_States==3){
                        // 3 Stage Solve
//                    r=[0,6.9e-3,0,160e-3,4.7e-3,190e-3]
                        r6 = 190.0;//190e-3;
                        a11 = 1+dtms*6.9e-3;
                        a12 = -dtms*160e-3;
                        a21 = dtms*(r6*state_vars->c[c_index(x,y,Nc-1,3,Nx)]);
                        a22 = 1+dtms*(r6*state_vars->c[c_index(x,y,Nc-1,3,Nx)]+160e-3+4.7e-3);

                        detA = a11*a22-a12*a21;
                        gate_vars->yNMDA[xy_index(x,y,Nx)] = (a22*gate_vars_past->yNMDA[xy_index(x,y,Nx)]
                                                              -a12*(gate_vars_past->zNMDA[xy_index(x,y,Nx)]+
                                                                    dtms*r6*state_vars->c[c_index(x,y,Nc-1,3,Nx)]));
                        gate_vars->zNMDA[xy_index(x,y,Nx)] = (-a21*gate_vars_past->yNMDA[xy_index(x,y,Nx)]
                                                              +a11*(gate_vars_past->zNMDA[xy_index(x,y,Nx)]+
                                                                    dtms*r6*state_vars->c[c_index(x,y,Nc-1,3,Nx)]));

                        gate_vars->yNMDA[xy_index(x,y,Nx)] /= detA;
                        gate_vars->zNMDA[xy_index(x,y,Nx)] /= detA;
//                    Gphi = 1/(1+0.28*exp(-0.062*v));
                        Gphi = 1.0/(1+0.56*exp(-0.062*v));
                        gate_vars->gNMDA[xy_index(x,y,Nx)] = gate_vars->yNMDA[xy_index(x,y,Nx)]*Gphi;
                    }

                    // 4 Stages
                    if(Num_NMDA_States==4){
                        K_r = 2.3e-6;//34.9e-6;
                        npow = 1.5; //1.4;
                        Fglu = pow(state_vars->c[c_index(x,y,Nc-1,3,Nx)],npow)
                               /(pow(state_vars->c[c_index(x,y,Nc-1,3,Nx)],npow)+pow(K_r,npow));
                        k1 = 3.94e-3*Fglu;
                        k2 = 1.94e-3;
                        k3 = 0.0213e-3;
                        k4 = 0.00277e-3;

                        // Calculate determinant
                        detA = dtms*k1+dtms*k2+dtms*k3+dtms*k4+dtms*dtms*k1*k3+dtms*dtms*k1*k4+dtms*dtms*k2*k4+1;


                        Rstar = gate_vars_past->yNMDA[xy_index(x,y,Nx)];
                        D1 = gate_vars_past->zNMDA[xy_index(x,y,Nx)];
                        D2 = gate_vars_past->dNMDA[xy_index(x,y,Nx)];
                        // Explicitly calculate inverse
                        // First row
                        a11 = dtms*k2+dtms*k3+dtms*k4+dtms*dtms*k2*k4+1;
                        a12 = dtms*k2*(dtms*k4+1);
                        a21 = dtms*dtms*k2*k4;

                        gate_vars->yNMDA[xy_index(x,y,Nx)] = (a11*Rstar+a12*D1+a21*D2)/detA;

                        // Second Row
                        a11 = dtms*k1*(dtms*k4+1);
                        a12 = (dtms*k1+1)*(dtms*k4+1);
                        a21 = dtms*k4*(dtms*k1+1);
                        gate_vars->zNMDA[xy_index(x,y,Nx)] = (a11*Rstar+a12*D1+a21*D2)/detA;

                        //Third Row
                        a11 = dtms*dtms*k1*k3;
                        a12 = dtms*k3*(dtms*k1+1);
                        a21 = dtms*k1+dtms*k2+dtms*k3+dtms*dtms*k1*k3+1;
                        gate_vars->dNMDA[xy_index(x,y,Nx)] = (a11*Rstar+a12*D1+a21*D2)/detA;


                        Gphi = 1/(1+0.56*exp(-0.062*v));
                        gate_vars->gNMDA[xy_index(x,y,Nx)] = (gate_vars->yNMDA[xy_index(x,y,Nx)]*Fglu
                                +Desensitize[0]*gate_vars->zNMDA[xy_index(x,y,Nx)]
                                +Desensitize[1]*gate_vars->dNMDA[xy_index(x,y,Nx)])*Gphi;
                    }
                }
            }

        }
    }
    if(Profiling_on){
        PetscLogEventEnd(event[4],0,0,0,0);
    }
}

void excitation(struct AppCtx* user,PetscReal t)
{
    //compute excitation conductance to trigger csd
    //Leak conductances in mS/cm^2
    //all units converted to mmol/cm^2/sec
    PetscReal pexct,pany;
    PetscReal xexct;
    PetscReal radius;
    struct ExctType *exct = user->gexct;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    PetscReal dx = user->dx;
    PetscReal dy = user->dy;
    PetscInt num_points = 0;
// t<0 happens during initialization
    if(t<0){
        for(PetscInt i=0;i<Nx*Ny;i++){
            exct->pNa[i]=0;
            exct->pK[i]=0;
            exct->pCl[i]=0;
            exct->pGlu[i]=0;
        }
        return;
    }
    if((t<=texct+1.0))
    for (PetscInt i = 0; i < Nx; i++){
        for(PetscInt j = 0; j < Ny; j++){
            if(one_point_exct){
                if(t < texct && i == 0 && j == 0){
                    num_points++;
                    pany = pmax*pow(sin(pi*t/texct),2)*RTFC/FC;
                    exct->pNa[xy_index(i,j,Nx)] = pany;
                    exct->pK[xy_index(i,j,Nx)] = pany;
                    exct->pCl[xy_index(i,j,Nx)] = pany;
                    exct->pGlu[xy_index(i,j,Nx)] = pany;
                }else{
                    //pexct=0*RTFC/FC
                    exct->pNa[xy_index(i,j,Nx)] = 0;
                    exct->pK[xy_index(i,j,Nx)] = 0;
                    exct->pCl[xy_index(i,j,Nx)] = 0;
                    exct->pGlu[xy_index(i,j,Nx)] = 0;
                }

            }
            if(mid_points_exct){
                radius = sqrt(pow((i+0.5)*dx-Lx/2,2)+pow((j+0.5)*dy-Lx/2,2));
                if(t < texct && radius < Lexct){
                    num_points++;
                    pexct = pmax*pow(sin(pi*t/texct),2)*RTFC/FC;
                    xexct = pow(cos(pi/2*(radius/Lexct)),2);
                    pany = pexct*xexct;
                    exct->pNa[xy_index(i,j,Nx)] = pany;
                    exct->pK[xy_index(i,j,Nx)] = pany;
                    exct->pCl[xy_index(i,j,Nx)] = pany;
                    exct->pGlu[xy_index(i,j,Nx)] = pany;
                }else{
                    //pexct=0*RTFC/FC
                    exct->pNa[xy_index(i,j,Nx)] = 0;
                    exct->pK[xy_index(i,j,Nx)] = 0;
                    exct->pCl[xy_index(i,j,Nx)] = 0;
                    exct->pGlu[xy_index(i,j,Nx)] = 0;
                }
            }
            if(plane_wave_exct){
                //plane wave at left side
                if(t < texct && j == 0){
                    num_points++;
                    pexct = pmax*pow(sin(pi*t/texct),2)*RTFC/FC;
                    pany = pexct;
                    exct->pNa[xy_index(i,j,Nx)] = pany;
                    exct->pK[xy_index(i,j,Nx)] = pany;
                    exct->pCl[xy_index(i,j,Nx)] = pany;
                    exct->pGlu[xy_index(i,j,Nx)] = pany;
                }else{
                    //pexct=0*RTFC/FC
                    exct->pNa[xy_index(i,j,Nx)] = 0;
                    exct->pK[xy_index(i,j,Nx)] = 0;
                    exct->pCl[xy_index(i,j,Nx)] = 0;
                    exct->pGlu[xy_index(i,j,Nx)] = 0;
                }
            }
            if(!one_point_exct && !mid_points_exct && !plane_wave_exct){
                radius = sqrt(pow((i+0.5)*dx,2)+pow((j+0.5)*dy,2));
                if(t < texct && radius < Lexct){
                    num_points++;
                    pexct = pmax*pow(sin(pi*t/texct),2)*RTFC/FC;
//	    		xexct=pow((cos(pi/2*(i+.5)/Nexct))*(cos(pi/2*(j+.5)/Nexct)),2);
                    xexct = pow(cos(pi/2*(radius/Lexct)),2);
//				xexct=pow((cos(pi/2*((i+.5)*dx)/Lexct))*(cos(pi/2*((j+.5)*dy)/Lexct)),2);
                    pany = pexct*xexct;
                    exct->pNa[xy_index(i,j,Nx)] = pany;
                    exct->pK[xy_index(i,j,Nx)] = pany;
                    exct->pCl[xy_index(i,j,Nx)] = pany;
                    exct->pGlu[xy_index(i,j,Nx)] = pany;
                }else{
                    //pexct=0*RTFC/FC
                    exct->pNa[xy_index(i,j,Nx)] = 0;
                    exct->pK[xy_index(i,j,Nx)] = 0;
                    exct->pCl[xy_index(i,j,Nx)] = 0;
                    exct->pGlu[xy_index(i,j,Nx)] = 0;
                }
            }
        }
    }



}

void ionmflux(struct AppCtx* user)
{
    if(Profiling_on) {
        PetscLogEventBegin(event[5], 0, 0, 0, 0);
    }
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    struct FluxData *flux = user->flux;
    struct SimState *state_vars=user->state_vars;
    struct SimState *state_vars_past = user->state_vars_past;
    struct GateType *gvars = user->gate_vars_past;
    struct ExctType *gexct = user->gexct;
    struct ConstVars *con_vars = user->con_vars;
    //Variables to save to for ease of notation
    PetscReal vm,vmg,vmgp;
    PetscReal ci,cg,ce,cgp,cep,cnp;
    PetscReal Na,K;//Variables for pump (so it's clear)

    //For calculationg permeabilities
    PetscReal pGHK,pLin;
    PetscReal Ipump,NaKCl;
    for(PetscInt x=0;x<Nx;x++) {
        for(PetscInt y=0;y<Ny;y++) {
            vm = state_vars->phi[phi_index(x,y,0,Nx)]-state_vars->phi[phi_index(x,y,Nc-1,Nx)];
            vmg = state_vars->phi[phi_index(x,y,1,Nx)]-state_vars->phi[phi_index(x,y,Nc-1,Nx)];
            vmgp = state_vars_past->phi[phi_index(x,y,1,Nx)]-state_vars_past->phi[phi_index(x,y,Nc-1,Nx)];

            //Compute Na Channel currents
            ci = state_vars->c[c_index(x,y,0,0,Nx)];
            cg = state_vars->c[c_index(x,y,1,0,Nx)];
            ce = state_vars->c[c_index(x,y,Nc-1,0,Nx)];

            //Neurons
            pGHK = con_vars->pNaT[xy_index(x,y,Nx)]*gvars->gNaT[xy_index(x,y,Nx)]+
                    con_vars->pNaP[xy_index(x,y,Nx)]*gvars->gNaP[xy_index(x,y,Nx)]+
                    con_vars->pNMDA[xy_index(x,y,Nx)]*gvars->gNMDA[xy_index(x,y,Nx)]*(2.0/3.0);
            pLin = con_vars->pNaLeak[xy_index(x,y,Nx)] + gexct->pNa[xy_index(x,y,Nx)]; //Add excitation
            //Initialize GHK Flux
            mcGoldman(flux,c_index(x,y,0,0,Nx),pGHK,1,ci,ce,vm,0);
            //Add leak current to that.
            mclin(flux,c_index(x,y,0,0,Nx),pLin,1,ci,ce,vm,1);
            //Glial NaLeak
            mclin(flux,c_index(x,y,1,0,Nx),con_vars->pNaLeakg[xy_index(x,y,Nx)],1,cg,ce,vmg,0);

            // Compute K channel Currents
            ci = state_vars->c[c_index(x,y,0,1,Nx)];
            cg = state_vars->c[c_index(x,y,1,1,Nx)];
            ce = state_vars->c[c_index(x,y,Nc-1,1,Nx)];

            //Neurons
            pGHK = con_vars->pKDR[xy_index(x,y,Nx)]*gvars->gKDR[xy_index(x,y,Nx)]+
                    con_vars->pKA[xy_index(x,y,Nx)]*gvars->gKA[xy_index(x,y,Nx)]+
                    con_vars->pNMDA[xy_index(x,y,Nx)]*gvars->gNMDA[xy_index(x,y,Nx)]*(1.0/3.0);
            pLin = pKLeak+gexct->pK[xy_index(x,y,Nx)]; //add excitation
            mcGoldman(flux,c_index(x,y,0,1,Nx),pGHK,1,ci,ce,vm,0);
            mclin(flux,c_index(x,y,0,1,Nx),pLin,1,ci,ce,vm,1);

            // Glial K Leak(using past)
            cgp = state_vars_past->c[c_index(x,y,1,1,Nx)];
            cep = state_vars_past->c[c_index(x,y,Nc-1,1,Nx)];

            pLin = con_vars->pKIR[xy_index(x,y,Nx)]*inwardrect(cgp,cep,vmgp)*pKLeakadjust;
            mclin(flux,c_index(x,y,1,1,Nx),pLin,1,cg,ce,vmg,0);

            //Compute Cl Channel Current
            ci = state_vars->c[c_index(x,y,0,2,Nx)];
            cg = state_vars->c[c_index(x,y,1,2,Nx)];
            ce = state_vars->c[c_index(x,y,Nc-1,2,Nx)];

            //Neurons
            pLin = pClLeak+gexct->pCl[xy_index(x,y,Nx)]; //add excitation
            mclin(flux,c_index(x,y,0,2,Nx),pLin,-1,ci,ce,vm,0);

            //Glia
            mclin(flux,c_index(x,y,1,2,Nx),pClLeakg,-1,cg,ce,vmg,0);

            //Pump Currents(all past values)

            //Neurons
            Na = state_vars_past->c[c_index(x,y,0,0,Nx)];
            K = state_vars_past->c[c_index(x,y,Nc-1,1,Nx)];

            Ipump = npump*con_vars->Imax[xy_index(x,y,Nx)]/(pow(1+mK/K,2)*pow(1+mNa/Na,3));

            //Add to flux(it's explicit so no derivatives)
            flux->mflux[c_index(x,y,0,0,Nx)]+=3*Ipump; //Na part
            flux->mflux[c_index(x,y,0,1,Nx)]-=2*Ipump; //K part

            //Glia
            Na = state_vars_past->c[c_index(x,y,1,0,Nx)];
            //K is the same(extracellular)
            Ipump = glpump*con_vars->Imaxg[xy_index(x,y,Nx)]/(pow(1+mK/K,2)*pow(1+mNa/Na,3));
            //Add to flux(it's explicit so no derivatives)
            flux->mflux[c_index(x,y,1,0,Nx)]+=3*Ipump; //Na part
            flux->mflux[c_index(x,y,1,1,Nx)]-=2*Ipump; //K part

            //NaKCl Cotransporter
            //I'm going to reuse variables names...
            Na = state_vars_past->c[c_index(x,y,1,0,Nx)];//glia Na
            K = state_vars_past->c[c_index(x,y,1,1,Nx)]; // glia K.
            cgp = state_vars_past->c[c_index(x,y,1,2,Nx)]; //glia Cl

            cep = state_vars_past->c[c_index(x,y,Nc-1,0,Nx)];//Ext Na
            ce = state_vars_past->c[c_index(x,y,Nc-1,1,Nx)]; // Ext K.
            ci = state_vars_past->c[c_index(x,y,Nc-1,2,Nx)]; //Ext Cl

            NaKCl = con_vars->pNaKCl[xy_index(x,y,Nx)]*log(Na*K*pow(cgp,2)/(cep*ce*pow(ci,2)));
            //Add to flux
            flux->mflux[c_index(x,y,1,0,Nx)]+=NaKCl; //Sodium
            flux->mflux[c_index(x,y,1,1,Nx)]+=NaKCl; //K
            flux->mflux[c_index(x,y,1,2,Nx)]+=2*NaKCl; //Cl

            //Glutamate transport(Sets glutamate flux and adds to Sodium+K fluxes in both Neurons and glia
            if(Ni>3){
                glutamate_flux(flux,x,y,state_vars,state_vars_past,Nx,gexct->pGlu[xy_index(x,y,Nx)]);
            }


            //Change units of flux from mmol/cm^2 to mmol/cm^3/s
            for(PetscInt ion=0;ion<Ni;ion++) {
                flux->mflux[c_index(x,y,Nc-1,ion,Nx)] = 0;
                for(PetscInt comp = 0;comp<Nc-1;comp++) {
                    flux->mflux[c_index(x,y,comp,ion,Nx)]=flux->mflux[c_index(x,y,comp,ion,Nx)]/ell;
                    flux->dfdci[c_index(x,y,comp,ion,Nx)]=flux->dfdci[c_index(x,y,comp,ion,Nx)]/ell;
                    flux->dfdce[c_index(x,y,comp,ion,Nx)]=flux->dfdce[c_index(x,y,comp,ion,Nx)]/ell;
                    flux->dfdphim[c_index(x,y,comp,ion,Nx)]=flux->dfdphim[c_index(x,y,comp,ion,Nx)]/ell;

                    //And calculate the extracellular flux
                    flux->mflux[c_index(x,y,Nc-1,ion,Nx)] -= flux->mflux[c_index(x,y,comp,ion,Nx)];
                }
            }

        }
    }
    if(Profiling_on) {
        PetscLogEventEnd(event[5], 0, 0, 0, 0);
    }
}
void wflowm(struct AppCtx *user)
{
    //piw = sum of c over ions + ao/alpha
    // piw is the total number of ions in a compartment
    //outward transmembrane water flow seen as a function of
    //osmotic pressure and volume fraction or pressure.
    if(Profiling_on) {
        PetscLogEventBegin(event[6], 0, 0, 0, 0);
    }
    struct FluxData *flux = user->flux;
    struct SimState *state_vars = user->state_vars;
    struct ConstVars *con_vars = user->con_vars;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;

    PetscReal dwdpi,dwdal,piw,piwNc;
    for(PetscInt x=0;x<Nx;x++) {
        for(PetscInt y=0;y<Ny;y++) {
            //Calculate the pi for extracellular
            piwNc = 0;
            for(PetscInt ion=0;ion<Ni;ion++) {
                piwNc +=state_vars->c[c_index(x,y,Nc-1,ion,Nx)];
            }
            piwNc +=con_vars->ao[phi_index(x,y,Nc-1,Nx)]/(1-state_vars->alpha[al_index(x,y,0,Nx)]-state_vars->alpha[al_index(x,y,1,Nx)]);
            for(PetscInt comp = 0;comp<Nc-1;comp++) {
                piw = 0;
                for(PetscInt ion=0;ion<Ni;ion++) {
                    piw +=state_vars->c[c_index(x,y,comp,ion,Nx)];
                }
                piw +=con_vars->ao[phi_index(x,y,comp,Nx)]/state_vars->alpha[al_index(x,y,comp,Nx)];
                //ao, zeta1, and zetalpha are currently constant in space
                dwdpi = con_vars->zeta1[al_index(x,y,comp,Nx)];
                dwdal = con_vars->zeta1[al_index(x,y,comp,Nx)]*con_vars->zetaalpha[al_index(0,0,comp,Nx)];

                flux->wflow[al_index(x,y,comp,Nx)] = dwdpi*(piwNc-piw)+dwdal*(state_vars->alpha[al_index(x,y,comp,Nx)]-alphao[comp]);
                flux->dwdpi[al_index(x,y,comp,Nx)] = dwdpi;
                flux->dwdal[al_index(x,y,comp,Nx)] = dwdal;
            }
        }
    }
    if(Profiling_on) {
        PetscLogEventEnd(event[6], 0, 0, 0, 0);
    }
}

void grid_wflowm(struct AppCtx *user)
{
    //piw = sum of c over ions + ao/alpha
    // piw is the total number of ions in a compartment
    //outward transmembrane water flow seen as a function of
    //osmotic pressure and volume fraction or pressure.
    if(Profiling_on) {
        PetscLogEventBegin(event[6], 0, 0, 0, 0);
    }
    struct FluxData *flux = user->flux;
    struct SimState *state_vars = user->grid_vars;
    struct ConstVars *con_vars = user->con_vars;
    PetscInt Nx = 2*width_size+1;
    PetscInt Ny = 2*width_size+1;

    PetscReal dwdpi,dwdal,piw,piwNc;
    for(PetscInt x=0;x<Nx;x++) {
        for (PetscInt y = 0; y < Ny; y++) {
            //Calculate the pi for extracellular
            piwNc = 0;
            for (PetscInt ion = 0; ion < Ni; ion++) {
                piwNc += state_vars->c[c_index(x, y, Nc - 1, ion, Nx)];
            }
            piwNc += con_vars->ao[al_index(0, 0, Nc - 1, Nx)] /
                     (1 - state_vars->alpha[al_index(x, y, 0, Nx)] - state_vars->alpha[al_index(x, y, 1, Nx)]);
            for (PetscInt comp = 0; comp < Nc - 1; comp++) {
                piw = 0;
                for (PetscInt ion = 0; ion < Ni; ion++) {
                    piw += state_vars->c[c_index(x, y, comp, ion, Nx)];
                }
                piw += con_vars->ao[al_index(0, 0, comp, Nx)] / state_vars->alpha[al_index(x, y, comp, Nx)];
                //ao, zeta1, and zetalpha are currently constant in space
                dwdpi = con_vars->zeta1[al_index(0, 0, comp, Nx)];
                dwdal = con_vars->zeta1[al_index(0, 0, comp, Nx)] * con_vars->zetaalpha[al_index(0, 0, comp, Nx)];

                flux->wflow[al_index(x, y, comp, Nx)] =
                        dwdpi * (piwNc - piw) + dwdal * (state_vars->alpha[al_index(x, y, comp, Nx)] - alphao[comp]);
                flux->dwdpi[al_index(x, y, comp, Nx)] = dwdpi;
                flux->dwdal[al_index(x, y, comp, Nx)] = dwdal;
            }
        }
    }
    if(Profiling_on) {
        PetscLogEventEnd(event[6], 0, 0, 0, 0);
    }
}
void grid_ionmflux(struct AppCtx* user,PetscInt xi,PetscInt yi)
{
    if(Profiling_on) {
        PetscLogEventBegin(event[5], 0, 0, 0, 0);
    }
    PetscInt Nx = 2*width_size+1;
    PetscInt Ny = 2*width_size+1;
    struct FluxData *flux = user->flux;
    struct SimState *state_vars=user->grid_vars;
    struct SimState *state_vars_past = user->grid_vars_past;
    struct GateType *gvars = user->grid_gate_vars;
    struct ExctType *gexct = user->gexct;
    struct ConstVars *con_vars = user->con_vars;
    //Variables to save to for ease of notation
    PetscReal vm,vmg,vmgp;
    PetscReal ci,cg,ce,cgp,cep,cnp;
    PetscReal Na,K;//Variables for pump (so it's clear)

    //For calculationg permeabilities
    PetscReal pGHK,pLin;
    PetscReal Ipump,NaKCl;
    for(PetscInt x=0;x<Nx;x++){
        for(PetscInt y=0;y<Ny;y++) {
            vm = state_vars->phi[phi_index(x, y, 0, Nx)] - state_vars->phi[phi_index(x, y, Nc - 1, Nx)];
            vmg = state_vars->phi[phi_index(x, y, 1, Nx)] - state_vars->phi[phi_index(x, y, Nc - 1, Nx)];
            vmgp = state_vars_past->phi[phi_index(x, y, 1, Nx)] - state_vars_past->phi[phi_index(x, y, Nc - 1, Nx)];

            //Compute Na Channel currents
            ci = state_vars->c[c_index(x, y, 0, 0, Nx)];
            cg = state_vars->c[c_index(x, y, 1, 0, Nx)];
            ce = state_vars->c[c_index(x, y, Nc - 1, 0, Nx)];

            //Neurons
            pGHK = con_vars->pNaT[xy_index(xi,yi,Nx)] * gvars->gNaT[xy_index(x, y, Nx)] +
                    con_vars->pNaP[xy_index(xi,yi,Nx)] * gvars->gNaP[xy_index(x, y, Nx)]+
                    con_vars->pNMDA[xy_index(xi,yi,Nx)] * gvars->gNMDA[xy_index(x,y,Nx)]*(2.0/3);
            pLin = con_vars->pNaLeak[xy_index(xi,yi,Nx)] + gexct->pNa[xy_index(x, y, Nx)]; //Add excitation
            //Initialize GHK Flux
            mcGoldman(flux, c_index(x, y, 0, 0, Nx), pGHK, 1, ci, ce, vm, 0);
            //Add leak current to that.
            mclin(flux, c_index(x, y, 0, 0, Nx), pLin, 1, ci, ce, vm, 1);
            //Glial NaLeak
            mclin(flux, c_index(x, y, 1, 0, Nx), con_vars->pNaLeakg[xy_index(xi,yi,Nx)], 1, cg, ce, vmg, 0);

            // Compute K channel Currents
            ci = state_vars->c[c_index(x, y, 0, 1, Nx)];
            cg = state_vars->c[c_index(x, y, 1, 1, Nx)];
            ce = state_vars->c[c_index(x, y, Nc - 1, 1, Nx)];

            //Neurons
            pGHK = con_vars->pKDR[xy_index(xi,yi,Nx)] * gvars->gKDR[xy_index(x, y, Nx)] +
                    con_vars->pKA[xy_index(xi,yi,Nx)] * gvars->gKA[xy_index(x, y, Nx)] +
                    con_vars->pNMDA[xy_index(xi,yi,Nx)] * gvars->gNMDA[xy_index(x,y,Nx)]*(1.0/3);
            pLin = pKLeak + gexct->pK[xy_index(x, y, Nx)]; //add excitation
            mcGoldman(flux, c_index(x, y, 0, 1, Nx), pGHK, 1, ci, ce, vm, 0);
            mclin(flux, c_index(x, y, 0, 1, Nx), pLin, 1, ci, ce, vm, 1);

            // Glial K Leak(using past)
            cgp = state_vars_past->c[c_index(x, y, 1, 1, Nx)];
            cep = state_vars_past->c[c_index(x, y, Nc - 1, 1, Nx)];

            pLin = con_vars->pKIR[xy_index(xi,yi,Nx)] * inwardrect(cgp, cep, vmgp) * pKLeakadjust;
            mclin(flux, c_index(x, y, 1, 1, Nx), pLin, 1, cg, ce, vmg, 0);

            //Compute Cl Channel Current
            ci = state_vars->c[c_index(x, y, 0, 2, Nx)];
            cg = state_vars->c[c_index(x, y, 1, 2, Nx)];
            ce = state_vars->c[c_index(x, y, Nc - 1, 2, Nx)];

            //Neurons
            pLin = pClLeak + gexct->pCl[xy_index(x, y, Nx)]; //add excitation
            mclin(flux, c_index(x, y, 0, 2, Nx), pLin, -1, ci, ce, vm, 0);

            //Glia
            mclin(flux, c_index(x, y, 1, 2, Nx), pClLeakg, -1, cg, ce, vmg, 0);

            //Pump Currents(all past values)

            //Neurons
            Na = state_vars_past->c[c_index(x, y, 0, 0, Nx)];
            K = state_vars_past->c[c_index(x, y, Nc - 1, 1, Nx)];

            Ipump = npump * con_vars->Imax[xy_index(xi,yi,Nx)] / (pow(1 + mK / K, 2) * pow(1 + mNa / Na, 3));

            //Add to flux(it's explicit so no derivatives)
            flux->mflux[c_index(x, y, 0, 0, Nx)] += 3 * Ipump; //Na part
            flux->mflux[c_index(x, y, 0, 1, Nx)] -= 2 * Ipump; //K part

            //Glia
            Na = state_vars_past->c[c_index(x, y, 1, 0, Nx)];
            //K is the same(extracellular)
            Ipump = glpump * con_vars->Imaxg[xy_index(xi,yi,Nx)] / (pow(1 + mK / K, 2) * pow(1 + mNa / Na, 3));
            //Add to flux(it's explicit so no derivatives)
            flux->mflux[c_index(x, y, 1, 0, Nx)] += 3 * Ipump; //Na part
            flux->mflux[c_index(x, y, 1, 1, Nx)] -= 2 * Ipump; //K part

            //NaKCl Cotransporter
            //I'm going to reuse variables names...
            Na = state_vars_past->c[c_index(x, y, 1, 0, Nx)];//glia Na
            K = state_vars_past->c[c_index(x, y, 1, 1, Nx)]; // glia K.
            cgp = state_vars_past->c[c_index(x, y, 1, 2, Nx)]; //glia Cl

            cep = state_vars_past->c[c_index(x, y, Nc - 1, 0, Nx)];//Ext Na
            ce = state_vars_past->c[c_index(x, y, Nc - 1, 1, Nx)]; // Ext K.
            ci = state_vars_past->c[c_index(x, y, Nc - 1, 2, Nx)]; //Ext Cl

            NaKCl = con_vars->pNaKCl[xy_index(xi,yi,Nx)] * log(Na * K * pow(cgp, 2) / (cep * ce * pow(ci, 2)));
            //Add to flux
            flux->mflux[c_index(x, y, 1, 0, Nx)] += NaKCl; //Sodium
            flux->mflux[c_index(x, y, 1, 1, Nx)] += NaKCl; //K
            flux->mflux[c_index(x, y, 1, 2, Nx)] += 2 * NaKCl; //Cl

            //Glutamate transport
            if(Ni>3){
                glutamate_flux(flux,x,y,state_vars,state_vars_past,Nx,0);
            }


            //Change units of flux from mmol/cm^2 to mmol/cm^3/s
            for (PetscInt ion = 0; ion < Ni; ion++) {
                flux->mflux[c_index(x, y, Nc - 1, ion, Nx)] = 0;
                for (PetscInt comp = 0; comp < Nc - 1; comp++) {
                    flux->mflux[c_index(x, y, comp, ion, Nx)] = flux->mflux[c_index(x, y, comp, ion, Nx)] / ell;
                    flux->dfdci[c_index(x, y, comp, ion, Nx)] = flux->dfdci[c_index(x, y, comp, ion, Nx)] / ell;
                    flux->dfdce[c_index(x, y, comp, ion, Nx)] = flux->dfdce[c_index(x, y, comp, ion, Nx)] / ell;
                    flux->dfdphim[c_index(x, y, comp, ion, Nx)] = flux->dfdphim[c_index(x, y, comp, ion, Nx)] / ell;

                    //And calculate the extracellular flux
                    flux->mflux[c_index(x, y, Nc - 1, ion, Nx)] -= flux->mflux[c_index(x, y, comp, ion, Nx)];
                }

            }
        }
    }
    if(Profiling_on) {
        PetscLogEventEnd(event[5], 0, 0, 0, 0);
    }
}

void gatevars_update_grid(struct GateType *gate_vars,struct SimState *state_vars,PetscReal dtms,struct AppCtx *user)
{
    if(Profiling_on) {
        PetscLogEventBegin(event[4], 0, 0, 0, 0);
    }
    PetscInt Nx = 2*width_size+1;
    PetscInt Ny = 2*width_size+1;

    PetscReal v, alpha,beta,Gphi;
    int Num_NMDA_States = 4;
    PetscReal a11,a12,a21,a22,r6,detA;
    PetscReal k1,k2,k3,k4,k5,k6,Fglu,npow,K_r,Rstar,D1,D2,S;
    for(PetscInt x=0;x<Nx;x++) {
        for (PetscInt y = 0; y < Ny; y++) {
            //membrane potential in mV
            v = (state_vars->phi[phi_index(x, y, 0, Nx)] - state_vars->phi[phi_index(x, y, Nc - 1, Nx)]) * RTFC;

            //compute current NaT
            //gating variables mNaT
            alpha = xoverexpminusone(v, 0.32, 51.9, 0.25, 1); //0.32*(Vm+51.9)./(1-exp(-0.25*(Vm+51.9)))
            beta = xoverexpminusone(v, 0.28, 24.89, 0.2, 0); //0.28*(Vm+24.89)./(exp(0.2*(Vm+24.89))-1)
            gate_vars->mNaT[xy_index(x, y, Nx)] =
                    (gate_vars->mNaT[xy_index(x, y, Nx)] + alpha * dtms) / (1 + (alpha + beta) * dtms);

            //gating variable hNaT
            alpha = 0.128 * exp(-(0.056 * v + 2.94));
            beta = 4 / (exp(-(0.2 * v + 6)) + 1);
            gate_vars->hNaT[xy_index(x, y, Nx)] = alpha / (alpha + beta);
            gate_vars->hNaT[xy_index(x, y, Nx)] =
                    (gate_vars->hNaT[xy_index(x, y, Nx)] + alpha * dtms) / (1 + (alpha + beta) * dtms);

            gate_vars->gNaT[xy_index(x, y, Nx)] =
                    pow(gate_vars->mNaT[xy_index(x, y, Nx)], 3) * gate_vars->hNaT[xy_index(x, y, Nx)];
            //compute current NaP
            //gating variable mNaP
            alpha = 1 / (1 + exp(-(0.143 * v + 5.67))) / 6;
            beta = 1.0 / 6.0 - alpha; //1./(1+exp(0.143*Vm+5.67))/6
            gate_vars->mNaP[xy_index(x, y, Nx)] =
                    (gate_vars->mNaP[xy_index(x, y, Nx)] + alpha * dtms) / (1 + (alpha + beta) * dtms);

            //gating variable hNaP
            alpha = 5.12e-6 * exp(-(0.056 * v + 2.94));
            beta = 1.6e-4 / (1 + exp(-(0.2 * v + 8)));
            gate_vars->hNaP[xy_index(x, y, Nx)] =
                    (gate_vars->hNaP[xy_index(x, y, Nx)] + alpha * dtms) / (1 + (alpha + beta) * dtms);

            gate_vars->gNaP[xy_index(x, y, Nx)] =
                    pow(gate_vars->mNaP[xy_index(x, y, Nx)], 2) * gate_vars->hNaP[xy_index(x, y, Nx)];

            //compute KDR current
            //gating variable mKDR
            alpha = xoverexpminusone(v, 0.016, 34.9, 0.2, 1); //0.016*(Vm+34.9)./(1-exp(-0.2*(Vm+34.9)))
            beta = 0.25 * exp(-(0.025 * v + 1.25));
            gate_vars->mKDR[xy_index(x, y, Nx)] =
                    (gate_vars->mKDR[xy_index(x, y, Nx)] + alpha * dtms) / (1 + (alpha + beta) * dtms);

            gate_vars->gKDR[xy_index(x, y, Nx)] = pow(gate_vars->mKDR[xy_index(x, y, Nx)], 2);

            //compute KA current
            //gating variable mKA
            alpha = xoverexpminusone(v, 0.02, 56.9, 0.1, 1); //0.02*(Vm+56.9)./(1-exp(-0.1*(Vm+56.9)))
            beta = xoverexpminusone(v, 0.0175, 29.9, 0.1, 0); //0.0175*(Vm+29.9)./(exp(0.1*(Vm+29.9))-1)
            gate_vars->mKA[xy_index(x, y, Nx)] =
                    (gate_vars->mKA[xy_index(x, y, Nx)] + alpha * dtms) / (1 + (alpha + beta) * dtms);

            //gating variable hKA
            alpha = 0.016 * exp(-(0.056 * v + 4.61));
            beta = 0.5 / (exp(-(0.2 * v + 11.98)) + 1);
            gate_vars->hKA[xy_index(x, y, Nx)] =
                    (gate_vars->hKA[xy_index(x, y, Nx)] + alpha * dtms) / (1 + (alpha + beta) * dtms);

            gate_vars->gKA[xy_index(x, y, Nx)] =
                    pow(gate_vars->mKA[xy_index(x, y, Nx)], 2) * gate_vars->hKA[xy_index(x, y, Nx)];

            //gating variable NMDA
            if(Ni>3){
                if(Num_NMDA_States==2){
                    alpha = 72*state_vars->c[c_index(x,y,Nc-1,3,Nx)];
                    beta = 6.6e-3; // 6.6 (sec)^-1->6.6e-3 msec^-1
                    gate_vars->yNMDA[xy_index(x,y,Nx)] =
                            (gate_vars->yNMDA[xy_index(x,y,Nx)]+alpha*dtms)/(1+(alpha+beta)*dtms);

//                Gphi = 1/(1+0.28*exp(-0.062*v)); //Other gating "variable" given by just this.
                    Gphi = 1/(1+0.56*exp(-0.062*v)); //From Rossi/Atwell

                    gate_vars->gNMDA[xy_index(x,y,Nx)] = gate_vars->yNMDA[xy_index(x,y,Nx)]*Gphi;
                }
                if(Num_NMDA_States==3){
                    // 3 Stage Solve
//                    r=[0,6.9e-3,0,160e-3,4.7e-3,190e-3]
                    r6 = 190.0;//190e-3;
                    a11 = 1+dtms*6.9e-3;
                    a12 = -dtms*160e-3;
                    a21 = dtms*(r6*state_vars->c[c_index(x,y,Nc-1,3,Nx)]);
                    a22 = 1+dtms*(r6*state_vars->c[c_index(x,y,Nc-1,3,Nx)]+160e-3+4.7e-3);

                    Rstar=gate_vars->yNMDA[xy_index(x,y,Nx)];
                    D1=gate_vars->zNMDA[xy_index(x,y,Nx)];

                    detA = a11*a22-a12*a21;
                    gate_vars->yNMDA[xy_index(x,y,Nx)] = (a22*Rstar
                                                          -a12*(D1+dtms*r6*state_vars->c[c_index(x,y,Nc-1,3,Nx)]));
                    gate_vars->zNMDA[xy_index(x,y,Nx)] = (-a21*Rstar
                                                          +a11*(D1+dtms*r6*state_vars->c[c_index(x,y,Nc-1,3,Nx)]));

                    gate_vars->yNMDA[xy_index(x,y,Nx)] /= detA;
                    gate_vars->zNMDA[xy_index(x,y,Nx)] /= detA;
//                    Gphi = 1/(1+0.28*exp(-0.062*v));
                    Gphi = 1.0/(1+0.56*exp(-0.062*v));
                    gate_vars->gNMDA[xy_index(x,y,Nx)] = gate_vars->yNMDA[xy_index(x,y,Nx)]*Gphi;
                }

                // 4 Stages
                if(Num_NMDA_States==4){
                    K_r = 2.3e-6;//34.9e-6;
                    npow = 1.5; //1.4;
                    Fglu = pow(state_vars->c[c_index(x,y,Nc-1,3,Nx)],npow)
                           /(pow(state_vars->c[c_index(x,y,Nc-1,3,Nx)],npow)+pow(K_r,npow));
                    k1 = 3.94e-3*Fglu;
                    k2 = 1.94e-3;
                    k3 = 0.0213e-3;
                    k4 = 0.00277e-3;

                    // Calculate determinant
                    detA = dtms*k1+dtms*k2+dtms*k3+dtms*k4+dtms*dtms*k1*k3+dtms*dtms*k1*k4+dtms*dtms*k2*k4+1;


                    Rstar = gate_vars->yNMDA[xy_index(x,y,Nx)];
                    D1 = gate_vars->zNMDA[xy_index(x,y,Nx)];
                    D2 = gate_vars->dNMDA[xy_index(x,y,Nx)];
                    // Explicitly calculate inverse
                    // First row
                    a11 = dtms*k2+dtms*k3+dtms*k4+dtms*dtms*k2*k4+1;
                    a12 = dtms*k2*(dtms*k4+1);
                    a21 = dtms*dtms*k2*k4;

                    gate_vars->yNMDA[xy_index(x,y,Nx)] = (a11*Rstar+a12*D1+a21*D2)/detA;

                    // Second Row
                    a11 = dtms*k1*(dtms*k4+1);
                    a12 = (dtms*k1+1)*(dtms*k4+1);
                    a21 = dtms*k4*(dtms*k1+1);
                    gate_vars->zNMDA[xy_index(x,y,Nx)] = (a11*Rstar+a12*D1+a21*D2)/detA;

                    //Third Row
                    a11 = dtms*dtms*k1*k3;
                    a12 = dtms*k3*(dtms*k1+1);
                    a21 = dtms*k1+dtms*k2+dtms*k3+dtms*dtms*k1*k3+1;
                    gate_vars->dNMDA[xy_index(x,y,Nx)] = (a11*Rstar+a12*D1+a21*D2)/detA;


                    Gphi = 1/(1+0.56*exp(-0.062*v));
                    gate_vars->gNMDA[xy_index(x,y,Nx)] = (gate_vars->yNMDA[xy_index(x,y,Nx)]*Fglu
                                                          +Desensitize[0]*gate_vars->zNMDA[xy_index(x,y,Nx)]
                                                          +Desensitize[1]*gate_vars->dNMDA[xy_index(x,y,Nx)])*Gphi;
                }
            }

        }
    }
    if(Profiling_on) {
        PetscLogEventEnd(event[4], 0, 0, 0, 0);
    }
}
void excitation_grid(struct AppCtx* user,PetscReal t,PetscInt xi,PetscInt yi)
{
    //compute excitation conductance to trigger csd
    //Leak conductances in mS/cm^2
    //all units converted to mmol/cm^2/sec
    PetscReal pexct,pany;
    PetscReal xexct;
    PetscReal radius;
    struct ExctType *exct = user->gexct;
    PetscInt Nx = 2*width_size+1;
    PetscInt Ny = 2*width_size+1;
    PetscReal dx = user->dx;
    PetscReal dy = user->dy;


    for (PetscInt i = 0; i < Nx; i++){
        for(PetscInt j = 0; j < Ny; j++){
            if(i+xi == 0 && j+yi == 0){
                pany = pmax*pow(sin(pi*t/texct),2)*RTFC/FC;
                exct->pNa[xy_index(i,j,Nx)] = pany;
                exct->pK[xy_index(i,j,Nx)] = pany;
                exct->pCl[xy_index(i,j,Nx)] = pany;
            }

        }
    }

//    printf("Number of excited points: %d\n",num_points);
}
void grid_diff_coef(PetscReal *Dc,const PetscReal *alp,PetscReal scale,struct AppCtx* user,PetscInt xi,PetscInt yi)
{
    //diffusion coefficients at all points, for all ions, in all compartments, in both x and y directions
    PetscReal tortuosity=1.6;
    struct ConstVars *con_vars = user->con_vars;
    PetscInt Nx = 2*width_size+1;
    PetscInt Ny = 2*width_size+1;
    PetscReal alNcL,alNcR,alNcU;
    for(PetscInt x=0;x<Nx;x++) {
        for(PetscInt y=0;y<Ny;y++) {
            alNcL=1-alp[al_index(x,y,0,Nx)]-alp[al_index(x,y,1,Nx)]; //Left extracell
            alNcR = 0;
            if(x<Nx-1) {
                alNcR = 1 - alp[al_index(x + 1, y, 0,Nx)] - alp[al_index(x + 1, y, 1,Nx)]; //Right extracell
            }
            alNcU = 0;
            if(y<Ny-1) {
                alNcU = 1 - alp[al_index(x, y + 1, 0,Nx)] - alp[al_index(x, y + 1, 1,Nx)];
            }
            for(PetscInt ion = 0; ion<Ni;ion++) {
                //diffusion coefficients in x direction
                if(x==(Nx-1)) {
                    //Boundary is zero
                    Dc[c_index(x,y,Nc-1,ion,Nx)*2] = con_vars->DExtracellScale[xy_index(x+xi,y+yi,Nx)*2]*scale*D[ion]*(alNcL)/(tortuosity*tortuosity);
                } else {
                    //diffusion coefficients in the extracellular space proportional to volume fraction
                    Dc[c_index(x,y,Nc-1,ion,Nx)*2] = con_vars->DExtracellScale[xy_index(x+xi,y+yi,Nx)*2]*scale*D[ion]/2*(alNcL+alNcR)/(tortuosity*tortuosity);
                }
                //diffusion coefficients in neuronal compartment equal to 0
                Dc[c_index(x,y,0,ion,Nx)*2] =  con_vars->DNeuronScale[xy_index(x+xi,y+yi,Nx)*2]*scale*D[ion]*alphao[al_index(0,0,0,Nx)]/pow(tortuosity,2);
                //diffusion coefficients in glial compartment proportional to default volume fraction
                Dc[c_index(x,y,1,ion,Nx)*2] = con_vars->DGliaScale[xy_index(x+xi,y+yi,Nx)*2]*scale*D[ion]*alphao[al_index(0,0,1,Nx)]/pow(tortuosity,2);
                //diffusion coefficients in y direction
                if(y==(Ny-1)) {
                    //Boundary is zero
                    Dc[c_index(x,y,Nc-1,ion,Nx)*2+1] = con_vars->DExtracellScale[xy_index(x+xi,y+yi,Nx)*2+1]*scale*D[ion]*(alNcL)/pow(tortuosity,2);
                } else {
                    //diffusion coefficients in the extracellular space proportional to volume fraction
                    Dc[c_index(x,y,Nc-1,ion,Nx)*2+1] = con_vars->DExtracellScale[xy_index(x+xi,y+yi,Nx)*2+1]*scale*D[ion]/2*(alNcL+alNcU)/pow(tortuosity,2);

                }
                //diffusion coefficients in neuronal compartment equal to 0
                Dc[c_index(x,y,0,ion,Nx)*2+1] = con_vars->DNeuronScale[xy_index(x+xi,y+yi,Nx)*2+1]*scale*D[ion]*alphao[al_index(0,0,0,Nx)]/pow(tortuosity,2);
                //diffusion coefficients in glial compartment proportional to default volume fraction
//			    Dc[c_index(x,y,1,ion,Nx)*2+1] = 0.25*scale*D[ion]*alphao[al_index(0,0,1,Nx)]/pow(tortuosity,2); //0.25
                Dc[c_index(x,y,1,ion,Nx)*2+1] = con_vars->DGliaScale[xy_index(x+xi,y+yi,Nx)*2+1]*scale*D[ion]*alphao[al_index(0,0,1,Nx)]/pow(tortuosity,2);

            }
        }
    }
}


