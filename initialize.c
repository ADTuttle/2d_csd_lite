#include "constants.h"
#include "functions.h"


#include <stdlib.h>
#include <stdio.h>
#include <math.h>



void init(Vec state,struct SimState *state_vars,struct AppCtx*user)
{
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    extract_subarray(state,state_vars);
    for(PetscInt x=0;x<Nx;x++) {
        for(PetscInt y=0;y<Ny;y++) {
            //initial volume fractions
            state_vars->alpha[al_index(x,y,0,Nx)]=alphao[0];
            state_vars->alpha[al_index(x,y,1,Nx)]=alphao[1];
            //initial voltages (dimensionless)
            state_vars->phi[phi_index(x,y,0,Nx)] = -70/RTFC; //neuronal voltage
            state_vars->phi[phi_index(x,y,1,Nx)] = -85/RTFC; //glial voltage
            state_vars->phi[phi_index(x,y,2,Nx)] = -0/RTFC; //extracell voltage
            //initial concentrations in mmol/cm^3=1e-3 mmol/l
            state_vars->c[c_index(x,y,0,0,Nx)] = 10e-3;     //neuronal Na concentration
            state_vars->c[c_index(x,y,1,0,Nx)] = 10e-3;      //glial Na concentration
            state_vars->c[c_index(x,y,2,0,Nx)] = 140e-3;     //extracellular Na concentration
            state_vars->c[c_index(x,y,0,1,Nx)] = 130e-3;     //neuronal K concentration
            state_vars->c[c_index(x,y,1,1,Nx)] = 130e-3;     //glial K concentration
            state_vars->c[c_index(x,y,2,1,Nx)] = 3.4e-3;     //extracellular K concentration
            state_vars->c[c_index(x,y,0,2,Nx)] = 10e-3;       //neuronal Cl concentration
            state_vars->c[c_index(x,y,1,2,Nx)] = 10e-3; 		//glial Cl concentraion
            state_vars->c[c_index(x,y,2,2,Nx)] = 120e-3;       //143.5e-3%extracellular Cl

            //glutamage taken from K. Moussawi, A. Riegel, et al
            if(Ni>3){
                state_vars->c[c_index(x,y,0,3,Nx)] = 10e-3;//10e-5; //10e-3;   //glutamate concentrations
                state_vars->c[c_index(x,y,1,3,Nx)] = (10e-3*1e-3); //(10.0/6)*1e-3; //1e-10; //1e-5; //10e-3;  //glial glu
                state_vars->c[c_index(x,y,Nc-1,3,Nx)] = 1e-8;//2.8089e-05;;//3.6227e-13;//2.2411e-10;//1e-10;//2e-8;    //.02 muM-> 2e-5mM or .1muM ->10e-5
            }
        }
    }
    restore_subarray(state,state_vars);
}


void set_params(Vec state,struct SimState* state_vars,struct ConstVars* con_vars,struct GateType* gate_vars,struct FluxData *flux,struct AppCtx*user)
{
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    PetscReal vm,vmg,NaKCl,pKGHK,pNaGHK,Ipump,pKLinG,Ipumpg,osmotic,alNc,zetaadjust;
    extract_subarray(state,state_vars);
    PetscReal *c = state_vars->c;
    PetscReal *phi = state_vars->phi;
    PetscReal *alpha = state_vars->alpha;
    PetscReal cmphi[Nc]={0,0,0}; //initializing cmphi

    // Compute Chloride concentration
    for(PetscInt x=0;x<Nx;x++){
        for(PetscInt y=0;y<Ny;y++){
            vm = phi[phi_index(x, y, 0, Nx)] - phi[phi_index(x, y, 2, Nx)]; //neuronal membrane potential
            vmg = phi[phi_index(x, y, 1, Nx)] - phi[phi_index(x, y, 2, Nx)]; //glial membrane potential

            //compute neuronal Cl concentration (since neuron has only leak conductance, must be at reversal potential for Cl)

            c[c_index(x, y, 0, 2, Nx)] = c[c_index(x, y, 2, 2, Nx)] * exp(vm);
            //set glial Cl concentration equal to neuronal Cl concentration
            c[c_index(x, y, 1, 2, Nx)] = c[c_index(x, y, 0, 2, Nx)];
        }
    }

    //compute gating variables
    gatevars_update(gate_vars, gate_vars, state_vars, 0, user, 1);


    for(PetscInt x=0;x<Nx;x++) {
        for (PetscInt y = 0; y < Ny; y++) {
            vm = phi[phi_index(x, y, 0, Nx)] - phi[phi_index(x, y, 2, Nx)]; //neuronal membrane potential
            vmg = phi[phi_index(x, y, 1, Nx)] - phi[phi_index(x, y, 2, Nx)]; //glial membrane potential

            //compute cotransporter permeability so that glial Cl is at rest
            mclin(flux, c_index(x, y, 1, 2, Nx),pClLeakg, -1,c[c_index(x, y, 1, 2, Nx)],c[c_index(x, y, 2, 2, Nx)],vmg, 0);
            con_vars->pNaKCl[xy_index(x,y,Nx)] =-flux->mflux[c_index(x, y, 1, 2, Nx)]/2/
                                                log(c[c_index(x, y, 1, 0, Nx)] * c[c_index(x, y, 1, 1, Nx)] *
                                                    c[c_index(x, y, 1, 2, Nx)] * c[c_index(x, y, 1, 2, Nx)] /
                                                    (c[c_index(x, y, 2, 0, Nx)] * c[c_index(x, y, 2, 1, Nx)] *
                                                     c[c_index(x, y, 2, 2, Nx)] * c[c_index(x, y, 2, 2, Nx)]));
            NaKCl = -flux->mflux[c_index(x, y, 1, 2, Nx)] / 2;

            //compute K channel currents (neuron)
            pKGHK = con_vars->pKDR[xy_index(x,y,Nx)] * gate_vars->gKDR[xy_index(x,y,Nx)] +
                    con_vars->pKA[xy_index(x,y,Nx)] * gate_vars->gKA[xy_index(x,y,Nx)]+
                    con_vars->pNMDA[xy_index(x,y,Nx)] * gate_vars->gNMDA[xy_index(x,y,Nx)]*(1.0/3);
            //Initialize the KGHK flux
            mcGoldman(flux, c_index(x, y, 0, 1, Nx), pKGHK, 1, c[c_index(x, y, 0, 1, Nx)],
                      c[c_index(x, y, Nc - 1, 1, Nx)], vm, 0);
            //Add the KLeak flux to it
            mclin(flux, c_index(x, y, 0, 1, Nx), pKLeak, 1, c[c_index(x, y, 0, 1, Nx)], c[c_index(x, y, Nc - 1, 1, Nx)],
                  vm, 1);

            //compute neuronal ATPase value
            con_vars->Imax[xy_index(x,y,Nx)] = flux->mflux[c_index(x, y, 0, 1, Nx)] * (pow(1 + mK / c[c_index(x, y, Nc - 1, 1, Nx)], 2) *
                                                                                       pow(1 + mNa / c[c_index(x, y, 0, 0, Nx)], 3)) / 2;


            //compute neuronal sodium currents and leak permeability value
            pNaGHK = con_vars->pNaT[xy_index(x,y,Nx)]*gate_vars->gNaT[xy_index(x,y,Nx)] +
                    con_vars->pNaP[xy_index(x,y,Nx)]*gate_vars->gNaP[xy_index(x,y,Nx)]+
                    con_vars->pNMDA[xy_index(x,y,Nx)] * gate_vars->gNMDA[xy_index(x,y,Nx)]*(2.0/3);
            mcGoldman(flux, c_index(x, y, 0, 0, Nx), pNaGHK, 1, c[c_index(x, y, 0, 0, Nx)],
                      c[c_index(x, y, Nc - 1, 0, Nx)], vm, 0);
            Ipump = npump * con_vars->Imax[xy_index(x,y,Nx)] / (pow((1 + mK / c[c_index(x, y, Nc - 1, 1, Nx)]), 2) *
                                                                pow((1 + mNa / c[c_index(x, y, 0, 0, Nx)]), 3));
            con_vars->pNaLeak[xy_index(x,y,Nx)] = (-flux->mflux[c_index(x, y, 0, 0, Nx)] - 3 * Ipump) /
                                                  (log(c[c_index(x, y, 0, 0, Nx)] / c[c_index(x, y, Nc - 1, 0, Nx)]) + vm);

            //compute K channel currents (glial)
            pKLinG = con_vars->pKIR[xy_index(x,y,Nx)] * inwardrect(c[c_index(x, y, 1, 1, Nx)], c[c_index(x, y, Nc - 1, 1, Nx)], vmg) * pKLeakadjust;
            mclin(flux, c_index(x, y, 1, 1, Nx), pKLinG, 1, c[c_index(x, y, 1, 1, Nx)], c[c_index(x, y, Nc - 1, 1, Nx)],
                  vmg, 0);
            flux->mflux[c_index(x, y, 1, 1, Nx)] += NaKCl;

            //compute glial ATPase value
            con_vars->Imaxg[xy_index(x,y,Nx)] =
                    flux->mflux[c_index(x, y, 1, 1, Nx)] * pow((1 + mK / c[c_index(x, y, Nc - 1, 1, Nx)]), 2) *
                    pow((1 + mNa / c[c_index(x, y, 1, 0, Nx)]), 3) / 2;

            //compute glial sodium current and leak permeability value
            Ipumpg = glpump * con_vars->Imaxg[xy_index(x,y,Nx)] / (pow((1 + mK / c[c_index(x, y, Nc - 1, 1, Nx)]), 2) *
                                                                   pow((1 + mNa / c[c_index(x, y, 1, 0, Nx)]), 3));
            con_vars->pNaLeakg[xy_index(x,y,Nx)] =
                    (-NaKCl - 3 * Ipumpg) / (log(c[c_index(x, y, 1, 0, Nx)] / c[c_index(x, y, Nc - 1, 0, Nx)]) + vmg);

            //Compute resting organic anion amounts and average valences
            //set extracellular organic anion amounts and valence to ensure electroneutrality
            con_vars->ao[phi_index(x,y,Nc-1,Nx)] = 5e-4;
            alNc = 1- alpha[al_index(x,y,0,Nx)]-alpha[al_index(x,y,1,Nx)];
            cmphi[Nc-1] = 0; //initializing extracell cmphi

            for (PetscInt k = 0; k < Nc - 1; k++) {
                cmphi[k] = cm[k] * (phi[phi_index(x,y,k,Nx)] - phi[phi_index(x,y,Nc-1,Nx)]);
                cmphi[Nc - 1] += cmphi[k];
                //set intracellular organic anion amounts to ensure osmotic pressure balance
                osmotic = 0;
                for (PetscInt ion = 0; ion < Ni; ion++) {
                    osmotic += c[c_index(x, y, Nc - 1, ion, Nx)] - c[c_index(x, y, k, ion, Nx)];
                }
                con_vars->ao[phi_index(x,y,k,Nx)] = alpha[al_index(x,y,k,Nx)]
                        * (con_vars->ao[phi_index(x,y,Nc-1,Nx)] / alNc + osmotic);
                //set average valence to ensure electroneutrality
                con_vars->zo[phi_index(x,y,k,Nx)] = (-cz(c, z, x, y, Nx, k, user)
                        * alpha[al_index(x,y,k,Nx)] + cmphi[k]) / con_vars->ao[phi_index(x,y,k,Nx)];
            }
            con_vars->zo[phi_index(x,y,Nc-1,Nx)]=(-cz(c,z,x,y,Nx,Nc-1,user)*alNc - cmphi[Nc - 1])
                    / con_vars->ao[phi_index(x,y,Nc-1,Nx)];

            //Set kappa to 0 for no flow
            con_vars->kappa = 0;


            //parameters for osmotic water flow

            zetaadjust = 1; //modify glial permeability
            for (PetscInt comp = 0; comp < Nc - 1; comp++) {
                //based on B.E. Shapiro dissertation (2000)
                con_vars->zeta1[al_index(x,y,comp,Nx)] = 5.4e-5;  //hydraulic permeability in cm/sec/(mmol/cm^3)
                con_vars->zeta1[al_index(x,y,comp,Nx)] /= ell;  //conversion to 1/sec/(mmol/cm^3)
                //based on Strieter, Stephenson, Palmer,
                //Weinstein, Journal or General Physiology, 1990.
                //zeta=7e-8%6e-10%hydraulic permeability in cm/sec/mmHg
                //zeta=zeta*7.501e-6%conversion to cm/sec/mPa
                //zeta=zeta*R*T%conversion to cm/sec/(mmol/cm^3)
                //zeta=zeta/ell%conversion to 1/sec/(mmol/cm^3)
                if (comp == 1) {          //parameter for varying glial hydraulic permeability
                    con_vars->zeta1[al_index(x,y,comp,Nx)] *= zetaadjust; //adjust glial hydraulic permeability
                }
                con_vars->zetaalpha[comp] = 0;  //stiffness constant or 1/stiffness constant
            }

            con_vars->S = 1;  //Indicates whether zetaalpha is the stiffness (true) or 1/stiffness (false)
        }
    }

    restore_subarray(state,state_vars);
}

void initialize_data(Vec current_state,struct AppCtx *user)
{

    //Make a temp solver for just a 1x1 grid for speed
    PetscInt temp_Nx = user->Nx;
    PetscInt temp_Ny = user->Ny;
    user->Nx = 1;
    user->Ny = 1;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;

    struct Solver *slvr = (struct Solver*)malloc(sizeof(struct Solver));
    //Create Vectors
    VecCreate(PETSC_COMM_WORLD,&slvr->Q);
    VecSetType(slvr->Q,VECSEQ);
    VecSetSizes(slvr->Q,PETSC_DECIDE,Nx*Ny*Nv);
    VecDuplicate(slvr->Q,&slvr->Res);
    MatCreate(PETSC_COMM_WORLD,&slvr->A);
    MatSetType(slvr->A,MATSEQAIJ);
    MatSetSizes(slvr->A,PETSC_DECIDE,PETSC_DECIDE,Nx*Ny*Nv,Nx*Ny*Nv);
    MatSeqAIJSetPreallocation(slvr->A,5*Nv,NULL);
    MatSetUp(slvr->A);
    initialize_jacobian(slvr->A,user,0);
    KSPCreate(PETSC_COMM_WORLD,&slvr->ksp);
    KSPSetType(slvr->ksp,KSPPREONLY);
    KSPGetPC(slvr->ksp,&slvr->pc);
    PCSetType(slvr->pc,PCLU);
//    PCFactorSetMatSolverPackage(slvr->pc, MATSOLVERSUPERLU);
    PCFactorSetMatSolverType(slvr->pc, MATSOLVERSUPERLU);



    PetscReal convtol = 1e-9;
    extract_subarray(current_state,user->state_vars);
    PetscReal tol =convtol;
    PetscReal rsd = 1.0;
    PetscReal rsd_v[3];
    //Compute Gating variables
    //compute gating variables
    gatevars_update(user->gate_vars,user->gate_vars,user->state_vars,0,user,1);
    gatevars_update(user->gate_vars_past,user->gate_vars_past,user->state_vars,0,user,1);
    restore_subarray(current_state,user->state_vars);

    //Initialize and compute the excitation (it's zeros here)
    excitation(user,-1.0);
    PetscReal dt_temp = user->dt;
    PetscInt k = 0;
    user->dt = 0.01;

    // For 1x1 grid Dcs is zeros
    memset(user->Dcs,0,sizeof(PetscReal)*2*temp_Nx*temp_Ny*Nc*Ni);

    while(rsd>tol && k<1e5)
    {
        extract_subarray(current_state,user->state_vars);
        //Save the "current" aka past state
        restore_subarray(user->state_vars_past->v, user->state_vars_past);
        copy_simstate(current_state, user->state_vars_past);
        if (separate_vol) {
            memcpy(user->state_vars_past->alpha, user->state_vars->alpha, sizeof(PetscReal) * user->Nx * user->Ny * (Nc - 1));
            //Update volume
            volume_update(user->state_vars, user->state_vars_past, user);
        }
        //compute diffusion coefficients (Dcs is not used for 1x1)
        //Bath diffusion
        diff_coef(user->Dcb,user->state_vars->alpha,Batheps,user);
        restore_subarray(current_state, user->state_vars);

        // Use own solver since we reuse data structs to update a subset
        newton_solve(current_state,slvr,user);
        //Update gating variables
        extract_subarray(current_state,user->state_vars);


        // Set to be "firstpass" (that's the 1)
        // So that we set to alpha/beta infinity values as if it came to rest
        gatevars_update(user->gate_vars,user->gate_vars,user->state_vars,user->dt*1e3,user,1);


        rsd_v[0] = array_diff_max(user->state_vars->c,user->state_vars_past->c,(size_t)Nx*Ny*Nc*Ni);
        rsd_v[1] = array_diff_max(user->state_vars->phi,user->state_vars_past->phi,(size_t)Nx*Ny*Nc);
        rsd_v[2] = array_diff_max(user->state_vars->alpha,user->state_vars_past->alpha,(size_t)Nx*Ny*(Nc-1));
        rsd = array_max(rsd_v,3);
        restore_subarray(current_state,user->state_vars);

        if(details|| k%1000==0) {
            printf("Tol: %.10e: rsd: c: %.10e, phi: %.10e, al: %.10e\n",tol,rsd_v[0],rsd_v[1],rsd_v[2]);
        }
        k++;
    }
    //Save the one set of variables we solved for (x=1,y=1 because that's the midpoint of the 3x3 system)

    extract_subarray(current_state,user->state_vars);
    PetscReal c[Ni*Nc];
    PetscReal phi[Nc];
    PetscReal al[Nc-1];
    PetscInt comp,ion,x,y;
    for(comp=0;comp<Nc;comp++){
        for(ion=0;ion<Ni;ion++){
            c[c_index(0,0,comp,ion,Nx)]=user->state_vars->c[c_index(0,0,comp,ion,Nx)];
        }
        phi[phi_index(0,0,comp,Nx)]=user->state_vars->phi[phi_index(0,0,comp,Nx)];
    }
    for(comp=0;comp<Nc-1;comp++){
        al[al_index(0,0,comp,Nx)] = user->state_vars->alpha[al_index(0,0,comp,Nx)];
    }
    user->dt = dt_temp;
    user->Nx = temp_Nx;
    user->Ny = temp_Ny;

    //Copy over the saved variables.
    for(x=0;x<temp_Nx;x++){
        for(y=0;y<temp_Ny;y++){
            for(comp=0;comp<Nc;comp++){
                for(ion=0;ion<Ni;ion++){
                    user->state_vars->c[c_index(x,y,comp,ion,Nx)]=c[c_index(0,0,comp,ion,Nx)];
                }
                user->state_vars->phi[phi_index(x,y,comp,Nx)]=phi[phi_index(0,0,comp,Nx)];
            }
            for(comp=0;comp<Nc-1;comp++){
                user->state_vars->alpha[al_index(x,y,comp,Nx)]=al[al_index(0,0,comp,Nx)];
            }
        }
    }
    // Reset gating variables to rest values at this voltage.
    gatevars_update(user->gate_vars,user->gate_vars,user->state_vars,0,user,1);
    gatevars_update(user->gate_vars_past,user->gate_vars_past,user->state_vars,0,user,1);

    restore_subarray(current_state,user->state_vars);
    free(slvr);
    printf("Initialization stopped at %.10e in %d iterations\n",rsd,k);
    if(rsd>tol) {
        fprintf(stderr, "Did not converge! Continuing anyway...\n");
//    	exit(EXIT_FAILURE); /* indicate failure.*/
        return;
    } else {
        return;
    }

}


PetscErrorCode initialize_petsc(struct Solver *slvr,int argc, char **argv,struct AppCtx *user)
{
    PetscErrorCode ierr;
    //Init Petsc
    PetscInitialize(&argc,&argv,(char*)0,NULL);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&slvr->size);CHKERRQ(ierr);
    //    Get Nx, Ny, and dt from options if possible

    user->Nx = 32;
    user->Ny = 32;
    user->dt =0.01;
//    user->dt =1e-4;

    PetscOptionsGetInt(NULL,NULL,"-Nx",&user->Nx,NULL);
    PetscOptionsGetInt(NULL,NULL,"-Ny",&user->Ny,NULL);
    PetscOptionsGetReal(NULL,NULL,"-dt",&user->dt,NULL);


    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;

    user->dx = Lx/Nx;
    user->dy = Ly/Ny;
    slvr->NA = Nv*Nx*Ny;//total number of unknowns
    user->Nz = (Ni*Nc*(4*(Nx-1)*Ny+4*(Ny-1)*Nx+2*Nx*Ny)+Ni*(Nc-1)*6*Nx*Ny+(Nc*Ni+1)*Nx*Ny+(Nc-1)*(6*Nx*Ny+Nx*Ny*(Nc-2)+Ni*2*Nx*Ny)); //number of nonzeros in Jacobian

    PetscInt NA = slvr->NA;

    //Create Vectors
    ierr = VecCreate(PETSC_COMM_WORLD,&slvr->Q);CHKERRQ(ierr);
    ierr = VecSetType(slvr->Q,VECSEQ);CHKERRQ(ierr);
    ierr = VecSetSizes(slvr->Q,PETSC_DECIDE,NA);CHKERRQ(ierr);
    ierr = VecDuplicate(slvr->Q,&slvr->Res);CHKERRQ(ierr);

    //Create Matrix
    //Get number of nonzeros in each row
    int *nnz = (int*) malloc(sizeof(int)*NA);
    Get_Nonzero_in_Rows(nnz,user,0);
    //Construct matrix using that
    ierr = MatCreate(PETSC_COMM_WORLD,&slvr->A);CHKERRQ(ierr);
    ierr = MatSetType(slvr->A,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(slvr->A,PETSC_DECIDE,PETSC_DECIDE,NA,NA);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(slvr->A,5*Nv,nnz);CHKERRQ(ierr);
    ierr = MatSetUp(slvr->A);CHKERRQ(ierr);

    //Initialize Space

    ierr = initialize_jacobian(slvr->A,user,0); CHKERRQ(ierr);
    ierr = MatSetOption(slvr->A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE); CHKERRQ(ierr);

    //Create Solver Contexts
    ierr = SNESCreate(PETSC_COMM_WORLD,&slvr->snes); CHKERRQ(ierr);
    ierr = SNESGetKSP(slvr->snes,&slvr->ksp); CHKERRQ(ierr);
    ierr = KSPGetPC(slvr->ksp,&slvr->pc);CHKERRQ(ierr);

    //Choose solver based on constants.h options.
    if(Linear_Diffusion){
        if(use_en_deriv){
            //Set Function eval
            ierr = SNESSetFunction(slvr->snes, slvr->Res, calc_residual_linear_deriv, user);
            CHKERRQ(ierr);
            //Set Jacobian eval
            ierr = SNESSetJacobian(slvr->snes, slvr->A, slvr->A, calc_jacobian_linear_deriv, user);
            CHKERRQ(ierr);
        } else {
            //Set Function eval
            ierr = SNESSetFunction(slvr->snes, slvr->Res, calc_residual_linear_algebraic, user);
            CHKERRQ(ierr);
            //Set Jacobian eval
            ierr = SNESSetJacobian(slvr->snes, slvr->A, slvr->A, calc_jacobian_linear_algebraic, user);
            CHKERRQ(ierr);
        }
    }else {
        if (separate_vol && use_en_deriv) {
            //Set Function eval
            ierr = SNESSetFunction(slvr->snes, slvr->Res, calc_residual_no_vol, user);
            CHKERRQ(ierr);
            //Set Jacobian eval
            ierr = SNESSetJacobian(slvr->snes, slvr->A, slvr->A, calc_jacobian_no_vol, user);
            CHKERRQ(ierr);
        } else if (!separate_vol && !use_en_deriv) {
            //Set Function eval
            ierr = SNESSetFunction(slvr->snes, slvr->Res, calc_residual_algebraic, user);
            CHKERRQ(ierr);
            //Set Jacobian eval
            ierr = SNESSetJacobian(slvr->snes, slvr->A, slvr->A, calc_jacobian_algebraic, user);
            CHKERRQ(ierr);
        } else if (separate_vol && !use_en_deriv) {
            //Set Function eval
            ierr = SNESSetFunction(slvr->snes, slvr->Res, calc_residual_algebraic_no_vol, user);
            CHKERRQ(ierr);
            //Set Jacobian eval
            ierr = SNESSetJacobian(slvr->snes, slvr->A, slvr->A, calc_jacobian_algebraic_no_vol, user);
            CHKERRQ(ierr);
        } else {
            //Set Function eval
            ierr = SNESSetFunction(slvr->snes, slvr->Res, calc_residual, user);
            CHKERRQ(ierr);
            //Set Jacobian eval
            ierr = SNESSetJacobian(slvr->snes, slvr->A, slvr->A, calc_jacobian, user);
            CHKERRQ(ierr);
        }
    }
    //Set SNES types
    ierr = SNESSetType(slvr->snes,SNESNEWTONLS); CHKERRQ(ierr);
//    ierr = SNESSetType(slvr->snes,SNESNEWTONTR); CHKERRQ(ierr);



//    ierr = KSPSetType(slvr->ksp,KSPPREONLY);CHKERRQ(ierr);
//     ierr = KSPSetType(slvr->ksp,KSPBCGS);CHKERRQ(ierr);

    //Gmres type methods
//     ierr = KSPSetType(slvr->ksp,KSPGMRES);CHKERRQ(ierr);
//    ierr = KSPSetType(slvr->ksp,KSPFGMRES);CHKERRQ(ierr);
//    /*
    ierr = KSPSetType(slvr->ksp,KSPDGMRES); CHKERRQ(ierr);

    ierr = KSPGMRESSetRestart(slvr->ksp,40); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL,"-ksp_dgmres_eigen","10"); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL,"-ksp_dgmres_max_eigen","100"); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL,"-ksp_dgmres_force",""); CHKERRQ(ierr);
//*/


    //Multigrid precond
//    ierr = Initialize_PCMG(slvr->pc,slvr->A,user); CHKERRQ(ierr);

    //LU Direct solve
    /*
    ierr = PCSetType(slvr->pc,PCLU);CHKERRQ(ierr);
    ierr = KSPSetPC(slvr->ksp,slvr->pc);CHKERRQ(ierr);
     ierr = PCFactorSetMatSolverPackage(slvr->pc, MATSOLVERSUPERLU); CHKERRQ(ierr);
    */
    // ILU Precond
//    /*
    ierr = PCSetType(slvr->pc,PCILU);CHKERRQ(ierr);
    ierr = PCFactorSetFill(slvr->pc,3.0);CHKERRQ(ierr);
    ierr = PCFactorSetLevels(slvr->pc,1);CHKERRQ(ierr);
    ierr = PCFactorSetAllowDiagonalFill(slvr->pc,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PCFactorSetMatOrderingType(slvr->pc,MATORDERINGNATURAL); CHKERRQ(ierr);
//    */


    ierr = SNESSetFromOptions(slvr->snes);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(slvr->ksp);CHKERRQ(ierr);
    ierr = PCSetFromOptions(slvr->pc);CHKERRQ(ierr);


    return ierr;
}

PetscErrorCode initialize_grid_slvr(struct Solver *slvr,int argc, char **argv,struct AppCtx *user)
{
    PetscErrorCode ierr;
    //    Get Nx, Ny, and dt from options if possible

    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;

    user->dx = Lx/Nx;
    user->dy = Ly/Ny;
    slvr->NA = ((Ni+2)*Nc-1)*(2*width_size+1)*(2*width_size+1);//total number of unknowns

    PetscInt NA = slvr->NA;


    //Create Vectors
    ierr = VecCreate(PETSC_COMM_WORLD,&slvr->Q);CHKERRQ(ierr);
    ierr = VecSetType(slvr->Q,VECSEQ);CHKERRQ(ierr);
    ierr = VecSetSizes(slvr->Q,PETSC_DECIDE,NA);CHKERRQ(ierr);
    ierr = VecDuplicate(slvr->Q,&slvr->Res);CHKERRQ(ierr);

    //Create Matrix
    //Get number of nonzeros in each row
    int *nnz = (int*) malloc(sizeof(int)*NA);
    Get_Nonzero_in_Rows(nnz,user,1);
    //Construct matrix using that
    ierr = MatCreate(PETSC_COMM_WORLD,&slvr->A);CHKERRQ(ierr);
    ierr = MatSetType(slvr->A,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(slvr->A,PETSC_DECIDE,PETSC_DECIDE,NA,NA);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(slvr->A,5*((Ni+2)*Nc-1),nnz);CHKERRQ(ierr);
    ierr = MatSetUp(slvr->A);CHKERRQ(ierr);

    //Initialize Space

    ierr = initialize_grid_jacobian(slvr->A,user,1); CHKERRQ(ierr);
    ierr = MatSetOption(slvr->A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE); CHKERRQ(ierr);

    //Create Solver Contexts

    ierr = KSPCreate(PETSC_COMM_WORLD,&slvr->ksp);CHKERRQ(ierr);


//    ierr = KSPSetType(slvr->ksp,KSPPREONLY);CHKERRQ(ierr);
//    ierr = KSPSetType(slvr->ksp,KSPBCGS);CHKERRQ(ierr);
    //Gmres type methods
//     ierr = KSPSetType(slvr->ksp,KSPGMRES);CHKERRQ(ierr);
//    ierr = KSPSetType(slvr->ksp,KSPFGMRES);CHKERRQ(ierr);
    //    /*
    ierr = KSPSetType(slvr->ksp,KSPDGMRES); CHKERRQ(ierr);

    ierr = KSPGMRESSetRestart(slvr->ksp,40); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL,"-ksp_dgmres_eigen","10"); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL,"-ksp_dgmres_max_eigen","100"); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL,"-ksp_dgmres_force",""); CHKERRQ(ierr);
//*/

    ierr = KSPGetPC(slvr->ksp,&slvr->pc);CHKERRQ(ierr);

    //LU Direct solve
    /*
    ierr = PCSetType(slvr->pc,PCLU);CHKERRQ(ierr);
    ierr = KSPSetPC(slvr->ksp,slvr->pc);CHKERRQ(ierr);
    ierr = PCFactorSetMatSolverPackage(slvr->pc, MATSOLVERSUPERLU); CHKERRQ(ierr);
    */


    // ILU Precond
//    /*
    ierr = PCSetType(slvr->pc,PCILU);CHKERRQ(ierr);
    ierr = PCFactorSetFill(slvr->pc,3.0);CHKERRQ(ierr);
    ierr = PCFactorSetLevels(slvr->pc,1);CHKERRQ(ierr);
    ierr = PCFactorSetAllowDiagonalFill(slvr->pc,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PCFactorSetMatOrderingType(slvr->pc,MATORDERINGNATURAL); CHKERRQ(ierr);
//    */


    return ierr;
}

void Get_Nonzero_in_Rows(int *nnz,struct AppCtx *user,int grid)
{
    PetscInt Nx;
    PetscInt Ny;
    PetscInt NA;
    PetscInt (*Ind)(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt);
    if(grid) {
        Nx = 2 * width_size + 1;
        Ny = 2 * width_size + 1;
        NA = user->grid_slvr->NA;
        Ind = &Ind_2;
    }else{
        Nx = user->Nx;
        Ny = user->Ny;
        NA = user->slvr->NA;
        Ind = &Ind_1;
    }
    //Make sure nnz is initialized to zero
    for(int i=0;i<NA;i++) {
        nnz[i]=0;
    }
    int ind = 0;
    int x,y,comp,ion;
    //Ionic concentration equations
    for(x=0;x<Nx;x++) {
        for(y=0;y<Ny;y++) {
            for(ion=0;ion<Ni;ion++) {
                for(comp=0;comp<Nc-1;comp++) {
                    //Electrodiffusion contributions
                    if(x<Nx-1) {
                        nnz[Ind(x+1,y,ion,comp,Nx)]++; //Ind_1(x,y,ion,comp,Nx)
                        ind++;
                        //Right c with left phi (-Fph0x)
                        nnz[Ind(x+1,y,ion,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                        ind++;
                        nnz[Ind(x+1,y,Ni,comp,Nx)]++;
                        ind++;
                    }
                    if(x>0) {
                        //left c with right c (-Fc1x)
                        nnz[Ind(x-1,y,ion,comp,Nx)]++;//Ind_1(x,y,ion,comp,Nx)
                        ind++;
                        //Left c with right phi (-Fph1x)
                        nnz[Ind(x-1,y,ion,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                        ind++;
                        nnz[Ind(x-1,y,Ni,comp,Nx)]++;
                        ind++;
                    }
                    if(y<Ny-1) {
                        // Upper c with lower c (-Fc0y)
                        nnz[Ind(x,y+1,ion,comp,Nx)]++;//Ind_1(x,y,ion,comp,Nx);
                        ind++;
                        //Upper c with lower phi (-Fph0y)
                        nnz[Ind(x,y+1,ion,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                        ind++;
                        nnz[Ind(x,y+1,Ni,comp,Nx)]++;
                        ind++;
                    }
                    if(y>0) {
                        //Lower c with Upper c (-Fc1y)
                        nnz[Ind(x,y-1,ion,comp,Nx)]++;//Ind_1(x,y,ion,comp,Nx)
                        ind++;
                        //Lower c with Upper phi (-Fph1y)
                        nnz[Ind(x,y-1,ion,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                        ind++;
                        nnz[Ind(x,y-1,Ni,comp,Nx)]++;
                        ind++;
                    }

                    //membrane current contributions
                    // Different Compartment Terms
                    // C Extracellular with C Inside
                    nnz[Ind(x,y,ion,Nc-1,Nx)]++;//Ind_1(x,y,ion,comp,Nx)
                    ind++;
                    // C Intra with C Extra
                    nnz[Ind(x,y,ion,comp,Nx)]++;//Ind_1(x,y,ion,Nc-1,Nx)
                    ind++;
                    // C Extracellular with Phi Inside
                    nnz[Ind(x,y,ion,Nc-1,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                    ind++;
                    // C Intra with Phi Extra
                    nnz[Ind(x,y,ion,comp,Nx)]++;//Ind_1(x,y,Ni,Nc-1,Nx)
                    ind++;
                    if(!separate_vol||grid) {
                        //Volume terms
                        //C extra with intra alpha
                        nnz[Ind(x, y, ion, Nc - 1,Nx)]++;//Ind_1(x,y,Ni+1,comp,Nx)
                        ind++;
                        //C intra with intra alpha
                        nnz[Ind(x, y, ion, comp,Nx)]++;//Ind_1(x,y,Ni+1,comp,Nx)
                        ind++;
                    }
                    //Same compartment terms
                    // c with c
                    nnz[Ind(x,y,ion,comp,Nx)]++;//Ind_1(x,y,ion,comp,Nx)
                    ind++;
                    // c with phi
                    nnz[Ind(x,y,ion,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                    ind++;

                    //Intra-Phi with c (voltage eqn)
                    nnz[Ind(x,y,Ni,comp,Nx)]++;//Ind_1(x,y,ion,comp,Nx)
                    //IntraPhi with c extra(volt eqn)
                    nnz[Ind(x,y,Ni,comp,Nx)]++;//Ind_1(x,y,ion,Nc-1,Nx)

                    //Extra-Phi with intra-c (voltage eqn)
                    nnz[Ind(x,y,Ni,Nc-1,Nx)]++;//Ind_1(x,y,ion,comp,Nx)


                }
                //Extracellular terms
                comp = Nc-1;
                //Electrodiffusion contributions
                if(x<Nx-1) {
                    // Right c with left c (-Fc0x)
                    nnz[Ind(x+1,y,ion,comp,Nx)]++;//Ind_1(x,y,ion,comp,Nx)
                    ind++;
                    //Right c with left phi (-Fph0x)
                    nnz[Ind(x+1,y,ion,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                    ind++;
                    nnz[Ind(x+1,y,Ni,comp,Nx)]++;
                    ind++;
                }
                if(x>0) {
                    //left c with right c (-Fc1x)
                    nnz[Ind(x-1,y,ion,comp,Nx)]++;//Ind_1(x,y,ion,comp,Nx)
                    ind++;
                    //Left c with right phi (-Fph1x)
                    nnz[Ind(x-1,y,ion,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                    ind++;
                    nnz[Ind(x-1,y,Ni,comp,Nx)]++;
                    ind++;
                }
                if(y<Ny-1) {
                    // Upper c with lower c (-Fc0y)
                    nnz[Ind(x,y+1,ion,comp,Nx)]++;//Ind_1(x,y,ion,comp,Nx)
                    ind++;
                    //Upper c with lower phi (-Fph0y)
                    nnz[Ind(x,y+1,ion,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                    ind++;
                    nnz[Ind(x,y+1,Ni,comp,Nx)]++;
                    ind++;
                }
                if(y>0) {
                    //Lower c with Upper c (-Fc1y)
                    nnz[Ind(x,y-1,ion,comp,Nx)]++;//Ind_1(x,y,ion,comp,Nx)
                    ind++;
                    //Lower c with Upper phi (-Fph1y)
                    nnz[Ind(x,y-1,ion,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                    ind++;
                    nnz[Ind(x,y-1,Ni,comp,Nx)]++;
                    ind++;
                }

                //Membrane current contribution
                //Add bath contributions
                //Insert extracell to extracell parts
                // c with c
                nnz[Ind(x,y,ion,Nc-1,Nx)]++;//Ind_1(x,y,ion,Nc-1,Nx)
                ind++;
                // c with phi
                nnz[Ind(x,y,ion,Nc-1,Nx)]++;//Ind_1(x,y,Ni,Nc-1,Nx)
                ind++;
                //Extra phi with c (volt eqn)
                nnz[Ind(x,y,Ni,Nc-1,Nx)]++;
                ind++;
            }
            //Derivative of charge-capacitance
            for(comp=0;comp<Nc-1;comp++){
                if(x<Nx-1) {
                    //Right phi with left phi (-Fph0x)
                    nnz[Ind(x+1,y,Ni,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                    ind++;
                }
                if(x>0) {
                    //Left phi with right phi (-Fph1x)
                    nnz[Ind(x-1,y,Ni,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                    ind++;
                }
                if(y<Ny-1) {
                    //Upper phi with lower phi (-Fph0y)
                    nnz[Ind(x,y+1,Ni,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                    ind++;
                }
                if(y>0) {
                    //Lower phi with upper phi (-Fph1y)
                    nnz[Ind(x,y-1,Ni,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                    ind++;
                }
                //Intra-phi with Intra-phi
                nnz[Ind(x,y,Ni,comp,Nx)]++;
                ind++;
                //Intra-phi with extra-phi
                nnz[Ind(x,y,Ni,comp,Nx)]++;//Ind_1(x,y,Ni,Nc-1,Nx)
                ind++;
            }
            //Extracellular terms
            comp = Nc-1;
            if(x<Nx-1) {
                //Right phi with left phi (-Fph0x)
                nnz[Ind(x+1,y,Ni,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                ind++;
            }
            if(x>0) {
                //Left phi with right phi (-Fph1x)
                nnz[Ind(x-1,y,Ni,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                ind++;
            }
            if(y<Ny-1) {
                //Upper phi with lower phi (-Fph0y)
                nnz[Ind(x,y+1,Ni,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                ind++;
            }
            if(y>0) {
                //Lower phi with upper phi (-Fph1y)
                nnz[Ind(x,y-1,Ni,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
                ind++;
            }
            for(int k=0;k<Nc-1;k++){

                //Extra-phi with Intra-phi
                nnz[Ind(x,y,Ni,comp,Nx)]++;//Ind_1(x,y,Ni,k,Nx)
                ind++;
            }
            //extra-phi with extra-phi
            nnz[Ind(x,y,Ni,comp,Nx)]++;//Ind_1(x,y,Ni,comp,Nx)
            ind++;

            // Neuron-Glia glutamate exchange
//            nnz[Ind(x,y,3,0,Nx)]++;//Ind_1(x,y,3,1,Nx)
//            ind++;
//            nnz[Ind(x,y,3,1,Nx)]++;//Ind_1(x,y,3,0,Nx),
//            ind++;
        }
    }
    if(!separate_vol||grid) {
        //water flow
        for (x = 0; x < Nx; x++) {
            for (y = 0; y < Ny; y++) {
                for (comp = 0; comp < Nc - 1; comp++) {
                    //Water flow volume fraction entries
                    //Volume to Volume
                    nnz[Ind(x, y, Ni + 1, comp,Nx)]++;//Ind_1(x,y,Ni+1,comp,Nx)
                    ind++;
                    //Off diagonal (from aNc=1-sum(ak))
                    for (PetscInt l = 0; l < comp; l++) {
                        nnz[Ind(x, y, Ni + 1, comp,Nx)]++;//Ind_1(x,y,Ni+1,l,Nx)
                        ind++;
                    }
                    for (PetscInt l = comp + 1; l < Nc - 1; l++) {
                        nnz[Ind(x, y, Ni + 1, comp,Nx)]++;//Ind_1(x,y,Ni+1,l,Nx)
                        ind++;
                    }
                    for (ion = 0; ion < Ni; ion++) {
                        //Volume to extra c
                        nnz[Ind(x, y, Ni + 1, comp,Nx)]++;//Ind_1(x,y,ion,Nc-1,Nx)
                        ind++;
                        //Volume to intra c
                        nnz[Ind(x, y, Ni + 1, comp,Nx)]++;//Ind_1(x,y,ion,comp,Nx)
                        ind++;
                    }
                }
            }
        }
    }
    printf("Base Nz: %d, Actual Nz: %d\n",user->Nz,ind);
}


PetscErrorCode initialize_jacobian(Mat Jac,struct AppCtx *user,int grid) {
    printf("Initializing Jacobian Memory\n");
    PetscErrorCode ierr;
    PetscInt Nx;
    PetscInt Ny;
    if(grid) {
        Nx = 2 * width_size + 1;
        Ny = 2 * width_size + 1;
    }else{
        Nx = user->Nx;
        Ny = user->Ny;
    }
    PetscInt ind = 0;
    PetscInt x,y,ion,comp;

    //Ionic concentration equations
    for(x=0;x<Nx;x++) {
        for(y=0;y<Ny;y++) {
            for(ion=0;ion<Ni;ion++) {
                for(comp=0;comp<Nc-1;comp++) {
                    //Electrodiffusion contributions
                    if(x<Nx-1) {
                        // Right c with left c (-Fc0x)
                        ierr = MatSetValue(Jac,Ind_1(x+1,y,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Right c with left phi (-Fph0x)
                        ierr = MatSetValue(Jac,Ind_1(x+1,y,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        if (use_en_deriv&&!grid) {
                            //Right phi with left c in voltage eqn
                            ierr = MatSetValue(Jac,Ind_1(x+1,y,Ni,comp,Nx),Ind_1(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;
                        }
                    }
                    if(x>0) {
                        //left c with right c (-Fc1x)
                        ierr = MatSetValue(Jac,Ind_1(x-1,y,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Left c with right phi (-Fph1x)
                        ierr = MatSetValue(Jac,Ind_1(x-1,y,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        if (use_en_deriv&&!grid) {
                            //Left phi with right c in voltage eqn
                            ierr = MatSetValue(Jac, Ind_1(x - 1, y, Ni, comp,Nx), Ind_1(x, y, ion, comp,Nx), 0,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                        }
                    }
                    if(y<Ny-1) {
                        // Upper c with lower c (-Fc0y)
                        ierr = MatSetValue(Jac,Ind_1(x,y+1,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Upper c with lower phi (-Fph0y)
                        ierr = MatSetValue(Jac,Ind_1(x,y+1,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        if (use_en_deriv&&!grid) {
                            //Upper phi with lower c in voltage eqn
                            ierr = MatSetValue(Jac, Ind_1(x, y + 1, Ni, comp,Nx), Ind_1(x, y, ion, comp,Nx), 0,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                        }
                    }
                    if(y>0) {
                        //Lower c with Upper c (-Fc1y)
                        ierr = MatSetValue(Jac,Ind_1(x,y-1,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Lower c with Upper phi (-Fph1y)
                        ierr = MatSetValue(Jac,Ind_1(x,y-1,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        if (use_en_deriv&&!grid) {
                            //Lower phi with upper c in voltage eqn
                            ierr = MatSetValue(Jac, Ind_1(x, y - 1, Ni, comp,Nx), Ind_1(x, y, ion, comp,Nx), 0,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                        }
                    }
                    // Different Compartment Terms
                    // C Extracellular with C Inside
                    ierr = MatSetValue(Jac,Ind_1(x,y,ion,Nc-1,Nx),Ind_1(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // C Intra with C Extra
                    ierr = MatSetValue(Jac,Ind_1(x,y,ion,comp,Nx),Ind_1(x,y,ion,Nc-1,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // C Extracellular with Phi Inside
                    ierr = MatSetValue(Jac,Ind_1(x,y,ion,Nc-1,Nx),Ind_1(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // C Intra with Phi Extra
                    ierr = MatSetValue(Jac,Ind_1(x,y,ion,comp,Nx),Ind_1(x,y,Ni,Nc-1,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    if(!separate_vol||grid) {
                        //Volume terms
                        //C extra with intra alpha
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, Ni + 1, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //C intra with intra alpha
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, Ni + 1, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    //Same compartment terms
                    // c with c
                    ierr = MatSetValue(Jac,Ind_1(x,y,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // c with phi
                    ierr = MatSetValue(Jac,Ind_1(x,y,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    if (use_en_deriv&&!grid) {
                        //Intra-Phi with c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //IntraPhi with c extra(volt eqn)
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, ion, Nc - 1,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;

                        //Extra-Phi with intra-c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni, Nc - 1,Nx), Ind_1(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                }
                //Extracellular terms
                comp = Nc-1;
                //Electrodiffusion contributions
                if(x<Nx-1) {
                    // Right c with left c (-Fc0x)
                    ierr = MatSetValue(Jac,Ind_1(x+1,y,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Right c with left phi (-Fph0x)
                    ierr = MatSetValue(Jac,Ind_1(x+1,y,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    if (use_en_deriv&&!grid) {
                        // left Phi with right c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_1(x + 1, y, Ni, comp,Nx), Ind_1(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                }
                if(x>0) {
                    //left c with right c (-Fc1x)
                    ierr = MatSetValue(Jac,Ind_1(x-1,y,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Left c with right phi (-Fph1x)
                    ierr = MatSetValue(Jac,Ind_1(x-1,y,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    if (use_en_deriv&&!grid) {
                        // left Phi with right c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_1(x - 1, y, Ni, comp,Nx), Ind_1(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                }
                if(y<Ny-1) {
                    // Upper c with lower c (-Fc0y)
                    ierr = MatSetValue(Jac,Ind_1(x,y+1,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Upper c with lower phi (-Fph0y)
                    ierr = MatSetValue(Jac,Ind_1(x,y+1,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    if (use_en_deriv&&!grid) {
                        // Upper Phi with lower c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_1(x, y + 1, Ni, comp,Nx), Ind_1(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                }
                if(y>0) {
                    //Lower c with Upper c (-Fc1y)
                    ierr = MatSetValue(Jac,Ind_1(x,y-1,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Lower c with Upper phi (-Fph1y)
                    ierr = MatSetValue(Jac,Ind_1(x,y-1,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    if (use_en_deriv&&!grid) {
                        // Lower Phi with upper c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_1(x, y - 1, Ni, comp,Nx), Ind_1(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                }
                //Insert extracell to extracell parts
                // c with c
                ierr = MatSetValue(Jac,Ind_1(x,y,ion,Nc-1,Nx),Ind_1(x,y,ion,Nc-1,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                ind++;
                // c with phi
                ierr = MatSetValue(Jac,Ind_1(x,y,ion,Nc-1,Nx),Ind_1(x,y,Ni,Nc-1,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                ind++;
                if (use_en_deriv&&!grid) {
                    //phi with c (voltage eqn)
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, Nc - 1,Nx), Ind_1(x, y, ion, Nc - 1,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
            }
            if (use_en_deriv&&!grid) {
                //Derivative of charge-capacitance
                for (comp = 0; comp < Nc - 1; comp++) {
                    if (x < Nx - 1) {
                        //Right phi with left phi (-Fph0x)
                        ierr = MatSetValue(Jac, Ind_1(x + 1, y, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if (x > 0) {
                        //Left phi with right phi (-Fph1x)
                        ierr = MatSetValue(Jac, Ind_1(x - 1, y, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if (y < Ny - 1) {
                        //Upper phi with lower phi (-Fph0y)
                        ierr = MatSetValue(Jac, Ind_1(x, y + 1, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if (y > 0) {
                        //Lower phi with upper phi (-Fph1y)
                        ierr = MatSetValue(Jac, Ind_1(x, y - 1, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    //Intra-phi with Intra-phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Intra-phi with extra-phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni, Nc - 1,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                //Extracellular terms
                comp = Nc - 1;
                if (x < Nx - 1) {
                    //Right phi with left phi (-Fph0x)
                    ierr = MatSetValue(Jac, Ind_1(x + 1, y, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                if (x > 0) {
                    //Left phi with right phi (-Fph1x)
                    ierr = MatSetValue(Jac, Ind_1(x - 1, y, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                if (y < Ny - 1) {
                    //Upper phi with lower phi (-Fph0y)
                    ierr = MatSetValue(Jac, Ind_1(x, y + 1, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                if (y > 0) {
                    //Lower phi with upper phi (-Fph1y)
                    ierr = MatSetValue(Jac, Ind_1(x, y - 1, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }

                for (int k = 0; k < Nc - 1; k++) {
                    //Extra-phi with Intra-phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni, k,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                //extra-phi with extra-phi
                ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
            }

        }
    }
    if(!use_en_deriv||grid) {
        //Electroneutrality charge-capcitance condition
        for (x = 0; x < Nx; x++) {
            for (y = 0; y < Ny; y++) {
                //electroneutral-concentration entries
                for (ion = 0; ion < Ni; ion++) {
                    for (comp = 0; comp < Nc - 1; comp++) {
                        //Phi with C entries
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    //Phi with C extracellular one
                    comp = Nc - 1;
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                //electroneutrality-voltage entries

                //extraphi with extra phi
                ierr = MatSetValue(Jac, Ind_1(x, y, Ni, Nc - 1,Nx), Ind_1(x, y, Ni, Nc - 1,Nx), 0, INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
                for (comp = 0; comp < Nc - 1; comp++) {
                    //Extra phi with intra phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, Nc - 1,Nx), Ind_1(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    // Intra phi with Extraphi
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni, Nc - 1,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Intra phi with Intra phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    if(!separate_vol||grid) {
                        //Extra phi with intra-Volume
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni, Nc - 1,Nx), Ind_1(x, y, Ni + 1, comp,Nx), 0,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Intra phi with Intra Vol
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni + 1, comp,Nx), 0,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if(grid) {
                        //Extra phi with intra phi
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni, Nc - 1, Nx), Ind_1(x, y, Ni, comp, Nx), cm[comp],
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // Intra phi with Extraphi
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp, Nx), Ind_1(x, y, Ni, Nc - 1, Nx), cm[comp],
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Intra phi with Intra phi
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp, Nx), Ind_1(x, y, Ni, comp, Nx), -cm[comp],
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                }
                // Neuron-Glia glutamate exchange
//                ierr = MatSetValue(Jac,Ind_1(x,y,3,0,Nx),Ind_1(x,y,3,1,Nx),-glut_Bg*user->dt,INSERT_VALUES);CHKERRQ(ierr); ind++;

//                ierr = MatSetValue(Jac,Ind_1(x,y,3,1,Nx),Ind_1(x,y,3,0,Nx),((1-glut_gamma)*glut_Bn*glut_Re-glut_Bg*glut_Rg)*user->dt,INSERT_VALUES);CHKERRQ(ierr);
//                ind++;
            }
        }
    }
    if(!separate_vol||grid) {
        //water flow
        for (x = 0; x < Nx; x++) {
            for (y = 0; y < Ny; y++) {
                for (comp = 0; comp < Nc - 1; comp++) {
                    //Water flow volume fraction entries
                    //Volume to Volume
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni + 1, comp,Nx), Ind_1(x, y, Ni + 1, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Off diagonal (from aNc=1-sum(ak))
                    for (PetscInt l = 0; l < comp; l++) {
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni + 1, comp,Nx), Ind_1(x, y, Ni + 1, l,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    for (PetscInt l = comp + 1; l < Nc - 1; l++) {
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni + 1, comp,Nx), Ind_1(x, y, Ni + 1, l,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    for (ion = 0; ion < Ni; ion++) {
                        //Volume to extra c
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni + 1, comp,Nx), Ind_1(x, y, ion, Nc - 1,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Volume to intra c
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni + 1, comp,Nx), Ind_1(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                }
            }
        }
    }
    ierr = MatAssemblyBegin(Jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


    return ierr;
}

PetscErrorCode Create_Restriction(Mat R,PetscInt nx, PetscInt ny)
{
    PetscErrorCode  ierr;
    int x,y,ion,comp;
    for(x=1;x<nx/2-1;x++) {
        for (y = 1; y < ny / 2-1; y++) {
            //Restriction for concentrations
            for (ion = 0; ion < Ni; ion++) {
                for (comp = 0; comp < Nc; comp++) {
                    //Center point
                    ierr = MatSetValue(R, Ind_1(x, y, ion, comp, nx / 2), Ind_1(2 * x, 2 * y, ion, comp, nx), 1.0 / 4,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);

                    //Up/down/left/right
                    ierr = MatSetValue(R, Ind_1(x, y, ion, comp, nx / 2), Ind_1(2 * x, 2 * y - 1, ion, comp, nx),
                                       1.0 / 8, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(x, y, ion, comp, nx / 2), Ind_1(2 * x - 1, 2 * y, ion, comp, nx),
                                       1.0 / 8, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(x, y, ion, comp, nx / 2), Ind_1(2 * x, 2 * y + 1, ion, comp, nx),
                                       1.0 / 8, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(x, y, ion, comp, nx / 2), Ind_1(2 * x + 1, 2 * y, ion, comp, nx),
                                       1.0 / 8, INSERT_VALUES);
                    CHKERRQ(ierr);

                    //Four diagonals
                    ierr = MatSetValue(R, Ind_1(x, y, ion, comp, nx / 2), Ind_1(2 * x - 1, 2 * y - 1, ion, comp, nx),
                                       1.0 / 16, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(x, y, ion, comp, nx / 2), Ind_1(2 * x - 1, 2 * y + 1, ion, comp, nx),
                                       1.0 / 16, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(x, y, ion, comp, nx / 2), Ind_1(2 * x + 1, 2 * y - 1, ion, comp, nx),
                                       1.0 / 16, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(x, y, ion, comp, nx / 2), Ind_1(2 * x + 1, 2 * y + 1, ion, comp, nx),
                                       1.0 / 16, INSERT_VALUES);
                    CHKERRQ(ierr);

                }
            }
            //Restriction for Voltage
            for (comp = 0; comp < Nc; comp++) {
                //Center point
                ierr = MatSetValue(R, Ind_1(x, y, Ni, comp, nx / 2), Ind_1(2 * x, 2 * y, ion, comp, nx), 1.0 / 4,
                                   INSERT_VALUES);
                CHKERRQ(ierr);

                //Up/down/left/right
                ierr = MatSetValue(R, Ind_1(x, y, Ni, comp, nx / 2), Ind_1(2 * x, 2 * y - 1, Ni, comp, nx), 1.0 / 8,
                                   INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni, comp, nx / 2), Ind_1(2 * x - 1, 2 * y, Ni, comp, nx), 1.0 / 8,
                                   INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni, comp, nx / 2), Ind_1(2 * x, 2 * y + 1, Ni, comp, nx), 1.0 / 8,
                                   INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni, comp, nx / 2), Ind_1(2 * x + 1, 2 * y, Ni, comp, nx), 1.0 / 8,
                                   INSERT_VALUES);
                CHKERRQ(ierr);

                //Four diagonals
                ierr = MatSetValue(R, Ind_1(x, y, Ni, comp, nx / 2), Ind_1(2 * x - 1, 2 * y - 1, Ni, comp, nx),
                                   1.0 / 16, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni, comp, nx / 2), Ind_1(2 * x - 1, 2 * y + 1, Ni, comp, nx),
                                   1.0 / 16, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni, comp, nx / 2), Ind_1(2 * x + 1, 2 * y - 1, Ni, comp, nx),
                                   1.0 / 16, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni, comp, nx / 2), Ind_1(2 * x + 1, 2 * y + 1, Ni, comp, nx),
                                   1.0 / 16, INSERT_VALUES);
                CHKERRQ(ierr);

            }
            if(!separate_vol) {
                //Restriction for Volume
                for (comp = 0; comp < Nc - 1; comp++) {
                    //Center point
                    ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y, ion, comp, nx),
                                       1.0 / 4,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);

                    //Up/down/left/right
                    ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                       Ind_1(2 * x, 2 * y - 1, Ni + 1, comp, nx),
                                       1.0 / 8, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                       Ind_1(2 * x - 1, 2 * y, Ni + 1, comp, nx),
                                       1.0 / 8, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                       Ind_1(2 * x, 2 * y + 1, Ni + 1, comp, nx),
                                       1.0 / 8, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                       Ind_1(2 * x + 1, 2 * y, Ni + 1, comp, nx),
                                       1.0 / 8, INSERT_VALUES);
                    CHKERRQ(ierr);

                    //Four diagonals
                    ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                       Ind_1(2 * x - 1, 2 * y - 1, Ni + 1, comp, nx), 1.0 / 16, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                       Ind_1(2 * x - 1, 2 * y + 1, Ni + 1, comp, nx), 1.0 / 16, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                       Ind_1(2 * x + 1, 2 * y - 1, Ni + 1, comp, nx), 1.0 / 16, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                       Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, nx), 1.0 / 16, INSERT_VALUES);
                    CHKERRQ(ierr);

                }
            }

        }
    }
    x=0;
    for(y=1;y<ny/2-1;y++) {
        //Restriction for concentrations
        for(ion=0;ion<Ni;ion++)
        {
            for(comp=0;comp<Nc;comp++) {
                //Center point
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),1.0/3,INSERT_VALUES); CHKERRQ(ierr);

                //Up/down/left/right
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y-1,ion,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y+1,ion,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x+1,2*y,ion,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);

                //Two diagonals
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x+1,2*y-1,ion,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x+1,2*y+1,ion,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);

            }
        }
        //Restriction for Voltage
        for(comp=0;comp<Nc;comp++) {
            //Center point
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),1.0/3,INSERT_VALUES); CHKERRQ(ierr);

            //Up/down/left/right
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y-1,Ni,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y+1,Ni,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x+1,2*y,Ni,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);

            //Four diagonals
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x+1,2*y-1,Ni,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x+1,2*y+1,Ni,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);

        }
        if(!separate_vol) {
            //Restriction for Volume
            for (comp = 0; comp < Nc - 1; comp++) {
                //Center point
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y, ion, comp, nx), 1.0 / 3,
                                   INSERT_VALUES);
                CHKERRQ(ierr);

                //Up/down/left/right
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y - 1, Ni + 1, comp, nx),
                                   1.0 / 6, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y + 1, Ni + 1, comp, nx),
                                   1.0 / 6, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x + 1, 2 * y, Ni + 1, comp, nx),
                                   1.0 / 6, INSERT_VALUES);
                CHKERRQ(ierr);

                //Four diagonals
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                   Ind_1(2 * x + 1, 2 * y - 1, Ni + 1, comp, nx), 1.0 / 12, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                   Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, nx), 1.0 / 12, INSERT_VALUES);
                CHKERRQ(ierr);

            }
        }

    }
    x=nx/2-1;
    for(y=1;y<ny/2-1;y++) {
        //Restriction for concentrations
        for(ion=0;ion<Ni;ion++)
        {
            for(comp=0;comp<Nc;comp++) {
                //Center point
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),1.0/3,INSERT_VALUES); CHKERRQ(ierr);

                //Up/down/left/right
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y-1,ion,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y+1,ion,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x-1,2*y,ion,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);

                //Two diagonals
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x-1,2*y-1,ion,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x-1,2*y+1,ion,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);

            }
        }
        //Restriction for Voltage
        for(comp=0;comp<Nc;comp++) {
            //Center point
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),1.0/3,INSERT_VALUES); CHKERRQ(ierr);

            //Up/down/left/right
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y-1,Ni,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y+1,Ni,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x-1,2*y,Ni,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);

            //Four diagonals
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x-1,2*y-1,Ni,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x-1,2*y+1,Ni,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);

        }
        if(!separate_vol) {
            //Restriction for Volume
            for (comp = 0; comp < Nc - 1; comp++) {
                //Center point
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y, ion, comp, nx), 1.0 / 3,
                                   INSERT_VALUES);
                CHKERRQ(ierr);

                //Up/down/left/right
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y - 1, Ni + 1, comp, nx),
                                   1.0 / 6, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y + 1, Ni + 1, comp, nx),
                                   1.0 / 6, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x - 1, 2 * y, Ni + 1, comp, nx),
                                   1.0 / 6, INSERT_VALUES);
                CHKERRQ(ierr);

                //Four diagonals
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                   Ind_1(2 * x - 1, 2 * y - 1, Ni + 1, comp, nx), 1.0 / 12, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                   Ind_1(2 * x - 1, 2 * y + 1, Ni + 1, comp, nx), 1.0 / 12, INSERT_VALUES);
                CHKERRQ(ierr);

            }
        }

    }
    y=0;
    for(x=1;x<nx/2-1;x++) {
        //Restriction for concentrations
        for(ion=0;ion<Ni;ion++)
        {
            for(comp=0;comp<Nc;comp++) {
                //Center point
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),1.0/3,INSERT_VALUES); CHKERRQ(ierr);

                //Up/down/left/right
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x-1,2*y,ion,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y+1,ion,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x+1,2*y,ion,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);

                //Two diagonals
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x-1,2*y+1,ion,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x+1,2*y+1,ion,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);

            }
        }
        //Restriction for Voltage
        for(comp=0;comp<Nc;comp++) {
            //Center point
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),1.0/3,INSERT_VALUES); CHKERRQ(ierr);

            //Up/down/left/right
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x-1,2*y,Ni,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y+1,Ni,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x+1,2*y,Ni,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);

            //Four diagonals
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x-1,2*y+1,Ni,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x+1,2*y+1,Ni,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);

        }
        if(!separate_vol) {
            //Restriction for Volume
            for (comp = 0; comp < Nc - 1; comp++) {
                //Center point
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y, ion, comp, nx), 1.0 / 3,
                                   INSERT_VALUES);
                CHKERRQ(ierr);

                //Up/down/left/right
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x - 1, 2 * y, Ni + 1, comp, nx),
                                   1.0 / 6, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y + 1, Ni + 1, comp, nx),
                                   1.0 / 6, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x + 1, 2 * y, Ni + 1, comp, nx),
                                   1.0 / 6, INSERT_VALUES);
                CHKERRQ(ierr);

                //Four diagonals
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                   Ind_1(2 * x - 1, 2 * y + 1, Ni + 1, comp, nx), 1.0 / 12, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                   Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, nx), 1.0 / 12, INSERT_VALUES);
                CHKERRQ(ierr);

            }
        }

    }
    y=nx/2-1;
    for(x=1;x<nx/2-1;x++) {
        //Restriction for concentrations
        for(ion=0;ion<Ni;ion++)
        {
            for(comp=0;comp<Nc;comp++) {
                //Center point
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),1.0/3,INSERT_VALUES); CHKERRQ(ierr);

                //Up/down/left/right
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y-1,ion,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x+1,2*y,ion,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x-1,2*y,ion,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);

                //Two diagonals
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x-1,2*y-1,ion,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x+1,2*y-1,ion,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);

            }
        }
        //Restriction for Voltage
        for(comp=0;comp<Nc;comp++) {
            //Center point
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),1.0/3,INSERT_VALUES); CHKERRQ(ierr);

            //Up/down/left/right
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y-1,Ni,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x+1,2*y,Ni,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x-1,2*y,Ni,comp,nx),1.0/6,INSERT_VALUES); CHKERRQ(ierr);

            //Four diagonals
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x-1,2*y-1,Ni,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x+1,2*y-1,Ni,comp,nx),1.0/12,INSERT_VALUES); CHKERRQ(ierr);

        }
        if(!separate_vol) {
            //Restriction for Volume
            for (comp = 0; comp < Nc - 1; comp++) {
                //Center point
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y, ion, comp, nx), 1.0 / 3,
                                   INSERT_VALUES);
                CHKERRQ(ierr);

                //Up/down/left/right
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y - 1, Ni + 1, comp, nx),
                                   1.0 / 6, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x + 1, 2 * y, Ni + 1, comp, nx),
                                   1.0 / 6, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x - 1, 2 * y, Ni + 1, comp, nx),
                                   1.0 / 6, INSERT_VALUES);
                CHKERRQ(ierr);

                //Four diagonals
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                   Ind_1(2 * x - 1, 2 * y - 1, Ni + 1, comp, nx), 1.0 / 12, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2),
                                   Ind_1(2 * x + 1, 2 * y - 1, Ni + 1, comp, nx), 1.0 / 12, INSERT_VALUES);
                CHKERRQ(ierr);

            }
        }

    }
    x=0;y=0;
    //Restriction for concentrations
    for(ion=0;ion<Ni;ion++)
    {
        for(comp=0;comp<Nc;comp++) {
            //Center point
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),4.0/9,INSERT_VALUES); CHKERRQ(ierr);

            //Up/down/left/right
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y+1,ion,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x+1,2*y,ion,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);

            //Two diagonals
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x+1,2*y+1,ion,comp,nx),1.0/9,INSERT_VALUES); CHKERRQ(ierr);

        }
    }
    //Restriction for Voltage
    for(comp=0;comp<Nc;comp++) {
        //Center point
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),4.0/9,INSERT_VALUES); CHKERRQ(ierr);

        //Up/down/left/right
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y+1,Ni,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x+1,2*y,Ni,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);

        //Four diagonals
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x+1,2*y+1,Ni,comp,nx),1.0/9,INSERT_VALUES); CHKERRQ(ierr);

    }
    if(!separate_vol) {
        //Restriction for Volume
        for (comp = 0; comp < Nc - 1; comp++) {
            //Center point
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y, ion, comp, nx), 4.0 / 9,
                               INSERT_VALUES);
            CHKERRQ(ierr);

            //Up/down/left/right
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y + 1, Ni + 1, comp, nx),
                               2.0 / 9, INSERT_VALUES);
            CHKERRQ(ierr);
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x + 1, 2 * y, Ni + 1, comp, nx),
                               2.0 / 9, INSERT_VALUES);
            CHKERRQ(ierr);

            //Four diagonals
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, nx),
                               1.0 / 9, INSERT_VALUES);
            CHKERRQ(ierr);

        }
    }
    x=nx/2-1;y=0;
    //Restriction for concentrations
    for(ion=0;ion<Ni;ion++)
    {
        for(comp=0;comp<Nc;comp++) {
            //Center point
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),4.0/9,INSERT_VALUES); CHKERRQ(ierr);

            //Up/down/left/right
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y+1,ion,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x-1,2*y,ion,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);

            //Two diagonals
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x-1,2*y+1,ion,comp,nx),1.0/9,INSERT_VALUES); CHKERRQ(ierr);

        }
    }
    //Restriction for Voltage
    for(comp=0;comp<Nc;comp++) {
        //Center point
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),4.0/9,INSERT_VALUES); CHKERRQ(ierr);

        //Up/down/left/right
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y+1,Ni,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x-1,2*y,Ni,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);

        //Four diagonals
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x-1,2*y+1,Ni,comp,nx),1.0/9,INSERT_VALUES); CHKERRQ(ierr);

    }
    if(!separate_vol) {
        //Restriction for Volume
        for (comp = 0; comp < Nc - 1; comp++) {
            //Center point
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y, ion, comp, nx), 4.0 / 9,
                               INSERT_VALUES);
            CHKERRQ(ierr);

            //Up/down/left/right
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y + 1, Ni + 1, comp, nx),
                               2.0 / 9, INSERT_VALUES);
            CHKERRQ(ierr);
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x - 1, 2 * y, Ni + 1, comp, nx),
                               2.0 / 9, INSERT_VALUES);
            CHKERRQ(ierr);

            //Four diagonals
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x - 1, 2 * y + 1, Ni + 1, comp, nx),
                               1.0 / 9, INSERT_VALUES);
            CHKERRQ(ierr);

        }
    }
    x=0;y=ny/2-1;
    //Restriction for concentrations
    for(ion=0;ion<Ni;ion++)
    {
        for(comp=0;comp<Nc;comp++) {
            //Center point
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),4.0/9,INSERT_VALUES); CHKERRQ(ierr);

            //Up/down/left/right
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y-1,ion,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x+1,2*y,ion,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);

            //Two diagonals
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x+1,2*y-1,ion,comp,nx),1.0/9,INSERT_VALUES); CHKERRQ(ierr);

        }
    }
    //Restriction for Voltage
    for(comp=0;comp<Nc;comp++) {
        //Center point
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),4.0/9,INSERT_VALUES); CHKERRQ(ierr);

        //Up/down/left/right
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y-1,Ni,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x+1,2*y,Ni,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);

        //Four diagonals
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x+1,2*y-1,Ni,comp,nx),1.0/9,INSERT_VALUES); CHKERRQ(ierr);

    }
    if(!separate_vol) {
        //Restriction for Volume
        for (comp = 0; comp < Nc - 1; comp++) {
            //Center point
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y, ion, comp, nx), 4.0 / 9,
                               INSERT_VALUES);
            CHKERRQ(ierr);

            //Up/down/left/right
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y - 1, Ni + 1, comp, nx),
                               2.0 / 9, INSERT_VALUES);
            CHKERRQ(ierr);
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x + 1, 2 * y, Ni + 1, comp, nx),
                               2.0 / 9, INSERT_VALUES);
            CHKERRQ(ierr);

            //Four diagonals
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x + 1, 2 * y - 1, Ni + 1, comp, nx),
                               1.0 / 9, INSERT_VALUES);
            CHKERRQ(ierr);

        }
    }
    x=nx/2-1;y=ny/2-1;
    //Restriction for concentrations
    for(ion=0;ion<Ni;ion++)
    {
        for(comp=0;comp<Nc;comp++) {
            //Center point
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),4.0/9,INSERT_VALUES); CHKERRQ(ierr);

            //Up/down/left/right
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x,2*y-1,ion,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x-1,2*y,ion,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);

            //Two diagonals
            ierr = MatSetValue(R,Ind_1(x,y,ion,comp,nx/2),Ind_1(2*x-1,2*y-1,ion,comp,nx),1.0/9,INSERT_VALUES); CHKERRQ(ierr);

        }
    }
    //Restriction for Voltage
    for(comp=0;comp<Nc;comp++) {
        //Center point
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y,ion,comp,nx),4.0/9,INSERT_VALUES); CHKERRQ(ierr);

        //Up/down/left/right
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x,2*y-1,Ni,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x-1,2*y,Ni,comp,nx),2.0/9,INSERT_VALUES); CHKERRQ(ierr);

        //Four diagonals
        ierr = MatSetValue(R,Ind_1(x,y,Ni,comp,nx/2),Ind_1(2*x-1,2*y-1,Ni,comp,nx),1.0/9,INSERT_VALUES); CHKERRQ(ierr);

    }
    if(!separate_vol) {
        //Restriction for Volume
        for (comp = 0; comp < Nc - 1; comp++) {
            //Center point
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y, ion, comp, nx), 4.0 / 9,
                               INSERT_VALUES);
            CHKERRQ(ierr);

            //Up/down/left/right
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x, 2 * y - 1, Ni + 1, comp, nx),
                               2.0 / 9, INSERT_VALUES);
            CHKERRQ(ierr);
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x - 1, 2 * y, Ni + 1, comp, nx),
                               2.0 / 9, INSERT_VALUES);
            CHKERRQ(ierr);

            //Four diagonals
            ierr = MatSetValue(R, Ind_1(x, y, Ni + 1, comp, nx / 2), Ind_1(2 * x - 1, 2 * y - 1, Ni + 1, comp, nx),
                               1.0 / 9, INSERT_VALUES);
            CHKERRQ(ierr);

        }
    }
    ierr = MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    return ierr;
}
PetscErrorCode Create_Interpolation(Mat R,PetscInt nx, PetscInt ny)
{
    PetscErrorCode  ierr;
    int x,y,ion,comp;
    for(x=0;x<nx-1;x++) {
        for(y=0;y<ny-1;y++) {
            //Interpolation for concentrations
            for(ion=0;ion<Ni;ion++) {
                for(comp=0;comp<Nc;comp++) {
                    ierr = MatSetValue(R,Ind_1(2*x+1,2*y,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);
                    ierr = MatSetValue(R,Ind_1(2*x+1,2*y,ion,comp,2*nx),Ind_1(x+1,y,ion,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);

                    ierr = MatSetValue(R,Ind_1(2*x,2*y+1,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);
                    ierr = MatSetValue(R,Ind_1(2*x,2*y+1,ion,comp,2*nx),Ind_1(x,y+1,ion,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);

                    ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),0.25,INSERT_VALUES); CHKERRQ(ierr);
                    ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,ion,comp,2*nx),Ind_1(x+1,y,ion,comp,nx),0.25,INSERT_VALUES); CHKERRQ(ierr);
                    ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,ion,comp,2*nx),Ind_1(x,y+1,ion,comp,nx),0.25,INSERT_VALUES); CHKERRQ(ierr);
                    ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,ion,comp,2*nx),Ind_1(x+1,y+1,ion,comp,nx),0.25,INSERT_VALUES); CHKERRQ(ierr);

                    ierr = MatSetValue(R,Ind_1(2*x,2*y,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),1,INSERT_VALUES); CHKERRQ(ierr);

                }
            }
            //Interpolation for Voltage
            for(comp=0;comp<Nc;comp++) {
                ierr = MatSetValue(R,Ind_1(2*x+1,2*y,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(2*x+1,2*y,Ni,comp,2*nx),Ind_1(x+1,y,Ni,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);

                ierr = MatSetValue(R,Ind_1(2*x,2*y+1,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(2*x,2*y+1,Ni,comp,2*nx),Ind_1(x,y+1,Ni,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);

                ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),0.25,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,Ni,comp,2*nx),Ind_1(x+1,y,Ni,comp,nx),0.25,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,Ni,comp,2*nx),Ind_1(x,y+1,Ni,comp,nx),0.25,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,Ni,comp,2*nx),Ind_1(x+1,y+1,Ni,comp,nx),0.25,INSERT_VALUES); CHKERRQ(ierr);

                ierr = MatSetValue(R,Ind_1(2*x,2*y,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),1,INSERT_VALUES); CHKERRQ(ierr);
            }
            if(!separate_vol) {
                //Interpolation for Volume
                for (comp = 0; comp < Nc - 1; comp++) {

                    ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y, Ni + 1, comp, 2 * nx),
                                       Ind_1(x, y, Ni + 1, comp, nx), 0.5, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y, Ni + 1, comp, 2 * nx),
                                       Ind_1(x + 1, y, Ni + 1, comp, nx), 0.5, INSERT_VALUES);
                    CHKERRQ(ierr);

                    ierr = MatSetValue(R, Ind_1(2 * x, 2 * y + 1, Ni + 1, comp, 2 * nx),
                                       Ind_1(x, y, Ni + 1, comp, nx), 0.5, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(2 * x, 2 * y + 1, Ni + 1, comp, 2 * nx),
                                       Ind_1(x, y + 1, Ni + 1, comp, nx), 0.5, INSERT_VALUES);
                    CHKERRQ(ierr);

                    ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, 2 * nx),
                                       Ind_1(x, y, Ni + 1, comp, nx), 0.25, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, 2 * nx),
                                       Ind_1(x + 1, y, Ni + 1, comp, nx), 0.25, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, 2 * nx),
                                       Ind_1(x, y + 1, Ni + 1, comp, nx), 0.25, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, 2 * nx),
                                       Ind_1(x + 1, y + 1, Ni + 1, comp, nx), 0.25, INSERT_VALUES);
                    CHKERRQ(ierr);

                    ierr = MatSetValue(R, Ind_1(2 * x, 2 * y, Ni + 1, comp, 2 * nx), Ind_1(x, y, Ni + 1, comp, nx), 1,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                }
            }

        }
    }
    x=nx-1;
    for(y=0;y<ny-1;y++) {
        //Interpolation for concentrations
        for(ion=0;ion<Ni;ion++) {
            for(comp=0;comp<Nc;comp++) {
                ierr = MatSetValue(R,Ind_1(2*x+1,2*y,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),1.0,INSERT_VALUES); CHKERRQ(ierr);

                ierr = MatSetValue(R,Ind_1(2*x,2*y+1,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(2*x,2*y+1,ion,comp,2*nx),Ind_1(x,y+1,ion,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);

                ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,ion,comp,2*nx),Ind_1(x,y+1,ion,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);

                ierr = MatSetValue(R,Ind_1(2*x,2*y,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),1,INSERT_VALUES); CHKERRQ(ierr);

            }
        }
        //Interpolation for Voltage
        for(comp=0;comp<Nc;comp++) {
            ierr = MatSetValue(R,Ind_1(2*x+1,2*y,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),1.0,INSERT_VALUES); CHKERRQ(ierr);

            ierr = MatSetValue(R,Ind_1(2*x,2*y+1,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(2*x,2*y+1,Ni,comp,2*nx),Ind_1(x,y+1,Ni,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);

            ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,Ni,comp,2*nx),Ind_1(x,y+1,Ni,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);

            ierr = MatSetValue(R,Ind_1(2*x,2*y,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),1,INSERT_VALUES); CHKERRQ(ierr);
        }
        if(!separate_vol) {
            //Interpolation for Volume
            for (comp = 0; comp < Nc - 1; comp++) {

                ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y, Ni + 1, comp, 2 * nx), Ind_1(x, y, Ni + 1, comp, nx),
                                   1.0, INSERT_VALUES);
                CHKERRQ(ierr);

                ierr = MatSetValue(R, Ind_1(2 * x, 2 * y + 1, Ni + 1, comp, 2 * nx), Ind_1(x, y, Ni + 1, comp, nx),
                                   0.5, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(2 * x, 2 * y + 1, Ni + 1, comp, 2 * nx),
                                   Ind_1(x, y + 1, Ni + 1, comp, nx), 0.5, INSERT_VALUES);
                CHKERRQ(ierr);

                ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, 2 * nx),
                                   Ind_1(x, y, Ni + 1, comp, nx), 0.5, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, 2 * nx),
                                   Ind_1(x, y + 1, Ni + 1, comp, nx), 0.5, INSERT_VALUES);
                CHKERRQ(ierr);

                ierr = MatSetValue(R, Ind_1(2 * x, 2 * y, Ni + 1, comp, 2 * nx), Ind_1(x, y, Ni + 1, comp, nx), 1,
                                   INSERT_VALUES);
                CHKERRQ(ierr);
            }
        }
    }
    y=ny-1;
    for(x=0;x<nx-1;x++) {
        //Interpolation for concentrations
        for(ion=0;ion<Ni;ion++) {
            for(comp=0;comp<Nc;comp++) {
                ierr = MatSetValue(R,Ind_1(2*x+1,2*y,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(2*x+1,2*y,ion,comp,2*nx),Ind_1(x+1,y,ion,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);

                ierr = MatSetValue(R,Ind_1(2*x,2*y+1,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),1.0,INSERT_VALUES); CHKERRQ(ierr);

                ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);
                ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,ion,comp,2*nx),Ind_1(x+1,y,ion,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);

                ierr = MatSetValue(R,Ind_1(2*x,2*y,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),1,INSERT_VALUES); CHKERRQ(ierr);

            }
        }
        //Interpolation for Voltage
        for(comp=0;comp<Nc;comp++) {
            ierr = MatSetValue(R,Ind_1(2*x+1,2*y,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(2*x+1,2*y,Ni,comp,2*nx),Ind_1(x+1,y,Ni,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);

            ierr = MatSetValue(R,Ind_1(2*x,2*y+1,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),1.0,INSERT_VALUES); CHKERRQ(ierr);

            ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);
            ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,Ni,comp,2*nx),Ind_1(x+1,y,Ni,comp,nx),0.5,INSERT_VALUES); CHKERRQ(ierr);

            ierr = MatSetValue(R,Ind_1(2*x,2*y,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),1,INSERT_VALUES); CHKERRQ(ierr);
        }
        if(!separate_vol) {
            //Interpolation for Volume
            for (comp = 0; comp < Nc - 1; comp++) {

                ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y, Ni + 1, comp, 2 * nx), Ind_1(x, y, Ni + 1, comp, nx),
                                   0.5, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y, Ni + 1, comp, 2 * nx),
                                   Ind_1(x + 1, y, Ni + 1, comp, nx), 0.5, INSERT_VALUES);
                CHKERRQ(ierr);

                ierr = MatSetValue(R, Ind_1(2 * x, 2 * y + 1, Ni + 1, comp, 2 * nx), Ind_1(x, y, Ni + 1, comp, nx),
                                   1.0, INSERT_VALUES);
                CHKERRQ(ierr);

                ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, 2 * nx),
                                   Ind_1(x, y, Ni + 1, comp, nx), 0.5, INSERT_VALUES);
                CHKERRQ(ierr);
                ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, 2 * nx),
                                   Ind_1(x + 1, y, Ni + 1, comp, nx), 0.5, INSERT_VALUES);
                CHKERRQ(ierr);

                ierr = MatSetValue(R, Ind_1(2 * x, 2 * y, Ni + 1, comp, 2 * nx), Ind_1(x, y, Ni + 1, comp, nx), 1,
                                   INSERT_VALUES);
                CHKERRQ(ierr);
            }
        }
    }
    x=nx-1;y=ny-1;
    //Interpolation for concentrations
    for(ion=0;ion<Ni;ion++) {
        for(comp=0;comp<Nc;comp++) {
            ierr = MatSetValue(R,Ind_1(2*x+1,2*y,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),1.0,INSERT_VALUES); CHKERRQ(ierr);

            ierr = MatSetValue(R,Ind_1(2*x,2*y+1,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),1.0,INSERT_VALUES); CHKERRQ(ierr);

            ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),1.0,INSERT_VALUES); CHKERRQ(ierr);

            ierr = MatSetValue(R,Ind_1(2*x,2*y,ion,comp,2*nx),Ind_1(x,y,ion,comp,nx),1,INSERT_VALUES); CHKERRQ(ierr);

        }
    }
    //Interpolation for Voltage
    for(comp=0;comp<Nc;comp++) {
        ierr = MatSetValue(R,Ind_1(2*x+1,2*y,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),1.0,INSERT_VALUES); CHKERRQ(ierr);

        ierr = MatSetValue(R,Ind_1(2*x,2*y+1,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),1.0,INSERT_VALUES); CHKERRQ(ierr);

        ierr = MatSetValue(R,Ind_1(2*x+1,2*y+1,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),1.0,INSERT_VALUES); CHKERRQ(ierr);

        ierr = MatSetValue(R,Ind_1(2*x,2*y,Ni,comp,2*nx),Ind_1(x,y,Ni,comp,nx),1,INSERT_VALUES); CHKERRQ(ierr);
    }
    if(!separate_vol) {
        //Interpolation for Volume
        for (comp = 0; comp < Nc - 1; comp++) {

            ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y, Ni + 1, comp, 2 * nx), Ind_1(x, y, Ni + 1, comp, nx), 1.0,
                               INSERT_VALUES);
            CHKERRQ(ierr);

            ierr = MatSetValue(R, Ind_1(2 * x, 2 * y + 1, Ni + 1, comp, 2 * nx), Ind_1(x, y, Ni + 1, comp, nx), 1.0,
                               INSERT_VALUES);
            CHKERRQ(ierr);

            ierr = MatSetValue(R, Ind_1(2 * x + 1, 2 * y + 1, Ni + 1, comp, 2 * nx), Ind_1(x, y, Ni + 1, comp, nx),
                               1.0, INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValue(R, Ind_1(2 * x, 2 * y, Ni + 1, comp, 2 * nx), Ind_1(x, y, Ni + 1, comp, nx), 1,
                               INSERT_VALUES);
            CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode Initialize_PCMG(PC pc,Mat A,struct AppCtx*user)
{
    PetscErrorCode  ierr;

    KSP coarse_ksp,sksp;
    PC coarse_pc,spc;


    PetscInt nx = user->Nx;
    PetscInt ny = user->Ny;
    PetscInt nlevels = (PetscInt) log2(nx);
    Mat R,P;

    ierr = PCSetType(pc,PCMG); CHKERRQ(ierr);
    ierr = PCSetOperators(pc,A,A);CHKERRQ(ierr);
    ierr = PCMGSetType(pc,PC_MG_MULTIPLICATIVE); CHKERRQ(ierr);
//    ierr = PCMGSetType(pc,PC_MG_KASKADE); CHKERRQ(ierr);
    ierr = PCMGSetGalerkin(pc,PC_MG_GALERKIN_BOTH); CHKERRQ(ierr);
    PCMGSetLevels(pc,nlevels,PETSC_NULL);
//    ierr = PCMGSetCycleType(pc,	PC_MG_CYCLE_V); CHKERRQ(ierr);
    ierr = PCMGSetCycleType(pc,PC_MG_CYCLE_W); CHKERRQ(ierr);



    //Set coarse solve
    ierr = PCMGGetCoarseSolve(pc,&coarse_ksp); CHKERRQ(ierr);
    ierr = KSPSetType(coarse_ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(coarse_ksp,&coarse_pc);CHKERRQ(ierr);
    ierr = PCSetType(coarse_pc,PCLU); CHKERRQ(ierr);
//    ierr = PCFactorSetMatSolverPackage(coarse_pc, MATSOLVERSUPERLU); CHKERRQ(ierr);
    PCFactorSetMatSolverType(coarse_pc, MATSOLVERSUPERLU);

    //Make restriction operators
    for (int i=nlevels-1; i>0; i--) {
        ierr = MatCreate(PETSC_COMM_WORLD,&R);CHKERRQ(ierr);
        ierr = MatSetType(R,MATSEQAIJ);CHKERRQ(ierr);
        ierr = MatSetSizes(R,PETSC_DECIDE,PETSC_DECIDE,nx/2*ny/2*Nv,nx*ny*Nv);CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(R,9,NULL);CHKERRQ(ierr);
        ierr = MatSetOption(R,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE); CHKERRQ(ierr);

        ierr = Create_Restriction(R,nx,ny); CHKERRQ(ierr);

        ierr = PCMGSetRestriction(pc,i,R); CHKERRQ(ierr);
//        ierr = PCMGSetResidual(pc,i,PCMGResidualDefault,A); CHKERRQ(ierr);

        ierr = MatDestroy(&R); CHKERRQ(ierr);

        nx = nx/2;
        ny = ny/2;
    }

    // Make interpolation Ops
    for (int i=1; i<nlevels; i++) {
        ierr = MatCreate(PETSC_COMM_WORLD,&P);CHKERRQ(ierr);
        ierr = MatSetType(P,MATSEQAIJ);CHKERRQ(ierr);
        ierr = MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,nx*2*ny*2*Nv,nx*ny*Nv);CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(P,4,NULL);CHKERRQ(ierr);
        ierr = MatSetOption(P,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE); CHKERRQ(ierr);

        ierr = Create_Interpolation(P,nx,ny); CHKERRQ(ierr);

        ierr = PCMGSetInterpolation(pc,i,P); CHKERRQ(ierr);

        ierr = MatDestroy(&P); CHKERRQ(ierr);

        nx = nx*2;
        ny = ny*2;
    }


    //Modify the smoother (default KSP is chebyshev with SOR)
    for(int i=1;i<nlevels;i++){
        ierr = PCMGGetSmoother(pc,i,&sksp); CHKERRQ(ierr);
        ierr = KSPGetPC(sksp,&spc); CHKERRQ(ierr);


        //Smoother KSP
        ierr = KSPSetType(sksp,KSPRICHARDSON); CHKERRQ(ierr);
        ierr = KSPRichardsonSetScale(sksp,1.0); CHKERRQ(ierr);
//        ierr = KSPSetType(sksp,KSPBCGS); CHKERRQ(ierr);
//        ierr = KSPSetType(sksp,KSPGMRES); CHKERRQ(ierr);
//        ierr = KSPSetType(sksp,KSPPREONLY); CHKERRQ(ierr);
        //Smoother Precond
//        /*
        ierr = PCSetType(spc,PCSOR); CHKERRQ(ierr);
//        ierr = PCSORSetSymmetric(spc,SOR_LOCAL_BACKWARD_SWEEP); CHKERRQ(ierr);
//        ierr = PCSORSetSymmetric(spc,SOR_LOCAL_SYMMETRIC_SWEEP); CHKERRQ(ierr);
        ierr = PCSORSetSymmetric(spc,SOR_LOCAL_FORWARD_SWEEP); CHKERRQ(ierr);
        ierr = PCSORSetIterations(spc,2,2); CHKERRQ(ierr);
        ierr = PCSORSetOmega(spc,1.0);
//         */
//        ierr = PCSetType(spc, PCJACOBI);CHKERRQ(ierr);
//        ierr = PCJacobiSetType(spc,PC_JACOBI_ROWMAX); CHKERRQ(ierr);
        /*
        ierr = PCSetType(spc, PCASM); CHKERRQ(ierr);
        ierr = PCASMSetType(spc,PC_ASM_BASIC); CHKERRQ(ierr);
        ierr = PCASMSetLocalType(spc,PC_COMPOSITE_ADDITIVE); CHKERRQ(ierr);
//        ierr = PCASMSetLocalType(spc,PC_COMPOSITE_MULTIPLICATIVE); CHKERRQ(ierr);
         */

        /*
        ierr = PCSetType(spc,PCILU);CHKERRQ(ierr);
        ierr = PCFactorSetFill(spc,3.0);CHKERRQ(ierr);
        ierr = PCFactorSetLevels(spc,1);CHKERRQ(ierr);
        ierr = PCFactorSetAllowDiagonalFill(spc,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PCFactorSetMatOrderingType(spc,MATORDERINGRCM); CHKERRQ(ierr);
//        ierr = PCFactorSetReuseOrdering(spc,PETSC_TRUE); CHKERRQ(ierr);
        */
    }

    return ierr;
}