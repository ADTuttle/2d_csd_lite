#include "constants.h"
#include "functions.h"

//Array functions and things

PetscInt c_index(PetscInt x,PetscInt y,PetscInt comp,PetscInt ion,PetscInt Nx)
{
    return Nc*Ni* (Nx * y + x) + comp*Ni+ion;
}
PetscInt phi_index(PetscInt x,PetscInt y,PetscInt comp,PetscInt Nx)
{
    return Nc* (Nx * y + x) + comp;
}
PetscInt al_index(PetscInt x,PetscInt y,PetscInt comp,PetscInt Nx)
{
    return (Nc-1)* (Nx * y + x) + comp;
}
PetscInt xy_index(PetscInt x,PetscInt y,PetscInt Nx)
{
    return Nx*y+x;
}
//Index based on Nv, which can change to either include or exclude alpha
PetscInt Ind_1(PetscInt x,PetscInt y,PetscInt ion,PetscInt comp,PetscInt Nx)
{
    return Nv*(Nx*y+x)+ion*Nc+comp;
}
// Index based on solving c,phi, and alpha.
PetscInt Ind_2(PetscInt x,PetscInt y,PetscInt ion,PetscInt comp, PetscInt nx)
{
    return ((Ni+2)*Nc-1)*(nx*y+x)+ion*Nc+comp;
}

PetscErrorCode init_simstate(Vec state,struct SimState *state_vars,struct AppCtx *user)
{
    PetscErrorCode ierr;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    //Setup indices
    int x,y,comp,ion;
    PetscInt *c_ind = (PetscInt *) malloc(sizeof(PetscInt)*Nx*Ny*Nc*Ni);
    PetscInt *phi_ind = (PetscInt *) malloc(sizeof(PetscInt)*Nx*Ny*Nc);
    for(x=0;x<Nx;x++){
        for(y=0;y<Ny;y++){
            for(comp=0;comp<Nc;comp++)
            {
                for(ion=0;ion<Ni;ion++)
                {
                    c_ind[c_index(x,y,comp,ion,Nx)] = Ind_1(x,y,ion,comp,Nx);
                }
                phi_ind[phi_index(x,y,comp,Nx)] = Ind_1(x,y,Ni,comp,Nx);
            }
        }
    }
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,Nx*Ny*Ni*Nc,c_ind,PETSC_COPY_VALUES,&state_vars->c_ind); CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_WORLD,Nx*Ny*Nc,phi_ind,PETSC_COPY_VALUES,&state_vars->phi_ind); CHKERRQ(ierr);

    free(phi_ind);free(c_ind);
    if(!separate_vol) {
        PetscInt *al_ind = (PetscInt *) malloc(sizeof(PetscInt)*Nx*Ny*(Nc-1));
        for (x = 0; x < Nx; x++) {
            for (y = 0; y < Ny; y++) {
                for (comp = 0; comp < Nc - 1; comp++) {
                    al_ind[al_index(x, y, comp,Nx)] = Ind_1(x, y, Ni + 1, comp,Nx);
                }
            }
        }
        ierr = ISCreateGeneral(PETSC_COMM_WORLD, Nx * Ny * (Nc - 1), al_ind, PETSC_COPY_VALUES, &state_vars->al_ind);
        CHKERRQ(ierr);
        free(al_ind);
    }
    else{
        state_vars->alpha = (PetscReal*)malloc(sizeof(PetscReal)*Nx*Ny*(Nc-1));
    }
    extract_subarray(state,state_vars);
    return ierr;
}

PetscErrorCode extract_subarray(Vec state,struct SimState *state_vars)
{
    if(Profiling_on) {
        PetscLogEventBegin(event[2], 0, 0, 0, 0);
    }
    PetscErrorCode ierr;
    ierr = VecGetSubVector(state,state_vars->c_ind,&state_vars->c_vec); CHKERRQ(ierr);
    ierr = VecGetArray(state_vars->c_vec,&state_vars->c); CHKERRQ(ierr);

    ierr = VecGetSubVector(state,state_vars->phi_ind,&state_vars->phi_vec); CHKERRQ(ierr);
    ierr = VecGetArray(state_vars->phi_vec,&state_vars->phi); CHKERRQ(ierr);
    if(!separate_vol) {
        ierr = VecGetSubVector(state, state_vars->al_ind, &state_vars->al_vec);
        CHKERRQ(ierr);
        ierr = VecGetArray(state_vars->al_vec, &state_vars->alpha);
        CHKERRQ(ierr);
    }
    if(Profiling_on) {
        PetscLogEventEnd(event[2], 0, 0, 0, 0);
    }

    return ierr;

}

PetscErrorCode restore_subarray(Vec state,struct SimState *state_vars)
{
    if(Profiling_on) {
        PetscLogEventBegin(event[3], 0, 0, 0, 0);
    }
    PetscErrorCode ierr;

    ierr = VecRestoreArray(state_vars->c_vec,&state_vars->c); CHKERRQ(ierr);
    ierr = VecRestoreSubVector(state,state_vars->c_ind,&state_vars->c_vec); CHKERRQ(ierr);

    ierr = VecRestoreArray(state_vars->phi_vec,&state_vars->phi); CHKERRQ(ierr);
    ierr = VecRestoreSubVector(state,state_vars->phi_ind,&state_vars->phi_vec); CHKERRQ(ierr);

    if(!separate_vol) {
        ierr = VecRestoreArray(state_vars->al_vec, &state_vars->alpha);
        CHKERRQ(ierr);
        ierr = VecRestoreSubVector(state, state_vars->al_ind, &state_vars->al_vec);
        CHKERRQ(ierr);
        state_vars->alpha = NULL;
    }

    state_vars->c = NULL;
    state_vars->phi = NULL;
    if(Profiling_on) {
        PetscLogEventEnd(event[3], 0, 0, 0, 0);
    }

    return ierr;

}
PetscErrorCode extract_subarray_Read(Vec state,struct SimState *state_vars)
{
    if(Profiling_on) {
        PetscLogEventBegin(event[2], 0, 0, 0, 0);
    }
    PetscErrorCode ierr;
    ierr = VecGetSubVector(state,state_vars->c_ind,&state_vars->c_vec); CHKERRQ(ierr);
    ierr = VecGetArrayRead(state_vars->c_vec,&state_vars->c); CHKERRQ(ierr);

    ierr = VecGetSubVector(state,state_vars->phi_ind,&state_vars->phi_vec); CHKERRQ(ierr);
    ierr = VecGetArrayRead(state_vars->phi_vec,&state_vars->phi); CHKERRQ(ierr);
    if(!separate_vol) {
        ierr = VecGetSubVector(state, state_vars->al_ind, &state_vars->al_vec);
        CHKERRQ(ierr);
        ierr = VecGetArrayRead(state_vars->al_vec, &state_vars->alpha);
        CHKERRQ(ierr);
    }
    if(Profiling_on) {
        PetscLogEventEnd(event[2], 0, 0, 0, 0);
    }

    return ierr;

}

PetscErrorCode restore_subarray_Read(Vec state,struct SimState *state_vars)
{
    if(Profiling_on) {
        PetscLogEventBegin(event[3], 0, 0, 0, 0);
    }
    PetscErrorCode ierr;

    ierr = VecRestoreArrayRead(state_vars->c_vec,&state_vars->c); CHKERRQ(ierr);
    ierr = VecRestoreSubVector(state,state_vars->c_ind,&state_vars->c_vec); CHKERRQ(ierr);

    ierr = VecRestoreArrayRead(state_vars->phi_vec,&state_vars->phi); CHKERRQ(ierr);
    ierr = VecRestoreSubVector(state,state_vars->phi_ind,&state_vars->phi_vec); CHKERRQ(ierr);

    if(!separate_vol) {
        ierr = VecRestoreArrayRead(state_vars->al_vec, &state_vars->alpha);
        CHKERRQ(ierr);
        ierr = VecRestoreSubVector(state, state_vars->al_ind, &state_vars->al_vec);
        CHKERRQ(ierr);
        state_vars->alpha = NULL;
    }

    state_vars->c = NULL;
    state_vars->phi = NULL;
    if(Profiling_on) {
        PetscLogEventEnd(event[3], 0, 0, 0, 0);
    }

    return ierr;

}
PetscErrorCode copy_simstate(Vec current_state,struct SimState *state_vars_past)
{
    PetscErrorCode ierr;
    ierr = VecCopy(current_state,state_vars_past->v); CHKERRQ(ierr);
    ierr = extract_subarray(state_vars_past->v,state_vars_past); CHKERRQ(ierr);
    return ierr;
}

void init_arrays(struct AppCtx*user)
{
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    PetscInt nx = 2*width_size+1;
    PetscInt ny = 2*width_size+1;
    //Flux quantities
    user->flux->mflux = (PetscReal*) malloc(Nx*Ny*Ni*Nc*sizeof(PetscReal));
    user->flux->dfdci = (PetscReal*) malloc(Nx*Ny*Ni*Nc*sizeof(PetscReal));
    user->flux->dfdce = (PetscReal*) malloc(Nx*Ny*Ni*Nc*sizeof(PetscReal));
    user->flux->dfdphim = (PetscReal*) malloc(Nx*Ny*Ni*Nc*sizeof(PetscReal));
    user->flux->wflow = (PetscReal*) malloc(Nx*Ny*(Nc-1)*sizeof(PetscReal));
    user->flux->dwdpi = (PetscReal*) malloc(Nx*Ny*(Nc-1)*sizeof(PetscReal));
    user->flux->dwdal = (PetscReal*) malloc(Nx*Ny*(Nc-1)*sizeof(PetscReal));

    //Gating variables (present)
    user->gate_vars->mNaT = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->hNaT = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->gNaT = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->mNaP = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->hNaP = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->gNaP = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->mKDR = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->gKDR = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->mKA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->hKA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->gKA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->yNMDA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->zNMDA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->dNMDA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars->gNMDA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));

    //Gating variables (past)
    user->gate_vars_past->mNaT = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->hNaT = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->gNaT = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->mNaP = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->hNaP = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->gNaP = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->mKDR = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->gKDR = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->mKA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->hKA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->gKA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->yNMDA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->zNMDA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->dNMDA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gate_vars_past->gNMDA = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));


    //Excitation
    user->gexct->pNa = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gexct->pK = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gexct->pCl = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    user->gexct->pGlu = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));

    //Constant params
    user->con_vars->ao = (PetscReal*) malloc(Nx*Ny*Nc*sizeof(PetscReal));
    user->con_vars->zo = (PetscReal*) malloc(Nx*Ny*Nc*sizeof(PetscReal));
    user->con_vars->zeta1 = (PetscReal*) malloc(Nx*Ny*(Nc-1)*sizeof(PetscReal));
    user->con_vars->zetaalpha = (PetscReal*) malloc((Nc-1)*sizeof(PetscReal));

    //Diffusion in ctx
    user->Dcs = (PetscReal*) malloc(Nx*Ny*Ni*Nc*2*sizeof(PetscReal));
    user->Dcb = (PetscReal*) malloc(Nx*Ny*Ni*Nc*2*sizeof(PetscReal));

    //Small Grid variables

    // Past membrane voltage storage
    user->vm_past = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    //Grid Gating variables
    user->grid_gate_vars->mNaT = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->hNaT = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->gNaT = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->mNaP = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->hNaP = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->gNaP = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->mKDR = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->gKDR = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->mKA = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->hKA = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->gKA = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->yNMDA = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->zNMDA = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->dNMDA = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));
    user->grid_gate_vars->gNMDA = (PetscReal*) malloc(nx*ny*sizeof(PetscReal));

    //Grid state_vars
    user->grid_vars->c = (PetscReal*) malloc(Nc*Ni*nx*ny*sizeof(PetscReal));
    user->grid_vars->phi = (PetscReal*) malloc(Nc*nx*ny*sizeof(PetscReal));
    user->grid_vars->alpha = (PetscReal*) malloc((Nc-1)*nx*ny*sizeof(PetscReal));
    user->grid_vars->v = NULL;
    user->grid_vars->phi_ind = NULL;
    user->grid_vars->phi_vec = NULL;
    user->grid_vars->c_ind = NULL;
    user->grid_vars->c_vec = NULL;
    user->grid_vars->al_ind = NULL;
    user->grid_vars->al_vec = NULL;

    //Grid past state_Vars

    user->grid_vars_past->c = (PetscReal*) malloc(Nc*Ni*nx*ny*sizeof(PetscReal));
    user->grid_vars_past->phi = (PetscReal*) malloc(Nc*nx*ny*sizeof(PetscReal));
    user->grid_vars_past->alpha = (PetscReal*) malloc((Nc-1)*nx*ny*sizeof(PetscReal));
    user->grid_vars_past->v = NULL;
    user->grid_vars_past->phi_ind = NULL;
    user->grid_vars_past->phi_vec = NULL;
    user->grid_vars_past->c_ind = NULL;
    user->grid_vars_past->c_vec = NULL;
    user->grid_vars_past->al_ind = NULL;
    user->grid_vars_past->al_vec = NULL;

    //dt saving
    user->dt_space = (PetscReal*) malloc(Nx*Ny*sizeof(PetscReal));
    for(int x=0;x<Nx;x++){
        for(int y=0;y<Ny;y++){
            user->dt_space[xy_index(x,y,Nx)]=user->dt;
        }
    }



}

void parameter_dependence(struct AppCtx *user)
{
    struct ConstVars *con_vars = user->con_vars;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    PetscInt x,y;

    //Gating variables
    con_vars->pNaT = (PetscReal*)malloc(sizeof(PetscReal)*Nx*Ny);
    con_vars->pNaP = (PetscReal*)malloc(sizeof(PetscReal)*Nx*Ny);
    con_vars->pKDR = (PetscReal*)malloc(sizeof(PetscReal)*Nx*Ny);
    con_vars->pKA = (PetscReal*)malloc(sizeof(PetscReal)*Nx*Ny);
    con_vars->pNMDA = (PetscReal*)malloc(sizeof(PetscReal)*Nx*Ny);

    con_vars->pKIR = (PetscReal*)malloc(sizeof(PetscReal)*Nx*Ny);
    //Glial diffusion scaling
    con_vars->DNeuronScale = (PetscReal*)malloc(sizeof(PetscReal)*2*Nx*Ny);
    con_vars->DGliaScale = (PetscReal*)malloc(sizeof(PetscReal)*2*Nx*Ny);
    con_vars->DExtracellScale = (PetscReal*)malloc(sizeof(PetscReal)*2*Nx*Ny);
    for(x=0;x<Nx;x++){
        for(y=0;y<Ny;y++){
            con_vars->pNaT[xy_index(x,y,Nx)]=basepNaT;
            con_vars->pNaP[xy_index(x,y,Nx)]=basepNaP;//*((double)x)/((double)Nx-1);
            con_vars->pKDR[xy_index(x,y,Nx)]=basepKDR;
            con_vars->pKA[xy_index(x,y,Nx)]=basepKA;
            con_vars->pNMDA[xy_index(x,y,Nx)]=basepNMDA;//*((double)x)/((double)Nx-1);

            con_vars->pKIR[xy_index(x,y,Nx)]=basepKIR;

            con_vars->DNeuronScale[xy_index(x,y,Nx)*2]=DNeuronMult[0]; //x-direction Neurons
            con_vars->DNeuronScale[xy_index(x,y,Nx)*2+1]=DNeuronMult[1]; //y-direction Neurons
            con_vars->DGliaScale[xy_index(x,y,Nx)*2]=DGliaMult[0]; //x-direction scale Glia
            con_vars->DGliaScale[xy_index(x,y,Nx)*2+1]=DGliaMult[1]; // y-direction scale glia
            con_vars->DExtracellScale[xy_index(x,y,Nx)*2]=DExtraMult[0]; //x-direction scale extracell
            con_vars->DExtracellScale[xy_index(x,y,Nx)*2+1]=DExtraMult[1]; // y-direction scale Extracell

        }
    }




    //Variables that get set during set_params to steady state necessary values
    con_vars->pNaKCl = (PetscReal*)malloc(sizeof(PetscReal)*Nx*Ny);
    con_vars->Imax = (PetscReal*)malloc(sizeof(PetscReal)*Nx*Ny);
    con_vars->Imaxg = (PetscReal*)malloc(sizeof(PetscReal)*Nx*Ny);
    con_vars->pNaLeak = (PetscReal*)malloc(sizeof(PetscReal)*Nx*Ny);
    con_vars->pNaLeakg = (PetscReal*)malloc(sizeof(PetscReal)*Nx*Ny);

//    con_vars->zo = (PetscReal*)malloc(sizeof(PetscReal)*Nc);
//    con_vars->ao = (PetscReal*)malloc(sizeof(PetscReal)*Nc);
//    con_vars->zeta1 = (PetscReal*)malloc(sizeof(PetscReal)*(Nc-1));


}

