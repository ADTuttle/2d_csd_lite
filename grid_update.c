#include "constants.h"
#include "functions.h"

void Load_Grid(struct AppCtx *user,PetscInt xi,PetscInt yi){
    if(Profiling_on) {
        PetscLogEventBegin(event[12], 0, 0, 0, 0);
    }
    struct SimState *state_vars = user->state_vars_past;
    struct SimState * grid_vars = user->grid_vars;

    struct GateType *gate_vars = user->gate_vars_past;
    struct GateType *grid_gate = user->grid_gate_vars;

    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    PetscInt nx = 2*width_size+1;
    PetscInt ny = 2*width_size+1;
    PetscInt xind,yind,ion,comp,x,y;

    for( x=0;x<nx;x++) {
        for ( y = 0; y < ny; y++) {
            xind = x-width_size+xi;
            yind = y-width_size+yi;
            //If in interior just copy
            if(xind>-1 && yind>-1 && xind<Nx && yind<Ny) {

                grid_gate->mNaT[xy_index(x, y, nx)] = gate_vars->mNaT[xy_index(xind, yind, Nx)];
                grid_gate->hNaT[xy_index(x, y, nx)] = gate_vars->hNaT[xy_index(xind, yind, Nx)];
                grid_gate->gNaT[xy_index(x, y, nx)] = gate_vars->gNaT[xy_index(xind, yind, Nx)];
                grid_gate->mNaP[xy_index(x, y, nx)] = gate_vars->mNaP[xy_index(xind, yind, Nx)];
                grid_gate->hNaP[xy_index(x, y, nx)] = gate_vars->hNaP[xy_index(xind, yind, Nx)];
                grid_gate->gNaP[xy_index(x, y, nx)] = gate_vars->gNaP[xy_index(xind, yind, Nx)];
                grid_gate->mKDR[xy_index(x, y, nx)] = gate_vars->mKDR[xy_index(xind, yind, Nx)];
                grid_gate->gKDR[xy_index(x, y, nx)] = gate_vars->gKDR[xy_index(xind, yind, Nx)];
                grid_gate->mKA[xy_index(x, y, nx)] = gate_vars->mKA[xy_index(xind, yind, Nx)];
                grid_gate->hKA[xy_index(x, y, nx)] = gate_vars->hKA[xy_index(xind, yind, Nx)];
                grid_gate->gKA[xy_index(x, y, nx)] = gate_vars->gKA[xy_index(xind, yind, Nx)];
                grid_gate->yNMDA[xy_index(x, y, nx)] = gate_vars->yNMDA[xy_index(xind, yind, Nx)];
                grid_gate->zNMDA[xy_index(x, y, nx)] = gate_vars->zNMDA[xy_index(xind, yind, Nx)];
                grid_gate->dNMDA[xy_index(x, y, nx)] = gate_vars->dNMDA[xy_index(xind, yind, Nx)];
                grid_gate->gNMDA[xy_index(x, y, nx)] = gate_vars->gNMDA[xy_index(xind, yind, Nx)];

                for (comp = 0; comp < Nc; comp++) {
                    for (ion = 0; ion < Ni; ion++) {
                        grid_vars->c[c_index(x, y, comp, ion, nx)] = state_vars->c[c_index(xind, yind, comp, ion, Nx)];
                    }
                    grid_vars->phi[phi_index(x, y, comp, nx)] = state_vars->phi[phi_index(xind, yind, comp, Nx)];
                }
                for (comp = 0; comp < Nc - 1; comp++) {
                    grid_vars->alpha[al_index(x, y, comp, nx)] = state_vars->alpha[al_index(xind, yind, comp, Nx)];
                }
            } else{
                //If not the interior get closest point
                //Right side
                if(xind==Nx && yind>-1 && yind<Ny){
                    grid_gate->mNaT[xy_index(x, y, nx)] = gate_vars->mNaT[xy_index(xind-1, yind, Nx)];
                    grid_gate->hNaT[xy_index(x, y, nx)] = gate_vars->hNaT[xy_index(xind-1, yind, Nx)];
                    grid_gate->gNaT[xy_index(x, y, nx)] = gate_vars->gNaT[xy_index(xind-1, yind, Nx)];
                    grid_gate->mNaP[xy_index(x, y, nx)] = gate_vars->mNaP[xy_index(xind-1, yind, Nx)];
                    grid_gate->hNaP[xy_index(x, y, nx)] = gate_vars->hNaP[xy_index(xind-1, yind, Nx)];
                    grid_gate->gNaP[xy_index(x, y, nx)] = gate_vars->gNaP[xy_index(xind-1, yind, Nx)];
                    grid_gate->mKDR[xy_index(x, y, nx)] = gate_vars->mKDR[xy_index(xind-1, yind, Nx)];
                    grid_gate->gKDR[xy_index(x, y, nx)] = gate_vars->gKDR[xy_index(xind-1, yind, Nx)];
                    grid_gate->mKA[xy_index(x, y, nx)] = gate_vars->mKA[xy_index(xind-1, yind, Nx)];
                    grid_gate->hKA[xy_index(x, y, nx)] = gate_vars->hKA[xy_index(xind-1, yind, Nx)];
                    grid_gate->gKA[xy_index(x, y, nx)] = gate_vars->gKA[xy_index(xind-1, yind, Nx)];
                    grid_gate->yNMDA[xy_index(x, y, nx)] = gate_vars->yNMDA[xy_index(xind-1, yind, Nx)];
                    grid_gate->zNMDA[xy_index(x, y, nx)] = gate_vars->zNMDA[xy_index(xind-1, yind, Nx)];
                    grid_gate->dNMDA[xy_index(x, y, nx)] = gate_vars->dNMDA[xy_index(xind-1, yind, Nx)];
                    grid_gate->gNMDA[xy_index(x, y, nx)] = gate_vars->gNMDA[xy_index(xind-1, yind, Nx)];

                    for (comp = 0; comp < Nc; comp++) {
                        for (ion = 0; ion < Ni; ion++) {
                            grid_vars->c[c_index(x, y, comp, ion, nx)] = state_vars->c[c_index(xind-1, yind, comp, ion, Nx)];
                        }
                        grid_vars->phi[phi_index(x, y, comp, nx)] = state_vars->phi[phi_index(xind-1, yind, comp, Nx)];
                    }
                    for (comp = 0; comp < Nc - 1; comp++) {
                        grid_vars->alpha[al_index(x, y, comp, nx)] = state_vars->alpha[al_index(xind-1, yind, comp, Nx)];
                    }
                }
                //Top side
                if(yind==Ny && xind>-1 && xind<Nx){
                    grid_gate->mNaT[xy_index(x, y, nx)] = gate_vars->mNaT[xy_index(xind, yind-1, Nx)];
                    grid_gate->hNaT[xy_index(x, y, nx)] = gate_vars->hNaT[xy_index(xind, yind-1, Nx)];
                    grid_gate->gNaT[xy_index(x, y, nx)] = gate_vars->gNaT[xy_index(xind, yind-1, Nx)];
                    grid_gate->mNaP[xy_index(x, y, nx)] = gate_vars->mNaP[xy_index(xind, yind-1, Nx)];
                    grid_gate->hNaP[xy_index(x, y, nx)] = gate_vars->hNaP[xy_index(xind, yind-1, Nx)];
                    grid_gate->gNaP[xy_index(x, y, nx)] = gate_vars->gNaP[xy_index(xind, yind-1, Nx)];
                    grid_gate->mKDR[xy_index(x, y, nx)] = gate_vars->mKDR[xy_index(xind, yind-1, Nx)];
                    grid_gate->gKDR[xy_index(x, y, nx)] = gate_vars->gKDR[xy_index(xind, yind-1, Nx)];
                    grid_gate->mKA[xy_index(x, y, nx)] = gate_vars->mKA[xy_index(xind, yind-1, Nx)];
                    grid_gate->hKA[xy_index(x, y, nx)] = gate_vars->hKA[xy_index(xind, yind-1, Nx)];
                    grid_gate->gKA[xy_index(x, y, nx)] = gate_vars->gKA[xy_index(xind, yind-1, Nx)];
                    grid_gate->yNMDA[xy_index(x, y, nx)] = gate_vars->yNMDA[xy_index(xind, yind-1, Nx)];
                    grid_gate->zNMDA[xy_index(x, y, nx)] = gate_vars->zNMDA[xy_index(xind, yind-1, Nx)];
                    grid_gate->dNMDA[xy_index(x, y, nx)] = gate_vars->dNMDA[xy_index(xind, yind-1, Nx)];
                    grid_gate->gNMDA[xy_index(x, y, nx)] = gate_vars->gNMDA[xy_index(xind, yind-1, Nx)];

                    for (comp = 0; comp < Nc; comp++) {
                        for (ion = 0; ion < Ni; ion++) {
                            grid_vars->c[c_index(x, y, comp, ion, nx)] = state_vars->c[c_index(xind, yind-1, comp, ion, Nx)];
                        }
                        grid_vars->phi[phi_index(x, y, comp, nx)] = state_vars->phi[phi_index(xind, yind-1, comp, Nx)];
                    }
                    for (comp = 0; comp < Nc - 1; comp++) {
                        grid_vars->alpha[al_index(x, y, comp, nx)] = state_vars->alpha[al_index(xind, yind-1, comp, Nx)];
                    }
                }
                //left side
                if(xind==-1 && yind>-1 && yind<Ny){
                    grid_gate->mNaT[xy_index(x, y, nx)] = gate_vars->mNaT[xy_index(xind+1, yind, Nx)];
                    grid_gate->hNaT[xy_index(x, y, nx)] = gate_vars->hNaT[xy_index(xind+1, yind, Nx)];
                    grid_gate->gNaT[xy_index(x, y, nx)] = gate_vars->gNaT[xy_index(xind+1, yind, Nx)];
                    grid_gate->mNaP[xy_index(x, y, nx)] = gate_vars->mNaP[xy_index(xind+1, yind, Nx)];
                    grid_gate->hNaP[xy_index(x, y, nx)] = gate_vars->hNaP[xy_index(xind+1, yind, Nx)];
                    grid_gate->gNaP[xy_index(x, y, nx)] = gate_vars->gNaP[xy_index(xind+1, yind, Nx)];
                    grid_gate->mKDR[xy_index(x, y, nx)] = gate_vars->mKDR[xy_index(xind+1, yind, Nx)];
                    grid_gate->gKDR[xy_index(x, y, nx)] = gate_vars->gKDR[xy_index(xind+1, yind, Nx)];
                    grid_gate->mKA[xy_index(x, y, nx)] = gate_vars->mKA[xy_index(xind+1, yind, Nx)];
                    grid_gate->hKA[xy_index(x, y, nx)] = gate_vars->hKA[xy_index(xind+1, yind, Nx)];
                    grid_gate->gKA[xy_index(x, y, nx)] = gate_vars->gKA[xy_index(xind+1, yind, Nx)];
                    grid_gate->yNMDA[xy_index(x, y, nx)] = gate_vars->yNMDA[xy_index(xind+1, yind, Nx)];
                    grid_gate->zNMDA[xy_index(x, y, nx)] = gate_vars->zNMDA[xy_index(xind+1, yind, Nx)];
                    grid_gate->dNMDA[xy_index(x, y, nx)] = gate_vars->dNMDA[xy_index(xind+1, yind, Nx)];
                    grid_gate->gNMDA[xy_index(x, y, nx)] = gate_vars->gNMDA[xy_index(xind+1, yind, Nx)];


                    for (comp = 0; comp < Nc; comp++) {
                        for (ion = 0; ion < Ni; ion++) {
                            grid_vars->c[c_index(x, y, comp, ion, nx)] = state_vars->c[c_index(xind+1, yind, comp, ion, Nx)];
                        }
                        grid_vars->phi[phi_index(x, y, comp, nx)] = state_vars->phi[phi_index(xind+1, yind, comp, Nx)];
                    }
                    for (comp = 0; comp < Nc - 1; comp++) {
                        grid_vars->alpha[al_index(x, y, comp, nx)] = state_vars->alpha[al_index(xind+1, yind, comp, Nx)];
                    }
                }
                //Bottom side
                if(yind==-1 && xind>-1 && xind<Nx){
                    grid_gate->mNaT[xy_index(x, y, nx)] = gate_vars->mNaT[xy_index(xind, yind+1, Nx)];
                    grid_gate->hNaT[xy_index(x, y, nx)] = gate_vars->hNaT[xy_index(xind, yind+1, Nx)];
                    grid_gate->gNaT[xy_index(x, y, nx)] = gate_vars->gNaT[xy_index(xind, yind+1, Nx)];
                    grid_gate->mNaP[xy_index(x, y, nx)] = gate_vars->mNaP[xy_index(xind, yind+1, Nx)];
                    grid_gate->hNaP[xy_index(x, y, nx)] = gate_vars->hNaP[xy_index(xind, yind+1, Nx)];
                    grid_gate->gNaP[xy_index(x, y, nx)] = gate_vars->gNaP[xy_index(xind, yind+1, Nx)];
                    grid_gate->mKDR[xy_index(x, y, nx)] = gate_vars->mKDR[xy_index(xind, yind+1, Nx)];
                    grid_gate->gKDR[xy_index(x, y, nx)] = gate_vars->gKDR[xy_index(xind, yind+1, Nx)];
                    grid_gate->mKA[xy_index(x, y, nx)] = gate_vars->mKA[xy_index(xind, yind+1, Nx)];
                    grid_gate->hKA[xy_index(x, y, nx)] = gate_vars->hKA[xy_index(xind, yind+1, Nx)];
                    grid_gate->gKA[xy_index(x, y, nx)] = gate_vars->gKA[xy_index(xind, yind+1, Nx)];
                    grid_gate->yNMDA[xy_index(x, y, nx)] = gate_vars->yNMDA[xy_index(xind, yind+1, Nx)];
                    grid_gate->zNMDA[xy_index(x, y, nx)] = gate_vars->zNMDA[xy_index(xind, yind+1, Nx)];
                    grid_gate->dNMDA[xy_index(x, y, nx)] = gate_vars->dNMDA[xy_index(xind, yind+1, Nx)];
                    grid_gate->gNMDA[xy_index(x, y, nx)] = gate_vars->gNMDA[xy_index(xind, yind+1, Nx)];

                    for (comp = 0; comp < Nc; comp++) {
                        for (ion = 0; ion < Ni; ion++) {
                            grid_vars->c[c_index(x, y, comp, ion, nx)] = state_vars->c[c_index(xind, yind+1, comp, ion, Nx)];
                        }
                        grid_vars->phi[phi_index(x, y, comp, nx)] = state_vars->phi[phi_index(xind, yind+1, comp, Nx)];
                    }
                    for (comp = 0; comp < Nc - 1; comp++) {
                        grid_vars->alpha[al_index(x, y, comp, nx)] = state_vars->alpha[al_index(xind, yind+1, comp, Nx)];
                    }
                }
                //Top Right corner
                if(xind==Nx &&yind==Ny){
                    grid_gate->mNaT[xy_index(x, y, nx)] = gate_vars->mNaT[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->hNaT[xy_index(x, y, nx)] = gate_vars->hNaT[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->gNaT[xy_index(x, y, nx)] = gate_vars->gNaT[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->mNaP[xy_index(x, y, nx)] = gate_vars->mNaP[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->hNaP[xy_index(x, y, nx)] = gate_vars->hNaP[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->gNaP[xy_index(x, y, nx)] = gate_vars->gNaP[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->mKDR[xy_index(x, y, nx)] = gate_vars->mKDR[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->gKDR[xy_index(x, y, nx)] = gate_vars->gKDR[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->mKA[xy_index(x, y, nx)] = gate_vars->mKA[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->hKA[xy_index(x, y, nx)] = gate_vars->hKA[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->gKA[xy_index(x, y, nx)] = gate_vars->gKA[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->yNMDA[xy_index(x, y, nx)] = gate_vars->yNMDA[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->zNMDA[xy_index(x, y, nx)] = gate_vars->zNMDA[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->dNMDA[xy_index(x, y, nx)] = gate_vars->dNMDA[xy_index(xind-1, yind-1, Nx)];
                    grid_gate->gNMDA[xy_index(x, y, nx)] = gate_vars->gNMDA[xy_index(xind-1, yind-1, Nx)];

                    for (comp = 0; comp < Nc; comp++) {
                        for (ion = 0; ion < Ni; ion++) {
                            grid_vars->c[c_index(x, y, comp, ion, nx)] = state_vars->c[c_index(xind-1, yind-1, comp, ion, Nx)];
                        }
                        grid_vars->phi[phi_index(x, y, comp, nx)] = state_vars->phi[phi_index(xind-1, yind-1, comp, Nx)];
                    }
                    for (comp = 0; comp < Nc - 1; comp++) {
                        grid_vars->alpha[al_index(x, y, comp, nx)] = state_vars->alpha[al_index(xind-1, yind-1, comp, Nx)];
                    }
                }
                // Top left corner
                if(yind==Ny && xind==-1){
                    grid_gate->mNaT[xy_index(x, y, nx)] = gate_vars->mNaT[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->hNaT[xy_index(x, y, nx)] = gate_vars->hNaT[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->gNaT[xy_index(x, y, nx)] = gate_vars->gNaT[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->mNaP[xy_index(x, y, nx)] = gate_vars->mNaP[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->hNaP[xy_index(x, y, nx)] = gate_vars->hNaP[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->gNaP[xy_index(x, y, nx)] = gate_vars->gNaP[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->mKDR[xy_index(x, y, nx)] = gate_vars->mKDR[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->gKDR[xy_index(x, y, nx)] = gate_vars->gKDR[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->mKA[xy_index(x, y, nx)] = gate_vars->mKA[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->hKA[xy_index(x, y, nx)] = gate_vars->hKA[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->gKA[xy_index(x, y, nx)] = gate_vars->gKA[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->yNMDA[xy_index(x, y, nx)] = gate_vars->yNMDA[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->dNMDA[xy_index(x, y, nx)] = gate_vars->dNMDA[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->zNMDA[xy_index(x, y, nx)] = gate_vars->zNMDA[xy_index(xind+1, yind-1, Nx)];
                    grid_gate->gNMDA[xy_index(x, y, nx)] = gate_vars->gNMDA[xy_index(xind+1, yind-1, Nx)];

                    for (comp = 0; comp < Nc; comp++) {
                        for (ion = 0; ion < Ni; ion++) {
                            grid_vars->c[c_index(x, y, comp, ion, nx)] = state_vars->c[c_index(xind+1, yind-1, comp, ion, Nx)];
                        }
                        grid_vars->phi[phi_index(x, y, comp, nx)] = state_vars->phi[phi_index(xind+1, yind-1, comp, Nx)];
                    }
                    for (comp = 0; comp < Nc - 1; comp++) {
                        grid_vars->alpha[al_index(x, y, comp, nx)] = state_vars->alpha[al_index(xind+1, yind-1, comp, Nx)];
                    }
                }
                //Bottom left
                if(xind==-1 && yind==-1){
                    grid_gate->mNaT[xy_index(x, y, nx)] = gate_vars->mNaT[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->hNaT[xy_index(x, y, nx)] = gate_vars->hNaT[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->gNaT[xy_index(x, y, nx)] = gate_vars->gNaT[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->mNaP[xy_index(x, y, nx)] = gate_vars->mNaP[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->hNaP[xy_index(x, y, nx)] = gate_vars->hNaP[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->gNaP[xy_index(x, y, nx)] = gate_vars->gNaP[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->mKDR[xy_index(x, y, nx)] = gate_vars->mKDR[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->gKDR[xy_index(x, y, nx)] = gate_vars->gKDR[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->mKA[xy_index(x, y, nx)] = gate_vars->mKA[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->hKA[xy_index(x, y, nx)] = gate_vars->hKA[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->gKA[xy_index(x, y, nx)] = gate_vars->gKA[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->yNMDA[xy_index(x, y, nx)] = gate_vars->yNMDA[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->zNMDA[xy_index(x, y, nx)] = gate_vars->zNMDA[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->dNMDA[xy_index(x, y, nx)] = gate_vars->dNMDA[xy_index(xind+1, yind+1, Nx)];
                    grid_gate->gNMDA[xy_index(x, y, nx)] = gate_vars->gNMDA[xy_index(xind+1, yind+1, Nx)];

                    for (comp = 0; comp < Nc; comp++) {
                        for (ion = 0; ion < Ni; ion++) {
                            grid_vars->c[c_index(x, y, comp, ion, nx)] = state_vars->c[c_index(xind+1, yind+1, comp, ion, Nx)];
                        }
                        grid_vars->phi[phi_index(x, y, comp, nx)] = state_vars->phi[phi_index(xind+1, yind+1, comp, Nx)];
                    }
                    for (comp = 0; comp < Nc - 1; comp++) {
                        grid_vars->alpha[al_index(x, y, comp, nx)] = state_vars->alpha[al_index(xind+1, yind+1, comp, Nx)];
                    }
                }
                //Bottom right
                if(xind==Nx && yind==-1){
                    grid_gate->mNaT[xy_index(x, y, nx)] = gate_vars->mNaT[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->hNaT[xy_index(x, y, nx)] = gate_vars->hNaT[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->gNaT[xy_index(x, y, nx)] = gate_vars->gNaT[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->mNaP[xy_index(x, y, nx)] = gate_vars->mNaP[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->hNaP[xy_index(x, y, nx)] = gate_vars->hNaP[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->gNaP[xy_index(x, y, nx)] = gate_vars->gNaP[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->mKDR[xy_index(x, y, nx)] = gate_vars->mKDR[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->gKDR[xy_index(x, y, nx)] = gate_vars->gKDR[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->mKA[xy_index(x, y, nx)] = gate_vars->mKA[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->hKA[xy_index(x, y, nx)] = gate_vars->hKA[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->gKA[xy_index(x, y, nx)] = gate_vars->gKA[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->yNMDA[xy_index(x, y, nx)] = gate_vars->yNMDA[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->zNMDA[xy_index(x, y, nx)] = gate_vars->zNMDA[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->dNMDA[xy_index(x, y, nx)] = gate_vars->dNMDA[xy_index(xind-1, yind+1, Nx)];
                    grid_gate->gNMDA[xy_index(x, y, nx)] = gate_vars->gNMDA[xy_index(xind-1, yind+1, Nx)];

                    for (comp = 0; comp < Nc; comp++) {
                        for (ion = 0; ion < Ni; ion++) {
                            grid_vars->c[c_index(x, y, comp, ion, nx)] = state_vars->c[c_index(xind-1, yind+1, comp, ion, Nx)];
                        }
                        grid_vars->phi[phi_index(x, y, comp, nx)] = state_vars->phi[phi_index(xind-1, yind+1, comp, Nx)];
                    }
                    for (comp = 0; comp < Nc - 1; comp++) {
                        grid_vars->alpha[al_index(x, y, comp, nx)] = state_vars->alpha[al_index(xind-1, yind+1, comp, Nx)];
                    }
                }


            }
        }
    }
    if(Profiling_on) {
        PetscLogEventEnd(event[12], 0, 0, 0, 0);
    }
}

void Unload_Grid(struct AppCtx *user,PetscInt x, PetscInt y){
    if(Profiling_on) {
        PetscLogEventBegin(event[13], 0, 0, 0, 0);
    }
    PetscInt comp,ion;
    PetscInt Nx = user->Nx;
    PetscInt nx = 2*width_size+1;
    for ( comp = 0; comp < Nc; comp++) {
        for ( ion = 0; ion < Ni; ion++) {
            user->state_vars->c[c_index(x,y,comp,ion,Nx)]=user->grid_vars->c[c_index(width_size,width_size,comp,ion,nx)];
        }
        user->state_vars->phi[phi_index(x,y,comp,Nx)]=user->grid_vars->phi[phi_index(width_size,width_size,comp,nx)];
    }
    //Save the held variable
    for ( comp = 0; comp < Nc-1; comp++) {
        user->state_vars->alpha[al_index(x,y,comp,Nx)]=user->grid_vars->alpha[al_index(width_size,width_size,comp,nx)];
    }
    //Save the gating variables
    user->gate_vars->mNaT[xy_index(x, y, Nx)] = user->grid_gate_vars->mNaT[xy_index(width_size, width_size, nx)];
    user->gate_vars->hNaT[xy_index(x, y, Nx)] = user->grid_gate_vars->hNaT[xy_index(width_size, width_size, nx)];
    user->gate_vars->gNaT[xy_index(x, y, Nx)] = user->grid_gate_vars->gNaT[xy_index(width_size, width_size, nx)];
    user->gate_vars->mNaP[xy_index(x, y, Nx)] = user->grid_gate_vars->mNaP[xy_index(width_size, width_size, nx)];
    user->gate_vars->hNaP[xy_index(x, y, Nx)] = user->grid_gate_vars->hNaP[xy_index(width_size, width_size, nx)];
    user->gate_vars->gNaP[xy_index(x, y, Nx)] = user->grid_gate_vars->gNaP[xy_index(width_size, width_size, nx)];
    user->gate_vars->mKDR[xy_index(x, y, Nx)] = user->grid_gate_vars->mKDR[xy_index(width_size, width_size, nx)];
    user->gate_vars->gKDR[xy_index(x, y, Nx)] = user->grid_gate_vars->gKDR[xy_index(width_size, width_size, nx)];
    user->gate_vars->mKA[xy_index(x, y, Nx)] = user->grid_gate_vars->mKA[xy_index(width_size, width_size, nx)];
    user->gate_vars->hKA[xy_index(x, y, Nx)] = user->grid_gate_vars->hKA[xy_index(width_size, width_size, nx)];
    user->gate_vars->gKA[xy_index(x, y, Nx)] = user->grid_gate_vars->gKA[xy_index(width_size, width_size, nx)];
    user->gate_vars->yNMDA[xy_index(x, y, Nx)] = user->grid_gate_vars->yNMDA[xy_index(width_size, width_size, nx)];
    user->gate_vars->zNMDA[xy_index(x, y, Nx)] = user->grid_gate_vars->zNMDA[xy_index(width_size, width_size, nx)];
    user->gate_vars->dNMDA[xy_index(x, y, Nx)] = user->grid_gate_vars->dNMDA[xy_index(width_size, width_size, nx)];
    user->gate_vars->gNMDA[xy_index(x, y, Nx)] = user->grid_gate_vars->gNMDA[xy_index(width_size, width_size, nx)];

    if(Profiling_on) {
        PetscLogEventEnd(event[13], 0, 0, 0, 0);
    }
}

PetscErrorCode Grid_Residual(Vec Res,PetscInt xi,PetscInt yi,void *ctx)
{
    //Residual equation using derivative of the charge-capacitance relation
    // Volume not solved for here
    struct AppCtx * user = (struct AppCtx *) ctx;
    PetscErrorCode ierr;
    if(Profiling_on) {
        PetscLogEventBegin(event[1], 0, 0, 0, 0);
    }
    //Compute membrane ionic flux relation quantitites
    grid_ionmflux(user,xi,yi);

    //Compute membrane water flow related quantities
    grid_wflowm(user);

    PetscReal *c = user->grid_vars->c;
    PetscReal *phi = user->grid_vars->phi;
    PetscReal *al = user->grid_vars->alpha;
    PetscReal *cp = user->grid_vars_past->c;
    PetscReal *alp = user->grid_vars_past->alpha;
    PetscReal *phip = user->grid_vars_past->phi;

    PetscReal *Dcs = user->Dcs;
    PetscReal *Dcb = user->Dcb;
    struct FluxData *flux = user->flux;
    PetscReal dt = user->dt;
    PetscReal dx = user->dx;
    PetscReal dy = user->dy;
    PetscInt Nx = 2*width_size+1;
    PetscInt Ny = 2*width_size+1;

    //Residual for concentration equations
    PetscReal Rcvx,Rcvy,Resc;
    PetscReal RcvxRight,RcvyUp;

    //Residual for fluxes in voltage differential equations
    PetscReal Rphx[Nc], Rphy[Nc], RphxRight[Nc], RphyUp[Nc];
    PetscReal Resph,ResphN;

    PetscReal alNc,alpNc;
    PetscInt ion,comp,x,y;


    for(x=0;x<Nx;x++) {
        for(y=0;y<Ny;y++) {
            //Init voltage tracking to zero
            for(comp=0;comp<Nc;comp++) {
                Rphx[comp]=0;
                Rphy[comp]=0;
                RphxRight[comp]=0;
                RphyUp[comp]=0;
            }
            for(ion=0;ion<Ni;ion++) {
                for(comp=0;comp<Nc-1;comp++) {
                    Rcvx = 0;
                    RcvxRight = 0;
                    if(x>0) {
                        //First difference term
                        Rcvx = Dcs[c_index(x-1,y,comp,ion,Nx)*2]*(cp[c_index(x-1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2;
                        Rcvx = Rcvx*(log(c[c_index(x,y,comp,ion,Nx)])-log(c[c_index(x-1,y,comp,ion,Nx)])+z[ion]*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x-1,y,comp,Nx)]))/dx*dt/dx;
                    }
                    //Add Second right moving difference
                    if(x<Nx-1) {
                        RcvxRight = Dcs[c_index(x,y,comp,ion,Nx)*2]*(cp[c_index(x,y,comp,ion,Nx)]+cp[c_index(x+1,y,comp,ion,Nx)])/2;
                        RcvxRight = RcvxRight*(log(c[c_index(x+1,y,comp,ion,Nx)])-log(c[c_index(x,y,comp,ion,Nx)])+z[ion]*(phi[phi_index(x+1,y,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]))/dx*dt/dx;
                    }
                    Rcvy = 0;
                    RcvyUp = 0;
                    //Up down difference
                    if(y>0) {
                        Rcvy = Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*(cp[c_index(x,y-1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2;
                        Rcvy = Rcvy*(log(c[c_index(x,y,comp,ion,Nx)])-log(c[c_index(x,y-1,comp,ion,Nx)])+z[ion]*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x,y-1,comp,Nx)]))/dy*dt/dy;
                    }
                    //Next upward difference
                    if(y<Ny-1) {
                        RcvyUp = Dcs[c_index(x,y,comp,ion,Nx)*2+1]*(cp[c_index(x,y,comp,ion,Nx)]+cp[c_index(x,y+1,comp,ion,Nx)])/2;
                        RcvyUp = RcvyUp*(log(c[c_index(x,y+1,comp,ion,Nx)])-log(c[c_index(x,y,comp,ion,Nx)])+z[ion]*(phi[phi_index(x,y+1,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]))/dy*dt/dy;
                    }
                    Resc = al[al_index(x,y,comp,Nx)]*c[c_index(x,y,comp,ion,Nx)]-alp[al_index(x,y,comp,Nx)]*cp[c_index(x,y,comp,ion,Nx)];
                    Resc += Rcvx - RcvxRight + Rcvy - RcvyUp + flux->mflux[c_index(x,y,comp,ion,Nx)]*dt;

                    ierr = VecSetValue(Res,Ind_2(x,y,ion,comp,Nx),Resc,INSERT_VALUES);CHKERRQ(ierr);

                    //Save values for voltage
                    Rphx[comp]+=z[ion]*Rcvx;
                    Rphy[comp]+=z[ion]*Rcvy;
                    RphxRight[comp]+=z[ion]*RcvxRight;
                    RphyUp[comp]+=z[ion]*RcvyUp;

                }
                //Set Extracellular values
                alNc = 1 - al[al_index(x,y,0,Nx)] - al[al_index(x,y,1,Nx)];
                alpNc = 1 - alp[al_index(x,y,0,Nx)] - alp[al_index(x,y,1,Nx)];
                comp = Nc-1;
                Rcvx = 0;
                RcvxRight = 0;
                if(x>0) {
                    //First difference term
                    Rcvx = Dcs[c_index(x-1,y,comp,ion,Nx)*2]*(cp[c_index(x-1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2;
                    Rcvx = Rcvx*(log(c[c_index(x,y,comp,ion,Nx)])-log(c[c_index(x-1,y,comp,ion,Nx)])+z[ion]*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x-1,y,comp,Nx)]))/dx*dt/dx;
                }
                //Add Second right moving difference
                if(x<Nx-1) {
                    RcvxRight = Dcs[c_index(x,y,comp,ion,Nx)*2]*(cp[c_index(x,y,comp,ion,Nx)]+cp[c_index(x+1,y,comp,ion,Nx)])/2;
                    RcvxRight = RcvxRight*(log(c[c_index(x+1,y,comp,ion,Nx)])-log(c[c_index(x,y,comp,ion,Nx)])+z[ion]*(phi[phi_index(x+1,y,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]))/dx*dt/dx;
                }
                Rcvy = 0;
                RcvyUp = 0;
                //Up down difference
                if(y>0) {
                    Rcvy = Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*(cp[c_index(x,y-1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2;
                    Rcvy = Rcvy*(log(c[c_index(x,y,comp,ion,Nx)])-log(c[c_index(x,y-1,comp,ion,Nx)])+z[ion]*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x,y-1,comp,Nx)]))/dy*dt/dy;
                }
                //Next upward difference
                if(y<Ny-1) {
                    RcvyUp = Dcs[c_index(x,y,comp,ion,Nx)*2+1]*(cp[c_index(x,y,comp,ion,Nx)]+cp[c_index(x,y+1,comp,ion,Nx)])/2;
                    RcvyUp = RcvyUp*(log(c[c_index(x,y+1,comp,ion,Nx)])-log(c[c_index(x,y,comp,ion,Nx)])+z[ion]*(phi[phi_index(x,y+1,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]))/dy*dt/dy;
                }
                Resc = alNc*c[c_index(x,y,comp,ion,Nx)]-alpNc*cp[c_index(x,y,comp,ion,Nx)];
                Resc += Rcvx - RcvxRight + Rcvy - RcvyUp + flux->mflux[c_index(x,y,comp,ion,Nx)]*dt;
                //Add bath variables

                Resc -= sqrt(pow(Dcb[c_index(x,y,comp,ion,Nx)*2],2)+pow(Dcb[c_index(x,y,comp,ion,Nx)*2+1],2))*(cp[c_index(x,y,comp,ion,Nx)]+cbath[ion])/2.0*(log(c[c_index(x,y,comp,ion,Nx)])-log(cbath[ion])+z[ion]*phi[phi_index(x,y,comp,Nx)]-z[ion]*phibath)*dt;
                ierr = VecSetValue(Res,Ind_2(x,y,ion,comp,Nx),Resc,INSERT_VALUES);CHKERRQ(ierr);

                //Save values for voltage
                Rphx[comp]+=z[ion]*Rcvx;
                Rphy[comp]+=z[ion]*Rcvy;
                RphxRight[comp]+=z[ion]*RcvxRight;
                RphyUp[comp]+=z[ion]*RcvyUp;
            }

            //Voltage Equations
            ResphN = 0;
            for(comp=0;comp<Nc-1;comp++) {
                Resph = cm[comp]*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x,y,Nc-1,Nx)])-cm[comp]*(phip[phi_index(x,y,comp,Nx)]-phip[phi_index(x,y,Nc-1,Nx)]);
                for(ion=0;ion<Ni;ion++){
                    //Ion channel
                    Resph +=z[ion]*flux->mflux[c_index(x,y,comp,ion,Nx)]*dt;
                }
                //Add the terms shared with extracell
                ResphN -= Resph; // Subtract total capacitance, subtract total ion channel flux
                Resph += Rphx[comp] - RphxRight[comp] + Rphy[comp] - RphyUp[comp];
                ierr = VecSetValue(Res,Ind_2(x,y,Ni,comp,Nx),Resph,INSERT_VALUES); CHKERRQ(ierr);
            }

            //Finish adding extracell
            comp = Nc-1;
            //Add bath contribution
            for(ion=0;ion<Ni;ion++){

                ResphN -=z[ion]*sqrt(pow(Dcb[c_index(x,y,comp,ion,Nx)*2],2)+pow(Dcb[c_index(x,y,comp,ion,Nx)*2+1],2))*(cp[c_index(x,y,comp,ion,Nx)]+cbath[ion])/2.0*(log(c[c_index(x,y,comp,ion,Nx)])-log(cbath[ion])+z[ion]*phi[phi_index(x,y,comp,Nx)]-z[ion]*phibath)*dt;
            }
            ResphN += Rphx[comp] - RphxRight[comp] + Rphy[comp] - RphyUp[comp];
            ierr = VecSetValue(Res,Ind_2(x,y,Ni,comp,Nx),ResphN,INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = VecAssemblyBegin(Res);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Res);CHKERRQ(ierr);

    if(Profiling_on) {
        PetscLogEventEnd(event[1], 0, 0, 0, 0);
    }
    return ierr;
}

PetscErrorCode Grid_Jacobian(Mat Jac,PetscInt xi,PetscInt yi,void *ctx) {
    //Jacobian equation using derivative of the charge-capacitance relation
    // Alpha is not solved here

    struct AppCtx *user = (struct AppCtx *) ctx;
    PetscErrorCode ierr;
    if (Profiling_on) {
        PetscLogEventBegin(event[0], 0, 0, 0, 0);
    }
    PetscReal *c = user->grid_vars->c;
    PetscReal *al = user->grid_vars->alpha;
    PetscReal *cp = user->grid_vars_past->c;

    PetscReal *Dcs = user->Dcs;
    PetscReal *Dcb = user->Dcb;
    struct FluxData *flux = user->flux;
    PetscReal dt = user->dt;
    PetscReal dx = user->dx;
    PetscReal dy = user->dy;
    PetscInt Nx = 2*width_size+1;
    PetscInt Ny = 2*width_size+1;
    struct ConstVars *con_vars = user->con_vars;

    PetscInt ind = 0;
    PetscInt x, y, ion, comp;

    PetscReal Ftmpx, Fc0x, Fc1x, Fph0x, Fph1x;
    PetscReal Ftmpy, Fc0y, Fc1y, Fph0y, Fph1y;
    PetscReal Ac, Aphi, Avolt, AvoltN;

    PetscReal Fphph0x[Nc], Fphph1x[Nc];
    PetscReal Fphph0y[Nc], Fphph1y[Nc];

    //Ionic concentration equations
    for (x = 0; x < Nx; x++) {
        for (y = 0; y < Ny; y++) {
            for (comp = 0; comp < Nc; comp++) {
                Fphph0x[comp] = 0;
                Fphph1x[comp] = 0;
                Fphph0y[comp] = 0;
                Fphph1y[comp] = 0;
            }
            for (ion = 0; ion < Ni; ion++) {
                for (comp = 0; comp < Nc - 1; comp++) {
                    //Electrodiffusion contributions
                    Ftmpx = 0;
                    Fc0x = 0;
                    Fc1x = 0;
                    Fph0x = 0;
                    Fph1x = 0;
                    Ftmpy = 0;
                    Fc0y = 0;
                    Fc1y = 0;
                    Fph0y = 0;
                    Fph1y = 0;
                    if (x < Nx - 1) {
                        Ftmpx = Dcs[c_index(x, y, comp, ion, Nx) * 2] *
                                (cp[c_index(x, y, comp, ion, Nx)] + cp[c_index(x + 1, y, comp, ion, Nx)]) / 2 / dx *
                                dt / dx;
                        Fc0x = Ftmpx / c[c_index(x, y, comp, ion, Nx)];
                        Fph0x = z[ion] * Ftmpx;
                        // Right c with left c (-Fc0x)

                        ierr = MatSetValue(Jac, Ind_2(x + 1, y, ion, comp, Nx), Ind_2(x, y, ion, comp, Nx), -Fc0x,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Right c with left phi (-Fph0x)
                        ierr = MatSetValue(Jac, Ind_2(x + 1, y, ion, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fph0x,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;

                        //Right phi with left c in voltage eqn
                        ierr = MatSetValue(Jac, Ind_2(x + 1, y, Ni, comp, Nx), Ind_2(x, y, ion, comp, Nx),
                                           -z[ion] * Fc0x, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if (x > 0) {
                        Ftmpx = Dcs[c_index(x - 1, y, comp, ion, Nx) * 2] *
                                (cp[c_index(x - 1, y, comp, ion, Nx)] + cp[c_index(x, y, comp, ion, Nx)]) / 2 / dx *
                                dt / dx;
                        Fc1x = Ftmpx / c[c_index(x, y, comp, ion, Nx)];
                        Fph1x = z[ion] * Ftmpx;
                        //left c with right c (-Fc1x)
                        ierr = MatSetValue(Jac, Ind_2(x - 1, y, ion, comp, Nx), Ind_2(x, y, ion, comp, Nx), -Fc1x,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Left c with right phi (-Fph1x)
                        ierr = MatSetValue(Jac, Ind_2(x - 1, y, ion, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fph1x,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;

                        //Left phi with right c in voltage eqn
                        ierr = MatSetValue(Jac, Ind_2(x - 1, y, Ni, comp, Nx), Ind_2(x, y, ion, comp, Nx),
                                           -z[ion] * Fc1x, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if (y < Ny - 1) {
                        Ftmpy = Dcs[c_index(x, y, comp, ion, Nx) * 2 + 1] *
                                (cp[c_index(x, y, comp, ion, Nx)] + cp[c_index(x, y + 1, comp, ion, Nx)]) / 2 / dy *
                                dt / dy;
                        Fc0y = Ftmpy / c[c_index(x, y, comp, ion, Nx)];
                        Fph0y = z[ion] * Ftmpy;
                        // Upper c with lower c (-Fc0y)
                        ierr = MatSetValue(Jac, Ind_2(x, y + 1, ion, comp, Nx), Ind_2(x, y, ion, comp, Nx), -Fc0y,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Upper c with lower phi (-Fph0y)
                        ierr = MatSetValue(Jac, Ind_2(x, y + 1, ion, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fph0y,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;

                        //Upper phi with lower c in voltage eqn
                        ierr = MatSetValue(Jac, Ind_2(x, y + 1, Ni, comp, Nx), Ind_2(x, y, ion, comp, Nx),
                                           -z[ion] * Fc0y, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if (y > 0) {
                        Ftmpy = Dcs[c_index(x, y - 1, comp, ion, Nx) * 2 + 1] *
                                (cp[c_index(x, y - 1, comp, ion, Nx)] + cp[c_index(x, y, comp, ion, Nx)]) / 2 / dy *
                                dt / dy;
                        Fc1y = Ftmpy / c[c_index(x, y, comp, ion, Nx)];
                        Fph1y = z[ion] * Ftmpy;
                        //Lower c with Upper c (-Fc1y)
                        ierr = MatSetValue(Jac, Ind_2(x, y - 1, ion, comp, Nx), Ind_2(x, y, ion, comp, Nx), -Fc1y,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Lower c with Upper phi (-Fph1y)
                        ierr = MatSetValue(Jac, Ind_2(x, y - 1, ion, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fph1y,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;

                        //Lower phi with upper c in voltage eqn
                        ierr = MatSetValue(Jac, Ind_2(x, y - 1, Ni, comp, Nx), Ind_2(x, y, ion, comp, Nx),
                                           -z[ion] * Fc1y, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    //Diagonal term contribution
                    Ac = al[al_index(x, y, comp, Nx)] + Fc0x + Fc1x + Fc0y + Fc1y;
                    Aphi = Fph0x + Fph1x + Fph0y + Fph1y;

                    //Add up terms for voltage eqns
                    Fphph0x[comp] += z[ion] * Fph0x;
                    Fphph1x[comp] += z[ion] * Fph1x;
                    Fphph0y[comp] += z[ion] * Fph0y;
                    Fphph1y[comp] += z[ion] * Fph1y;

                    //membrane current contributions
                    Ac += flux->dfdci[c_index(x, y, comp, ion, Nx)] * dt;
                    Aphi += flux->dfdphim[c_index(x, y, comp, ion, Nx)] * dt;
                    // Different Compartment Terms
                    // C Extracellular with C Inside
                    ierr = MatSetValue(Jac, Ind_2(x, y, ion, Nc - 1, Nx), Ind_2(x, y, ion, comp, Nx),
                                       -flux->dfdci[c_index(x, y, comp, ion, Nx)] * dt, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    // C Intra with C Extra
                    ierr = MatSetValue(Jac, Ind_2(x, y, ion, comp, Nx), Ind_2(x, y, ion, Nc - 1, Nx),
                                       flux->dfdce[c_index(x, y, comp, ion, Nx)] * dt, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    // C Extracellular with Phi Inside
                    ierr = MatSetValue(Jac, Ind_2(x, y, ion, Nc - 1, Nx), Ind_2(x, y, Ni, comp, Nx),
                                       -flux->dfdphim[c_index(x, y, comp, ion, Nx)] * dt, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    // C Intra with Phi Extra
                    ierr = MatSetValue(Jac, Ind_2(x, y, ion, comp, Nx), Ind_2(x, y, Ni, Nc - 1, Nx),
                                       -flux->dfdphim[c_index(x, y, comp, ion, Nx)] * dt, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Same compartment terms
                    // c with c
                    ierr = MatSetValue(Jac, Ind_2(x, y, ion, comp, Nx), Ind_2(x, y, ion, comp, Nx), Ac, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    // c with phi
                    ierr = MatSetValue(Jac, Ind_2(x, y, ion, comp, Nx), Ind_2(x, y, Ni, comp, Nx), Aphi, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;

                    //Intra-Phi with c (voltage eqn)
                    ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp, Nx), Ind_2(x, y, ion, comp, Nx), z[ion] *
                                                                                                   (Fc0x + Fc1x + Fc0y +
                                                                                                    Fc1y +
                                                                                                    flux->dfdci[c_index(
                                                                                                            x, y, comp,
                                                                                                            ion, Nx)] *
                                                                                                    dt), INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //IntraPhi with c extra(volt eqn)
                    ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp, Nx), Ind_2(x, y, ion, Nc - 1, Nx),
                                       z[ion] * (flux->dfdce[c_index(x, y, comp, ion, Nx)] * dt), INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Extra-Phi with intra-c (voltage eqn)
                    ierr = MatSetValue(Jac, Ind_2(x, y, Ni, Nc - 1, Nx), Ind_2(x, y, ion, comp, Nx),
                                       -z[ion] * (flux->dfdci[c_index(x, y, comp, ion, Nx)] * dt), INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;

                }
                //Extracellular terms
                comp = Nc - 1;
                //Electrodiffusion contributions
                Ftmpx = 0;
                Fc0x = 0;
                Fc1x = 0;
                Fph0x = 0;
                Fph1x = 0;
                Ftmpy = 0;
                Fc0y = 0;
                Fc1y = 0;
                Fph0y = 0;
                Fph1y = 0;
                if (x < Nx - 1) {
                    Ftmpx = Dcs[c_index(x, y, comp, ion, Nx) * 2] *
                            (cp[c_index(x, y, comp, ion, Nx)] + cp[c_index(x + 1, y, comp, ion, Nx)]) / 2 / dx * dt /
                            dx;
                    Fc0x = Ftmpx / c[c_index(x, y, comp, ion, Nx)];
                    Fph0x = z[ion] * Ftmpx;
                    // Right c with left c (-Fc0x)
                    ierr = MatSetValue(Jac, Ind_2(x + 1, y, ion, comp, Nx), Ind_2(x, y, ion, comp, Nx), -Fc0x,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Right c with left phi (-Fph0x)
                    ierr = MatSetValue(Jac, Ind_2(x + 1, y, ion, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fph0x,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;

                    // Right Phi with left c (voltage eqn)
                    ierr = MatSetValue(Jac, Ind_2(x + 1, y, Ni, comp, Nx), Ind_2(x, y, ion, comp, Nx), -z[ion] * Fc0x,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                if (x > 0) {
                    Ftmpx = Dcs[c_index(x - 1, y, comp, ion, Nx) * 2] *
                            (cp[c_index(x - 1, y, comp, ion, Nx)] + cp[c_index(x, y, comp, ion, Nx)]) / 2 / dx * dt /
                            dx;
                    Fc1x = Ftmpx / c[c_index(x, y, comp, ion, Nx)];
                    Fph1x = z[ion] * Ftmpx;
                    //left c with right c (-Fc1x)
                    ierr = MatSetValue(Jac, Ind_2(x - 1, y, ion, comp, Nx), Ind_2(x, y, ion, comp, Nx), -Fc1x,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Left c with right phi (-Fph1x)
                    ierr = MatSetValue(Jac, Ind_2(x - 1, y, ion, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fph1x,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;

                    // left Phi with right c (voltage eqn)
                    ierr = MatSetValue(Jac, Ind_2(x - 1, y, Ni, comp, Nx), Ind_2(x, y, ion, comp, Nx), -z[ion] * Fc1x,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                if (y < Ny - 1) {
                    Ftmpy = Dcs[c_index(x, y, comp, ion, Nx) * 2 + 1] *
                            (cp[c_index(x, y, comp, ion, Nx)] + cp[c_index(x, y + 1, comp, ion, Nx)]) / 2 / dy * dt /
                            dy;
                    Fc0y = Ftmpy / c[c_index(x, y, comp, ion, Nx)];
                    Fph0y = z[ion] * Ftmpy;
                    // Upper c with lower c (-Fc0y)
                    ierr = MatSetValue(Jac, Ind_2(x, y + 1, ion, comp, Nx), Ind_2(x, y, ion, comp, Nx), -Fc0y,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Upper c with lower phi (-Fph0y)
                    ierr = MatSetValue(Jac, Ind_2(x, y + 1, ion, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fph0y,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;

                    // Upper Phi with lower c (voltage eqn)
                    ierr = MatSetValue(Jac, Ind_2(x, y + 1, Ni, comp, Nx), Ind_2(x, y, ion, comp, Nx), -z[ion] * Fc0y,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                if (y > 0) {
                    Ftmpy = Dcs[c_index(x, y - 1, comp, ion, Nx) * 2 + 1] *
                            (cp[c_index(x, y - 1, comp, ion, Nx)] + cp[c_index(x, y, comp, ion, Nx)]) / 2 / dy * dt /
                            dy;
                    Fc1y = Ftmpy / c[c_index(x, y, comp, ion, Nx)];
                    Fph1y = z[ion] * Ftmpy;
                    //Lower c with Upper c (-Fc1y)
                    ierr = MatSetValue(Jac, Ind_2(x, y - 1, ion, comp, Nx), Ind_2(x, y, ion, comp, Nx), -Fc1y,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Lower c with Upper phi (-Fph1y)
                    ierr = MatSetValue(Jac, Ind_2(x, y - 1, ion, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fph1y,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;

                    // Lower Phi with upper c (voltage eqn)
                    ierr = MatSetValue(Jac, Ind_2(x, y - 1, Ni, comp, Nx), Ind_2(x, y, ion, comp, Nx), -z[ion] * Fc1y,
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }

                //Diagonal term contribution
                Ac = (1 - al[al_index(x, y, 0, Nx)] - al[al_index(x, y, 1, Nx)]) + Fc0x + Fc1x + Fc0y + Fc1y;
                Aphi = Fph0x + Fph1x + Fph0y + Fph1y;

                Avolt = z[ion] * (Fc0x + Fc1x + Fc0y + Fc1y);

                //Add up terms for voltage eqns
                Fphph0x[comp] += z[ion] * Fph0x;
                Fphph1x[comp] += z[ion] * Fph1x;
                Fphph0y[comp] += z[ion] * Fph0y;
                Fphph1y[comp] += z[ion] * Fph1y;

                //Membrane current contribution
                for (comp = 0; comp < Nc - 1; comp++) {
                    Ac -= flux->dfdce[c_index(x, y, comp, ion, Nx)] * dt;
                    Aphi += flux->dfdphim[c_index(x, y, comp, ion, Nx)] * dt;
                    Avolt -= z[ion] * flux->dfdce[c_index(x, y, comp, ion, Nx)] * dt;
                }
                //Add bath contributions
                Ftmpx = sqrt(pow(Dcb[c_index(x, y, Nc - 1, ion, Nx) * 2], 2) +
                             pow(Dcb[c_index(x, y, Nc - 1, ion, Nx) * 2 + 1], 2));
                Ac -= Ftmpx * (cp[c_index(x, y, Nc - 1, ion, Nx)] + cbath[ion]) /
                      (2 * c[c_index(x, y, Nc - 1, ion, Nx)]) * dt;
                Aphi -= Ftmpx * (cp[c_index(x, y, Nc - 1, ion, Nx)] + cbath[ion]) * z[ion] / 2 * dt;

                Avolt -= z[ion] * Ftmpx * (cp[c_index(x, y, Nc - 1, ion, Nx)] + cbath[ion]) /
                         (2 * c[c_index(x, y, Nc - 1, ion, Nx)]) * dt;

                //Insert extracell to extracell parts
                // c with c
                ierr = MatSetValue(Jac, Ind_2(x, y, ion, Nc - 1, Nx), Ind_2(x, y, ion, Nc - 1, Nx), Ac, INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
                // c with phi
                ierr = MatSetValue(Jac, Ind_2(x, y, ion, Nc - 1, Nx), Ind_2(x, y, Ni, Nc - 1, Nx), Aphi, INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;

                //phi with c (voltage eqn)
                ierr = MatSetValue(Jac, Ind_2(x, y, Ni, Nc - 1, Nx), Ind_2(x, y, ion, Nc - 1, Nx), Avolt,
                                   INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
            }
            //Derivative of charge-capacitance
            for (comp = 0; comp < Nc - 1; comp++) {
                if (x < Nx - 1) {
                    //Right phi with left phi (-Fph0x)
                    ierr = MatSetValue(Jac, Ind_2(x + 1, y, Ni, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fphph0x[comp],
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                if (x > 0) {
                    //Left phi with right phi (-Fph1x)
                    ierr = MatSetValue(Jac, Ind_2(x - 1, y, Ni, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fphph1x[comp],
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                if (y < Ny - 1) {
                    //Upper phi with lower phi (-Fph0y)
                    ierr = MatSetValue(Jac, Ind_2(x, y + 1, Ni, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fphph0y[comp],
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                if (y > 0) {
                    //Lower phi with upper phi (-Fph1y)
                    ierr = MatSetValue(Jac, Ind_2(x, y - 1, Ni, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fphph1y[comp],
                                       INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                Avolt = cm[comp] + Fphph0x[comp] + Fphph1x[comp] + Fphph0y[comp] + Fphph1y[comp];
                AvoltN = -cm[comp];
                for (ion = 0; ion < Ni; ion++) {
                    Avolt += z[ion] * flux->dfdphim[c_index(x, y, comp, ion, Nx)] * dt;
                    AvoltN -= z[ion] * flux->dfdphim[c_index(x, y, comp, ion, Nx)] * dt;
                }

                //Intra-phi with Intra-phi
                ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp, Nx), Ind_2(x, y, Ni, comp, Nx), Avolt, INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
                //Intra-phi with extra-phi
                ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp, Nx), Ind_2(x, y, Ni, Nc - 1, Nx), AvoltN, INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
            }
            //Extracellular terms
            comp = Nc - 1;
            if (x < Nx - 1) {
                //Right phi with left phi (-Fph0x)
                ierr = MatSetValue(Jac, Ind_2(x + 1, y, Ni, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fphph0x[comp],
                                   INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
            }
            if (x > 0) {
                //Left phi with right phi (-Fph1x)
                ierr = MatSetValue(Jac, Ind_2(x - 1, y, Ni, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fphph1x[comp],
                                   INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
            }
            if (y < Ny - 1) {
                //Upper phi with lower phi (-Fph0y)
                ierr = MatSetValue(Jac, Ind_2(x, y + 1, Ni, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fphph0y[comp],
                                   INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
            }
            if (y > 0) {
                //Lower phi with upper phi (-Fph1y)
                ierr = MatSetValue(Jac, Ind_2(x, y - 1, Ni, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -Fphph1y[comp],
                                   INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
            }
            AvoltN = 0;

            for (int k = 0; k < Nc - 1; k++) {
                AvoltN += cm[k];
                Avolt = -cm[k];
                for (ion = 0; ion < Ni; ion++) {
                    Avolt -= z[ion] * flux->dfdphim[c_index(x, y, k, ion, Nx)] * dt;
                    AvoltN += z[ion] * flux->dfdphim[c_index(x, y, k, ion, Nx)] * dt;
                }
                //Extra-phi with Intra-phi
                ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp, Nx), Ind_2(x, y, Ni, k, Nx), Avolt, INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
            }

            AvoltN += Fphph0x[comp] + Fphph1x[comp] + Fphph0y[comp] + Fphph1y[comp];

            //Bath terms
            for (ion = 0; ion < Ni; ion++) {
                Ftmpx = sqrt(pow(Dcb[c_index(x, y, Nc - 1, ion, Nx) * 2], 2) +
                             pow(Dcb[c_index(x, y, Nc - 1, ion, Nx) * 2 + 1], 2));
                AvoltN -= z[ion] * Ftmpx * (cp[c_index(x, y, Nc - 1, ion, Nx)] + cbath[ion]) * z[ion] / 2 * dt;
            }
            //extra-phi with extra-phi
            ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp, Nx), Ind_2(x, y, Ni, comp, Nx), AvoltN, INSERT_VALUES);
            CHKERRQ(ierr);
            ind++;

        }
    }

    ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    if (Profiling_on) {
        PetscLogEventEnd(event[0], 0, 0, 0, 0);
    }
    return ierr;

}

PetscErrorCode Grid_Residual_algebraic(Vec Res,PetscInt xi,PetscInt yi,void *ctx)
{
    //Residual equation using algebraic version of the charge-capacitance relation
    //Alpha is solved for here
    struct AppCtx * user = (struct AppCtx *) ctx;
    PetscErrorCode ierr;
    if(Profiling_on) {
        PetscLogEventBegin(event[10], 0, 0, 0, 0);
    }
    //Compute membrane ionic flux relation quantitites
    grid_ionmflux(user,xi,yi);

    //Compute membrane water flow related quantities
    grid_wflowm(user);

    PetscReal *c = user->grid_vars->c;
    PetscReal *phi = user->grid_vars->phi;
    PetscReal *al = user->grid_vars->alpha;
    PetscReal *cp = user->grid_vars_past->c;
    PetscReal *alp = user->grid_vars_past->alpha;

    PetscReal *Dcs = user->Dcs;
    PetscReal *Dcb = user->Dcb;
    struct FluxData *flux = user->flux;
    PetscReal dt = user->dt;
    PetscReal dx = user->dx;
    PetscReal dy = user->dy;
    PetscInt Nx = 2*width_size+1;
    PetscInt Ny = 2*width_size+1;

    //Residual for concentration equations
    PetscReal Rcvx,Rcvy,Resc;
    PetscReal RcvxRight,RcvyUp;

    PetscReal alNc,alpNc;
    PetscInt ion,comp,x,y;


    for(x=0;x<Nx;x++) {
        for(y=0;y<Ny;y++) {
            for(ion=0;ion<Ni;ion++) {
                for(comp=0;comp<Nc-1;comp++) {
                    Rcvx = 0;
                    RcvxRight = 0;
                    if(x>0) {
                        //First difference term
                        Rcvx = Dcs[c_index(x-1,y,comp,ion,Nx)*2]*(cp[c_index(x-1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2;
                        Rcvx = Rcvx*(log(c[c_index(x,y,comp,ion,Nx)])-log(c[c_index(x-1,y,comp,ion,Nx)])+z[ion]*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x-1,y,comp,Nx)]))/dx*dt/dx;
                    }
                    //Add Second right moving difference
                    if(x<Nx-1) {
                        RcvxRight = Dcs[c_index(x,y,comp,ion,Nx)*2]*(cp[c_index(x,y,comp,ion,Nx)]+cp[c_index(x+1,y,comp,ion,Nx)])/2;
                        RcvxRight = RcvxRight*(log(c[c_index(x+1,y,comp,ion,Nx)])-log(c[c_index(x,y,comp,ion,Nx)])+z[ion]*(phi[phi_index(x+1,y,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]))/dx*dt/dx;
                    }
                    Rcvy = 0;
                    RcvyUp = 0;
                    //Up down difference
                    if(y>0) {
                        Rcvy = Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*(cp[c_index(x,y-1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2;
                        Rcvy = Rcvy*(log(c[c_index(x,y,comp,ion,Nx)])-log(c[c_index(x,y-1,comp,ion,Nx)])+z[ion]*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x,y-1,comp,Nx)]))/dy*dt/dy;
                    }
                    //Next upward difference
                    if(y<Ny-1) {
                        RcvyUp = Dcs[c_index(x,y,comp,ion,Nx)*2+1]*(cp[c_index(x,y,comp,ion,Nx)]+cp[c_index(x,y+1,comp,ion,Nx)])/2;
                        RcvyUp = RcvyUp*(log(c[c_index(x,y+1,comp,ion,Nx)])-log(c[c_index(x,y,comp,ion,Nx)])+z[ion]*(phi[phi_index(x,y+1,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]))/dy*dt/dy;
                    }
                    Resc = al[al_index(x,y,comp,Nx)]*c[c_index(x,y,comp,ion,Nx)]-alp[al_index(x,y,comp,Nx)]*cp[c_index(x,y,comp,ion,Nx)];
                    Resc += Rcvx - RcvxRight + Rcvy - RcvyUp + flux->mflux[c_index(x,y,comp,ion,Nx)]*dt;

                    ierr = VecSetValue(Res,Ind_2(x,y,ion,comp,Nx),Resc,INSERT_VALUES);CHKERRQ(ierr);

                }
                //Set Extracellular values
                alNc = 1 - al[al_index(x,y,0,Nx)] - al[al_index(x,y,1,Nx)];
                alpNc = 1 - alp[al_index(x,y,0,Nx)] - alp[al_index(x,y,1,Nx)];
                comp = Nc-1;
                Rcvx = 0;
                RcvxRight = 0;
                if(x>0) {
                    //First difference term
                    Rcvx = Dcs[c_index(x-1,y,comp,ion,Nx)*2]*(cp[c_index(x-1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2;
                    Rcvx = Rcvx*(log(c[c_index(x,y,comp,ion,Nx)])-log(c[c_index(x-1,y,comp,ion,Nx)])+z[ion]*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x-1,y,comp,Nx)]))/dx*dt/dx;
                }
                //Add Second right moving difference
                if(x<Nx-1) {
                    RcvxRight = Dcs[c_index(x,y,comp,ion,Nx)*2]*(cp[c_index(x,y,comp,ion,Nx)]+cp[c_index(x+1,y,comp,ion,Nx)])/2;
                    RcvxRight = RcvxRight*(log(c[c_index(x+1,y,comp,ion,Nx)])-log(c[c_index(x,y,comp,ion,Nx)])+z[ion]*(phi[phi_index(x+1,y,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]))/dx*dt/dx;
                }
                Rcvy = 0;
                RcvyUp = 0;
                //Up down difference
                if(y>0) {
                    Rcvy = Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*(cp[c_index(x,y-1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2;
                    Rcvy = Rcvy*(log(c[c_index(x,y,comp,ion,Nx)])-log(c[c_index(x,y-1,comp,ion,Nx)])+z[ion]*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x,y-1,comp,Nx)]))/dy*dt/dy;
                }
                //Next upward difference
                if(y<Ny-1) {
                    RcvyUp = Dcs[c_index(x,y,comp,ion,Nx)*2+1]*(cp[c_index(x,y,comp,ion,Nx)]+cp[c_index(x,y+1,comp,ion,Nx)])/2;
                    RcvyUp = RcvyUp*(log(c[c_index(x,y+1,comp,ion,Nx)])-log(c[c_index(x,y,comp,ion,Nx)])+z[ion]*(phi[phi_index(x,y+1,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]))/dy*dt/dy;
                }
                Resc = alNc*c[c_index(x,y,comp,ion,Nx)]-alpNc*cp[c_index(x,y,comp,ion,Nx)];
                Resc += Rcvx - RcvxRight + Rcvy - RcvyUp + flux->mflux[c_index(x,y,comp,ion,Nx)]*dt;
                //Add bath variables

                Resc -= sqrt(pow(Dcb[c_index(x,y,comp,ion,Nx)*2],2)+pow(Dcb[c_index(x,y,comp,ion,Nx)*2+1],2))*(cp[c_index(x,y,comp,ion,Nx)]+cbath[ion])/2.0*(log(c[c_index(x,y,comp,ion,Nx)])-log(cbath[ion])+z[ion]*phi[phi_index(x,y,comp,Nx)]-z[ion]*phibath)*dt;
                ierr = VecSetValue(Res,Ind_2(x,y,ion,comp,Nx),Resc,INSERT_VALUES);CHKERRQ(ierr);

            }
        }
    }


    for(x=0;x<Nx;x++) {
        for(y=0;y<Ny;y++) {

            //Residual for electroneutrality condition
            for(comp=0;comp<Nc-1;comp++) {

                Resc = al[al_index(x,y,comp,Nx)]*cz(c,z,x,y,Nx,comp,user)+user->con_vars->zo[phi_index(0,0,comp,Nx)]*user->con_vars->ao[phi_index(0,0,comp,Nx)];
                ierr = VecSetValue(Res,Ind_2(x,y,Ni,comp,Nx),Resc,INSERT_VALUES); CHKERRQ(ierr);
            }
            //Extracellular term
            comp=Nc-1;
            Resc = (1-al[al_index(x,y,0,Nx)]-al[al_index(x,y,1,Nx)])*cz(c,z,x,y,Nx,comp,user)+user->con_vars->zo[phi_index(0,0,comp,Nx)]*user->con_vars->ao[phi_index(0,0,comp,Nx)];
            ierr = VecSetValue(Res,Ind_2(x,y,Ni,comp,Nx),Resc,INSERT_VALUES); CHKERRQ(ierr);

            //Residual for water flow
            //Plus modification to electroneutrality for non-zero mem.compacitance
            for(comp=0;comp<Nc-1;comp++) {
                //Water flow
                ierr = VecSetValue(Res,Ind_2(x,y,Ni+1,comp,Nx),al[al_index(x,y,comp,Nx)]-alp[al_index(x,y,comp,Nx)]+flux->wflow[al_index(x,y,comp,Nx)]*dt,INSERT_VALUES);CHKERRQ(ierr);

            }
        }
    }
    //Assemble before we add values in on top to modify the electroneutral.
    ierr = VecAssemblyBegin(Res);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Res);CHKERRQ(ierr);

    for(x=0;x<Nx;x++) {
        for(y=0;y<Ny;y++) {
            // Add Modification to electroneutrality for non-zero mem.compacitance
            for(comp=0;comp<Nc-1;comp++) {
                //Extracell voltage
                ierr = VecSetValue(Res,Ind_2(x,y,Ni,Nc-1,Nx),-cm[comp]*(phi[phi_index(x,y,Nc-1,Nx)]-phi[phi_index(x,y,comp,Nx)]),ADD_VALUES);CHKERRQ(ierr);
                //Intracell voltage mod
                ierr = VecSetValue(Res,Ind_2(x,y,Ni,comp,Nx),-cm[comp]*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x,y,Nc-1,Nx)]),ADD_VALUES);CHKERRQ(ierr);
            }
        }
    }

    ierr = VecAssemblyBegin(Res);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Res);CHKERRQ(ierr);
    if(Profiling_on) {
        PetscLogEventEnd(event[10], 0, 0, 0, 0);
    }
    return ierr;
}

PetscErrorCode
Grid_Jacobian_algebraic(Mat Jac,PetscInt xi, PetscInt yi,void *ctx)
{
    //Jacobian equation using algebraic version of the charge-capacitance relation
    // Alpha is solved for here
    struct AppCtx * user = (struct AppCtx *) ctx;
    PetscErrorCode ierr;
    if(Profiling_on) {
        PetscLogEventBegin(event[9], 0, 0, 0, 0);
    }
    PetscReal *c = user->grid_vars->c;
    PetscReal *al = user->grid_vars->alpha;
    PetscReal *cp = user->grid_vars_past->c;

    PetscReal *Dcs = user->Dcs;
    PetscReal *Dcb = user->Dcb;
    struct FluxData *flux = user->flux;
    PetscReal dt = user->dt;
    PetscReal dx = user->dx;
    PetscReal dy = user->dy;
    PetscInt Nx = 2*width_size+1;
    PetscInt Ny = 2*width_size+1;
    struct ConstVars *con_vars = user->con_vars;

    PetscInt ind = 0;
    PetscInt x,y,ion,comp;

    PetscReal Ftmpx,Fc0x,Fc1x,Fph0x,Fph1x;
    PetscReal Ftmpy,Fc0y,Fc1y,Fph0y,Fph1y;
    PetscReal Ac,Aphi;


    //Ionic concentration equations
    for(x=0;x<Nx;x++) {
        for(y=0;y<Ny;y++) {
            for(ion=0;ion<Ni;ion++) {
                for(comp=0;comp<Nc-1;comp++) {
                    //Electrodiffusion contributions
                    Ftmpx = 0;
                    Fc0x = 0;
                    Fc1x = 0;
                    Fph0x = 0;
                    Fph1x = 0;
                    Ftmpy = 0;
                    Fc0y = 0;
                    Fc1y = 0;
                    Fph0y = 0;
                    Fph1y = 0;
                    if(x<Nx-1) {
                        Ftmpx = Dcs[c_index(x,y,comp,ion,Nx)*2]*(cp[c_index(x,y,comp,ion,Nx)]+cp[c_index(x+1,y,comp,ion,Nx)])/2/dx*dt/dx;
                        Fc0x = Ftmpx/c[c_index(x,y,comp,ion,Nx)];
                        Fph0x = z[ion]*Ftmpx;
                        // Right c with left c (-Fc0x)

                        ierr = MatSetValue(Jac,Ind_2(x+1,y,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),-Fc0x,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Right c with left phi (-Fph0x)
                        ierr = MatSetValue(Jac,Ind_2(x+1,y,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),-Fph0x,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;

                    }
                    if(x>0) {
                        Ftmpx = Dcs[c_index(x-1,y,comp,ion,Nx)*2]*(cp[c_index(x-1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2/dx*dt/dx;
                        Fc1x = Ftmpx/c[c_index(x,y,comp,ion,Nx)];
                        Fph1x = z[ion]*Ftmpx;
                        //left c with right c (-Fc1x)
                        ierr = MatSetValue(Jac,Ind_2(x-1,y,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),-Fc1x,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Left c with right phi (-Fph1x)
                        ierr = MatSetValue(Jac,Ind_2(x-1,y,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),-Fph1x,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                    }
                    if(y<Ny-1) {
                        Ftmpy = Dcs[c_index(x,y,comp,ion,Nx)*2+1]*(cp[c_index(x,y,comp,ion,Nx)]+cp[c_index(x,y+1,comp,ion,Nx)])/2/dy*dt/dy;
                        Fc0y = Ftmpy/c[c_index(x,y,comp,ion,Nx)];
                        Fph0y = z[ion]*Ftmpy;
                        // Upper c with lower c (-Fc0y)
                        ierr = MatSetValue(Jac,Ind_2(x,y+1,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),-Fc0y,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Upper c with lower phi (-Fph0y)
                        ierr = MatSetValue(Jac,Ind_2(x,y+1,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),-Fph0y,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                    }
                    if(y>0) {
                        Ftmpy = Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*(cp[c_index(x,y-1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2/dy*dt/dy;
                        Fc1y = Ftmpy/c[c_index(x,y,comp,ion,Nx)];
                        Fph1y = z[ion]*Ftmpy;
                        //Lower c with Upper c (-Fc1y)
                        ierr = MatSetValue(Jac,Ind_2(x,y-1,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),-Fc1y,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Lower c with Upper phi (-Fph1y)
                        ierr = MatSetValue(Jac,Ind_2(x,y-1,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),-Fph1y,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                    }
                    //Diagonal term contribution
                    Ac = al[al_index(x,y,comp,Nx)]+Fc0x+Fc1x+Fc0y+Fc1y;
                    Aphi = Fph0x + Fph1x + Fph0y + Fph1y;


                    //membrane current contributions
                    Ac+=flux->dfdci[c_index(x,y,comp,ion,Nx)]*dt;
                    Aphi+=flux->dfdphim[c_index(x,y,comp,ion,Nx)]*dt;
                    // Different Compartment Terms
                    // C Extracellular with C Inside
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,Nc-1,Nx),Ind_2(x,y,ion,comp,Nx),-flux->dfdci[c_index(x,y,comp,ion,Nx)]*dt,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // C Intra with C Extra
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,comp,Nx),Ind_2(x,y,ion,Nc-1,Nx),flux->dfdce[c_index(x,y,comp,ion,Nx)]*dt,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // C Extracellular with Phi Inside
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,Nc-1,Nx),Ind_2(x,y,Ni,comp,Nx),-flux->dfdphim[c_index(x,y,comp,ion,Nx)]*dt,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // C Intra with Phi Extra
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,comp,Nx),Ind_2(x,y,Ni,Nc-1,Nx),-flux->dfdphim[c_index(x,y,comp,ion,Nx)]*dt,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Volume terms
                    //C extra with intra alpha
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,Nc-1,Nx),Ind_2(x,y,Ni+1,comp,Nx),-c[c_index(x,y,Nc-1,ion,Nx)],INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //C intra with intra alpha
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,comp,Nx),Ind_2(x,y,Ni+1,comp,Nx),c[c_index(x,y,comp,ion,Nx)],INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Same compartment terms
                    // c with c
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),Ac,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // c with phi
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),Aphi,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;

                }
                //Extracellular terms
                comp = Nc-1;
                //Electrodiffusion contributions
                Ftmpx = 0;
                Fc0x = 0;
                Fc1x = 0;
                Fph0x = 0;
                Fph1x = 0;
                Ftmpy = 0;
                Fc0y = 0;
                Fc1y = 0;
                Fph0y = 0;
                Fph1y = 0;
                if(x<Nx-1) {
                    Ftmpx = Dcs[c_index(x,y,comp,ion,Nx)*2]*(cp[c_index(x,y,comp,ion,Nx)]+cp[c_index(x+1,y,comp,ion,Nx)])/2/dx*dt/dx;
                    Fc0x = Ftmpx/c[c_index(x,y,comp,ion,Nx)];
                    Fph0x = z[ion]*Ftmpx;
                    // Right c with left c (-Fc0x)
                    ierr = MatSetValue(Jac,Ind_2(x+1,y,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),-Fc0x,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Right c with left phi (-Fph0x)
                    ierr = MatSetValue(Jac,Ind_2(x+1,y,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),-Fph0x,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }
                if(x>0) {
                    Ftmpx = Dcs[c_index(x-1,y,comp,ion,Nx)*2]*(cp[c_index(x-1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2/dx*dt/dx;
                    Fc1x = Ftmpx/c[c_index(x,y,comp,ion,Nx)];
                    Fph1x = z[ion]*Ftmpx;
                    //left c with right c (-Fc1x)
                    ierr = MatSetValue(Jac,Ind_2(x-1,y,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),-Fc1x,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Left c with right phi (-Fph1x)
                    ierr = MatSetValue(Jac,Ind_2(x-1,y,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),-Fph1x,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }
                if(y<Ny-1) {
                    Ftmpy = Dcs[c_index(x,y,comp,ion,Nx)*2+1]*(cp[c_index(x,y,comp,ion,Nx)]+cp[c_index(x,y+1,comp,ion,Nx)])/2/dy*dt/dy;
                    Fc0y = Ftmpy/c[c_index(x,y,comp,ion,Nx)];
                    Fph0y = z[ion]*Ftmpy;
                    // Upper c with lower c (-Fc0y)
                    ierr = MatSetValue(Jac,Ind_2(x,y+1,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),-Fc0y,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Upper c with lower phi (-Fph0y)
                    ierr = MatSetValue(Jac,Ind_2(x,y+1,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),-Fph0y,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }
                if(y>0) {
                    Ftmpy = Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*(cp[c_index(x,y-1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2/dy*dt/dy;
                    Fc1y = Ftmpy/c[c_index(x,y,comp,ion,Nx)];
                    Fph1y = z[ion]*Ftmpy;
                    //Lower c with Upper c (-Fc1y)
                    ierr = MatSetValue(Jac,Ind_2(x,y-1,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),-Fc1y,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Lower c with Upper phi (-Fph1y)
                    ierr = MatSetValue(Jac,Ind_2(x,y-1,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),-Fph1y,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }

                //Diagonal term contribution
                Ac = (1-al[al_index(x,y,0,Nx)]-al[al_index(x,y,1,Nx)])+Fc0x+Fc1x+Fc0y+Fc1y;
                Aphi = Fph0x + Fph1x + Fph0y + Fph1y;

                //Membrane current contribution
                for(comp=0;comp<Nc-1;comp++) {
                    Ac -= flux->dfdce[c_index(x,y,comp,ion,Nx)]*dt;
                    Aphi += flux->dfdphim[c_index(x,y,comp,ion,Nx)]*dt;
                }
                //Add bath contributions
                Ftmpx=sqrt(pow(Dcb[c_index(x,y,Nc-1,ion,Nx)*2],2)+pow(Dcb[c_index(x,y,Nc-1,ion,Nx)*2+1],2));
                Ac -= Ftmpx*(cp[c_index(x,y,Nc-1,ion,Nx)]+cbath[ion])/(2*c[c_index(x,y,Nc-1,ion,Nx)])*dt;
                Aphi -= Ftmpx*(cp[c_index(x,y,Nc-1,ion,Nx)]+cbath[ion])*z[ion]/2*dt;

                //Insert extracell to extracell parts
                // c with c
                ierr = MatSetValue(Jac,Ind_2(x,y,ion,Nc-1,Nx),Ind_2(x,y,ion,Nc-1,Nx),Ac,INSERT_VALUES);CHKERRQ(ierr);
                ind++;
                // c with phi
                ierr = MatSetValue(Jac,Ind_2(x,y,ion,Nc-1,Nx),Ind_2(x,y,Ni,Nc-1,Nx),Aphi,INSERT_VALUES);CHKERRQ(ierr);
                ind++;
            }

        }
    }

    //Electroneutrality charge-capcitance condition
    for(x=0;x<Nx;x++) {
        for(y=0;y<Ny;y++) {
            //electroneutral-concentration entries
            for(ion=0;ion<Ni;ion++) {
                for(comp=0;comp<Nc-1;comp++) {
                    //Phi with C entries
                    ierr = MatSetValue(Jac,Ind_2(x,y,Ni,comp,Nx),Ind_2(x,y,ion,comp,Nx),z[ion]*al[al_index(x,y,comp,Nx)],INSERT_VALUES); CHKERRQ(ierr);
                    ind++;
                }
                //Phi with C extracellular one
                comp = Nc-1;
                ierr = MatSetValue(Jac,Ind_2(x,y,Ni,comp,Nx),Ind_2(x,y,ion,comp,Nx),z[ion]*(1-al[al_index(x,y,0,Nx)]-al[al_index(x,y,1,Nx)]),INSERT_VALUES); CHKERRQ(ierr);
                ind++;

            }
            //electroneutrality-voltage entries
            Aphi = 0;
            for(comp=0;comp<Nc-1;comp++) {
                Aphi -= cm[comp];
            }
            //extraphi with extra phi
            ierr = MatSetValue(Jac,Ind_2(x,y,Ni,Nc-1,Nx),Ind_2(x,y,Ni,Nc-1,Nx),Aphi,INSERT_VALUES);CHKERRQ(ierr);
            ind++;
            for(comp=0;comp<Nc-1;comp++) {
                //The next 3 are inserted in init jacobian for the grid
                //Extra phi with intra phi
//                ierr = MatSetValue(Jac,Ind_2(x,y,Ni,Nc-1,Nx),Ind_2(x,y,Ni,comp,Nx),cm[comp],INSERT_VALUES);CHKERRQ(ierr);
//                ind++;
                // Intra phi with Extraphi
//                ierr = MatSetValue(Jac,Ind_2(x,y,Ni,comp,Nx),Ind_2(x,y,Ni,Nc-1,Nx),cm[comp],INSERT_VALUES);CHKERRQ(ierr);
//                ind++;
                //Intra phi with Intra phi
//                ierr = MatSetValue(Jac,Ind_2(x,y,Ni,comp,Nx),Ind_2(x,y,Ni,comp,Nx),-cm[comp],INSERT_VALUES);CHKERRQ(ierr);
//                ind++;
                //Extra phi with intra-Volume
                ierr = MatSetValue(Jac,Ind_2(x,y,Ni,Nc-1,Nx),Ind_2(x,y,Ni+1,comp,Nx),-cz(c,z,x,y,Nx,Nc-1,user),INSERT_VALUES);CHKERRQ(ierr);
                ind++;
                //Intra phi with Intra Vol
                ierr = MatSetValue(Jac,Ind_2(x,y,Ni,comp,Nx),Ind_2(x,y,Ni+1,comp,Nx),cz(c,z,x,y,Nx,comp,user),INSERT_VALUES);CHKERRQ(ierr);
                ind++;
            }
        }
    }
    //water flow
    for(x=0;x<Nx;x++) {
        for(y=0;y<Ny;y++) {
            for(comp=0;comp<Nc-1;comp++) {
                //Water flow volume fraction entries
                //Volume to Volume
                Ac=1+(flux->dwdpi[al_index(x,y,comp,Nx)]*(con_vars->ao[phi_index(0,0,Nc-1,Nx)]/(pow(1-al[al_index(x,y,0,Nx)]-al[al_index(x,y,1,Nx)],2))+con_vars->ao[phi_index(0,0,comp,Nx)]/pow(al[al_index(x,y,comp,Nx)],2))+flux->dwdal[al_index(x,y,comp,Nx)])*dt;
                ierr = MatSetValue(Jac,Ind_2(x,y,Ni+1,comp,Nx),Ind_2(x,y,Ni+1,comp,Nx),Ac,INSERT_VALUES);CHKERRQ(ierr);
                ind++;
                //Off diagonal (from aNc=1-sum(ak))
                for (PetscInt l=0; l<comp; l++) {
                    ierr = MatSetValue(Jac,Ind_2(x,y,Ni+1,comp,Nx),Ind_2(x,y,Ni+1,l,Nx),flux->dwdpi[al_index(x,y,comp,Nx)]*con_vars->ao[phi_index(0,0,Nc-1,Nx)]/pow(1-al[al_index(x,y,0,Nx)]-al[al_index(x,y,1,Nx)],2)*dt,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }
                for (PetscInt l=comp+1; l<Nc-1; l++) {
                    ierr = MatSetValue(Jac,Ind_2(x,y,Ni+1,comp,Nx),Ind_2(x,y,Ni+1,l,Nx),flux->dwdpi[al_index(x,y,comp,Nx)]*con_vars->ao[phi_index(0,0,Nc-1,Nx)]/((1-al[al_index(x,y,0,Nx)]-al[al_index(x,y,1,Nx)])*(1-al[al_index(x,y,0,Nx)]-al[al_index(x,y,1,Nx)]))*dt,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }
                for (ion=0; ion<Ni; ion++) {
                    //Volume to extra c
                    ierr = MatSetValue(Jac,Ind_2(x,y,Ni+1,comp,Nx),Ind_2(x,y,ion,Nc-1,Nx),flux->dwdpi[al_index(x,y,comp,Nx)]*dt,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Volume to intra c
                    ierr = MatSetValue(Jac,Ind_2(x,y,Ni+1,comp,Nx),Ind_2(x,y,ion,comp,Nx),-flux->dwdpi[al_index(x,y,comp,Nx)]*dt,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }
            }
        }
    }

    ierr = MatAssemblyBegin(Jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    if(Profiling_on) {
        PetscLogEventEnd(event[9], 0, 0, 0, 0);
    }
    return ierr;
}

int Newton_Solve_Grid(PetscInt xi, PetscInt yi,struct AppCtx *user) {


    if(Profiling_on) {
        PetscLogEventBegin(event[11], 0, 0, 0, 0);
    }

    PetscReal rsd;
    PetscErrorCode ierr = 0;
    PetscReal const *temp;

    PetscInt x,y,comp,ion;
    PetscInt Nx = 2*width_size+1;
    PetscInt Ny = 2*width_size+1;


    PetscReal tol = reltol * array_max(user->grid_vars_past->c, (size_t) Nx * Ny * Ni * Nc);
    rsd = tol + 1;

    for (PetscInt iter = 0; iter < itermax; iter++) {

        ierr = Grid_Residual_algebraic(user->grid_slvr->Res, xi, yi, user);CHKERRQ(ierr);

        ierr = VecNorm(user->grid_slvr->Res, NORM_MAX, &rsd);CHKERRQ(ierr);

//        printf("norm: %.10e\n",rsd);

        if (rsd < tol) {
            if(Profiling_on) {
                PetscLogEventEnd(event[11], 0, 0, 0, 0);
            }
            return iter;
        }
        ierr = Grid_Jacobian_algebraic(user->grid_slvr->A, xi, yi, user);CHKERRQ(ierr);

        //Set the new operator
        ierr = KSPSetOperators(user->grid_slvr->ksp, user->grid_slvr->A, user->grid_slvr->A);CHKERRQ(ierr);

        //Solve
        ierr = KSPSolve(user->grid_slvr->ksp, user->grid_slvr->Res, user->grid_slvr->Q);CHKERRQ(ierr);


        ierr = VecGetArrayRead(user->grid_slvr->Q, &temp);CHKERRQ(ierr);
        for (x = 0; x < Nx; x++) {
            for (y = 0; y < Ny; y++) {
                for (comp = 0; comp < Nc; comp++) {
                    for (ion = 0; ion < Ni; ion++) {
                        user->grid_vars->c[c_index(x, y, comp, ion, Nx)] -= temp[Ind_2(x,y,ion,comp,Nx)];
                    }
                    user->grid_vars->phi[phi_index(x, y, comp, Nx)] -= temp[Ind_2(x,y,Ni,comp,Nx)];
                }
                for (comp = 0; comp < Nc-1; comp++) {
                    user->grid_vars->alpha[al_index(x, y, comp, Nx)] -= temp[Ind_2(x,y,Ni+1,comp,Nx)];
                }
            }
        }
        ierr = VecRestoreArrayRead(user->grid_slvr->Q, &temp);CHKERRQ(ierr);

    }

    if (rsd > tol) {
        fprintf(stderr, "Netwon Iteration did not converge! Stopping...\n");
        exit(EXIT_FAILURE); /* indicate failure.*/
    }
    if(Profiling_on) {
        PetscLogEventEnd(event[11], 0, 0, 0, 0);
    }
    return itermax;
}
PetscErrorCode Update_Grid(PetscInt xi, PetscInt yi,PetscReal t,struct AppCtx *user)
{
    PetscErrorCode ierr = 0;


    PetscReal dt = user->dt;

    user->dt = user->dt_space[xy_index(xi,yi,user->Nx)];


    PetscInt Nx = 2*width_size+1;
    PetscInt Ny = 2*width_size+1;
    PetscInt ion,comp,x,y,iter;

    //Load current variable into past variable
    memcpy(user->grid_vars_past->c,user->grid_vars->c,sizeof(PetscReal)*Nx*Ny*Nc*Ni);
    memcpy(user->grid_vars_past->phi,user->grid_vars->phi,sizeof(PetscReal)*Nx*Ny*Nc);
    memcpy(user->grid_vars_past->alpha,user->grid_vars->alpha,sizeof(PetscReal)*Nx*Ny*(Nc-1));
    //Calculate diffusion
    //compute diffusion coefficients
    grid_diff_coef(user->Dcs, user->grid_vars_past->alpha, 1, user,xi,yi);
    //Bath diffusion
    grid_diff_coef(user->Dcb, user->grid_vars_past->alpha, Batheps, user,xi,yi);

    excitation_grid(user, t - dt, xi, yi);

    PetscInt steps = 0;
    PetscInt NSteps = (PetscInt)floor(dt/user->dt); //1;
    PetscInt accepted_step = 0;

    while(steps<NSteps) {

        //Perform Newton Solve
        iter = Newton_Solve_Grid(xi, yi, user);

        //Check if we accept the step
        if (iter < 3 || accepted_step || NSteps >= Max_Grid_Refine) {
            steps++;

            //Update Gating variable
            gatevars_update_grid(user->grid_gate_vars, user->grid_vars, user->dt * 1e3, user);

            //Update Excitation
            excitation_grid(user, t - dt+user->dt*steps, xi, yi);

            accepted_step = 1;

            //Load current variable into past variable
            memcpy(user->grid_vars_past->c,user->grid_vars->c,sizeof(PetscReal)*Nx*Ny*Nc*Ni);
            memcpy(user->grid_vars_past->phi,user->grid_vars->phi,sizeof(PetscReal)*Nx*Ny*Nc);
            memcpy(user->grid_vars_past->alpha,user->grid_vars->alpha,sizeof(PetscReal)*Nx*Ny*(Nc-1));
            //Calculate diffusion
            //compute diffusion coefficients
            grid_diff_coef(user->Dcs, user->grid_vars_past->alpha, 1, user,xi,yi);
            //Bath diffusion
            grid_diff_coef(user->Dcb, user->grid_vars_past->alpha, Batheps, user,xi,yi);

            if(xi==16&&yi==16) {
                write_point(user->fp, user, t - dt+user->dt*steps, 16, 16);
            }

        } else {
            //If we aren't below cutoff. Half the time step.
            user->dt = user->dt / 2;
            NSteps = 2 * NSteps;
//            printf("Reducing step at (%d,%d) to %f\n",xi,yi,user->dt);
            //Reset current vars
            //Load current variable into past variable
            memcpy(user->grid_vars->c,user->grid_vars_past->c,sizeof(PetscReal)*Nx*Ny*Nc*Ni);
            memcpy(user->grid_vars->phi,user->grid_vars_past->phi,sizeof(PetscReal)*Nx*Ny*Nc);
            memcpy(user->grid_vars->alpha,user->grid_vars_past->alpha,sizeof(PetscReal)*Nx*Ny*(Nc-1));
        }
    }

    if(user->dt<dt){
        user->dt_space[xy_index(xi,yi,user->Nx)] = 2*user->dt;
    } else{
        user->dt_space[xy_index(xi,yi,user->Nx)] = user->dt;
    }


    user->dt = dt;
    return ierr;

}

PetscErrorCode Update_Solution(Vec current_state,PetscReal t,struct AppCtx *user)
{

    PetscErrorCode ierr = 0;
    PetscInt x,y,ion,comp;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    PetscInt nx = 2*width_size+1;
    PetscInt ny = 2*width_size+1;
    PetscReal vm_new;
    PetscReal threshhold = 0.1;//0.1; //mV threshhold for update guess.



    for(x=0;x<Nx;x++){
        for(y=0;y<Ny;y++) {
            vm_new = (user->state_vars->phi[phi_index(x, y, 0, Nx)] -
                      user->state_vars->phi[phi_index(x, y, Nc - 1, Nx)]) * RTFC;

            //If it's above the threshhold. Or it previously was adaptively refined.
            if (fabs(vm_new - user->vm_past[xy_index(x, y, Nx)]) > threshhold ||user->dt_space[xy_index(x,y,Nx)]<user->dt) {
//            printf("Updating: (%d,%d)\n",x,y);
                // Load new gridpoint
                Load_Grid(user, x, y);
                //Update new grid
                Update_Grid(x, y, t, user);
                //Save the held variable
                Unload_Grid(user, x, y);
            }
        }
    }


    return ierr;

}

PetscErrorCode initialize_grid_jacobian(Mat Jac,struct AppCtx *user,int grid) {
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
                        ierr = MatSetValue(Jac,Ind_2(x+1,y,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Right c with left phi (-Fph0x)
                        ierr = MatSetValue(Jac,Ind_2(x+1,y,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        if (use_en_deriv&&!grid) {
                            //Right phi with left c in voltage eqn
                            ierr = MatSetValue(Jac,Ind_2(x+1,y,Ni,comp,Nx),Ind_2(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;
                        }


                    }
                    if(x>0) {
                        //left c with right c (-Fc1x)
                        ierr = MatSetValue(Jac,Ind_2(x-1,y,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Left c with right phi (-Fph1x)
                        ierr = MatSetValue(Jac,Ind_2(x-1,y,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        if (use_en_deriv&&!grid) {
                            //Left phi with right c in voltage eqn
                            ierr = MatSetValue(Jac, Ind_2(x - 1, y, Ni, comp,Nx), Ind_2(x, y, ion, comp,Nx), 0,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                        }
                    }
                    if(y<Ny-1) {
                        // Upper c with lower c (-Fc0y)
                        ierr = MatSetValue(Jac,Ind_2(x,y+1,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Upper c with lower phi (-Fph0y)
                        ierr = MatSetValue(Jac,Ind_2(x,y+1,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        if (use_en_deriv&&!grid) {
                            //Upper phi with lower c in voltage eqn
                            ierr = MatSetValue(Jac, Ind_2(x, y + 1, Ni, comp,Nx), Ind_2(x, y, ion, comp,Nx), 0,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                        }
                    }
                    if(y>0) {
                        //Lower c with Upper c (-Fc1y)
                        ierr = MatSetValue(Jac,Ind_2(x,y-1,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Lower c with Upper phi (-Fph1y)
                        ierr = MatSetValue(Jac,Ind_2(x,y-1,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        if (use_en_deriv&&!grid) {
                            //Lower phi with upper c in voltage eqn
                            ierr = MatSetValue(Jac, Ind_2(x, y - 1, Ni, comp,Nx), Ind_2(x, y, ion, comp,Nx), 0,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                        }
                    }

                    // Different Compartment Terms
                    // C Extracellular with C Inside
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,Nc-1,Nx),Ind_2(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // C Intra with C Extra
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,comp,Nx),Ind_2(x,y,ion,Nc-1,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // C Extracellular with Phi Inside
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,Nc-1,Nx),Ind_2(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // C Intra with Phi Extra
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,comp,Nx),Ind_2(x,y,Ni,Nc-1,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    if(!separate_vol||grid) {
                        //Volume terms
                        //C extra with intra alpha
                        ierr = MatSetValue(Jac, Ind_2(x, y, ion, Nc - 1,Nx), Ind_2(x, y, Ni + 1, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //C intra with intra alpha
                        ierr = MatSetValue(Jac, Ind_2(x, y, ion, comp,Nx), Ind_2(x, y, Ni + 1, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    //Same compartment terms
                    // c with c
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // c with phi
                    ierr = MatSetValue(Jac,Ind_2(x,y,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    if (use_en_deriv&&!grid) {
                        //Intra-Phi with c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp,Nx), Ind_2(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //IntraPhi with c extra(volt eqn)
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp,Nx), Ind_2(x, y, ion, Nc - 1,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;

                        //Extra-Phi with intra-c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni, Nc - 1,Nx), Ind_2(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }

                }
                //Extracellular terms
                comp = Nc-1;
                //Electrodiffusion contributions
                if(x<Nx-1) {
                    // Right c with left c (-Fc0x)
                    ierr = MatSetValue(Jac,Ind_2(x+1,y,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Right c with left phi (-Fph0x)
                    ierr = MatSetValue(Jac,Ind_2(x+1,y,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    if (use_en_deriv&&!grid) {
                        // left Phi with right c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_2(x + 1, y, Ni, comp,Nx), Ind_2(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                }
                if(x>0) {
                    //left c with right c (-Fc1x)
                    ierr = MatSetValue(Jac,Ind_2(x-1,y,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Left c with right phi (-Fph1x)
                    ierr = MatSetValue(Jac,Ind_2(x-1,y,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    if (use_en_deriv&&!grid) {
                        // left Phi with right c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_2(x - 1, y, Ni, comp,Nx), Ind_2(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                }
                if(y<Ny-1) {
                    // Upper c with lower c (-Fc0y)
                    ierr = MatSetValue(Jac,Ind_2(x,y+1,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Upper c with lower phi (-Fph0y)
                    ierr = MatSetValue(Jac,Ind_2(x,y+1,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    if (use_en_deriv&&!grid) {
                        // Upper Phi with lower c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_2(x, y + 1, Ni, comp,Nx), Ind_2(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                }
                if(y>0) {
                    //Lower c with Upper c (-Fc1y)
                    ierr = MatSetValue(Jac,Ind_2(x,y-1,ion,comp,Nx),Ind_2(x,y,ion,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Lower c with Upper phi (-Fph1y)
                    ierr = MatSetValue(Jac,Ind_2(x,y-1,ion,comp,Nx),Ind_2(x,y,Ni,comp,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    if (use_en_deriv&&!grid) {
                        // Lower Phi with upper c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_2(x, y - 1, Ni, comp,Nx), Ind_2(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                }
                //Insert extracell to extracell parts
                // c with c
                ierr = MatSetValue(Jac,Ind_2(x,y,ion,Nc-1,Nx),Ind_2(x,y,ion,Nc-1,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                ind++;
                // c with phi
                ierr = MatSetValue(Jac,Ind_2(x,y,ion,Nc-1,Nx),Ind_2(x,y,Ni,Nc-1,Nx),0,INSERT_VALUES);CHKERRQ(ierr);
                ind++;
                if (use_en_deriv&&!grid) {
                    //phi with c (voltage eqn)
                    ierr = MatSetValue(Jac, Ind_2(x, y, Ni, Nc - 1,Nx), Ind_2(x, y, ion, Nc - 1,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
            }
            if (use_en_deriv&&!grid) {
                //Derivative of charge-capacitance
                for (comp = 0; comp < Nc - 1; comp++) {
                    if (x < Nx - 1) {
                        //Right phi with left phi (-Fph0x)
                        ierr = MatSetValue(Jac, Ind_2(x + 1, y, Ni, comp,Nx), Ind_2(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if (x > 0) {
                        //Left phi with right phi (-Fph1x)
                        ierr = MatSetValue(Jac, Ind_2(x - 1, y, Ni, comp,Nx), Ind_2(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if (y < Ny - 1) {
                        //Upper phi with lower phi (-Fph0y)
                        ierr = MatSetValue(Jac, Ind_2(x, y + 1, Ni, comp,Nx), Ind_2(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if (y > 0) {
                        //Lower phi with upper phi (-Fph1y)
                        ierr = MatSetValue(Jac, Ind_2(x, y - 1, Ni, comp,Nx), Ind_2(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    //Intra-phi with Intra-phi
                    ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp,Nx), Ind_2(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Intra-phi with extra-phi
                    ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp,Nx), Ind_2(x, y, Ni, Nc - 1,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                //Extracellular terms
                comp = Nc - 1;
                if (x < Nx - 1) {
                    //Right phi with left phi (-Fph0x)
                    ierr = MatSetValue(Jac, Ind_2(x + 1, y, Ni, comp,Nx), Ind_2(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                if (x > 0) {
                    //Left phi with right phi (-Fph1x)
                    ierr = MatSetValue(Jac, Ind_2(x - 1, y, Ni, comp,Nx), Ind_2(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                if (y < Ny - 1) {
                    //Upper phi with lower phi (-Fph0y)
                    ierr = MatSetValue(Jac, Ind_2(x, y + 1, Ni, comp,Nx), Ind_2(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                if (y > 0) {
                    //Lower phi with upper phi (-Fph1y)
                    ierr = MatSetValue(Jac, Ind_2(x, y - 1, Ni, comp,Nx), Ind_2(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }

                for (int k = 0; k < Nc - 1; k++) {
                    //Extra-phi with Intra-phi
                    ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp,Nx), Ind_2(x, y, Ni, k,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                //extra-phi with extra-phi
                ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp,Nx), Ind_2(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
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
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp,Nx), Ind_2(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    //Phi with C extracellular one
                    comp = Nc - 1;
                    ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp,Nx), Ind_2(x, y, ion, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;

                }
                //electroneutrality-voltage entries

                //extraphi with extra phi
                ierr = MatSetValue(Jac, Ind_2(x, y, Ni, Nc - 1,Nx), Ind_2(x, y, Ni, Nc - 1,Nx), 0, INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
                for (comp = 0; comp < Nc - 1; comp++) {
                    //Extra phi with intra phi
                    ierr = MatSetValue(Jac, Ind_2(x, y, Ni, Nc - 1,Nx), Ind_2(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    // Intra phi with Extraphi
                    ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp,Nx), Ind_2(x, y, Ni, Nc - 1,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Intra phi with Intra phi
                    ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp,Nx), Ind_2(x, y, Ni, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    if(!separate_vol||grid) {
                        //Extra phi with intra-Volume
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni, Nc - 1,Nx), Ind_2(x, y, Ni + 1, comp,Nx), 0,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Intra phi with Intra Vol
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp,Nx), Ind_2(x, y, Ni + 1, comp,Nx), 0,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if(grid) {
                        //Extra phi with intra phi
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni, Nc - 1, Nx), Ind_2(x, y, Ni, comp, Nx), cm[comp],
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // Intra phi with Extraphi
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp, Nx), Ind_2(x, y, Ni, Nc - 1, Nx), cm[comp],
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Intra phi with Intra phi
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni, comp, Nx), Ind_2(x, y, Ni, comp, Nx), -cm[comp],
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }

                }
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
                    ierr = MatSetValue(Jac, Ind_2(x, y, Ni + 1, comp,Nx), Ind_2(x, y, Ni + 1, comp,Nx), 0, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Off diagonal (from aNc=1-sum(ak))
                    for (PetscInt l = 0; l < comp; l++) {
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni + 1, comp,Nx), Ind_2(x, y, Ni + 1, l,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    for (PetscInt l = comp + 1; l < Nc - 1; l++) {
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni + 1, comp,Nx), Ind_2(x, y, Ni + 1, l,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    for (ion = 0; ion < Ni; ion++) {
                        //Volume to extra c
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni + 1, comp,Nx), Ind_2(x, y, ion, Nc - 1,Nx), 0, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Volume to intra c
                        ierr = MatSetValue(Jac, Ind_2(x, y, Ni + 1, comp,Nx), Ind_2(x, y, ion, comp,Nx), 0, INSERT_VALUES);
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
