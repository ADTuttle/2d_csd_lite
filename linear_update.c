#include "constants.h"
#include "functions.h"

PetscErrorCode calc_residual_linear_algebraic(SNES snes,Vec current_state,Vec Res,void *ctx)
{
    //Linear discretization
    //Residual equation using algebraic version of the charge-capacitance relation
    //Alpha is not solved for here
    struct AppCtx * user = (struct AppCtx *) ctx;
    PetscErrorCode ierr;
    if(Profiling_on) {
        PetscLogEventBegin(event[1], 0, 0, 0, 0);
    }
    ierr = extract_subarray_Read(current_state,user->state_vars); CHKERRQ(ierr);
    //Compute membrane ionic flux relation quantitites
    ionmflux(user);


    //Compute membrane water flow related quantities
    wflowm(user);

    PetscReal *c = user->state_vars->c;
    PetscReal *phi = user->state_vars->phi;
    PetscReal *al = user->state_vars->alpha;
    PetscReal *cp = user->state_vars_past->c;
    PetscReal *alp = user->state_vars_past->alpha;
    PetscReal *phip = user->state_vars_past->phi;

    PetscReal *Dcs = user->Dcs;
    PetscReal *Dcb = user->Dcb;
    struct FluxData *flux = user->flux;
    PetscReal dt = user->dt;
    PetscReal dx = user->dx;
    PetscReal dy = user->dy;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;

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
                        // C term
                        Rcvx = Dcs[c_index(x-1,y,comp,ion,Nx)*2]*(c[c_index(x,y,comp,ion,Nx)]-c[c_index(x-1,y,comp,ion,Nx)]);
                        //Phi term
                        Rcvx += Dcs[c_index(x-1,y,comp,ion,Nx)*2]*z[ion]*(cp[c_index(x-1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x-1,y,comp,Nx)]);
                        Rcvx = Rcvx/dx*dt/dx;
                    }
                    //Add Second right moving difference
                    if(x<Nx-1) {
                        // C term
                        RcvxRight = Dcs[c_index(x,y,comp,ion,Nx)*2]*(c[c_index(x+1,y,comp,ion,Nx)]-c[c_index(x,y,comp,ion,Nx)]);
                        //Phi term
                        RcvxRight += Dcs[c_index(x,y,comp,ion,Nx)*2]*z[ion]*(cp[c_index(x+1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x+1,y,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]);
                        RcvxRight = RcvxRight/dx*dt/dx;
                    }
                    Rcvy = 0;
                    RcvyUp = 0;
                    //Up down difference
                    if(y>0) {
                        // C term
                        Rcvy = Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*(c[c_index(x,y,comp,ion,Nx)]-c[c_index(x,y-1,comp,ion,Nx)]);
                        //Phi term
                        Rcvy += Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*z[ion]*(cp[c_index(x,y-1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x,y-1,comp,Nx)]);
                        Rcvy = Rcvy/dy*dt/dy;
                    }
                    //Next upward difference
                    if(y<Ny-1) {
                        // C term
                        RcvyUp = Dcs[c_index(x,y,comp,ion,Nx)*2+1]*(c[c_index(x,y+1,comp,ion,Nx)]-c[c_index(x,y,comp,ion,Nx)]);
                        //Phi term
                        RcvyUp += Dcs[c_index(x,y,comp,ion,Nx)*2+1]*z[ion]*(cp[c_index(x,y+1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x,y+1,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]);
                        RcvyUp = RcvyUp/dy*dt/dy;
                    }
                    Resc = al[al_index(x,y,comp,Nx)]*c[c_index(x,y,comp,ion,Nx)]-alp[al_index(x,y,comp,Nx)]*cp[c_index(x,y,comp,ion,Nx)];
                    Resc += Rcvx - RcvxRight + Rcvy - RcvyUp + flux->mflux[c_index(x,y,comp,ion,Nx)]*dt;

                    ierr = VecSetValue(Res,Ind_1(x,y,ion,comp,Nx),Resc,INSERT_VALUES);CHKERRQ(ierr);

                }
                //Set Extracellular values
                alNc = 1 - al[al_index(x,y,0,Nx)] - al[al_index(x,y,1,Nx)];
                alpNc = 1 - alp[al_index(x,y,0,Nx)] - alp[al_index(x,y,1,Nx)];
                comp = Nc-1;
                Rcvx = 0;
                RcvxRight = 0;
                if(x>0) {
                    //First difference term
                    // C term
                    Rcvx = Dcs[c_index(x-1,y,comp,ion,Nx)*2]*(c[c_index(x,y,comp,ion,Nx)]-c[c_index(x-1,y,comp,ion,Nx)]);
                    //Phi term
                    Rcvx += Dcs[c_index(x-1,y,comp,ion,Nx)*2]*z[ion]*(cp[c_index(x-1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x-1,y,comp,Nx)]);
                    Rcvx = Rcvx/dx*dt/dx;
                }
                //Add Second right moving difference
                if(x<Nx-1) {
                    // C term
                    RcvxRight = Dcs[c_index(x,y,comp,ion,Nx)*2]*(c[c_index(x+1,y,comp,ion,Nx)]-c[c_index(x,y,comp,ion,Nx)]);
                    //Phi term
                    RcvxRight += Dcs[c_index(x,y,comp,ion,Nx)*2]*z[ion]*(cp[c_index(x+1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x+1,y,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]);
                    RcvxRight = RcvxRight/dx*dt/dx;
                }
                Rcvy = 0;
                RcvyUp = 0;
                //Up down difference
                if(y>0) {
                    // C term
                    Rcvy = Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*(c[c_index(x,y,comp,ion,Nx)]-c[c_index(x,y-1,comp,ion,Nx)]);
                    //Phi term
                    Rcvy += Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*z[ion]*(cp[c_index(x,y-1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x,y-1,comp,Nx)]);
                    Rcvy = Rcvy/dy*dt/dy;
                }
                //Next upward difference
                if(y<Ny-1) {
                    // C term
                    RcvyUp = Dcs[c_index(x,y,comp,ion,Nx)*2+1]*(c[c_index(x,y+1,comp,ion,Nx)]-c[c_index(x,y,comp,ion,Nx)]);
                    //Phi term
                    RcvyUp += Dcs[c_index(x,y,comp,ion,Nx)*2+1]*z[ion]*(cp[c_index(x,y+1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x,y+1,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]);
                    RcvyUp = RcvyUp/dy*dt/dy;
                }
                Resc = alNc*c[c_index(x,y,comp,ion,Nx)]-alpNc*cp[c_index(x,y,comp,ion,Nx)];
                Resc += Rcvx - RcvxRight + Rcvy - RcvyUp + flux->mflux[c_index(x,y,comp,ion,Nx)]*dt;
                //Add bath variables

                Resc -= sqrt(pow(Dcb[c_index(x,y,comp,ion,Nx)*2],2)+pow(Dcb[c_index(x,y,comp,ion,Nx)*2+1],2))*(c[c_index(x,y,comp,ion,Nx)]-cbath[ion]+z[ion]*(cp[c_index(x,y,comp,ion,Nx)]+cbath[ion])/2.0*(phi[phi_index(x,y,comp,Nx)]-phibath))*dt;
                ierr = VecSetValue(Res,Ind_1(x,y,ion,comp,Nx),Resc,INSERT_VALUES);CHKERRQ(ierr);

            }
        }
    }


    for(x=0;x<Nx;x++) {
        for(y=0;y<Ny;y++) {

            //Residual for electroneutrality condition
            for(comp=0;comp<Nc-1;comp++)
            {

                Resc = al[al_index(x,y,comp,Nx)]*cz(c,z,x,y,Nx,comp,user)+user->con_vars->zo[phi_index(0,0,comp,Nx)]*user->con_vars->ao[phi_index(0,0,comp,Nx)];
                ierr = VecSetValue(Res,Ind_1(x,y,Ni,comp,Nx),Resc,INSERT_VALUES); CHKERRQ(ierr);
            }
            //Extracellular term
            comp=Nc-1;
            Resc = (1-al[al_index(x,y,0,Nx)]-al[al_index(x,y,1,Nx)])*cz(c,z,x,y,Nx,comp,user)+user->con_vars->zo[phi_index(0,0,comp,Nx)]*user->con_vars->ao[phi_index(0,0,comp,Nx)];
            ierr = VecSetValue(Res,Ind_1(x,y,Ni,comp,Nx),Resc,INSERT_VALUES); CHKERRQ(ierr);
        }
    }
    //Assemble before we add values in on top to modify the electroneutral.
    ierr = VecAssemblyBegin(Res);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Res);CHKERRQ(ierr);

    for(x=0;x<Nx;x++)
    {
        for(y=0;y<Ny;y++)
        {
            // Add Modification to electroneutrality for non-zero mem.compacitance
            for(comp=0;comp<Nc-1;comp++)
            {
                //Extracell voltage
                ierr = VecSetValue(Res,Ind_1(x,y,Ni,Nc-1,Nx),-cm[comp]*(phi[phi_index(x,y,Nc-1,Nx)]-phi[phi_index(x,y,comp,Nx)]),ADD_VALUES);CHKERRQ(ierr);
                //Intracell voltage mod
                ierr = VecSetValue(Res,Ind_1(x,y,Ni,comp,Nx),-cm[comp]*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x,y,Nc-1,Nx)]),ADD_VALUES);CHKERRQ(ierr);
            }
        }
    }

    ierr = VecAssemblyBegin(Res);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Res);CHKERRQ(ierr);

    ierr = restore_subarray_Read(current_state,user->state_vars); CHKERRQ(ierr);
    if(Profiling_on) {
        PetscLogEventEnd(event[1], 0, 0, 0, 0);
    }
    return ierr;
}

PetscErrorCode
calc_jacobian_linear_algebraic(SNES snes,Vec current_state, Mat A, Mat Jac,void *ctx)
{
    //Linear discretization
    //Jacobian equation using algebraic version of the charge-capacitance relation
    // Alpha is not solved for here
    struct AppCtx * user = (struct AppCtx *) ctx;
    PetscErrorCode ierr;
    if(Profiling_on) {
        PetscLogEventBegin(event[0], 0, 0, 0, 0);
    }
    ierr = extract_subarray_Read(current_state,user->state_vars); CHKERRQ(ierr);
    PetscReal *c = user->state_vars->c;
    PetscReal *al = user->state_vars->alpha;
    PetscReal *cp = user->state_vars_past->c;

    PetscReal *Dcs = user->Dcs;
    PetscReal *Dcb = user->Dcb;
    struct FluxData *flux = user->flux;
    PetscReal dt = user->dt;
    PetscReal dx = user->dx;
    PetscReal dy = user->dy;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    struct ConstVars *con_vars = user->con_vars;

    PetscInt ind = 0;
    PetscInt x,y,ion,comp;

    PetscReal Fc0x,Fc1x,Fph0x,Fph1x;
    PetscReal Fc0y,Fc1y,Fph0y,Fph1y;
    PetscReal Ac,Aphi;

    PetscInt iter;
    ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);
    //Only calculate off diagonals on first iteration
    // For next iterations only diagonal changes
    if(iter==0) {
        //Ionic concentration equations
        for (x = 0; x < Nx; x++) {
            for (y = 0; y < Ny; y++) {
                for (ion = 0; ion < Ni; ion++) {
                    for (comp = 0; comp < Nc - 1; comp++) {
                        //Electrodiffusion contributions
                        Fc0x = 0;
                        Fc1x = 0;
                        Fph0x = 0;
                        Fph1x = 0;
                        Fc0y = 0;
                        Fc1y = 0;
                        Fph0y = 0;
                        Fph1y = 0;
                        if (x < Nx - 1) {
                            Fc0x = Dcs[c_index(x, y, comp, ion,Nx) * 2] / dx * dt / dx;
                            Fph0x = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2] *
                                    (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x + 1, y, comp, ion,Nx)]) / 2 / dx * dt /
                                    dx;
                            // Right c with left c (-Fc0x)
                            ierr = MatSetValue(Jac, Ind_1(x + 1, y, ion, comp,Nx), Ind_1(x, y, ion, comp,Nx), -Fc0x,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                            //Right c with left phi (-Fph0x)
                            ierr = MatSetValue(Jac, Ind_1(x + 1, y, ion, comp,Nx), Ind_1(x, y, Ni, comp,Nx), -Fph0x,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;

                        }
                        if (x > 0) {
                            Fc1x = Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] / dx * dt / dx;
                            Fph1x = z[ion] * Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] *
                                    (cp[c_index(x - 1, y, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)]) / 2 / dx * dt /
                                    dx;
                            //left c with right c (-Fc1x)
                            ierr = MatSetValue(Jac, Ind_1(x - 1, y, ion, comp,Nx), Ind_1(x, y, ion, comp,Nx), -Fc1x,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                            //Left c with right phi (-Fph1x)
                            ierr = MatSetValue(Jac, Ind_1(x - 1, y, ion, comp,Nx), Ind_1(x, y, Ni, comp,Nx), -Fph1x,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                        }
                        if (y < Ny - 1) {
                            Fc0y = Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                            Fph0y = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] *
                                    (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x, y + 1, comp, ion,Nx)]) / 2 / dy * dt /
                                    dy;
                            // Upper c with lower c (-Fc0y)
                            ierr = MatSetValue(Jac, Ind_1(x, y + 1, ion, comp,Nx), Ind_1(x, y, ion, comp,Nx), -Fc0y,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                            //Upper c with lower phi (-Fph0y)
                            ierr = MatSetValue(Jac, Ind_1(x, y + 1, ion, comp,Nx), Ind_1(x, y, Ni, comp,Nx), -Fph0y,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                        }
                        if (y > 0) {
                            Fc1y = Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                            Fph1y = z[ion] * Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] *
                                    (cp[c_index(x, y - 1, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)]) / 2 / dy * dt /
                                    dy;
                            //Lower c with Upper c (-Fc1y)
                            ierr = MatSetValue(Jac, Ind_1(x, y - 1, ion, comp,Nx), Ind_1(x, y, ion, comp,Nx), -Fc1y,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                            //Lower c with Upper phi (-Fph1y)
                            ierr = MatSetValue(Jac, Ind_1(x, y - 1, ion, comp,Nx), Ind_1(x, y, Ni, comp,Nx), -Fph1y,
                                               INSERT_VALUES);
                            CHKERRQ(ierr);
                            ind++;
                        }
                        //Diagonal term contribution
                        Ac = al[al_index(x, y, comp,Nx)] + Fc0x + Fc1x + Fc0y + Fc1y;
                        Aphi = Fph0x + Fph1x + Fph0y + Fph1y;


                        //membrane current contributions
                        Ac += flux->dfdci[c_index(x, y, comp, ion,Nx)] * dt;
                        Aphi += flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt;
                        // Different Compartment Terms
                        // C Extracellular with C Inside
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, ion, comp,Nx),
                                           -flux->dfdci[c_index(x, y, comp, ion,Nx)] * dt, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // C Intra with C Extra
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, ion, Nc - 1,Nx),
                                           flux->dfdce[c_index(x, y, comp, ion,Nx)] * dt, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // C Extracellular with Phi Inside
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, Ni, comp,Nx),
                                           -flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // C Intra with Phi Extra
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, Ni, Nc - 1,Nx),
                                           -flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Same compartment terms
                        // c with c
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, ion, comp,Nx), Ac, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // c with phi
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, Ni, comp,Nx), Aphi, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;

                    }
                    //Extracellular terms
                    comp = Nc - 1;
                    //Electrodiffusion contributions
                    Fc0x = 0;
                    Fc1x = 0;
                    Fph0x = 0;
                    Fph1x = 0;
                    Fc0y = 0;
                    Fc1y = 0;
                    Fph0y = 0;
                    Fph1y = 0;
                    if (x < Nx - 1) {
                        Fc0x = Dcs[c_index(x, y, comp, ion,Nx) * 2] / dx * dt / dx;
                        Fph0x = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2] *
                                (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x + 1, y, comp, ion,Nx)]) / 2 / dx * dt / dx;
                        // Right c with left c (-Fc0x)
                        ierr = MatSetValue(Jac, Ind_1(x + 1, y, ion, comp,Nx), Ind_1(x, y, ion, comp,Nx), -Fc0x,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Right c with left phi (-Fph0x)
                        ierr = MatSetValue(Jac, Ind_1(x + 1, y, ion, comp,Nx), Ind_1(x, y, Ni, comp,Nx), -Fph0x,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if (x > 0) {
                        Fc1x = Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] / dx * dt / dx;
                        Fph1x = z[ion] * Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] *
                                (cp[c_index(x - 1, y, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)]) / 2 / dx * dt / dx;
                        //left c with right c (-Fc1x)
                        ierr = MatSetValue(Jac, Ind_1(x - 1, y, ion, comp,Nx), Ind_1(x, y, ion, comp,Nx), -Fc1x,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Left c with right phi (-Fph1x)
                        ierr = MatSetValue(Jac, Ind_1(x - 1, y, ion, comp,Nx), Ind_1(x, y, Ni, comp,Nx), -Fph1x,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if (y < Ny - 1) {
                        Fc0y = Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                        Fph0y = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] *
                                (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x, y + 1, comp, ion,Nx)]) / 2 / dy * dt / dy;
                        // Upper c with lower c (-Fc0y)
                        ierr = MatSetValue(Jac, Ind_1(x, y + 1, ion, comp,Nx), Ind_1(x, y, ion, comp,Nx), -Fc0y,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Upper c with lower phi (-Fph0y)
                        ierr = MatSetValue(Jac, Ind_1(x, y + 1, ion, comp,Nx), Ind_1(x, y, Ni, comp,Nx), -Fph0y,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    if (y > 0) {
                        Fc1y = Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                        Fph1y = z[ion] * Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] *
                                (cp[c_index(x, y - 1, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)]) / 2 / dy * dt / dy;
                        //Lower c with Upper c (-Fc1y)
                        ierr = MatSetValue(Jac, Ind_1(x, y - 1, ion, comp,Nx), Ind_1(x, y, ion, comp,Nx), -Fc1y,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Lower c with Upper phi (-Fph1y)
                        ierr = MatSetValue(Jac, Ind_1(x, y - 1, ion, comp,Nx), Ind_1(x, y, Ni, comp,Nx), -Fph1y,
                                           INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }

                    //Diagonal term contribution
                    Ac = (1 - al[al_index(x, y, 0,Nx)] - al[al_index(x, y, 1,Nx)]) + Fc0x + Fc1x + Fc0y + Fc1y;
                    Aphi = Fph0x + Fph1x + Fph0y + Fph1y;

                    //Membrane current contribution
                    for (comp = 0; comp < Nc - 1; comp++) {
                        Ac -= flux->dfdce[c_index(x, y, comp, ion,Nx)] * dt;
                        Aphi += flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt;
                    }
                    //Add bath contributions
                    Ac -= sqrt(pow(Dcb[c_index(x, y, Nc - 1, ion,Nx) * 2], 2) +
                               pow(Dcb[c_index(x, y, Nc - 1, ion,Nx) * 2 + 1], 2)) * dt;
                    Aphi -= sqrt(pow(Dcb[c_index(x, y, Nc - 1, ion,Nx) * 2], 2) +
                                 pow(Dcb[c_index(x, y, Nc - 1, ion,Nx) * 2 + 1], 2)) *
                            (cp[c_index(x, y, Nc - 1, ion,Nx)] + cbath[ion]) * z[ion] / 2 * dt;

                    //Insert extracell to extracell parts
                    // c with c
                    ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, ion, Nc - 1,Nx), Ac, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    // c with phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, Ni, Nc - 1,Nx), Aphi, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }

            }
        }

        //Electroneutrality charge-capacitence condition
        for (x = 0; x < Nx; x++) {
            for (y = 0; y < Ny; y++) {
                //electroneutral-concentration entries
                for (ion = 0; ion < Ni; ion++) {
                    for (comp = 0; comp < Nc - 1; comp++) {
                        //Phi with C entries
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, ion, comp,Nx),
                                           z[ion] * al[al_index(x, y, comp,Nx)], INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                    }
                    //Phi with C extracellular one
                    comp = Nc - 1;
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, ion, comp,Nx),
                                       z[ion] * (1 - al[al_index(x, y, 0,Nx)] - al[al_index(x, y, 1,Nx)]), INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;

                }
                //electroneutrality-voltage entries
                Aphi = 0;
                for (comp = 0; comp < Nc - 1; comp++) {
                    Aphi -= cm[comp];
                }
                //extraphi with extra phi
                ierr = MatSetValue(Jac, Ind_1(x, y, Ni, Nc - 1,Nx), Ind_1(x, y, Ni, Nc - 1,Nx), Aphi, INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;
                for (comp = 0; comp < Nc - 1; comp++) {
                    //Extra phi with intra phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, Nc - 1,Nx), Ind_1(x, y, Ni, comp,Nx), cm[comp], INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    // Intra phi with Extraphi
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni, Nc - 1,Nx), cm[comp], INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Intra phi with Intra phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), -cm[comp], INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
            }
        }
    } else {
        //Ionic concentration equations
        for (x = 0; x < Nx; x++) {
            for (y = 0; y < Ny; y++) {
                for (ion = 0; ion < Ni; ion++) {
                    for (comp = 0; comp < Nc - 1; comp++) {
                        //Electrodiffusion contributions
                        Fc0x = 0;
                        Fc1x = 0;
                        Fph0x = 0;
                        Fph1x = 0;
                        Fc0y = 0;
                        Fc1y = 0;
                        Fph0y = 0;
                        Fph1y = 0;
                        if (x < Nx - 1) {
                            Fc0x = Dcs[c_index(x, y, comp, ion,Nx) * 2] / dx * dt / dx;
                            Fph0x = z[ion]*Dcs[c_index(x, y, comp, ion,Nx) * 2]*(cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x+1,y,comp,ion,Nx)])/2/dx*dt/dx;
                        }
                        if (x > 0) {
                            Fc1x = Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] / dx * dt / dx;
                            Fph1x = z[ion] * Dcs[c_index(x-1, y,comp,ion,Nx) * 2] * (cp[c_index(x-1, y, comp, ion,Nx)] + cp[c_index(x,y, comp, ion,Nx)])/2/dx*dt/dx;

                        }
                        if (y < Ny - 1) {
                            Fc0y = Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                            Fph0y = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] * (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x, y + 1, comp, ion,Nx)]) / 2 / dy * dt / dy;
                        }
                        if (y > 0) {
                            Fc1y = Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                            Fph1y = z[ion] * Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] * (cp[c_index(x, y - 1, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)]) / 2 / dy * dt / dy;
                        }
                        //Diagonal term contribution
                        Ac = al[al_index(x, y, comp,Nx)] + Fc0x + Fc1x + Fc0y + Fc1y;
                        Aphi = Fph0x + Fph1x + Fph0y + Fph1y;


                        //membrane current contributions
                        Ac += flux->dfdci[c_index(x, y, comp, ion,Nx)] * dt;
                        Aphi += flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt;
                        // Different Compartment Terms
                        // C Extracellular with C Inside
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, ion, comp,Nx), -flux->dfdci[c_index(x, y, comp, ion,Nx)] * dt, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // C Intra with C Extra
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, ion, Nc - 1,Nx),
                                           flux->dfdce[c_index(x, y, comp, ion,Nx)] * dt, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // C Extracellular with Phi Inside
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, Ni, comp,Nx),
                                           -flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // C Intra with Phi Extra
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, Ni, Nc - 1,Nx),
                                           -flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Same compartment terms
                        // c with c
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, ion, comp,Nx), Ac, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // c with phi
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, Ni, comp,Nx), Aphi, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;

                    }
                    //Extracellular terms
                    comp = Nc - 1;
                    //Electrodiffusion contributions
                    Fc0x = 0;
                    Fc1x = 0;
                    Fph0x = 0;
                    Fph1x = 0;
                    Fc0y = 0;
                    Fc1y = 0;
                    Fph0y = 0;
                    Fph1y = 0;
                    if (x < Nx - 1) {
                        Fc0x = Dcs[c_index(x, y, comp, ion,Nx) * 2] / dx * dt / dx;
                        Fph0x = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2] * (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x + 1, y, comp, ion,Nx)])/2/dx*dt/dx;
                    }
                    if (x > 0) {
                        Fc1x = Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] / dx * dt / dx;
                        Fph1x = z[ion] * Dcs[c_index(x - 1, y, comp, ion,Nx) * 2]*(cp[c_index(x - 1, y, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)])/2/dx*dt/dx;
                    }
                    if (y < Ny - 1) {
                        Fc0y = Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                        Fph0y = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] * (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x, y + 1, comp, ion,Nx)])/2/dy*dt/dy;
                    }
                    if (y > 0) {
                        Fc1y = Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                        Fph1y = z[ion] * Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] * (cp[c_index(x, y - 1, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)])/2/dy*dt/dy;
                    }

                    //Diagonal term contribution
                    Ac = (1 - al[al_index(x, y, 0,Nx)] - al[al_index(x, y, 1,Nx)]) + Fc0x + Fc1x + Fc0y + Fc1y;
                    Aphi = Fph0x + Fph1x + Fph0y + Fph1y;

                    //Membrane current contribution
                    for (comp = 0; comp < Nc - 1; comp++) {
                        Ac -= flux->dfdce[c_index(x, y, comp, ion,Nx)] * dt;
                        Aphi += flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt;
                    }
                    //Add bath contributions
                    Ac -= sqrt(pow(Dcb[c_index(x, y, Nc - 1, ion,Nx) * 2], 2) +
                               pow(Dcb[c_index(x, y, Nc - 1, ion,Nx) * 2 + 1], 2)) * dt;
                    Aphi -= sqrt(pow(Dcb[c_index(x, y, Nc - 1, ion,Nx) * 2], 2) +
                                 pow(Dcb[c_index(x, y, Nc - 1, ion,Nx) * 2 + 1], 2)) *
                            (cp[c_index(x, y, Nc - 1, ion,Nx)] + cbath[ion]) * z[ion] / 2 * dt;

                    //Insert extracell to extracell parts
                    // c with c
                    ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, ion, Nc - 1,Nx), Ac, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    // c with phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, Ni, Nc - 1,Nx), Aphi, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }

            }
        }
    }
    ierr = MatAssemblyBegin(Jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

//    printf("Iter: %d, Number of inserts: %d\n",iter,ind);
    if (A != Jac) {
        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); }

    ierr = restore_subarray_Read(current_state,user->state_vars); CHKERRQ(ierr);
    if(Profiling_on) {
        PetscLogEventEnd(event[0], 0, 0, 0, 0);
    }
    return ierr;
}

PetscErrorCode calc_residual_linear_deriv(SNES snes,Vec current_state,Vec Res,void *ctx)
{
    //Linear discretization
    //Residual equation using derivative of the charge-capacitance relation
    // Volume not solved for here
    struct AppCtx * user = (struct AppCtx *) ctx;
    PetscErrorCode ierr;
    if(Profiling_on) {
        PetscLogEventBegin(event[1], 0, 0, 0, 0);
    }
    ierr = extract_subarray_Read(current_state,user->state_vars); CHKERRQ(ierr);
    //Compute membrane ionic flux relation quantitites
    ionmflux(user);

    //Compute membrane water flow related quantities
    wflowm(user);

    PetscReal *c = user->state_vars->c;
    PetscReal *phi = user->state_vars->phi;
    PetscReal *al = user->state_vars->alpha;
    PetscReal *cp = user->state_vars_past->c;
    PetscReal *alp = user->state_vars_past->alpha;
    PetscReal *phip = user->state_vars_past->phi;

    PetscReal *Dcs = user->Dcs;
    PetscReal *Dcb = user->Dcb;
    struct FluxData *flux = user->flux;
    PetscReal dt = user->dt;
    PetscReal dx = user->dx;
    PetscReal dy = user->dy;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;

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
                        // C term
                        Rcvx = Dcs[c_index(x-1,y,comp,ion,Nx)*2]*(c[c_index(x,y,comp,ion,Nx)]-c[c_index(x-1,y,comp,ion,Nx)]);
                        //Phi term
                        Rcvx += Dcs[c_index(x-1,y,comp,ion,Nx)*2]*z[ion]*(cp[c_index(x-1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x-1,y,comp,Nx)]);
                        Rcvx = Rcvx/dx*dt/dx;
                    }
                    //Add Second right moving difference
                    if(x<Nx-1) {
                        // C term
                        RcvxRight = Dcs[c_index(x,y,comp,ion,Nx)*2]*(c[c_index(x+1,y,comp,ion,Nx)]-c[c_index(x,y,comp,ion,Nx)]);
                        //Phi term
                        RcvxRight += Dcs[c_index(x,y,comp,ion,Nx)*2]*z[ion]*(cp[c_index(x+1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x+1,y,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]);
                        RcvxRight = RcvxRight/dx*dt/dx;
                    }
                    Rcvy = 0;
                    RcvyUp = 0;
                    //Up down difference
                    if(y>0) {
                        // C term
                        Rcvy = Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*(c[c_index(x,y,comp,ion,Nx)]-c[c_index(x,y-1,comp,ion,Nx)]);
                        //Phi term
                        Rcvy += Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*z[ion]*(cp[c_index(x,y-1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x,y-1,comp,Nx)]);
                        Rcvy = Rcvy/dy*dt/dy;
                    }
                    //Next upward difference
                    if(y<Ny-1) {
                        // C term
                        RcvyUp = Dcs[c_index(x,y,comp,ion,Nx)*2+1]*(c[c_index(x,y+1,comp,ion,Nx)]-c[c_index(x,y,comp,ion,Nx)]);
                        //Phi term
                        RcvyUp += Dcs[c_index(x,y,comp,ion,Nx)*2+1]*z[ion]*(cp[c_index(x,y+1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x,y+1,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]);
                        RcvyUp = RcvyUp/dy*dt/dy;
                    }
                    Resc = al[al_index(x,y,comp,Nx)]*c[c_index(x,y,comp,ion,Nx)]-alp[al_index(x,y,comp,Nx)]*cp[c_index(x,y,comp,ion,Nx)];
                    Resc += Rcvx - RcvxRight + Rcvy - RcvyUp + flux->mflux[c_index(x,y,comp,ion,Nx)]*dt;

                    ierr = VecSetValue(Res,Ind_1(x,y,ion,comp,Nx),Resc,INSERT_VALUES);CHKERRQ(ierr);

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
                    // C term
                    Rcvx = Dcs[c_index(x-1,y,comp,ion,Nx)*2]*(c[c_index(x,y,comp,ion,Nx)]-c[c_index(x-1,y,comp,ion,Nx)]);
                    //Phi term
                    Rcvx += Dcs[c_index(x-1,y,comp,ion,Nx)*2]*z[ion]*(cp[c_index(x-1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x-1,y,comp,Nx)]);
                    Rcvx = Rcvx/dx*dt/dx;
                }
                //Add Second right moving difference
                if(x<Nx-1) {
                    // C term
                    RcvxRight = Dcs[c_index(x,y,comp,ion,Nx)*2]*(c[c_index(x+1,y,comp,ion,Nx)]-c[c_index(x,y,comp,ion,Nx)]);
                    //Phi term
                    RcvxRight += Dcs[c_index(x,y,comp,ion,Nx)*2]*z[ion]*(cp[c_index(x+1,y,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x+1,y,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]);
                    RcvxRight = RcvxRight/dx*dt/dx;
                }
                Rcvy = 0;
                RcvyUp = 0;
                //Up down difference
                if(y>0) {
                    // C term
                    Rcvy = Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*(c[c_index(x,y,comp,ion,Nx)]-c[c_index(x,y-1,comp,ion,Nx)]);
                    //Phi term
                    Rcvy += Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*z[ion]*(cp[c_index(x,y-1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x,y-1,comp,Nx)]);
                    Rcvy = Rcvy/dy*dt/dy;
                }
                //Next upward difference
                if(y<Ny-1) {
                    // C term
                    RcvyUp = Dcs[c_index(x,y,comp,ion,Nx)*2+1]*(c[c_index(x,y+1,comp,ion,Nx)]-c[c_index(x,y,comp,ion,Nx)]);
                    //Phi term
                    RcvyUp += Dcs[c_index(x,y,comp,ion,Nx)*2+1]*z[ion]*(cp[c_index(x,y+1,comp,ion,Nx)]+cp[c_index(x,y,comp,ion,Nx)])/2*(phi[phi_index(x,y+1,comp,Nx)]-phi[phi_index(x,y,comp,Nx)]);
                    RcvyUp = RcvyUp/dy*dt/dy;
                }
                Resc = alNc*c[c_index(x,y,comp,ion,Nx)]-alpNc*cp[c_index(x,y,comp,ion,Nx)];
                Resc += Rcvx - RcvxRight + Rcvy - RcvyUp + flux->mflux[c_index(x,y,comp,ion,Nx)]*dt;
                //Add bath variables

                Resc -= sqrt(pow(Dcb[c_index(x,y,comp,ion,Nx)*2],2)+pow(Dcb[c_index(x,y,comp,ion,Nx)*2+1],2))*(cp[c_index(x,y,comp,ion,Nx)]+cbath[ion])/2.0*(log(c[c_index(x,y,comp,ion,Nx)])-log(cbath[ion])+z[ion]*phi[phi_index(x,y,comp,Nx)]-z[ion]*phibath)*dt;
                ierr = VecSetValue(Res,Ind_1(x,y,ion,comp,Nx),Resc,INSERT_VALUES);CHKERRQ(ierr);

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
                ierr = VecSetValue(Res,Ind_1(x,y,Ni,comp,Nx),Resph,INSERT_VALUES); CHKERRQ(ierr);
            }

            //Finish adding extracell
            comp = Nc-1;
            //Add bath contribution
            for(ion=0;ion<Ni;ion++){

                ResphN -=z[ion]*sqrt(pow(Dcb[c_index(x,y,comp,ion,Nx)*2],2)+pow(Dcb[c_index(x,y,comp,ion,Nx)*2+1],2))*(cp[c_index(x,y,comp,ion,Nx)]+cbath[ion])/2.0*(log(c[c_index(x,y,comp,ion,Nx)])-log(cbath[ion])+z[ion]*phi[phi_index(x,y,comp,Nx)]-z[ion]*phibath)*dt;
            }
            ResphN += Rphx[comp] - RphxRight[comp] + Rphy[comp] - RphyUp[comp];
            ierr = VecSetValue(Res,Ind_1(x,y,Ni,comp,Nx),ResphN,INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = VecAssemblyBegin(Res);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Res);CHKERRQ(ierr);
    ierr = restore_subarray_Read(current_state,user->state_vars); CHKERRQ(ierr);
    if(Profiling_on) {
        PetscLogEventEnd(event[1], 0, 0, 0, 0);
    }
    return ierr;
}

PetscErrorCode
calc_jacobian_linear_deriv(SNES snes,Vec current_state, Mat A, Mat Jac,void *ctx)
{
    //Linear discretization
    //Jacobian equation using derivative of the charge-capacitance relation
    // Alpha is not solved here

    struct AppCtx * user = (struct AppCtx *) ctx;
    PetscErrorCode ierr;
    if(Profiling_on) {
        PetscLogEventBegin(event[0], 0, 0, 0, 0);
    }
    ierr = extract_subarray_Read(current_state,user->state_vars); CHKERRQ(ierr);
    PetscReal *c = user->state_vars->c;
    PetscReal *al = user->state_vars->alpha;
    PetscReal *cp = user->state_vars_past->c;

    PetscReal *Dcs = user->Dcs;
    PetscReal *Dcb = user->Dcb;
    struct FluxData *flux = user->flux;
    PetscReal dt = user->dt;
    PetscReal dx = user->dx;
    PetscReal dy = user->dy;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    struct ConstVars *con_vars = user->con_vars;

    PetscInt ind = 0;
    PetscInt x,y,ion,comp;

    PetscReal Ftmpx,Fc0x,Fc1x,Fph0x,Fph1x;
    PetscReal Fc0y,Fc1y,Fph0y,Fph1y;
    PetscReal Ac,Aphi,Avolt,AvoltN;

    PetscReal Fphph0x[Nc],Fphph1x[Nc];
    PetscReal Fphph0y[Nc],Fphph1y[Nc];

    PetscInt iter;
    ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);

    if(iter==0){
        for(x=0;x<Nx;x++) {
            for(y=0;y<Ny;y++) {
                for(comp=0;comp<Nc;comp++){
                    Fphph0x[comp]=0;
                    Fphph1x[comp]=0;
                    Fphph0y[comp]=0;
                    Fphph1y[comp]=0;
                }
                for(ion=0;ion<Ni;ion++) {
                    for(comp=0;comp<Nc-1;comp++) {
                        //Electrodiffusion contributions
                        Fc0x = 0;
                        Fc1x = 0;
                        Fph0x = 0;
                        Fph1x = 0;
                        Fc0y = 0;
                        Fc1y = 0;
                        Fph0y = 0;
                        Fph1y = 0;
                        if(x<Nx-1) {
                            Fc0x = Dcs[c_index(x, y, comp, ion,Nx) * 2] / dx * dt / dx;
                            Fph0x = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2] *
                                    (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x + 1, y, comp, ion,Nx)])/2/dx*dt/dx;
                            // Right c with left c (-Fc0x)
                            ierr = MatSetValue(Jac,Ind_1(x+1,y,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),-Fc0x,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;
                            //Right c with left phi (-Fph0x)
                            ierr = MatSetValue(Jac,Ind_1(x+1,y,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fph0x,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;

                            //Right phi with left c in voltage eqn
                            ierr = MatSetValue(Jac,Ind_1(x+1,y,Ni,comp,Nx),Ind_1(x,y,ion,comp,Nx),-z[ion]*Fc0x,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;
                        }
                        if(x>0) {
                            Fc1x = Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] / dx * dt / dx;
                            Fph1x = z[ion] * Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] *
                                    (cp[c_index(x - 1, y, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)])/2/dx*dt/dx;
                            //left c with right c (-Fc1x)
                            ierr = MatSetValue(Jac,Ind_1(x-1,y,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),-Fc1x,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;
                            //Left c with right phi (-Fph1x)
                            ierr = MatSetValue(Jac,Ind_1(x-1,y,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fph1x,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;

                            //Left phi with right c in voltage eqn
                            ierr = MatSetValue(Jac,Ind_1(x-1,y,Ni,comp,Nx),Ind_1(x,y,ion,comp,Nx),-z[ion]*Fc1x,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;
                        }
                        if(y<Ny-1) {
                            Fc0y = Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                            Fph0y = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] *
                                    (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x, y + 1, comp, ion,Nx)])/2/dy*dt/dy;
                            // Upper c with lower c (-Fc0y)
                            ierr = MatSetValue(Jac,Ind_1(x,y+1,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),-Fc0y,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;
                            //Upper c with lower phi (-Fph0y)
                            ierr = MatSetValue(Jac,Ind_1(x,y+1,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fph0y,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;

                            //Upper phi with lower c in voltage eqn
                            ierr = MatSetValue(Jac,Ind_1(x,y+1,Ni,comp,Nx),Ind_1(x,y,ion,comp,Nx),-z[ion]*Fc0y,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;
                        }
                        if(y>0) {
                            Fc1y = Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                            Fph1y = z[ion] * Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] *
                                    (cp[c_index(x, y - 1, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)])/2/dy*dt/dy;
                            //Lower c with Upper c (-Fc1y)
                            ierr = MatSetValue(Jac,Ind_1(x,y-1,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),-Fc1y,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;
                            //Lower c with Upper phi (-Fph1y)
                            ierr = MatSetValue(Jac,Ind_1(x,y-1,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fph1y,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;

                            //Lower phi with upper c in voltage eqn
                            ierr = MatSetValue(Jac,Ind_1(x,y-1,Ni,comp,Nx),Ind_1(x,y,ion,comp,Nx),-z[ion]*Fc1y,INSERT_VALUES);CHKERRQ(ierr);
                            ind++;
                        }
                        //Diagonal term contribution
                        Ac = al[al_index(x,y,comp,Nx)]+Fc0x+Fc1x+Fc0y+Fc1y;
                        Aphi = Fph0x + Fph1x + Fph0y + Fph1y;

                        //Add up terms for voltage eqns
                        Fphph0x[comp]+=z[ion]*Fph0x;
                        Fphph1x[comp]+=z[ion]*Fph1x;
                        Fphph0y[comp]+=z[ion]*Fph0y;
                        Fphph1y[comp]+=z[ion]*Fph1y;

                        //membrane current contributions
                        Ac+=flux->dfdci[c_index(x,y,comp,ion,Nx)]*dt;
                        Aphi+=flux->dfdphim[c_index(x,y,comp,ion,Nx)]*dt;
                        // Different Compartment Terms
                        // C Extracellular with C Inside
                        ierr = MatSetValue(Jac,Ind_1(x,y,ion,Nc-1,Nx),Ind_1(x,y,ion,comp,Nx),-flux->dfdci[c_index(x,y,comp,ion,Nx)]*dt,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        // C Intra with C Extra
                        ierr = MatSetValue(Jac,Ind_1(x,y,ion,comp,Nx),Ind_1(x,y,ion,Nc-1,Nx),flux->dfdce[c_index(x,y,comp,ion,Nx)]*dt,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        // C Extracellular with Phi Inside
                        ierr = MatSetValue(Jac,Ind_1(x,y,ion,Nc-1,Nx),Ind_1(x,y,Ni,comp,Nx),-flux->dfdphim[c_index(x,y,comp,ion,Nx)]*dt,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        // C Intra with Phi Extra
                        ierr = MatSetValue(Jac,Ind_1(x,y,ion,comp,Nx),Ind_1(x,y,Ni,Nc-1,Nx),-flux->dfdphim[c_index(x,y,comp,ion,Nx)]*dt,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Same compartment terms
                        // c with c
                        ierr = MatSetValue(Jac,Ind_1(x,y,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),Ac,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        // c with phi
                        ierr = MatSetValue(Jac,Ind_1(x,y,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),Aphi,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;

                        //Intra-Phi with c (voltage eqn)
                        ierr = MatSetValue(Jac,Ind_1(x,y,Ni,comp,Nx),Ind_1(x,y,ion,comp,Nx),z[ion]*(Fc0x+Fc1x+Fc0y+Fc1y+flux->dfdci[c_index(x,y,comp,ion,Nx)]*dt),INSERT_VALUES); CHKERRQ(ierr);
                        ind++;
                        //IntraPhi with c extra(volt eqn)
                        ierr = MatSetValue(Jac,Ind_1(x,y,Ni,comp,Nx),Ind_1(x,y,ion,Nc-1,Nx),z[ion]*(flux->dfdce[c_index(x,y,comp,ion,Nx)]*dt),INSERT_VALUES); CHKERRQ(ierr);
                        ind++;
                        //Extra-Phi with intra-c (voltage eqn)
                        ierr = MatSetValue(Jac,Ind_1(x,y,Ni,Nc-1,Nx),Ind_1(x,y,ion,comp,Nx),-z[ion]*(flux->dfdci[c_index(x,y,comp,ion,Nx)]*dt),INSERT_VALUES); CHKERRQ(ierr);
                        ind++;

                    }
                    //Extracellular terms
                    comp = Nc-1;
                    //Electrodiffusion contributions
                    Fc0x = 0;
                    Fc1x = 0;
                    Fph0x = 0;
                    Fph1x = 0;
                    Fc0y = 0;
                    Fc1y = 0;
                    Fph0y = 0;
                    Fph1y = 0;
                    if(x<Nx-1) {
                        Fc0x = Dcs[c_index(x, y, comp, ion,Nx) * 2] / dx * dt / dx;
                        Fph0x = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2] *
                                (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x + 1, y, comp, ion,Nx)])/2/dx*dt/dx;
                        // Right c with left c (-Fc0x)
                        ierr = MatSetValue(Jac,Ind_1(x+1,y,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),-Fc0x,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Right c with left phi (-Fph0x)
                        ierr = MatSetValue(Jac,Ind_1(x+1,y,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fph0x,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;

                        //Right phi with left c in voltage eqn
                        ierr = MatSetValue(Jac,Ind_1(x+1,y,Ni,comp,Nx),Ind_1(x,y,ion,comp,Nx),-z[ion]*Fc0x,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                    }
                    if(x>0) {
                        Fc1x = Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] / dx * dt / dx;
                        Fph1x = z[ion] * Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] *
                                (cp[c_index(x - 1, y, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)])/2/dx*dt/dx;
                        //left c with right c (-Fc1x)
                        ierr = MatSetValue(Jac,Ind_1(x-1,y,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),-Fc1x,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Left c with right phi (-Fph1x)
                        ierr = MatSetValue(Jac,Ind_1(x-1,y,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fph1x,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;

                        //Left phi with right c in voltage eqn
                        ierr = MatSetValue(Jac,Ind_1(x-1,y,Ni,comp,Nx),Ind_1(x,y,ion,comp,Nx),-z[ion]*Fc1x,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                    }
                    if(y<Ny-1) {
                        Fc0y = Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                        Fph0y = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] *
                                (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x, y + 1, comp, ion,Nx)])/2/dy*dt/dy;
                        // Upper c with lower c (-Fc0y)
                        ierr = MatSetValue(Jac,Ind_1(x,y+1,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),-Fc0y,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Upper c with lower phi (-Fph0y)
                        ierr = MatSetValue(Jac,Ind_1(x,y+1,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fph0y,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;

                        //Upper phi with lower c in voltage eqn
                        ierr = MatSetValue(Jac,Ind_1(x,y+1,Ni,comp,Nx),Ind_1(x,y,ion,comp,Nx),-z[ion]*Fc0y,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                    }
                    if(y>0) {
                        Fc1y = Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                        Fph1y = z[ion] * Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] *
                                (cp[c_index(x, y - 1, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)])/2/dy*dt/dy;
                        //Lower c with Upper c (-Fc1y)
                        ierr = MatSetValue(Jac,Ind_1(x,y-1,ion,comp,Nx),Ind_1(x,y,ion,comp,Nx),-Fc1y,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                        //Lower c with Upper phi (-Fph1y)
                        ierr = MatSetValue(Jac,Ind_1(x,y-1,ion,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fph1y,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;

                        //Lower phi with upper c in voltage eqn
                        ierr = MatSetValue(Jac,Ind_1(x,y-1,Ni,comp,Nx),Ind_1(x,y,ion,comp,Nx),-z[ion]*Fc1y,INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                    }

                    //Diagonal term contribution
                    Ac = (1-al[al_index(x,y,0,Nx)]-al[al_index(x,y,1,Nx)])+Fc0x+Fc1x+Fc0y+Fc1y;
                    Aphi = Fph0x + Fph1x + Fph0y + Fph1y;

                    Avolt = z[ion]*(Fc0x+Fc1x+Fc0y+Fc1y);

                    //Add up terms for voltage eqns
                    Fphph0x[comp]+=z[ion]*Fph0x;
                    Fphph1x[comp]+=z[ion]*Fph1x;
                    Fphph0y[comp]+=z[ion]*Fph0y;
                    Fphph1y[comp]+=z[ion]*Fph1y;

                    //Membrane current contribution
                    for(comp=0;comp<Nc-1;comp++) {
                        Ac -= flux->dfdce[c_index(x,y,comp,ion,Nx)]*dt;
                        Aphi += flux->dfdphim[c_index(x,y,comp,ion,Nx)]*dt;
                        Avolt -=z[ion]*flux->dfdce[c_index(x,y,comp,ion,Nx)]*dt;
                    }
                    //Add bath contributions
                    Ftmpx=sqrt(pow(Dcb[c_index(x,y,Nc-1,ion,Nx)*2],2)+pow(Dcb[c_index(x,y,Nc-1,ion,Nx)*2+1],2));
                    Ac -= Ftmpx*(cp[c_index(x,y,Nc-1,ion,Nx)]+cbath[ion])/(2*c[c_index(x,y,Nc-1,ion,Nx)])*dt;
                    Aphi -= Ftmpx*(cp[c_index(x,y,Nc-1,ion,Nx)]+cbath[ion])*z[ion]/2*dt;

                    Avolt -=z[ion]*Ftmpx*(cp[c_index(x,y,Nc-1,ion,Nx)]+cbath[ion])/(2*c[c_index(x,y,Nc-1,ion,Nx)])*dt;

                    //Insert extracell to extracell parts
                    // c with c
                    ierr = MatSetValue(Jac,Ind_1(x,y,ion,Nc-1,Nx),Ind_1(x,y,ion,Nc-1,Nx),Ac,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    // c with phi
                    ierr = MatSetValue(Jac,Ind_1(x,y,ion,Nc-1,Nx),Ind_1(x,y,Ni,Nc-1,Nx),Aphi,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;

                    //phi with c (voltage eqn)
                    ierr = MatSetValue(Jac,Ind_1(x,y,Ni,Nc-1,Nx),Ind_1(x,y,ion,Nc-1,Nx),Avolt,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }
                //Derivative of charge-capacitance
                for(comp=0;comp<Nc-1;comp++) {
                    if(x<Nx-1) {
                        //Right phi with left phi (-Fph0x)
                        ierr = MatSetValue(Jac,Ind_1(x+1,y,Ni,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fphph0x[comp],INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                    }
                    if(x>0) {
                        //Left phi with right phi (-Fph1x)
                        ierr = MatSetValue(Jac,Ind_1(x-1,y,Ni,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fphph1x[comp],INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                    }
                    if(y<Ny-1) {
                        //Upper phi with lower phi (-Fph0y)
                        ierr = MatSetValue(Jac,Ind_1(x,y+1,Ni,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fphph0y[comp],INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                    }
                    if(y>0) {
                        //Lower phi with upper phi (-Fph1y)
                        ierr = MatSetValue(Jac,Ind_1(x,y-1,Ni,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fphph1y[comp],INSERT_VALUES);CHKERRQ(ierr);
                        ind++;
                    }
                    Avolt = cm[comp]+Fphph0x[comp]+Fphph1x[comp]+Fphph0y[comp]+Fphph1y[comp];
                    AvoltN = -cm[comp];
                    for(ion=0;ion<Ni;ion++) {
                        Avolt+=z[ion]*flux->dfdphim[c_index(x,y,comp,ion,Nx)]*dt;
                        AvoltN-=z[ion]*flux->dfdphim[c_index(x,y,comp,ion,Nx)]*dt;
                    }

                    //Intra-phi with Intra-phi
                    ierr = MatSetValue(Jac,Ind_1(x,y,Ni,comp,Nx),Ind_1(x,y,Ni,comp,Nx),Avolt,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                    //Intra-phi with extra-phi
                    ierr = MatSetValue(Jac,Ind_1(x,y,Ni,comp,Nx),Ind_1(x,y,Ni,Nc-1,Nx),AvoltN,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }
                //Extracellular terms
                comp = Nc-1;
                if(x<Nx-1) {
                    //Right phi with left phi (-Fph0x)
                    ierr = MatSetValue(Jac,Ind_1(x+1,y,Ni,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fphph0x[comp],INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }
                if(x>0) {
                    //Left phi with right phi (-Fph1x)
                    ierr = MatSetValue(Jac,Ind_1(x-1,y,Ni,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fphph1x[comp],INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }
                if(y<Ny-1) {
                    //Upper phi with lower phi (-Fph0y)
                    ierr = MatSetValue(Jac,Ind_1(x,y+1,Ni,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fphph0y[comp],INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }
                if(y>0) {
                    //Lower phi with upper phi (-Fph1y)
                    ierr = MatSetValue(Jac,Ind_1(x,y-1,Ni,comp,Nx),Ind_1(x,y,Ni,comp,Nx),-Fphph1y[comp],INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }
                AvoltN = 0;

                for(int k=0;k<Nc-1;k++) {
                    AvoltN += cm[k];
                    Avolt = -cm[k];
                    for(ion=0;ion<Ni;ion++) {
                        Avolt-=z[ion]*flux->dfdphim[c_index(x,y,k,ion,Nx)]*dt;
                        AvoltN+=z[ion]*flux->dfdphim[c_index(x,y,k,ion,Nx)]*dt;
                    }
                    //Extra-phi with Intra-phi
                    ierr = MatSetValue(Jac,Ind_1(x,y,Ni,comp,Nx),Ind_1(x,y,Ni,k,Nx),Avolt,INSERT_VALUES);CHKERRQ(ierr);
                    ind++;
                }

                AvoltN += Fphph0x[comp]+Fphph1x[comp]+Fphph0y[comp]+Fphph1y[comp];

                //Bath terms
                for(ion=0;ion<Ni;ion++) {
                    Ftmpx = sqrt(pow(Dcb[c_index(x,y,Nc-1,ion,Nx)*2],2)+pow(Dcb[c_index(x,y,Nc-1,ion,Nx)*2+1],2));
                    AvoltN -= z[ion]*Ftmpx*(cp[c_index(x,y,Nc-1,ion,Nx)]+cbath[ion])*z[ion]/2*dt;
                }
                //extra-phi with extra-phi
                ierr = MatSetValue(Jac,Ind_1(x,y,Ni,comp,Nx),Ind_1(x,y,Ni,comp,Nx),AvoltN,INSERT_VALUES);CHKERRQ(ierr);
                ind++;

            }
        }

    }
    else {
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
                        Fc0x = 0;
                        Fc1x = 0;
                        Fph0x = 0;
                        Fph1x = 0;
                        Fc0y = 0;
                        Fc1y = 0;
                        Fph0y = 0;
                        Fph1y = 0;
                        if (x < Nx - 1) {
                            Fc0x = Dcs[c_index(x, y, comp, ion,Nx) * 2] / dx * dt / dx;
                            Fph0x = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2] *
                                    (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x + 1, y, comp, ion,Nx)])/2/dx*dt/dx;
                        }
                        if (x > 0) {
                            Fc1x = Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] / dx * dt / dx;
                            Fph1x = z[ion] * Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] *
                                    (cp[c_index(x - 1, y, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)])/2/dx*dt/dx;
                        }
                        if (y < Ny - 1) {
                            Fc0y = Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                            Fph0y = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] *
                                    (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x, y + 1, comp, ion,Nx)])/2/dy*dt/dy;
                        }
                        if (y > 0) {
                            Fc1y = Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                            Fph1y = z[ion] * Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] *
                                    (cp[c_index(x, y - 1, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)])/2/dy*dt/dy;
                        }
                        //Diagonal term contribution
                        Ac = al[al_index(x, y, comp,Nx)] + Fc0x + Fc1x + Fc0y + Fc1y;
                        Aphi = Fph0x + Fph1x + Fph0y + Fph1y;

                        //Add up terms for voltage eqns
                        Fphph0x[comp] += z[ion] * Fph0x;
                        Fphph1x[comp] += z[ion] * Fph1x;
                        Fphph0y[comp] += z[ion] * Fph0y;
                        Fphph1y[comp] += z[ion] * Fph1y;

                        //membrane current contributions
                        Ac += flux->dfdci[c_index(x, y, comp, ion,Nx)] * dt;
                        Aphi += flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt;
                        // Different Compartment Terms
                        // C Extracellular with C Inside
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, ion, comp,Nx),
                                           -flux->dfdci[c_index(x, y, comp, ion,Nx)] * dt, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // C Intra with C Extra
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, ion, Nc - 1,Nx),
                                           flux->dfdce[c_index(x, y, comp, ion,Nx)] * dt, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // C Extracellular with Phi Inside
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, Ni, comp,Nx),
                                           -flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // C Intra with Phi Extra
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, Ni, Nc - 1,Nx),
                                           -flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Same compartment terms
                        // c with c
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, ion, comp,Nx), Ac, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        // c with phi
                        ierr = MatSetValue(Jac, Ind_1(x, y, ion, comp,Nx), Ind_1(x, y, Ni, comp,Nx), Aphi, INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;

                        //Intra-Phi with c (voltage eqn)
                        ierr = MatSetValue(Jac,Ind_1(x,y,Ni,comp,Nx),Ind_1(x,y,ion,comp,Nx),z[ion]*(Fc0x+Fc1x+Fc0y+Fc1y+flux->dfdci[c_index(x,y,comp,ion,Nx)]*dt),INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //IntraPhi with c extra(volt eqn)
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, ion, Nc - 1,Nx),z[ion]*(flux->dfdce[c_index(x,y,comp,ion,Nx)]*dt),INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;
                        //Extra-Phi with intra-c (voltage eqn)
                        ierr = MatSetValue(Jac, Ind_1(x, y, Ni, Nc - 1,Nx), Ind_1(x, y, ion, comp,Nx),-z[ion]*(flux->dfdci[c_index(x,y,comp,ion,Nx)]*dt),INSERT_VALUES);
                        CHKERRQ(ierr);
                        ind++;

                    }
                    //Extracellular terms
                    comp = Nc - 1;
                    //Electrodiffusion contributions
                    Fc0x = 0;
                    Fc1x = 0;
                    Fph0x = 0;
                    Fph1x = 0;
                    Fc0y = 0;
                    Fc1y = 0;
                    Fph0y = 0;
                    Fph1y = 0;
                    if (x < Nx - 1) {
                        Fc0x = Dcs[c_index(x, y, comp, ion,Nx) * 2] / dx * dt / dx;
                        Fph0x = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2] *
                                (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x + 1, y, comp, ion,Nx)])/2/dx*dt/dx;
                    }
                    if (x > 0) {
                        Fc1x = Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] / dx * dt / dx;
                        Fph1x = z[ion] * Dcs[c_index(x - 1, y, comp, ion,Nx) * 2] *
                                (cp[c_index(x - 1, y, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)])/2/dx*dt/dx;
                    }
                    if (y < Ny - 1) {
                        Fc0y = Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                        Fph0y = z[ion] * Dcs[c_index(x, y, comp, ion,Nx) * 2 + 1] *
                                (cp[c_index(x, y, comp, ion,Nx)] + cp[c_index(x, y + 1, comp, ion,Nx)])/2/dy*dt/dy;
                    }
                    if (y > 0) {
                        Fc1y = Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] / dy * dt / dy;
                        Fph1y = z[ion] * Dcs[c_index(x, y - 1, comp, ion,Nx) * 2 + 1] *
                                (cp[c_index(x, y - 1, comp, ion,Nx)] + cp[c_index(x, y, comp, ion,Nx)])/2/dy*dt/dy;
                    }

                    //Diagonal term contribution
                    Ac = (1 - al[al_index(x, y, 0,Nx)] - al[al_index(x, y, 1,Nx)]) + Fc0x + Fc1x + Fc0y + Fc1y;
                    Aphi = Fph0x + Fph1x + Fph0y + Fph1y;

                    Avolt = z[ion] * (Fc0x + Fc1x + Fc0y + Fc1y);

                    //Add up terms for voltage eqns
                    Fphph0x[comp] += z[ion] * Fph0x;
                    Fphph1x[comp] += z[ion] * Fph1x;
                    Fphph0y[comp] += z[ion] * Fph0y;
                    Fphph1y[comp] += z[ion] * Fph1y;

                    //Membrane current contribution
                    for (comp = 0; comp < Nc - 1; comp++) {
                        Ac -= flux->dfdce[c_index(x, y, comp, ion,Nx)] * dt;
                        Aphi += flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt;
                        Avolt -= z[ion] * flux->dfdce[c_index(x, y, comp, ion,Nx)] * dt;
                    }
                    //Add bath contributions
                    Ftmpx = sqrt(pow(Dcb[c_index(x, y, Nc - 1, ion,Nx) * 2], 2) +
                                 pow(Dcb[c_index(x, y, Nc - 1, ion,Nx) * 2 + 1], 2));
                    Ac -= Ftmpx *(cp[c_index(x, y, Nc - 1, ion,Nx)] + cbath[ion])/(2 * c[c_index(x, y, Nc - 1, ion,Nx)])*dt;
                    Aphi -= Ftmpx *(cp[c_index(x, y, Nc - 1, ion,Nx)] + cbath[ion])*z[ion]/2*dt;

                    Avolt -= z[ion]*Ftmpx*(cp[c_index(x, y, Nc - 1, ion,Nx)]+cbath[ion])/(2*c[c_index(x, y, Nc - 1, ion,Nx)])*dt;

                    //Insert extracell to extracell parts
                    // c with c
                    ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, ion, Nc - 1,Nx), Ac, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    // c with phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, ion, Nc - 1,Nx), Ind_1(x, y, Ni, Nc - 1,Nx), Aphi, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;

                    //phi with c (voltage eqn)
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, Nc - 1,Nx), Ind_1(x, y, ion, Nc - 1,Nx), Avolt, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                //Derivative of charge-capacitance
                for (comp = 0; comp < Nc - 1; comp++) {
                    Avolt = cm[comp] + Fphph0x[comp] + Fphph1x[comp] + Fphph0y[comp] + Fphph1y[comp];
                    AvoltN = -cm[comp];
                    for (ion = 0; ion < Ni; ion++) {
                        Avolt += z[ion] * flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt;
                        AvoltN -= z[ion] * flux->dfdphim[c_index(x, y, comp, ion,Nx)] * dt;
                    }

                    //Intra-phi with Intra-phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), Avolt, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                    //Intra-phi with extra-phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni, Nc - 1,Nx), AvoltN, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }
                //Extracellular terms
                comp = Nc - 1;
                AvoltN = 0;

                for (int k = 0; k < Nc - 1; k++) {
                    AvoltN += cm[k];
                    Avolt = -cm[k];
                    for (ion = 0; ion < Ni; ion++) {
                        Avolt -= z[ion] * flux->dfdphim[c_index(x, y, k, ion,Nx)] * dt;
                        AvoltN += z[ion] * flux->dfdphim[c_index(x, y, k, ion,Nx)] * dt;
                    }
                    //Extra-phi with Intra-phi
                    ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni, k,Nx), Avolt, INSERT_VALUES);
                    CHKERRQ(ierr);
                    ind++;
                }

                AvoltN += Fphph0x[comp] + Fphph1x[comp] + Fphph0y[comp] + Fphph1y[comp];

                //Bath terms
                for (ion = 0; ion < Ni; ion++) {
                    Ftmpx = sqrt(pow(Dcb[c_index(x, y, Nc - 1, ion,Nx) * 2], 2) +
                                 pow(Dcb[c_index(x, y, Nc - 1, ion,Nx) * 2 + 1], 2));
                    AvoltN -= z[ion] * Ftmpx * (cp[c_index(x, y, Nc - 1, ion,Nx)] + cbath[ion]) * z[ion] / 2 * dt;
                }
                //extra-phi with extra-phi
                ierr = MatSetValue(Jac, Ind_1(x, y, Ni, comp,Nx), Ind_1(x, y, Ni, comp,Nx), AvoltN, INSERT_VALUES);
                CHKERRQ(ierr);
                ind++;

            }
        }
    }
//    printf("Iter:%d, inserts:%d\n",iter,ind);
    ierr = MatAssemblyBegin(Jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    if (A != Jac) {
        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); }

    ierr = restore_subarray_Read(current_state,user->state_vars); CHKERRQ(ierr);
    if(Profiling_on) {
        PetscLogEventEnd(event[0], 0, 0, 0, 0);
    }
    return ierr;
}