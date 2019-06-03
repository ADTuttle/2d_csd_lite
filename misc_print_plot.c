#include "functions.h"
#include "constants.h"
#include <string.h>
#include <math.h>


void print_all(struct AppCtx *user)
{
    PetscReal *Dcs = user->Dcs;
    PetscReal *Dcb = user->Dcb;
    struct ConstVars *con_vars = user->con_vars;
    struct FluxData *flux = user->flux;
    struct GateType *gvars = user->gate_vars;
    struct SimState *state_vars = user->state_vars;
    struct Solver *slvr = user->slvr;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    printf("ConstVars:\n");
    printf("%f,%f,%f\n",1e6*con_vars->pNaKCl[0],1e6*con_vars->Imax[0],1e6*con_vars->pNaLeak[0]);
    printf("%f,%f\n",1e6*con_vars->Imaxg[0],1e6*con_vars->pNaLeakg[0]);
    printf("%f,%f,%f\n",con_vars->ao[0],con_vars->ao[1],con_vars->ao[2]);
    printf("%f,%f,%f\n",con_vars->zo[0],con_vars->zo[1],con_vars->zo[2]);
    printf("%f,%f,%f\n",con_vars->kappa,con_vars->zeta1[0],con_vars->zeta1[1]);
    printf("%d,%f,%f\n",con_vars->S,1e6*con_vars->zetaalpha[0],1e6*con_vars->zetaalpha[1]);

    //Diffusion in each compartment
    //Has x and y components
    //x will be saved at even positions (0,2,4,...)
    //y at odd (1,3,5,...)
    //still use c_index(x,y,comp,ion,Nx), but with ind*2 or ind*2+1

    for(PetscInt ion=0;ion<Ni;ion++)
    {
        for(PetscInt comp=0;comp<Nc;comp++)
        {
            printf("Dcs: Ion %d, Comp %d ",ion,comp);
            printf("Dcs x: %.10e, Dcs y: %.10e\n",Dcs[c_index(0,0,comp,ion,Nx)*2],Dcs[c_index(0,0,comp,ion,Nx)*2+1]);
        }
    }
    printf("\n");

    //Bath diffusion

    for(PetscInt ion=0;ion<Ni;ion++)
    {
        for(PetscInt comp=0;comp<Nc;comp++)
        {
            printf("Dcb: Ion %d, Comp %d ",ion,comp);
            printf("Dcb x: %.10e, Dcb y: %.10e\n",Dcb[c_index(0,0,comp,ion,Nx)*2],Dcb[c_index(0,0,comp,ion,Nx)*2+1]);
        }
    }

    PetscInt x=0;PetscInt y=0;
    printf("\n");
    for(PetscInt ion=0;ion<Ni;ion++)
    {
        for(PetscInt comp=0;comp<Nc;comp++)
        {
            printf("Ion: %d, Comp %d, C: %10e\n",ion,comp,state_vars->c[c_index(0,0,comp,ion,Nx)]);
        }
    }
    for(PetscInt comp=0;comp<Nc;comp++)
    {
        printf("Comp %d, Phi: %f\n",comp,state_vars->phi[phi_index(0,0,comp,Nx)]*RTFC);
    }
    for(PetscInt comp=0;comp<Nc-1;comp++)
    {
        printf("Comp %d, alpha: %.10e\n",comp,state_vars->alpha[al_index(0,0,comp,Nx)]);
    }
    printf("Gvars:\n");
    printf("NaT :%.10e,%.10e,%.10e\n",gvars->mNaT[0],gvars->hNaT[0],gvars->gNaT[0]);
    printf("NaP :%.10e,%.10e,%.10e\n",gvars->mNaP[0],gvars->hNaP[0],gvars->gNaP[0]);
    printf("KDR :%.10e,%.10e\n",gvars->mKDR[0],gvars->gKDR[0]);
    printf("KA :%.10e,%.10e,%.10e\n",gvars->mKA[0],gvars->hKA[0],gvars->gKA[0]);
    printf("NMDA :%.10e,%.10e\n",gvars->yNMDA[0],gvars->gNMDA[0]);
    printf("\n");
    //Compute membrane ionic flux relation quantitites

    for(PetscInt ion=0;ion<Ni;ion++)
    {
        for(PetscInt comp=0;comp<Nc;comp++)
        {
            printf("Ion: %d, Comp %d\n",ion,comp);
            printf("Flux: %.10e, dfdci: %.10e, dfdce: %.10e, dfdphim: %.10e\n",flux->mflux[c_index(0,0,comp,ion,Nx)],flux->dfdci[c_index(0,0,comp,ion,Nx)],flux->dfdce[c_index(0,0,comp,ion,Nx)],flux->dfdphim[c_index(0,0,comp,ion,Nx)]);
        }
    }
    printf("\n");
    //Compute membrane water flow related quantities
    for(PetscInt comp=0;comp<Nc-1;comp++)
    {
        printf("Comp: %d\n",comp);
        printf("wFlux: %.10e,%.10e,%.10e\n",flux->wflow[al_index(x,y,comp,Nx)],flux->dwdpi[al_index(x,y,comp,Nx)],flux->dwdal[al_index(x,y,comp,Nx)]);
    }
    printf("\n");
    // VecView(slvr->Res,PETSC_VIEWER_STDOUT_SELF);

    // VecView(slvr->Q,PETSC_VIEWER_STDOUT_SELF);

    return;

}

const char* getfield(char* line, int num)
{
    const char* tok;
    for (tok = strtok(line, ",");
         tok && *tok;
         tok = strtok(NULL, ",\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}

void find_print(int rowC, int colC, double valC, int iter)
{

    if(iter!=-1){
        return;
    }
//    printf("%d,%d,%.10e\n",rowC,colC,valC);
//    return;
    if(rowC>70){
        return;
    }

    int row,col;
    double val;

    //Open Julia Mat file.
    FILE *fp;
    fp = fopen("../../../Julia_work/2d_modules/matrix.txt","r");
    if(fp==NULL)
    {
        fprintf(stderr, "File not found....\n");
        exit(EXIT_FAILURE); /* indicate failure.*/
    }

    //Begin searching for the right row and col.
    char line[1024];
    while (fgets(line, 1024, fp))
    {
        char* tmp = strdup(line);
        row = -1; col=-1; val=0;

        row = atoi(getfield(tmp,1));


        tmp = strdup(line);
        col = atoi(getfield(tmp,2));

        if(row==rowC && col==colC)
        {
            tmp = strdup(line);
            val = atof(getfield(tmp,3));
            if(val!=0) {
                if (fabs(val - valC)/fabs(val) > 5e-16) {
                    printf("Row: %d, Col: %d, J: %f, C: %f, Diff: %fe-16\n", row, col, val, valC,
                           1e16 * (val - valC) / val);
                }
            }
            /*
            if(val==0){
                    printf("Row: %d, Col: %d, J: %f, C: %f, Diff: %fe-16\n",row,col,val,valC,1e16*(val-valC));

            }
            else{
                    printf("Row: %d, Col: %d, J: %f, C: %f, Diff: %fe-16,Abs: %f\n",row,col,val,valC,1e16*(val-valC)/val,1e16*(val-valC));
            }
             */
            return;
        }
        free(tmp);
    }


    fclose(fp);
    return;

}

void compare_res(double *Res, int iter)
{
    if(iter!=-1){
        return;
    }

    FILE *fp;
    fp = fopen("../../../Julia_work/2d_modules/Res.txt","r");
    if(fp==NULL)
    {
        fprintf(stderr, "File not found....\n");
        exit(EXIT_FAILURE); /* indicate failure.*/
    }

    //Begin searching for the right row and col.
    char line[1024];
    int row=0;
    double val;
    while (fgets(line, 1024, fp))
    {
        char* tmp = strdup(line);
        val=0;
        val = atof(getfield(tmp,1));

        if(val==0){
            printf("Row: %d, J: %.10e, C: %.10e, diff: %.10e\n",row,val,Res[row],val-Res[row]);
        }else{
            printf("Row: %d, J: %.10e, C: %.10e,abs: %.10e, diff: %.10e\n",row,val,Res[row],val-Res[row],(val-Res[row])/val);
        }
        row++;
        if(row>70){
            break;
        }
        free(tmp);
    }


    fclose(fp);
    return;
}

void write_data(FILE *fp,struct AppCtx*user,PetscInt numrecords,int start)
{
    if(Profiling_on) {
        PetscLogEventBegin(event[8], 0, 0, 0, 0);
    }
    struct SimState *state_vars = user->state_vars;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    if(!save_one_var) {
        if (start) {
            fprintf(fp, "%d,%d,%d,%d,%d\n", Nx, Ny, numrecords, Nc, Ni);
            write_data(fp, user, numrecords,0);
        } else {
            int ion, comp, x, y;
            for (ion = 0; ion < Ni; ion++) {
                for (comp = 0; comp < Nc; comp++) {
                    for (y = 0; y < Ny; y++) {
                        for (x = 0; x < Nx; x++) {
                            if (x == Nx - 1 & y == Ny - 1) {
                                fprintf(fp, "%.10e\n", state_vars->c[c_index(x, y, comp, ion,Nx)]);
                            } else {
                                fprintf(fp, "%.10e,", state_vars->c[c_index(x, y, comp, ion,Nx)]);
                            }
//                            if (x == Nx - 1 & y == Ny - 1) {
//                                fprintf(fp, "%.10e\n", user->flux->mflux[c_index(x, y, comp, ion,Nx)]);
//                            } else {
//                                fprintf(fp, "%.10e,", user->flux->mflux[c_index(x, y, comp, ion,Nx)]);
//                            }
                        }
                    }
                }
            }
            for (comp = 0; comp < Nc; comp++) {
                for (y = 0; y < Ny; y++) {
                    for (x = 0; x < Nx; x++){
                        if (x == Nx - 1 & y == Ny - 1) {
                            fprintf(fp, "%.10e\n", state_vars->phi[phi_index(x, y, comp,Nx)] * RTFC);
                        } else {
                            fprintf(fp, "%.10e,", state_vars->phi[phi_index(x, y, comp,Nx)] * RTFC);
                        }
                    }
                }
            }
            for (comp = 0; comp < Nc - 1; comp++) {
                for (y = 0; y < Ny; y++) {
                    for (x = 0; x < Nx; x++) {
                        if (x == Nx - 1 & y == Ny - 1) {
                            fprintf(fp, "%.10e\n", state_vars->alpha[al_index(x, y, comp,Nx)]);
                        } else {
                            fprintf(fp, "%.10e,", state_vars->alpha[al_index(x, y, comp,Nx)]);
                        }
                    }
                }
            }
        }
    } else{
        if (start) {
            fprintf(fp, "%d,%d,%d,%d,%d\n", Nx, Ny, (int) floor(numrecords), 0, 0);
            write_data(fp, user,numrecords, 0);
        } else {
            int ion, comp, x, y;
            comp = 0;
            for (y = 0; y < Ny; y++) {
                for (x = 0; x < Nx; x++) {
                    if (x == Nx - 1 & y == Ny - 1) {
                        fprintf(fp, "%.10e\n", (state_vars->phi[phi_index(x, y, comp,Nx)]-state_vars->phi[phi_index(x, y, Nc-1,Nx)]) * RTFC);
//                            fprintf(fp, "%.10e\n", user->gate_vars->gNMDA[xy_index(x,y,Nx)]);
                    } else {
                        fprintf(fp, "%.10e,", (state_vars->phi[phi_index(x, y, comp,Nx)]-state_vars->phi[phi_index(x, y, Nc-1,Nx)]) * RTFC);
//                        fprintf(fp, "%.10e,", user->gate_vars->gNMDA[xy_index(x,y,Nx)]);
                    }
                }
            }
        }
    }
    if(Profiling_on) {
        PetscLogEventEnd(event[8], 0, 0, 0, 0);
    }
}
void write_point(FILE *fp,struct AppCtx* user,PetscReal t,PetscInt x,PetscInt y)
{
    if(Profiling_on) {
        PetscLogEventBegin(event[8], 0, 0, 0, 0);
    }
    struct SimState *state_vars = user->state_vars;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    int ion, comp;
    comp=0;

    fprintf(fp,"%f,%.10e\n",t, (state_vars->phi[phi_index(x, y, comp,Nx)]-state_vars->phi[phi_index(x,y,Nc-1,Nx)]) * RTFC);


}
void save_timestep(FILE *fp,struct AppCtx*user,PetscInt numrecords,int start)
{
    if(Profiling_on) {
        PetscLogEventBegin(event[8], 0, 0, 0, 0);
    }
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    if (start) {
        fprintf(fp, "%d,%d,%d,%d,%d\n", Nx, Ny, numrecords, 0, 0);
    } else {
        int x, y;

        for (y = 0; y < Ny; y++) {
            for (x = 0; x < Nx; x++) {
                if (x == Nx - 1 & y == Ny - 1) {
                    fprintf(fp, "%f\n", user->dt_space[xy_index(x,y,Nx)]);
                } else {
                    fprintf(fp, "%f,", user->dt_space[xy_index(x,y,Nx)]);
                }
            }
        }
    }



    if(Profiling_on) {
        PetscLogEventEnd(event[8], 0, 0, 0, 0);
    }
}

void record_measurements(FILE **fp_measures,struct AppCtx *user,PetscInt count,PetscInt numrecords,int start){
    if(start){
        if(start_at_steady) {
            fp_measures[0] = fopen("flux_csd.txt", "w");
            fp_measures[1] = fopen("grad_field.txt", "w");
            fp_measures[2] = fopen("measures.txt", "w");

            measure_flux(fp_measures[0],user,numrecords,start);
            velocity_field(fp_measures[1],user,numrecords,start);
//            calculate_measures(fp_measures[2],user,numrecords,start);
            calculate_energy(fp_measures[2],user,numrecords,start);
        }else{
            fp_measures[0] = fopen("flux_csd.txt", "a");
            fp_measures[1] = fopen("grad_field.txt", "a");
            fp_measures[2] = fopen("measures.txt", "a");
        }
    } else{
        measure_flux(fp_measures[0],user,numrecords,start);
        velocity_field(fp_measures[1],user,numrecords,start);
        //            calculate_measures(fp_measures[2],user,numrecords,start);
        calculate_energy(fp_measures[2],user,numrecords,start);

        if(count%100==0) {
            draw_csd(user);
        }
    }
}
void measure_flux(FILE *fp, struct AppCtx* user,PetscInt numrecords,int start)
{
    struct SimState *state_vars= user->state_vars;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    if (start) {
        fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d\n", Nx, Ny, numrecords, Nc, Ni, use_en_deriv, separate_vol,
                Linear_Diffusion);
        measure_flux(fp, user, numrecords, 0);
    } else{
        //compute diffusion coefficients
        diff_coef(user->Dcs,state_vars->alpha,1,user);
        //Bath diffusion
        diff_coef(user->Dcb,state_vars->alpha,Batheps,user);

        PetscReal *c = state_vars->c;
        PetscReal *phi = state_vars->phi;
        PetscReal *al = state_vars->alpha;
        PetscReal *Dcs = user->Dcs;
        PetscReal *Dcb = user->Dcb;
        PetscReal dx = user->dx;
        PetscReal dy = user->dy;

        PetscReal Ftmp;

        PetscReal Rcvx,RcvxRight,alNc;
        PetscReal Rcvy,RcvyUp;

        PetscReal Rphx,RphxRight;
        PetscReal Rphy,RphyUp;

        PetscReal RBath;

        PetscInt x,y,comp,ion;

        PetscReal Circ_Radius = 0.05;
        PetscReal radius;

        PetscReal Fluxc[Nc] = {0,0,0};
        PetscReal Fluxph[Nc] = {0,0,0};
        PetscReal Fluxbath[Nc] = {0,0,0};

        int num_points=0;

        for(x=0;x<Nx;x++){
            for(y=0;y<Ny;y++){
                radius = sqrt(pow((x + 0.5) * dx - Lx / 2, 2) + pow((y + 0.5) * dy - Lx / 2, 2));
                if(radius<=Circ_Radius) {
//                if(fabs((x+0.5)*dx-Lx/2)<=0.1 && fabs((y+0.5)*dy-Ly/2)<=0.1 ){
                    num_points++;
                    for(comp=0;comp<Nc;comp++) {
                        //Reset values
                        Rphx = 0;
                        RphxRight = 0;
                        Rphy = 0;
                        RphyUp = 0;
                        Rcvx = 0;
                        RcvxRight = 0;
                        Rcvy = 0;
                        RcvyUp = 0;
                        RBath = 0;
                        //Sum over all ions
                        for (ion = 0; ion < Ni; ion++) {
                            if (x > 0) {
                                //First difference term
                                Ftmp = z[ion] * Dcs[c_index(x - 1, y, comp, ion, Nx) * 2] / (dx * dx);
                                Rcvx += Ftmp * (c[c_index(x, y, comp, ion, Nx)] - c[c_index(x - 1, y, comp, ion, Nx)]);
                                Rphx += Ftmp * z[ion] * c[c_index(x - 1, y, comp, ion, Nx)] *
                                        (phi[phi_index(x, y, comp, Nx)] - phi[phi_index(x - 1, y, comp, Nx)]);
                            }
                            //Add Second right moving difference
                            if (x < Nx - 1) {
                                //Second difference term
                                Ftmp = z[ion] * Dcs[c_index(x, y, comp, ion, Nx) * 2] / (dx * dx);
                                RcvxRight +=
                                        Ftmp * (c[c_index(x + 1, y, comp, ion, Nx)] - c[c_index(x, y, comp, ion, Nx)]);
                                RphxRight += Ftmp * z[ion] * c[c_index(x, y, comp, ion, Nx)] *
                                             (phi[phi_index(x + 1, y, comp, Nx)] - phi[phi_index(x, y, comp, Nx)]);
                            }
                            if (y > 0) {
                                //Updown difference term
                                Ftmp = z[ion] * Dcs[c_index(x, y - 1, comp, ion, Nx) * 2 + 1] / (dy * dy);
                                Rcvy += Ftmp * (c[c_index(x, y, comp, ion, Nx)] - c[c_index(x, y - 1, comp, ion, Nx)]);
                                Rphy += Ftmp * z[ion] * c[c_index(x, y - 1, comp, ion, Nx)] *
                                        (phi[phi_index(x, y, comp, Nx)] - phi[phi_index(x, y - 1, comp, Nx)]);
                            }
                            //Next upward difference
                            if (y < Ny - 1) {
                                Ftmp = z[ion] * Dcs[c_index(x, y, comp, ion, Nx) * 2 + 1] / (dy * dy);
                                RcvyUp +=
                                        Ftmp * (c[c_index(x, y + 1, comp, ion, Nx)] - c[c_index(x, y, comp, ion, Nx)]);
                                RcvyUp += Ftmp * z[ion] * c[c_index(x, y, comp, ion, Nx)] *
                                          (phi[phi_index(x, y + 1, comp, Nx)] - phi[phi_index(x, y, comp, Nx)]);

                            }
                            if (comp == Nc - 1) {
                                Ftmp = z[ion] * sqrt(pow(Dcb[c_index(x, y, comp, ion, Nx) * 2], 2) +
                                                     pow(Dcb[c_index(x, y, comp, ion, Nx) * 2 + 1], 2));
                                RBath -= Ftmp * (c[c_index(x, y, comp, ion, Nx)] - cbath[ion]);
                                RBath -= Ftmp * (c[c_index(x, y, comp, ion, Nx)] + cbath[ion]) / 2.0 *
                                         (z[ion] * phi[phi_index(x, y, comp, Nx)] - z[ion] * phibath);
                            }
                        }

                        Fluxph[comp] += (Rphx - RphxRight + Rphy - RphyUp) * dx * dy;
                        Fluxc[comp] += (Rcvx - RcvxRight + Rcvy - RcvyUp) * dx * dy;
                        Fluxbath[comp] += RBath * dx * dy;
                    }
                }
            }
        }
        fprintf(fp,"%.10e,%.10e,%.10e\n",Fluxc[Nc-1], Fluxph[Nc-1], Fluxbath[Nc-1]);

    }
}
void init_events(struct AppCtx *user)
{
    PetscLogDefaultBegin();


    PetscClassId id;
    PetscClassIdRegister("CSD",&id);

    PetscLogEventRegister("Jacobian",id,&event[0]);
    PetscLogEventRegister("Residual",id,&event[1]);
    PetscLogEventRegister("Extract Subarray",id,&event[2]);
    PetscLogEventRegister("Restore Subarray",id,&event[3]);
    PetscLogEventRegister("Gating Variables",id,&event[4]);
    PetscLogEventRegister("Ion Flux",id,&event[5]);
    PetscLogEventRegister("Water Flux",id,&event[6]);
    PetscLogEventRegister("Volume Update",id,&event[7]);
    PetscLogEventRegister("Write to File",id,&event[8]);
    PetscLogEventRegister("Predict Jacobian",id,&event[9]);
    PetscLogEventRegister("Predict Residual",id,&event[10]);
    PetscLogEventRegister("Predict Solve",id,&event[11]);
    PetscLogEventRegister("Load Grid",id,&event[12]);
    PetscLogEventRegister("Unload Grid",id,&event[13]);

    //Deactivate Petsc tracking
    PetscLogEvent deactivate;

    PetscLogEventDeactivateClass(MAT_CLASSID);
    PetscLogEventDeactivateClass(KSP_CLASSID); /* includes PC and KSP */
    PetscLogEventDeactivateClass(VEC_CLASSID);
//    PetscLogEventDeactivateClass(SNES_CLASSID);

    //Some events are leftover somehow

    PetscLogEventGetId("PCApply",&deactivate);
    PetscLogEventDeactivate(deactivate);
    PetscLogEventGetId("VecSet",&deactivate);
    PetscLogEventDeactivate(deactivate);
    PetscLogEventGetId("MatAssemblyBegin",&deactivate);
    PetscLogEventDeactivate(deactivate);
    PetscLogEventGetId("MatAssemblyEnd",&deactivate);
    PetscLogEventDeactivate(deactivate);
    PetscLogEventGetId("SNESLineSearch",&deactivate);
    PetscLogEventDeactivate(deactivate);
    PetscLogEventGetId("PCSetUp",&deactivate);
    PetscLogEventDeactivate(deactivate);

    PetscLogEventGetId("SNESFunctionEval",&deactivate);
    PetscLogEventDeactivate(deactivate);
    PetscLogEventGetId("SNESJacobianEval",&deactivate);
    PetscLogEventDeactivate(deactivate);

}

void save_file(struct AppCtx *user){

    FILE *fp;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    PetscInt x,y,ion,comp;

    struct SimState *state_vars = user->state_vars;
    struct GateType *gate_vars = user->gate_vars;

    fp = fopen("save_csd.txt","w");


    //Header
    fprintf(fp, "%d,%d,%d,%d,%d\n", Nx, Ny, 1, Nc, Ni);

    //Macro-variables
    for (ion = 0; ion < Ni; ion++) {
        for (comp = 0; comp < Nc; comp++) {
            for (y = 0; y < Ny; y++) {
                for (x = 0; x < Nx; x++) {
                    if (x == Nx - 1 & y == Ny - 1) {
                        fprintf(fp, "%.20e\n", state_vars->c[c_index(x, y, comp, ion,Nx)]);
                    } else {
                        fprintf(fp, "%.20e,", state_vars->c[c_index(x, y, comp, ion,Nx)]);
                    }
                }
            }
        }
    }
    for (comp = 0; comp < Nc; comp++) {
        for (y = 0; y < Ny; y++) {
            for (x = 0; x < Nx; x++) {
                if (x == Nx - 1 & y == Ny - 1) {
                    fprintf(fp, "%.20e\n", state_vars->phi[phi_index(x, y, comp,Nx)]);
                } else {
                    fprintf(fp, "%.20e,", state_vars->phi[phi_index(x, y, comp,Nx)]);
                }
            }
        }
    }
    for (comp = 0; comp < Nc - 1; comp++) {
        for (y = 0; y < Ny; y++) {
            for (x = 0; x < Nx; x++) {
                if (x == Nx - 1 & y == Ny - 1) {
                    fprintf(fp, "%.20e\n", state_vars->alpha[al_index(x, y, comp,Nx)]);
                } else {
                    fprintf(fp, "%.20e,", state_vars->alpha[al_index(x, y, comp,Nx)]);
                }
            }
        }
    }

    //Gating Variables
    for(y=0;y<Ny;y++){
        for(x=0;x<Nx;x++){
            if (x == Nx - 1 && y == Ny - 1) {
                fprintf(fp, "%.20e\n", gate_vars->mNaT[xy_index(x,y,Nx)]);
            } else {
                fprintf(fp, "%.20e,", gate_vars->mNaT[xy_index(x,y,Nx)]);
            }
        }
    }
    for(y=0;y<Ny;y++){
        for(x=0;x<Nx;x++){
            if (x == Nx - 1 && y == Ny - 1) {
                fprintf(fp, "%.20e\n", gate_vars->hNaT[xy_index(x,y,Nx)]);
            } else {
                fprintf(fp, "%.20e,", gate_vars->hNaT[xy_index(x,y,Nx)]);
            }
        }
    }
    for(y=0;y<Ny;y++){
        for(x=0;x<Nx;x++){
            if (x == Nx - 1 && y == Ny - 1) {
                fprintf(fp, "%.20e\n", gate_vars->mNaP[xy_index(x,y,Nx)]);
            } else {
                fprintf(fp, "%.20e,", gate_vars->mNaP[xy_index(x,y,Nx)]);
            }
        }
    }
    for(y=0;y<Ny;y++){
        for(x=0;x<Nx;x++){
            if (x == Nx - 1 && y == Ny - 1) {
                fprintf(fp, "%.20e\n", gate_vars->hNaP[xy_index(x,y,Nx)]);
            } else {
                fprintf(fp, "%.20e,", gate_vars->hNaP[xy_index(x,y,Nx)]);
            }
        }
    }
    for(y=0;y<Ny;y++){
        for(x=0;x<Nx;x++){
            if (x == Nx - 1 && y == Ny - 1) {
                fprintf(fp, "%.20e\n", gate_vars->mKDR[xy_index(x,y,Nx)]);
            } else {
                fprintf(fp, "%.20e,", gate_vars->mKDR[xy_index(x,y,Nx)]);
            }
        }
    }
    for(y=0;y<Ny;y++){
        for(x=0;x<Nx;x++){
            if (x == Nx - 1 && y == Ny - 1) {
                fprintf(fp, "%.20e\n", gate_vars->mKA[xy_index(x,y,Nx)]);
            } else {
                fprintf(fp, "%.20e,", gate_vars->mKA[xy_index(x,y,Nx)]);
            }
        }
    }
    for(y=0;y<Ny;y++){
        for(x=0;x<Nx;x++){
            if (x == Nx - 1 && y == Ny - 1) {
                fprintf(fp, "%.20e\n", gate_vars->hKA[xy_index(x,y,Nx)]);
            } else {
                fprintf(fp, "%.20e,", gate_vars->hKA[xy_index(x,y,Nx)]);
            }
        }
    }
    for(y=0;y<Ny;y++){
        for(x=0;x<Nx;x++){
            if (x == Nx - 1 && y == Ny - 1) {
                fprintf(fp, "%.20e\n", gate_vars->yNMDA[xy_index(x,y,Nx)]);
            } else {
                fprintf(fp, "%.20e,", gate_vars->yNMDA[xy_index(x,y,Nx)]);
            }
        }
    }
    for(y=0;y<Ny;y++){
        for(x=0;x<Nx;x++){
            if (x == Nx - 1 && y == Ny - 1) {
                fprintf(fp, "%.20e\n", gate_vars->zNMDA[xy_index(x,y,Nx)]);
            } else {
                fprintf(fp, "%.20e,", gate_vars->zNMDA[xy_index(x,y,Nx)]);
            }
        }
    }
    for(y=0;y<Ny;y++){
        for(x=0;x<Nx;x++){
            if (x == Nx - 1 && y == Ny - 1) {
                fprintf(fp, "%.20e\n", gate_vars->dNMDA[xy_index(x,y,Nx)]);
            } else {
                fprintf(fp, "%.20e,", gate_vars->dNMDA[xy_index(x,y,Nx)]);
            }
        }
    }


    fclose(fp);

}
void read_file(struct AppCtx *user)
{
    //Read data_csd.txt and copy it into the current state.
    int Nx,Ny,numrecords;
    struct SimState *state_vars = user->state_vars;
    struct GateType *gate_vars = user->gate_vars;
    PetscInt x,y,comp,ion;


    char *line = (char *) malloc(sizeof(char) * 1024 * 1024);//[1024];
    char *tmp;

    FILE *fp;
    fp = fopen("save_csd.txt","r");


    if(fp==NULL) {
        fp = fopen("data_csd.txt", "r");
        if (fp == NULL) {
            fprintf(stderr, "File not found....\n");
            exit(EXIT_FAILURE); /* indicate failure.*/
        }

        //Read top file details
        fgets(line, 1024 * 1024, fp);

        tmp = strdup(line);

        Nx = atoi(getfield(tmp, 1));

        tmp = strdup(line);
        Ny = atoi(getfield(tmp, 2));

        tmp = strdup(line);
        numrecords = atoi(getfield(tmp, 3));


        printf("%d,%d,%d\n", Nx, Ny, numrecords);

        //Zero out for safety
        for (x = 0; x < Nx; x++) {
            for (y = 0; y < Ny; y++) {
                for (comp = 0; comp < Nc; comp++) {
                    for (ion = 0; ion < Ni; ion++) {
                        state_vars->c[c_index(x, y, comp, ion, Nx)] = 0;
                    }
                    state_vars->phi[phi_index(x, y, comp, Nx)] = 0;
                }
                for (comp = 0; comp < Nc - 1; comp++) {
                    state_vars->alpha[al_index(x, y, comp, Nx)] = 0;
                }
            }
        }


        //Get to the last recorded value
        for (int count = 0; count < ((Ni + 2) * Nc - 1) * (numrecords - 1); count++) {
            fgets(line, 1024 * 1024, fp);
        }

        for (ion = 0; ion < Ni; ion++) {
            for (comp = 0; comp < Nc; comp++) {
                fgets(line, 1024 * 1024, fp);
                for (y = 0; y < Ny; y++) {
                    for (x = 0; x < Nx; x++) {
                        tmp = strdup(line);

                        state_vars->c[c_index(x, y, comp, ion, Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));

                    }
                }
            }
        }
        for (comp = 0; comp < Nc; comp++) {
            fgets(line, 1024 * 1024, fp);
            for (y = 0; y < Ny; y++) {
                for (x = 0; x < Nx; x++) {
                    tmp = strdup(line);

                    state_vars->phi[phi_index(x, y, comp, Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));

                }
            }
        }
        for (comp = 0; comp < Nc - 1; comp++) {
            fgets(line, 1024 * 1024, fp);
            for (y = 0; y < Ny; y++) {
                for (x = 0; x < Nx; x++) {
                    tmp = strdup(line);

                    state_vars->alpha[al_index(x, y, comp, Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));

                }
            }
        }
        gatevars_update(user->gate_vars,user->gate_vars,state_vars,0,user,1);
        gatevars_update(user->gate_vars_past,user->gate_vars_past,state_vars,0,user,1);
    } else{
        //Read top file details
        fgets(line, 1024 * 1024, fp);

        tmp = strdup(line);

        Nx = atoi(getfield(tmp, 1));

        tmp = strdup(line);
        Ny = atoi(getfield(tmp, 2));

        tmp = strdup(line);
        numrecords = atoi(getfield(tmp, 3));


        printf("%d,%d,%d\n", Nx, Ny, numrecords);

        //Zero out for safety
        for (x = 0; x < Nx; x++) {
            for (y = 0; y < Ny; y++) {
                for (comp = 0; comp < Nc; comp++) {
                    for (ion = 0; ion < Ni; ion++) {
                        state_vars->c[c_index(x, y, comp, ion, Nx)] = 0;
                    }
                    state_vars->phi[phi_index(x, y, comp, Nx)] = 0;
                }
                for (comp = 0; comp < Nc - 1; comp++) {
                    state_vars->alpha[al_index(x, y, comp, Nx)] = 0;
                }
            }
        }


        for (ion = 0; ion < Ni; ion++) {
            for (comp = 0; comp < Nc; comp++) {
                fgets(line, 1024 * 1024, fp);
                for (y = 0; y < Ny; y++) {
                    for (x = 0; x < Nx; x++) {
                        tmp = strdup(line);
                        state_vars->c[c_index(x, y, comp, ion, Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
                    }
                }
            }
        }
        for (comp = 0; comp < Nc; comp++) {
            fgets(line, 1024 * 1024, fp);
            for (y = 0; y < Ny; y++) {
                for (x = 0; x < Nx; x++) {
                    tmp = strdup(line);
                    state_vars->phi[phi_index(x, y, comp, Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
                }
            }
        }
        for (comp = 0; comp < Nc - 1; comp++) {
            fgets(line, 1024 * 1024, fp);
            for (y = 0; y < Ny; y++) {
                for (x = 0; x < Nx; x++) {
                    tmp = strdup(line);
                    state_vars->alpha[al_index(x, y, comp, Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
                }
            }
        }
        //Gating variables
        fgets(line, 1024 * 1024, fp);
        for (y=0;y<Ny;y++){
            for(x=0;x<Nx;x++){
                tmp = strdup(line);
                gate_vars->mNaT[xy_index(x,y,Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
            }
        }
        fgets(line, 1024 * 1024, fp);
        for (y=0;y<Ny;y++){
            for(x=0;x<Nx;x++){
                tmp = strdup(line);
                gate_vars->hNaT[xy_index(x,y,Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
            }
        }
        fgets(line, 1024 * 1024, fp);
        for (y=0;y<Ny;y++){
            for(x=0;x<Nx;x++){
                tmp = strdup(line);
                gate_vars->mNaP[xy_index(x,y,Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
            }
        }
        fgets(line, 1024 * 1024, fp);
        for (y=0;y<Ny;y++){
            for(x=0;x<Nx;x++){
                tmp = strdup(line);
                gate_vars->hNaP[xy_index(x,y,Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
            }
        }
        fgets(line, 1024 * 1024, fp);
        for (y=0;y<Ny;y++){
            for(x=0;x<Nx;x++){
                tmp = strdup(line);
                gate_vars->mKDR[xy_index(x,y,Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
            }
        }
        fgets(line, 1024 * 1024, fp);
        for (y=0;y<Ny;y++){
            for(x=0;x<Nx;x++){
                tmp = strdup(line);
                gate_vars->mKA[xy_index(x,y,Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
            }
        }
        fgets(line, 1024 * 1024, fp);
        for (y=0;y<Ny;y++){
            for(x=0;x<Nx;x++){
                tmp = strdup(line);
                gate_vars->hKA[xy_index(x,y,Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
            }
        }
        fgets(line, 1024 * 1024, fp);
        for (y=0;y<Ny;y++){
            for(x=0;x<Nx;x++){
                tmp = strdup(line);
                gate_vars->yNMDA[xy_index(x,y,Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
            }
        }
        fgets(line, 1024 * 1024, fp);
        for (y=0;y<Ny;y++){
            for(x=0;x<Nx;x++){
                tmp = strdup(line);
                gate_vars->zNMDA[xy_index(x,y,Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
            }
        }
        fgets(line, 1024 * 1024, fp);
        for (y=0;y<Ny;y++){
            for(x=0;x<Nx;x++){
                tmp = strdup(line);
                gate_vars->dNMDA[xy_index(x,y,Nx)] = atof(getfield(tmp, xy_index(x, y, Nx) + 1));
            }
        }

        //Copy over past vars and calculate g.
        PetscReal Gphi,v;
        PetscReal K_r = 2.3e-6;//34.9e-6;
        PetscReal npow = 1.5; //1.4;
        PetscReal Fglu;
        for(y=0;y<Ny;y++){
            for(x=0;x<Nx;x++){
                gate_vars->gNaT[xy_index(x,y,Nx)] = pow(gate_vars->mNaT[xy_index(x,y,Nx)],3)*gate_vars->hNaT[xy_index(x,y,Nx)];
                gate_vars->gNaP[xy_index(x,y,Nx)] = pow(gate_vars->mNaP[xy_index(x,y,Nx)],2)*gate_vars->hNaP[xy_index(x,y,Nx)];
                gate_vars->gKDR[xy_index(x,y,Nx)] = pow(gate_vars->mKDR[xy_index(x,y,Nx)],2);
                gate_vars->gKA[xy_index(x,y,Nx)] = pow(gate_vars->mKA[xy_index(x,y,Nx)],2)*gate_vars->hKA[xy_index(x,y,Nx)];

                v = (state_vars->phi[phi_index(x,y,0,Nx)]-state_vars->phi[phi_index(x,y,Nc-1,Nx)])*RTFC;

                Fglu = pow(state_vars->c[c_index(x,y,Nc-1,3,Nx)],npow)
                       /(pow(state_vars->c[c_index(x,y,Nc-1,3,Nx)],npow)+pow(K_r,npow));
                Gphi = 1/(1+0.56*exp(-0.062*v));
                gate_vars->gNMDA[xy_index(x,y,Nx)] = (gate_vars->yNMDA[xy_index(x,y,Nx)]*Fglu
                                                      +Desensitize[0]*gate_vars->zNMDA[xy_index(x,y,Nx)]
                                                      +Desensitize[1]*gate_vars->dNMDA[xy_index(x,y,Nx)])*Gphi;

            }
        }
        //Copy old gating variables
        //Save the gating variables
        memcpy(user->gate_vars_past->mNaT,user->gate_vars->mNaT,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->hNaT,user->gate_vars->hNaT,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->gNaT,user->gate_vars->gNaT,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->mNaP,user->gate_vars->mNaP,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->hNaP,user->gate_vars->hNaP,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->gNaP,user->gate_vars->gNaP,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->gKA,user->gate_vars->gKA,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->hKA,user->gate_vars->hKA,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->mKA,user->gate_vars->mKA,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->mKDR,user->gate_vars->mKDR,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->gKDR,user->gate_vars->gKDR,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->yNMDA,user->gate_vars->yNMDA,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->zNMDA,user->gate_vars->zNMDA,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->dNMDA,user->gate_vars->dNMDA,sizeof(PetscReal)*user->Nx*user->Ny);
        memcpy(user->gate_vars_past->gNMDA,user->gate_vars->gNMDA,sizeof(PetscReal)*user->Nx*user->Ny);
    }



    free(tmp);
    free(line);
    fclose(fp);
    return;
    //Modify beginning of file
    fp = fopen("data_csd.txt", "r");
    if (fp == NULL) {
        fprintf(stderr, "File not found....\n");
        exit(EXIT_FAILURE); /* indicate failure.*/
    }
    //Read top file details
    fgets(line, 1024 * 1024, fp);

    tmp = strdup(line);

    Nx = atoi(getfield(tmp, 1));

    tmp = strdup(line);
    Ny = atoi(getfield(tmp, 2));

    tmp = strdup(line);
    numrecords = atoi(getfield(tmp, 3));
    fclose(fp);

    fp = fopen("data_csd.txt","r+");
    fseek( fp, 0, SEEK_SET );
    fprintf(fp, "%d,%d,%d,%d,%d\n", Nx, Ny, numrecords+(PetscInt)floor(Time/trecordstep)-1, Nc, Ni);

    fclose(fp);
}

void velocity_field(FILE *fp,struct AppCtx *user,PetscInt numrecords,int start) {


    if (start) {
            fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d\n", user->Nx, user->Ny, numrecords, Nc, Ni, use_en_deriv, separate_vol,
                    Linear_Diffusion);

        velocity_field(fp, user, numrecords, 0);
    } else {
        PetscReal *c = user->state_vars->c;
        PetscReal *phi = user->state_vars->phi;
        PetscReal *al = user->state_vars->alpha;

        diff_coef(user->Dcs, al, 1, user);
        PetscReal *Dcs = user->Dcs;


        PetscReal dt = user->dt;
        PetscReal dx = user->dx;
        PetscReal dy = user->dy;
        PetscInt Nx = user->Nx;
        PetscInt Ny = user->Ny;

        PetscInt x, y, ion, comp;

         PetscReal Gradleft, GradRight;
            PetscReal GradUp, GradDown;
            PetscInt count_x, count_y;
            PetscReal alNc, alNcRight, alNcUp;
            PetscReal vx, vy;

            PetscReal Gradx, Grady;


            for (ion = 0; ion < Ni; ion++) {
                for (comp = Nc - 1; comp < Nc; comp++) {
                    for (y = 0; y < Ny; y++) {
                        for (x = 0; x < Nx; x++) {
                            count_x = 0;
                            count_y = 0;

                            Gradleft = 0;
                            GradRight = 0;
                            if (x > 0) {
                                //First difference term
                                Gradleft = 1;//Dcs[c_index(x-1,y,comp,ion,Nx)*2]*(c[c_index(x-1,y,comp,ion,Nx)]+c[c_index(x,y,comp,ion,Nx)])/2;
                                Gradleft = Gradleft * (log(c[c_index(x, y, comp, ion, Nx)]) -
                                                       log(c[c_index(x - 1, y, comp, ion, Nx)]) +
                                                       z[ion] * (phi[phi_index(x, y, comp, Nx)] -
                                                                 phi[phi_index(x - 1, y, comp, Nx)])) / dx;
                                count_x++;
                            }
                            //Add Second right moving difference
                            if (x < Nx - 1) {
                                GradRight = 1;//Dcs[c_index(x,y,comp,ion,Nx)*2]*(c[c_index(x,y,comp,ion,Nx)]+c[c_index(x+1,y,comp,ion,Nx)])/2;
                                GradRight = GradRight * (log(c[c_index(x + 1, y, comp, ion, Nx)]) -
                                                         log(c[c_index(x, y, comp, ion, Nx)]) +
                                                         z[ion] * (phi[phi_index(x + 1, y, comp, Nx)] -
                                                                   phi[phi_index(x, y, comp, Nx)])) / dx;
                                count_x++;
                            }
                            GradDown = 0;
                            GradUp = 0;
                            //Up down difference
                            if (y > 0) {
                                GradDown = 1;//Dcs[c_index(x,y-1,comp,ion,Nx)*2+1]*(c[c_index(x,y-1,comp,ion,Nx)]+c[c_index(x,y,comp,ion,Nx)])/2;
                                GradDown = GradDown * (log(c[c_index(x, y, comp, ion, Nx)]) -
                                                       log(c[c_index(x, y - 1, comp, ion, Nx)]) +
                                                       z[ion] * (phi[phi_index(x, y, comp, Nx)] -
                                                                 phi[phi_index(x, y - 1, comp, Nx)])) / dy;
                                count_y++;
                            }
                            //Next upward difference
                            if (y < Ny - 1) {
                                GradUp = 1;//Dcs[c_index(x,y,comp,ion,Nx)*2+1]*(c[c_index(x,y,comp,ion,Nx)]+c[c_index(x,y+1,comp,ion,Nx)])/2;
                                GradUp = GradUp * (log(c[c_index(x, y + 1, comp, ion, Nx)]) -
                                                   log(c[c_index(x, y, comp, ion, Nx)]) +
                                                   z[ion] * (phi[phi_index(x, y + 1, comp, Nx)] -
                                                             phi[phi_index(x, y, comp, Nx)])) / dy;
                                count_y++;
                            }

                            Gradx = (Gradleft + GradRight) / count_x;
                            Grady = (GradUp + GradDown) / count_y;

                            if (x == Nx - 1 & y == Ny - 1) {
                                fprintf(fp, "%f,%f\n", Gradx, Grady);
                            } else {
                                fprintf(fp, "%f,%f,", Gradx, Grady);
                            }

                        }
                    }
                }
            }
    }
}

void calculate_measures(FILE *fp, struct AppCtx *user,PetscInt numrecords,int start)
{
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;
    PetscInt x,y;

    PetscReal dx = Lx/Nx;
    PetscReal dy = Ly/Ny;

    PetscReal *c= user->state_vars->c;
    PetscReal *al = user->state_vars->alpha;

    PetscReal Neu_Gli_K_diff = 0;
    PetscReal Glia_K_per_amt = 0;
    PetscReal total_amt_K=0;
    PetscReal alN;
    for(x=0;x<Nx;x++){
        for(y=0;y<Ny;y++){
            alN = 1-al[al_index(x,y,0,Nx)]-al[al_index(x,y,1,Nx)];
            total_amt_K += al[al_index(x,y,0,Nx)]*c[c_index(x,y,0,1,Nx)]+al[al_index(x,y,1,Nx)]*c[c_index(x,y,1,1,Nx)]+alN*c[c_index(x,y,Nc-1,1,Nx)];

            Neu_Gli_K_diff += al[al_index(x,y,0,Nx)]*c[c_index(x,y,0,1,Nx)]-al[al_index(x,y,1,Nx)]*c[c_index(x,y,1,1,Nx)];

            Glia_K_per_amt += al[al_index(x,y,1,Nx)]*c[c_index(x,y,1,1,Nx)];
        }
    }

    Neu_Gli_K_diff = Neu_Gli_K_diff*dx*dy;

    Glia_K_per_amt = (Glia_K_per_amt*dx*dy)/(total_amt_K*dx*dy);


    fprintf(fp,"%.10e,%.10e\n",Neu_Gli_K_diff,Glia_K_per_amt);
}
void calculate_energy(FILE *fp, struct AppCtx *user, PetscInt numrecords, int start){
    if (start) {
        fprintf(fp, "%d,%d,%d,%d,%d\n", user->Nx, user->Ny, numrecords, 0, 0);
        write_data(fp, user, numrecords, 0);
    }else{
        PetscScalar *c = user->state_vars->c;
        PetscScalar *phi = user->state_vars->phi;
        PetscScalar  *al = user->state_vars->alpha;
        PetscInt Nx = user->Nx;
        PetscInt Ny = user->Ny;

        PetscScalar Energy,alNc;
        PetscInt comp,ion;
        for(PetscInt y=0;y<Ny;y++){
            for(PetscInt x=0;x<Nx;x++){
                Energy = 0;
                //Ionic contribution
                for(comp=0;comp<Nc-1;comp++){
                    //Immobile ion part
                    Energy +=user->con_vars->ao[comp]*log(user->con_vars->ao[phi_index(x,y,comp,Nx)])
                            /al[al_index(x,y,comp,Nx)];
                    //Mobile ions
                    for(ion=0;ion<Ni;ion++){
                        Energy +=al[al_index(x,y,comp,Nx)]*c[c_index(x,y,comp,ion,Nx)]*log(c[c_index(x,y,comp,ion,Nx)]);
                    }
                    //ElectroPotential part
                    Energy += (cm[comp]/2)*pow((phi[phi_index(x,y,comp,Nx)]-phi[phi_index(x,y,Nc-1,Nx)])*RTFC,2);
                }
                //Extracellular ion term
                comp = Nc-1;
                alNc = 1-al[al_index(x,y,0,Nx)]-al[al_index(x,y,1,Nx)];
                //Immobile ion part
                Energy +=user->con_vars->ao[phi_index(x,y,comp,Nx)]*log(user->con_vars->ao[comp])/(alNc);
                //Mobile ions
                for(ion=0;ion<Ni;ion++){
                    Energy +=alNc*c[c_index(x,y,comp,ion,Nx)]*log(c[c_index(x,y,comp,ion,Nx)]);
                }
                //Write to file
                if (x == Nx - 1 & y == Ny - 1) {
                    fprintf(fp, "%.10e\n", Energy);
                } else {
                    fprintf(fp, "%.10e,", Energy);
                }

            }
        }

    }
}

void draw_csd(struct AppCtx *user)
{

    PetscReal vm,threshhold;
    threshhold = -20;
    PetscInt Nx = user->Nx;
    PetscInt Ny = user->Ny;

    for(PetscInt y=0;y<Ny;y++){
            printf("_");
    }
    printf("\n");

    for(PetscInt x=0;x<Nx;x++){
        printf("|");
        for(PetscInt y=0;y<Ny;y++){
            vm = user->state_vars->phi[phi_index(x,y,0,Nx)]-user->state_vars->phi[phi_index(x,y,Nc-1,Nx)];
            vm = vm * RTFC;
            if (vm > threshhold) {
                printf("x");
            } else {
                printf(" ");
            }

        }
        printf("|\n");
    }
    for(PetscInt y=0;y<Ny;y++){
        printf("_");
    }
    printf("\n");
}