#include "constants.h"
#ifndef __FUNCTIONS__
#define __FUNCTIONS__

#include <petsctime.h>
#include <petscsnes.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petsclog.h>
//Function to initialize the state
void init(Vec,struct SimState *,struct AppCtx*);
//Solve until steady state (mostly to update gating_vars)
void initialize_data(Vec,struct AppCtx*);

//Set the paramaters based on the constants
void set_params(Vec,struct SimState *,struct ConstVars*,struct GateType*,struct FluxData*,struct AppCtx*);
void parameter_dependence(struct AppCtx *);

//Data management functions
void init_arrays(struct AppCtx*);
PetscErrorCode init_simstate(Vec,struct SimState*,struct AppCtx*);
PetscErrorCode extract_subarray(Vec,struct SimState*);
PetscErrorCode restore_subarray(Vec,struct SimState*);
PetscErrorCode extract_subarray_Read(Vec,struct SimState*);
PetscErrorCode restore_subarray_Read(Vec,struct SimState*);
PetscErrorCode copy_simstate(Vec,struct SimState *state_vars_past);

//Linear current-voltage flux relation
void mclin(struct FluxData *,int,double,int,double,double,double,int);
//GHK Relation 
void mcGoldman(struct FluxData *,int,double,int,double,double,double,int);
//Glutamate membrane fluxes
void glutamate_flux(struct FluxData *,PetscInt,PetscInt,struct SimState *state_vars,struct SimState *state_vars_past,
                    PetscInt,PetscReal Glut_Excite);
//Conductance for potassium inward rectifier
double inwardrect(double,double,double);
//Returns of c_i*z_i
double cz(const double *,const PetscInt *,int,int,int,PetscInt,struct AppCtx*);
//Computes diffusion coef
void diff_coef(double *,const PetscReal *,PetscReal,struct AppCtx*);
//Calculate the ion fluxes and derivatives
void ionmflux(struct AppCtx*);
//Water flow
void wflowm(struct AppCtx*);

//Update the gating variables
void gatevars_update(struct GateType *,struct GateType *,struct SimState *,PetscReal,struct AppCtx *,int);
//Update Volume fraction
void volume_update(struct SimState*,struct SimState*,struct AppCtx*);
//Initial excitation
void excitation(struct AppCtx*,PetscReal);

// Indexing functions
PetscInt c_index(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt);
PetscInt phi_index(PetscInt,PetscInt,PetscInt,PetscInt);
PetscInt al_index(PetscInt,PetscInt,PetscInt,PetscInt);
PetscInt xy_index(PetscInt,PetscInt,PetscInt);
PetscInt Ind_1(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt);
PetscInt Ind_2(PetscInt,PetscInt ,PetscInt ,PetscInt , PetscInt );


//Create Petsc Structures

PetscErrorCode initialize_petsc(struct Solver *,int, char **,struct AppCtx*);
PetscErrorCode initialize_grid_slvr(struct Solver *,int, char **,struct AppCtx*);
void Get_Nonzero_in_Rows(int *,struct AppCtx*,int);
PetscErrorCode initialize_jacobian(Mat,struct AppCtx*,int);
PetscErrorCode initialize_grid_jacobian(Mat,struct AppCtx*,int);

//Multigrid functions
PetscErrorCode Create_Restriction(Mat R,PetscInt nx, PetscInt ny);
PetscErrorCode Create_Interpolation(Mat R,PetscInt nx, PetscInt ny);
PetscErrorCode Initialize_PCMG(PC pc,Mat A,struct AppCtx*);

//Newton Solver
PetscErrorCode newton_solve(Vec,struct Solver*,struct AppCtx*);
//Calculate residuals and jacobians

//Nonlinear discretizations
// Derivative of CC with volume
PetscErrorCode calc_residual(SNES,Vec,Vec,void*); //void is masked AppCtx
PetscErrorCode calc_jacobian(SNES, Vec, Mat,Mat, void*); //void is masked AppCtx
//Derivative of CC with no volume
PetscErrorCode calc_residual_no_vol(SNES,Vec,Vec,void*); //void is masked AppCtx
PetscErrorCode calc_jacobian_no_vol(SNES, Vec, Mat,Mat, void*); //void is masked AppCtx
//Algebraic CC with volume
PetscErrorCode calc_residual_algebraic(SNES,Vec,Vec,void*); //void is masked AppCtx
PetscErrorCode calc_jacobian_algebraic(SNES, Vec, Mat,Mat, void*); //void is masked AppCtx
//Algebraic CC with no volume
PetscErrorCode calc_residual_algebraic_no_vol(SNES,Vec,Vec,void*); //void is masked AppCtx
PetscErrorCode calc_jacobian_algebraic_no_vol(SNES, Vec, Mat,Mat, void*); //void is masked AppCtx

//Linear discretizations
// Linear system Algebraic CC with no volume
PetscErrorCode calc_residual_linear_algebraic(SNES,Vec,Vec,void*); //void is masked AppCtx
PetscErrorCode calc_jacobian_linear_algebraic(SNES, Vec, Mat,Mat, void*); //void is masked AppCtx
PetscErrorCode calc_residual_linear_deriv(SNES,Vec,Vec,void*); //void is masked AppCtx
PetscErrorCode calc_jacobian_linear_deriv(SNES, Vec, Mat,Mat, void*); //void is masked AppCtx


//Functions used in Grid solves
void grid_wflowm(struct AppCtx *);
void grid_ionmflux(struct AppCtx*,PetscInt,PetscInt);
void gatevars_update_grid(struct GateType *,struct SimState *,PetscReal ,struct AppCtx *);
void excitation_grid(struct AppCtx* ,PetscReal ,PetscInt, PetscInt);
void grid_diff_coef(PetscReal *,const PetscReal *,PetscReal ,struct AppCtx* ,PetscInt,PetscInt);

PetscErrorCode Grid_Residual(Vec ,PetscInt ,PetscInt ,void *);
PetscErrorCode Grid_Jacobian(Mat ,PetscInt ,PetscInt ,void *);
PetscErrorCode Grid_Residual_algebraic(Vec ,PetscInt ,PetscInt ,void *);
PetscErrorCode Grid_Jacobian_algebraic(Mat ,PetscInt ,PetscInt ,void *);
PetscErrorCode Update_Grid(PetscInt ,PetscInt,PetscReal ,struct AppCtx *);
PetscErrorCode Update_Solution(Vec,PetscReal t,struct AppCtx *);
int Newton_Solve_Grid(PetscInt, PetscInt,struct AppCtx *);
void Unload_Grid(struct AppCtx *,PetscInt , PetscInt );

void save_timestep(FILE *,struct AppCtx*,PetscInt ,int );


//Find abs. max value of an array
PetscReal array_max(PetscReal *,size_t);
// abs. max, but for difference
PetscReal array_diff_max(PetscReal *,PetscReal *,size_t);
//Calc l2_norm of one array
PetscReal l2_norm(PetscReal *,size_t);

void print_all(struct AppCtx*);
const char* getfield(char* , int );
void find_print(int, int, double, int iter);
void compare_res(double *, int );
void write_data(FILE *,struct AppCtx *,PetscInt,int );
void write_point(FILE *,struct AppCtx *,PetscReal,PetscInt,PetscInt );
void measure_flux(FILE *,struct AppCtx *,PetscInt,int );
void init_events(struct AppCtx *);
void read_file(struct AppCtx *);
void save_file(struct AppCtx *);
void velocity_field(FILE *,struct AppCtx *,PetscInt,int);
void record_measurements(FILE **,struct AppCtx *,PetscInt,PetscInt ,int );
void calculate_measures(FILE *, struct AppCtx *,PetscInt ,int );
void draw_csd(struct AppCtx *);
void calculate_energy(FILE *, struct AppCtx *, PetscInt , int );


#endif