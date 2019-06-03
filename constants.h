#ifndef __CONSTANTS__
#define __CONSTANTS__
#include <petsctime.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsnes.h>


//set global parameters here (constants)

// General options

#define details 0 //if true, will show how many iterations were necessary for each newton solve, and the residual
#define mid_points_exct 0
#define one_point_exct 0 //if true, triggers SD at origin
#define plane_wave_exct 1 //if true, initiates a uniform plane wave
#define Profiling_on 1 //Turns timing of functions on/off.
#define trecordstep 0.5 //determines how often to record
#define save_one_var 0 //Instead of saving all 14 vars, save just 1 (specified in write_data)
#define start_at_steady 1 //Start at steady state?


//Solver Type Options
#define use_en_deriv 1 //if true, will use the derivative of the electroneutrality condition for the system of equations
#define separate_vol 1 //if true, will solve c,phi separate from alpha.
#define Linear_Diffusion 0 //Changes to a linear discretization of electrodiffusion.
#define Predictor 0  // Turns on predictor. Adaptive single point estimated update
#define width_size  1 //Number of up,down,left,right neighbors to pair in the predictor.
#define Max_Grid_Refine 256 // Max number of time steps to refine in grid predictor

//basic ion extern constants
#define   Ni  4          //number of ion species (Na, K, Cl)
static const   PetscInt z[4] = {1,1,-1,0}; //valences of ion species
static const   PetscReal D[4] = {1.33e-5, 1.96e-5, 2.03e-5,7.6e-6};      //diffusion coefficients in cm^2/sec
//Diffusion multipliers {x-dir,y-dir}
static const PetscReal DNeuronMult[2] = {0,0};
static const PetscReal DGliaMult[2] = {0.25,0.25};
static const PetscReal DExtraMult[2] = {1.0,1.0};
#define Time 10.0   //total simulated time in seconds
//#define  Time  180.0//2e-2
#define   Nc 3           //number of compartments
//#define Lx 0.32        //width of domain in cm (x)
//#define Ly 0.32         //length of domain in cm (y)
#define Lx 0.5        //width of domain in cm (x)
#define Ly 0.5         //length of domain in cm (y)


//number of variables to be solved for at each grid point
//#define  Nv  ((Ni+2)*Nc-1) //version if volume is included
//#define  Nv  ((Ni+1)*Nc) //version if volume is excluded
#define Nv  (((Ni+2)*Nc-1)*(!separate_vol)+((Ni+1)*Nc)*separate_vol)  //combining the above two with the separate_vol

//Newton solve parameters
#define  itermax  20      //maximum Newton iterations allowed
#define  reltol  1e-11    //relative tolerance


//physical extern constants
#define   Rc  8.314472e6    //gas constant in nJ/K/mmol
#define   T (273.15+37)    //absolute temperature in K
#define  FC  9.64853399e7 //Faraday constant in muC/mmol
#define RTFC (Rc*T/FC)
#define pi 3.14159265358979323846 //M_PI is defined in some C implementations, but isn't in gen.
//So we define pi here.

//Bath variables
//extern PetscReal cbath[3]; //Na, K, and Cl
static const PetscReal cbath[4]={140*1e-3,3.4*1e-3,120*1e-3,1e-8}; //Na, K, Cl, and Glutamate
#define Batheps 1.0 //Bath diffusion multiplier
#define phibath (-0/RTFC) //Voltage of outside bath

//excitation parameters
//#define pmax  (1e-1/3)          //max value for excitation
#define pmax  (1e1)          //max value for excitation
//#define pmax  50          //max value for excitation
//#define texct 2         //time for excitation
#define texct 0.5
//#define texct 0.05         //time for excitation
#define Lexct 0.05          //Length of region for excitation in each direction
//#define Lexct 0.025          //Length of region for excitation in each direction

//initial state setup
#define rest_state  1        //if true, membrane parameters are set so that the initial voltages and concentrations are at a rest state
#define spatially_uniform  1 //if true, all initial values and parameters are spatially uniform


//set "relaxed" volume fractions and initial volume fractions
#define  alphaon 0.5         //base neuronal volume fraction
#define alphaog 0.300000       //base glial volume fraction
static const PetscReal alphao[2]={alphaon,alphaog};
static const PetscReal alpha0[2]={alphaon,alphaog};
// membrane parameters
#define  cmt  0.75e-3            //membrane capacitance in mF/cm^2
#define sa  1.586e-5          //membrane area in cm^2
#define voli  2.16e-9         //intracellular volume in cm^3
#define vole (0.15*voli)
#define ell ((voli+vole)/sa)    //average membrane separation in cm
static const PetscReal cm[2] ={cmt*RTFC/FC/ell,cmt*RTFC/FC/ell};     //membrane capacitance in mF/cm^2 converted to mmol/cm^3


//data for ion channel currents
//permeabilities in cm/s from Kager, 2000 and Yao, Huang, Miura, 2011.
#define basepNaT  0//1e-4               //1e-4%0%1e-3%if set to 0, recovery possible
#define basepNaP  2e-5 //8e-6 //2e-5
#define basepKDR  1e-3
#define basepKA  1e-4
#define basepNMDA  1e-5//3e-6 //6e-5//5e-5           //NMDA permeability (cm/sec)

//Leak conductances in mS/cm^2 from Kager, 2000 or Yao, Huang, Miura, 2011.
#define pKLeak  (7e-2*RTFC/FC)     //Kager:10e-2,Miura:7e-2%K Leak conductance in mS/cm^2 converted to mmol/cm^2/s
#define pClLeak  (10e-2*RTFC/FC)   //Kager:10e-2,Miura:20e-2%Cl Leak conductance in mS/cm^2 converted to mmol/cm^2/s
#define pClLeakg  (5e-2*RTFC/FC)

//Glial KIR from Steinberg et al 2005
#define basepKIR  (.13*RTFC/FC)        //conductance in mS/cm^2 converted to mmol/cm^2/s
#define pKLeakadjust  1.0       //parameter for varying glial permeability

//pump current, parameters from Yao, Huang, Miura, 2011
#define mK  2e-3                 //pump current constant in mmol/cm^3=mol/l
#define mNa  7.7e-3              //pump current constant in mmol/cm^3=mol/l
#define glpump  1.0            //multiplier to change glial pump rate
#define npump  1.0

// Glutamate parameters
#define glut_A  50e-3 //500e-3 //500e-5 //100e-5//500e-5 //(500e-3) //500       //Release rate in mmol/cm^3/sec
#define glut_eps 22.99e-6//5e-3//5e-6 //5e-3      //Small scaling factor muMol converted to millMol
#define glut_Rg (1e-3)//(1.0/6)      // Steady state glia/neuron concentration ratio
#define glut_Bg (1.0/84) //(1.0/42)//(1.0/20)//(1.0/42)//8e-2             //Decay rate(glia->neurons) in 1/sec
#define pNaGl_n 0 //(3e-5*RTFC/FC)              //Na-Glu transporter permeability
#define pNaGl_g (3e-5*RTFC/FC)              //Na-Glu transporter permeability
#define pHratio 2.0//0.5             //Ratio of extracell to intracell pH(for transporter)

//#define glut_Bg 1.0e-5 //19.2e-6             //Decay rate(glia->neurons) in 1/sec
//#define glut_Kg 99.940035978412951            //Neuronal fraction for glial reaction


#define glut_gamma 0.1 //0.2    //Reabsorbtion ratio (arbitrary)
#define glut_Bn (1.0/42) //(1.0/21) //(1.0/10)//(1.0/42)//10e-2//e-2              //Decay rate(extracell->intracell) in 1/sec
#define glut_Re (1e-3) //(1e-4/6)         // Steady state extracell/neuron concentration ratio

// NMDA glutamate interaction
static const PetscReal Desensitize[3] = {0.0,0.0};//{0.1,0.01};//{.30,0.1};//{.33,.16}; //{0.2,0.02};//{0.33,0.01}; // In the NMDA receptor it becomes desensitized over time


// Data Structures
struct SimState{
    PetscScalar *c; //Variable arrays
    PetscScalar *phi;
    PetscScalar *alpha;
    Vec v;  // Full variable vec
    Vec c_vec;  //Individual variable vecs
    Vec phi_vec;
    Vec al_vec;
    IS c_ind;    //Indices for c/phi/al.
    IS al_ind;
    IS phi_ind;

};
// Flux data plus derivatives
struct FluxData{
    PetscReal *mflux; //membrane flux
    PetscReal *dfdci; //intracellular deriv
    PetscReal *dfdce; //extracell. deriv
    PetscReal *dfdphim; //membrane voltage deriv (intra volt = + this, extra volt = -this)
    PetscReal *wflow;  // osmotic water flow
    PetscReal *dwdpi;  //concentration deriv.
    PetscReal *dwdal;  //volume deriv
};


struct GateType{
    PetscReal *mNaT;
    PetscReal *hNaT;
    PetscReal *gNaT;
    PetscReal *mNaP;
    PetscReal *hNaP;
    PetscReal *gNaP;
    PetscReal *mKDR;
    PetscReal *gKDR;
    PetscReal *mKA;
    PetscReal *hKA;
    PetscReal *gKA;
    PetscReal *yNMDA;
    PetscReal *zNMDA;
    PetscReal *dNMDA;
    PetscReal *gNMDA;
};
// Excitation permeabilities
struct ExctType{
    PetscReal *pNa;
    PetscReal *pK;
    PetscReal *pCl;
    PetscReal *pGlu;
};
// Constant params (vary in space set in constants.c)
struct ConstVars{
    PetscReal *pNaT; //Gating variable arrays
    PetscReal *pNaP;
    PetscReal *pKDR;
    PetscReal *pKA;
    PetscReal *pNaKCl; //Glial NaKCl Cotransporter array
    PetscReal *Imax;   // Neuronal pump strength
    PetscReal *pNaLeak;
    PetscReal *Imaxg;   //Glial pump strength
    PetscReal *pNaLeakg;
    PetscReal *pNMDA; //NMDA perm.
    PetscReal *pKIR;   //K inward rectifier in glia
    PetscReal *ao;  //Immobile ions
    PetscReal *zo;  //Avg valence
    PetscReal kappa;
    PetscReal *zeta1;
    int S; //boolean (not used)
    PetscReal *zetaalpha;
    PetscReal *DNeuronScale; // Glial diffusion scaling
    PetscReal *DGliaScale; // Glial diffusion scaling
    PetscReal *DExtracellScale; // Extracellular diffusion scaling
};
// All Solver data structs
struct Solver{
    Vec Q;      /* Update*/
    Vec Res;
    Mat A;            /* linear system matrix */
    SNES snes;       /*Nonlinear solver context*/
    KSP ksp;         /* krylov linear solver context */
    PC pc;           /* preconditioner context */
    PetscMPIInt   size;  // MPI ranks

    PetscInt NA;  // Total number of variables (Nx*Ny*Nv)
};

// Struct containing all arrays and params
// This gets passed as "context" to petsc functions
struct AppCtx{
    struct SimState *state_vars;
    struct SimState *state_vars_past;
    struct SimState *grid_vars;
    struct SimState *grid_vars_past;
    struct Solver *slvr;
    struct Solver *grid_slvr;
    struct FluxData *flux;
    struct GateType *gate_vars;
    struct GateType *gate_vars_past;
    struct GateType *grid_gate_vars;
    struct ExctType *gexct;
    struct ConstVars *con_vars;
    PetscReal *Dcs;
    PetscReal *Dcb;
    PetscReal dt;
    PetscReal dx;
    PetscReal dy;
    PetscInt Nx;
    PetscInt Ny;
    PetscInt Nz;
    PetscReal *dt_space;
    PetscReal *vm_past;
    PetscReal t;
    FILE *fp;
};

// Logging events during the code for timing
// Add additional events in misc_print_plot.c -> init_events function
// May need to allocate a longer array if so.
PetscLogEvent event[14];

#endif