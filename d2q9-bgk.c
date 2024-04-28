/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#define ALIGNMENT       64
#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

  const float c_sq = 3.f; /* square of speed of sound */
  const int lookup[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
  const float W[NSPEEDS] = {4.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f};

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float* speeds;
} t_speed;
/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
void pointer_swap(float** A, float** B);
int timestep(const t_param params, float* restrict cells, float* restrict tmp_cells, const int* restrict obstacles, float* av_velocity);
int accelerate_flow(const t_param params, float* restrict cells, const int* restrict obstacles);
int reision(const t_param params, const float* restrict cells, float* restrict tmp_cells, const int* restrict obstacles, float* av_velocity);
int write_values(const t_param params, const float* restrict cells, const int* restrict obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, const float* restrict cells);

/* compute average velocity */
float av_velocity(const t_param params, const float* restrict cells, const int* restrict obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  float* cells     = NULL;    /* grid containing fluid densities */
  float* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    timestep(params, cells, tmp_cells, obstacles, &av_vels[tt]);
    pointer_swap(&cells, &tmp_cells);
    // av_vels[tt] = av_velocity(params, cells, obstacles);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

void pointer_swap(float** A, float** B){
  float* aux = *A;
  *A = *B;
  *B = aux;
}

int timestep(const t_param params, float* restrict cells, float* restrict tmp_cells, const int* restrict obstacles, float* av_velocity)
{
  accelerate_flow(params, cells, obstacles);

  reision(params, cells, tmp_cells, obstacles, av_velocity);

  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, float* restrict cells, const int* restrict obstacles)
{
  __assume(params.nx % ALIGNMENT == 0);
  __assume(params.ny % ALIGNMENT == 0);
  __assume_aligned(cells, ALIGNMENT);
  __assume_aligned(obstacles, ALIGNMENT);

  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.0f;
  const float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2;

  #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    int boolean = !obstacles[ii + jj*params.nx]
        && (cells[3 * params.nx * params.ny + ii + jj*params.nx] - w1) > 0.f
        && (cells[6 * params.nx * params.ny + ii + jj*params.nx] - w2) > 0.f
        && (cells[7 * params.nx * params.ny + ii + jj*params.nx] - w2) > 0.f;
    
    /* increase 'east-side' densities */
    cells[params.nx * params.ny + ii + jj*params.nx] += w1 * boolean;
    cells[5 * params.nx * params.ny + ii + jj*params.nx] += w2 * boolean;
    cells[8 * params.nx * params.ny + ii + jj*params.nx] += w2 * boolean;
    /* decrease 'west-side' densities */
    cells[3 * params.nx * params.ny + ii + jj*params.nx] -= w1 * boolean;
    cells[6 * params.nx * params.ny + ii + jj*params.nx] -= w2 * boolean;
    cells[7 * params.nx * params.ny + ii + jj*params.nx] -= w2 * boolean;
    
  }

  return EXIT_SUCCESS;
}

int reision(const t_param params, const float*restrict cells,  float*restrict tmp_cells, const int* restrict obstacles, float* av_velocity){
  __assume(params.nx % ALIGNMENT == 0);
  __assume(params.ny % ALIGNMENT == 0);
  __assume_aligned(cells, ALIGNMENT);
  __assume_aligned(tmp_cells, ALIGNMENT);
  __assume_aligned(obstacles, ALIGNMENT);
  
  float counter = 0.0;
  float local_velocity = 0.0;


  // const float w0 = 4.f / 9.f;  /* weighting factor          */
  // const float w1 = 1.f / 9.f;  /* weighting factor          */
  // const float w2 = 1.f / 36.f; /*                           */

  for (int j = 0; j < params.ny; j++){
    const int y_n = (j + 1) % params.ny;
    const int y_s = (j == 0) ? (j + params.ny - 1) : (j - 1);

    #pragma omp simd reduction(+:local_velocity) reduction(+:counter)
    for (int i = 0; i < params.nx; i++){
      const int x_e = (i + 1) % params.nx;
      const int x_w = (i == 0) ? (i + params.nx - 1) : (i - 1);

      
      
        /* compute local density total */
        const float snapshot[9] = {
            cells[i + j * params.nx],
            cells[params.nx * params.ny + x_w + j*params.nx],
            cells[2 * params.nx * params.ny + i + y_s*params.nx],
            cells[3 * params.nx * params.ny + x_e + j*params.nx],
            cells[4 * params.nx * params.ny + i + y_n*params.nx],
            cells[5 * params.nx * params.ny + x_w + y_s*params.nx],
            cells[6 * params.nx * params.ny + x_e + y_s*params.nx],
            cells[7 * params.nx * params.ny + x_e + y_n*params.nx],
            cells[8 * params.nx * params.ny + x_w + y_n*params.nx]
          
        };



        /* compute x velocity component */
        const float u_x = (snapshot[1]
                      + snapshot[5]
                      + snapshot[8]
                      - (snapshot[3]
                         + snapshot[6]
                         + snapshot[7]));
        /* compute y velocity component */
        const float u_y = (snapshot[2]
                      + snapshot[5]
                      + snapshot[6]
                      - (snapshot[4]
                         + snapshot[7]
                         + snapshot[8]));

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        const float u[NSPEEDS] = {
          .0f,
          u_x,      
          u_y,
          - u_x,      
          - u_y,
          u_x + u_y,
          - u_x + u_y,
          - u_x - u_y,
          u_x - u_y
        };

        float local_density = 0.f;
        #pragma omp simd reduction(+:local_density)
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += snapshot[kk];
        }
        const float factor = local_density - (u_sq * c_sq) / (2.f * local_density);
        const int boolean = obstacles[i + j * params.nx];
        counter += 1 - boolean;

        float vel_density = 0.0;

        #pragma omp simd reduction(+:vel_density)
        for (int kk = 0; kk < NSPEEDS; kk++){
          tmp_cells[kk * params.nx * params.ny + i + j * params.nx] = boolean * snapshot[lookup[kk]] + !boolean * 
                                                    (snapshot[kk] * (1 - params.omega) + params.omega * W[kk] * 
                                                          (u[kk] * c_sq * (1 + (u[kk] * c_sq) / (2.f * local_density)) + factor));
          vel_density += tmp_cells[kk * params.nx * params.ny + i + j * params.nx];
        }

        local_velocity += !boolean * sqrtf(u_sq)/vel_density;
      
    }
  }

  // if (counter == 0) counter++;
  *av_velocity = local_velocity/counter;

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, const float* restrict cells, const int* restrict obstacles)
{
  __assume(params.nx % ALIGNMENT == 0);
  __assume(params.ny % ALIGNMENT == 0);
  __assume_aligned(cells, ALIGNMENT);
  __assume_aligned(obstacles, ALIGNMENT);
  
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma omp simd reduction(+:tot_cells) reduction(+:tot_u)
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[kk * params.nx * params.ny + ii + jj*params.nx];
        }

        /* x-component of velocity */
        float u_x = (cells[params.nx * params.ny + ii + jj*params.nx]
                      + cells[5 * params.nx * params.ny + ii + jj*params.nx]
                      + cells[8 * params.nx * params.ny + ii + jj*params.nx]
                      - (cells[3 * params.nx * params.ny + ii + jj*params.nx]
                         + cells[6 * params.nx * params.ny + ii + jj*params.nx]
                         + cells[7 * params.nx * params.ny + ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[2 * params.nx * params.ny + ii + jj*params.nx]
                      + cells[5 * params.nx * params.ny + ii + jj*params.nx]
                      + cells[6 * params.nx * params.ny + ii + jj*params.nx]
                      - (cells[4 * params.nx * params.ny + ii + jj*params.nx]
                         + cells[7 * params.nx * params.ny + ii + jj*params.nx]
                         + cells[8 * params.nx * params.ny + ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */

  *cells_ptr = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx * NSPEEDS), ALIGNMENT);

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */

  *tmp_cells_ptr = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx * NSPEEDS), ALIGNMENT);

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);


  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int) * (params->ny * params->nx), ALIGNMENT);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  __assume(params->nx % ALIGNMENT == 0);
  __assume(params->ny % ALIGNMENT == 0);
  __assume_aligned((*cells_ptr), ALIGNMENT);
  __assume_aligned((*tmp_cells_ptr), ALIGNMENT);
  __assume_aligned((*obstacles_ptr), ALIGNMENT);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx] = w0;
      (*tmp_cells_ptr)[ii + jj*params->nx] = 0;
      /* axis directions */
      (*cells_ptr)[params->nx * params->ny + ii + jj*params->nx] = w1;
      (*tmp_cells_ptr)[params->nx * params->ny + ii + jj*params->nx] = 0;
      (*cells_ptr)[2 * params->nx * params->ny + ii + jj*params->nx] = w1;
      (*tmp_cells_ptr)[2 * params->nx * params->ny + ii + jj*params->nx] = 0;
      (*cells_ptr)[3 * params->nx * params->ny + ii + jj*params->nx] = w1;
      (*tmp_cells_ptr)[3 * params->nx * params->ny + ii + jj*params->nx] = 0;
      (*cells_ptr)[4 * params->nx * params->ny + ii + jj*params->nx] = w1;
      (*tmp_cells_ptr)[4 * params->nx * params->ny + ii + jj*params->nx] = 0;
      /* diagonals */
      (*cells_ptr)[5 * params->nx * params->ny + ii + jj*params->nx] = w2;
      (*tmp_cells_ptr)[5 * params->nx * params->ny + ii + jj*params->nx] = 0;
      (*cells_ptr)[6 * params->nx * params->ny + ii + jj*params->nx] = w2;
      (*tmp_cells_ptr)[6 * params->nx * params->ny + ii + jj*params->nx] = 0;
      (*cells_ptr)[7 * params->nx * params->ny + ii + jj*params->nx] = w2;
      (*tmp_cells_ptr)[7 * params->nx * params->ny + ii + jj*params->nx] = 0;
      (*cells_ptr)[8 * params->nx * params->ny + ii + jj*params->nx] = w2;
      (*tmp_cells_ptr)[8 * params->nx * params->ny + ii + jj*params->nx] = 0;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    #pragma omp simd
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  _mm_free(*cells_ptr);
  *cells_ptr = NULL;

  _mm_free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, const float* restrict cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[kk * params.nx * params.ny + ii + jj*params.nx];
      }
    }
  }

  return total;
}

int write_values(const t_param params, const float* restrict cells, const int* restrict obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[kk * params.nx * params.ny + ii + jj*params.nx];
        }

        /* compute x velocity component */
        u_x = (cells[params.nx * params.ny + ii + jj*params.nx]
               + cells[5 * params.nx * params.ny + ii + jj*params.nx]
               + cells[8 * params.nx * params.ny + ii + jj*params.nx]
               - (cells[3 * params.nx * params.ny + ii + jj*params.nx]
                  + cells[6 * params.nx * params.ny + ii + jj*params.nx]
                  + cells[7 * params.nx * params.ny + ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[2 * params.nx * params.ny + ii + jj*params.nx]
               + cells[5 * params.nx * params.ny + ii + jj*params.nx]
               + cells[6 * params.nx * params.ny + ii + jj*params.nx]
               - (cells[4 * params.nx * params.ny + ii + jj*params.nx]
                  + cells[7 * params.nx * params.ny + ii + jj*params.nx]
                  + cells[8 * params.nx * params.ny + ii + jj*params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
