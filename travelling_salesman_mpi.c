/* Exercise to solve the Traveling Salesman Problem (TSP) using
 a brute force Monte Carlo approach. Parallelized using MPI


To compile:

 - serial:
cc -O2 tsp.c -o tsp


 - MPI:
mpicc -O2 tsp_mpi.c -o tsp_mpi


To run:
 - MPI:
mpirun -np 24 ./tsp_mpi


*/

#include <sys/time.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
//RK: Include MPI Library

// Number of cities:
#define N_CITIES 11

// Maximum number of Monte Carlo steps:
#define N_MC (long long int) 1e8


// Size of the square region (km) where the cities are randomly scattered:
#define SIZE 1000

// Cities coordinates:
float x[N_CITIES], y[N_CITIES];
// Distance array:
float dist[N_CITIES][N_CITIES];


/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  */

// It messes up with y!

int
timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
  struct timeval result0;

  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double restime;
  int devid, devcount, error, i, j, k, l, l1, ltemp;
  int perm[N_CITIES], perm_min[N_CITIES];
  float d;

 // RK: Create variables needed for MPI operations
int my_rank; /* rank of process */
int p; /* number of processes */
int source; /* rank of sender */
int dest; /* rank of receiver */
int tag = 0; /* tag for messages */
long long int local_nmc; /* local number of montecarlo simulations for each process*/
MPI_Status status; /* status for receive */

// RK: In order to use reduce later to find the rank with the smallest d_min but retain the rank of this process in Reduce, we use MPI_Minloc, and we create a struct for the value (d_min) and the rank of the affiliated process
 struct { 
              float val;
              int   rank;
           }  in, out;


  // Generating randomly scattered cities
  for (i=0; i<N_CITIES; i++)
    {
      x[i] = (float)SIZE * (float)rand()/((float)RAND_MAX+1.0);
      y[i] = (float)SIZE * (float)rand()/((float)RAND_MAX+1.0);
    }


  // Computing the distance matrix:
  for (i=N_CITIES-1; i>=0; i--)
    {
      for (j=0; j<N_CITIES; j++)
    	{
    	  if (j < i)
    	    dist[i][j] = sqrt(pow(x[j]-x[i],2) + pow(y[j]-y[i],2));
    	  else if (j == i)
    	    dist[i][j] = 0.0;
    	  else
    	    dist[i][j] = dist[j][i];
    	}
    }



// RK: Initialise MPI parallelisation once the distance matrix has been computed
MPI_Init(&argc, &argv);
  
  gettimeofday (&tdr0, NULL);


// RK: set the ranks for each process, as well as the number of processes
  /* Find out process rank */
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
/* Find out number of processes */
MPI_Comm_size(MPI_COMM_WORLD, &p);

// RK: divide the monte carlo simulations between the processes (we set this to 24 to equal the number of processors on Orca)
local_nmc = N_MC/p;

// RK: send the distance matrix from rank 0 to all other ranks via collective communication
// RK: with broadcast, no receive is needed and all the ranks receive the distance matrix in their buffer
MPI_Bcast(&dist, N_CITIES*N_CITIES, MPI_FLOAT, 0, MPI_COMM_WORLD);

 // RK: We assign a different seed to each rank, so that when each process computes local_nmc iterations, all the seeds will be different, hence each process is doing different simulations
  float d_min = 1e30;
  unsigned int seed = 111+ my_rank;

  // We always start from the same (0-th) city:
  perm[0] = 0;


  // Cycle for Monte Carlo steps:
  for (k=0; k<local_nmc; k++)
    {
      // Generating a random permutation:

      // Initially we have an ordered list:
      for (l=1; l<N_CITIES; l++)
	perm[l] = l;

      // Then we reshuffle it randomly, starting with l=1:
      d = 0.0;
      for (l=1; l<N_CITIES-1; l++)
	{
	  // This generates a random integer in the range [l ... N_CITIES-1]:
	  l1 = l + rand_r(&seed) % (N_CITIES-l);

	  // Swapping the l and l1 cities:
	  ltemp = perm[l];
	  perm[l] = perm[l1];
	  perm[l1] = ltemp;

	  // At this point, cities in perm[l-1] and perm[l] have already been reshuffled, so we 
	  // can compute their contribution to the total distance:
	  d = d + dist[perm[l-1]][perm[l]];
	}

      // At the final leg we are coming back to the original (0-th) city:
      d = d + dist[perm[N_CITIES-1]][0];
      d = d + dist[perm[N_CITIES-1]][perm[N_CITIES-2]];

      // RK: Find dmin for each process 
      // Finding globally shortest TSP distance:
      if (d < d_min)
	{
	  d_min = d;
    // RK: No need to print every iteration now
	  //printf ("%d %f ", k, d_min);
	  // Printing the itinerary corresponding to the current smallest distance:
	  for (l=0; l<N_CITIES; l++)
	    {
	     // printf ("%d ", perm[l]);
	      // Memorizing the shortest itinernary:
	      perm_min[l] = perm[l];
	    }
	  //printf ("\n");
	}

    // RK: Once we have a local d_min for every rank, we assign the rank and d_min value for each to struct in

   
    in.val = d_min; 
    in.rank = my_rank; 
   

    }  

// RK: Need to reduce d_min back to rank 0 from all other processes, so use reduce with min_loc to get rank of process with global d_min, as well as global_dmin
 MPI_Reduce( &in, &out, 1, MPI_FLOAT_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
// RK: The rank of the process giving the smallest d_min and its value is stored in the structure 'out'
 

// RK: Broadcast the value of this rank to all other ranks, so the scope of rank_min is global; so far only rank 0 is aware of what this is
 MPI_Bcast(&out.rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
// RK: Then use MPI_send for rank == rank_min and MPI_recv for rank == 0 to send the perm_min from rank_min to rank 0
  if (my_rank == out.rank)
      {
        MPI_Send(&perm_min,N_CITIES, MPI_INT, 0, 0, MPI_COMM_WORLD);
      }
  if (my_rank == 0) 
      { 
        MPI_Recv(&perm_min, N_CITIES, MPI_INT, out.rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  
      } 
 



// RK: Print the minimum distance and the associated permutation of cities
  
  gettimeofday (&tdr1, NULL);


if (my_rank == 0)
{
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);
  printf ("Shortest total distance: %f\n", out.val);
  printf ("Order of cities visited:");
for (l=0; l<N_CITIES; l++)
{
  printf (" %d ", perm_min[l]);
}
printf("\n");
  // Timing for the region to be parallelized:
  printf ("%e\n", restime);

  // End of the region to parallelize
  //--------------------------------------------------------------------------------


  // Writing a text file containing the itinerary (x,y coordinates) -
  // to be plotted in a plotting software.
  FILE *fp = fopen ("tsp.dat", "w");
  for (l=0; l<N_CITIES; l++)
    {
      fprintf (fp, "%f %f\n", x[perm_min[l]], y[perm_min[l]]);
    }
  // Going back to the original city:
  fprintf (fp, "%f %f\n", x[perm_min[0]], y[perm_min[0]]);
  fclose (fp);

}

MPI_Finalize();
  return 0;

}
