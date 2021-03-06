/* Exercise to solve the Traveling Salesman Problem (TSP) using
 a brute force Monte Carlo approach. 

To compile:

 - serial:
cc -O2 tsp.c -o tsp

 - OpenMP:
cc -O2 -openmp test_2.c -o tsp_omp

 
*/

#include <sys/time.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

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
  
  gettimeofday (&tdr0, NULL);


  //--------------------------------------------------------------------------------
  // This serial computation will have to be parallelized


    
  float d_min = 1e30;
  unsigned int seed = 111;
  int threadid;

   #pragma omp parallel default(none) private(k,l, ltemp,l1, d,threadid, perm) firstprivate(seed) shared(d_min, perm_min, dist) 
  {
  // RK: seed is firstprivate so all threads are initialised with the same value, and then updated so each thread gets a new seed, to prevent a duplicated rand sequence between threads
  // RK: d_min and perm_min are shared as each thread needs to be able to update these if they find the minumum path
  // We always start from the same (0-th) city:
  perm[0] = 0;

  
  // Cycle for Monte Carlo steps:

  // RK: Use a for schedule to split up iterations between threads. Guided worked best. Since all iterations should theoretically take a similar amount of time, having larger chunks in the beginning can reduce overhead
 #pragma omp for schedule(guided)
  for (k=0; k<N_MC; k++)
    {

      threadid = omp_get_thread_num();
      seed = seed + threadid;

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


      // Finding globally shortest TSP distance:

     
     // RK: Need to "flush" here to create consistent value of d_min and prevent race conditions, however this is inefficient, so we duplicate the if statement before and after entering critical region instead.
    
      if (d < d_min)
	{
    // RK: Critical region with an implicit flush, ensuring all threads have same d_min value and only one thread should be able to update d_min and perm_min at a time
    #pragma omp critical 

    if (d <d_min)
    {
	  d_min = d;
    // RK: All threads waiting to enter critical region should have a consistent value for d_min before executing the if statement checking for shortest distance
    #pragma omp flush (d_min)
	  printf ("%d %f ", k, d_min);
	  // Printing the itinerary corresponding to the current smallest distance:
	  for (l=0; l<N_CITIES; l++)
	    {
	      printf ("%d ", perm[l]);
	      // Memorizing the shortest itinernary:
	      perm_min[l] = perm[l];
	    }
	  printf ("\n");
	}
 }

    } 
  } 
  
  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);
  printf ("Shortest total distance: %f\n", d_min);

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


  return 0;

}
