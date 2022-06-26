/* Exercise to convert a simple serial code to count the number of prime
   numbers within a given interval into OpenMP (32-bit, int version).

   All prime numbers can be expressed as 6*k-1 or 6*k+1, k being an
   integer. We provide the range of k to probe as macro parameters
   KMIN and KMAX (see below).

   Check the parallel code correctness - it should produce the same number of prime
   numbers as the serial version, for the same range KMIN...KMAX. (The result
   is 3,562,113 for K=1...10,000,000.)

   Try to make the parallel code as efficient as possible.

   Your speedup should be close to the number of threads/ranks you are using.

OpenMP instructions:

   The best strategy is to start from KMAX and go down to KMIN. Why?

   The code should print the number of threads used.

   Use "default(none)" in parallel region(s).

   Used OpenMP directives:
 - parallel
 - for schedule
 - single


Compiling instructions:

 - Serial code:
  icc -O2 primes_count.c -o primes_count

 - OpenMP code:
  icc -openmp -O2 primes_count_omp_guided.c -o primes_count_omp_guided

*/

// Ramsha Khan - Assignment 1 

#include <sys/time.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <omp.h> 

// Range of k-numbers for primes search:
#define KMIN 1
// Should be smaller than 357,913,941 (because we are using signed int)
#define KMAX 10000000

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
  int devid, devcount, error, success;
  int xmax, ymax, x, y, k, j, count;

  gettimeofday (&tdr0, NULL);


  int nthreads;
  count = 0;

 // RK: Start a parallel block to allow worksharing between threads
 // RK: Variables declared within the for loop and indices are kept private to each thread
 // RK: Count is shared, as each thread will need to write to it once it has completed its task
  /*   The best strategy is to start from KMAX and go down to KMIN. Why?

   ANSWER: The primality test will require more work for large values of k. By starting near KMAX and using a 
   for schedule with a dynamic clause, and a specified chunk size of 250 (this gave me my best speedup), all threads
   start working on the large values of k, and thus more intensive computations are completed before moving
   on to smaller values of k. This ensures a better distribution of the workload balance.*/
#pragma omp parallel default(none) private(k,j,x,ymax,y,nthreads,success) shared(count)      
  {
      // RK: Use a 'for schedule' to divide the work into chunks, where chunks of size 250 are allocated to each thread as it moves up
      // ... in the work queue
      // RK: Each thread keeps a private copy of the total count and we use reduction to sum the total when work is completed
      #pragma omp for schedule(dynamic,250) reduction(+:count)      
            for (k=KMAX; k>=KMIN; k--)
              {
                // testing "-1" and "+1" cases:
                for (j=-1; j<2; j=j+2)
                {
                // Prime candidate:
                x = 6*k + j;
                // We should be dividing by numbers up to sqrt(x):
                ymax = (int)ceil(sqrt((double)x));

                // Primality test:
                for (y=3; y<=ymax; y=y+2)
                  {
                    // Tpo be a success, the modulus should not be equal to zero:
                    success = x % y;
                    if (!success)
                break;
                  }

                if (success)
                  {
                    count++;
                  }
                }
              }
    // RK: Once out of the for schedule, retrieve number of threads
    // RK: use 'single' to ensure only one thread computes and prints the total number of threads
    #pragma omp single nowait
      {

    // get the number of threads
      nthreads = omp_get_num_threads();
      printf("There are %d threads\n", nthreads);
    }

  }

  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);
  printf ("N_primes: %d\n", count);
  printf ("time: %e\n", restime);
  //--------------------------------------------------------------------------------



  return 0;

}
