/* Exercise to convert a simple serial code to count the number of prime
   numbers within a given interval into MPI (32-bit, int version).

   All prime numbers can be expressed as 6*k-1 or 6*k+1, k being an
   integer. We provide the range of k to probe as macro parameters
   KMIN and KMAX (see below).

  

Compiling instructions:

 - Serial code:
  icc -O2 primes_count.c -o primes_count

 - MPI code:
  mpicc -O2 primes_count.c -o primes_count
   To run: 
  mpirun -np 24 ./primes_count

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

// Range of k-numbers for primes search:
#define KMIN 1
// Should be smaller than 357,913,941 (because we are using signed int)
#define KMAX 10000000
#define dk 15000   // RK: This was the optimal chunk size I found

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
  int xmax, ymax, x, y, k, k0, i, j, b;
  int count = 0;
  int a = KMAX;  // RK: This is the total number of iterations to be handed out to workers


int my_rank; /* rank of process */
int p; /* number of processes */

int source; /* rank of sender */
int dest; /* rank of receiver */
int c_tag = 0; /* tag for messages */
int a_tag = 0;
int flag = 0;
int loc_count = 0;  // RK: Local count of primes for each work parcel done
int m_count  = 0;  // RK global work count updated by master each time work is completed by worker


MPI_Init(&argc, &argv);

MPI_Status status; /* status for receive */
MPI_Request request;

// RK: set the ranks for each process, as well as the number of processes
  /* Find out process rank */
MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
/* Find out number of processes */
MPI_Comm_size(MPI_COMM_WORLD, &p);



if (my_rank == 0)
{ gettimeofday (&tdr0, NULL);
  // RK: While there are still parcels of chunk size dk available
  while (a >= dk)
  {
     // RK: Non-blocking Receive of work done by worker, allowing master to do work till receive is returned
     MPI_Irecv(&loc_count, 1, MPI_INT, MPI_ANY_SOURCE, c_tag, MPI_COMM_WORLD, &request);
     // RK: Test to see whether receive is completed or not. Flag == 1 when it is completed
     MPI_Test( &request, &flag, &status);
     
     // RK: In the time taken for work to be received, master does iterations of size 1
     while (!flag)
     {           
        k0=a; 
        // testing "-1" and "+1" cases:
        for (j=-1; j<2; j=j+2)
        {
            // Prime candidate:
            x = 6*k0 + j;
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
                //  loc_count++;
                //  update the global count each time the master finds a prime 
                m_count++;
                }
         }
        
      //  RK: Can check how much prime count total is updated by master thread
      //  printf("a = %d, Master thread has done work. Global count = %d\n",a, m_count);
      // RK: Decrease each iteration by one
      a--;

      // RK: After each iteration, test to see if work from other threads have been received
      MPI_Test( &request, &flag, &status);
        
    }
    // RK: If the flag is not zero, i.e. a request has been received, then send work
    // printf ("Master successfully received local count = %d from rank %d\n",loc_count, status.MPI_SOURCE);
    // RK:  update the global count with the local count received by workers
    m_count = m_count + loc_count;
    //  printf("The updated global count is %d\n", m_count);
    // RK: Send workers the starting point a for each work parcel in order to complete iterations from a to a-dk 
    MPI_Send(&a, 1, MPI_INT, status.MPI_SOURCE, a_tag, MPI_COMM_WORLD);
    // printf("Master successfully sent a=%d to rank %d\n",a, status.MPI_SOURCE);
    // RK:  Decrease a by dk to get updated value of remaining iterations
    a -= dk;
    // printf("Master is attempting to send a=%d to rank %d\n", a, status.MPI_SOURCE);  

     }
    //  RK: Recieve the last bit of complete dk chunk from whichver worker sends it
    MPI_Recv(&loc_count, 1, MPI_INT, MPI_ANY_SOURCE, c_tag, MPI_COMM_WORLD, &status);
    // RK: Send remaining < dk iterations to this worker as it is now free 
    MPI_Send(&a, 1, MPI_INT, status.MPI_SOURCE, a_tag, MPI_COMM_WORLD);
  
   
    //  printf ("Master successfully received final local count = %d from rank %d\n",loc_count, status.MPI_SOURCE);
    // RK:  Update this final local count from to global count
     m_count = m_count + loc_count;


  // RK:  When the remaining iterations are less than chunk size dk, break out of while loop and make sure work sent from all workers is received for previous iterations
  for(i=1; i < p; i ++)
    {
       MPI_Recv(&loc_count,1, MPI_INT, MPI_ANY_SOURCE, c_tag, MPI_COMM_WORLD, &status);
       m_count += loc_count;
       loc_count = 0;
    }


    // RK: Reset remaining iterations to zero and let workers know no more work is available
    if (a - dk <0)
    {
      a = 0;
      // RK: Let all workers know there is no more work left
      for ( i = 1; i< p; i++)
          {
          MPI_Send(&a, 1, MPI_INT, i, a_tag, MPI_COMM_WORLD);
          }
    }
    else
    {
     a = a- dk;
     printf("Error: a should be zero instead of  %d\n", a);
    }



 }


// RK: for all workers
if(my_rank != 0) // 
{
// RK: While there are still jobs available
while(a >0)
{
   // printf("Rank %d is attempting to send local count=%d to master\n", my_rank, loc_count);
   // RK: Send work done for current iteration chunk dk to master and let it know worker is ready for more work 
   MPI_Send(&loc_count, 1, MPI_INT, 0, c_tag, MPI_COMM_WORLD);
   // printf("Rank %d successfully sent local count= %d to master\n", my_rank, loc_count);
   // RK: Reset the local count back to zero
   loc_count = 0;
    // printf("Rank %d is attemping to receive work from master\n",my_rank);
    // RK: Receive new iteration starting point from master
    MPI_Recv(&a, 1, MPI_INT, 0, a_tag, MPI_COMM_WORLD, &status);
    // printf("Rank %d successfully received a= %d from master\n", my_rank, a);

    // RK: Ensure that if remaining chunk size is less than dk, to stop after completing iteration KMIN
    if((a-dk) > KMIN)
    {
      b = a-dk;
    }
    else
    {
      b = KMIN-1;
    }
     // printf("rank %d is doing iterations %d to %d\n",my_rank, a, b);
      for (k = a; k > b; k--)
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
                   // To be a success, the modulus should not be equal to zero:
                    success = x % y;
                    if (!success)
                    break;
               }     

              if (success)
                {
                  loc_count++;
                }
         }
        
  
      }


  }

}





if(my_rank == 0)
{

  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);
  printf ("N_primes: %d\n",m_count);
  printf ("time: %e\n", restime);

}
  //--------------------------------------------------------------------------------


MPI_Finalize();
return 0;
}
