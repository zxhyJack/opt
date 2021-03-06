/***************Differential Evolution Algorithm*************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define D 10     // Number of variables
#define NP 50    // Number of Population
#define Iter 300 // Maximum number of iterations

struct DE
{
    double X[NP][D];
    double F;
    double CR;
    double (*function)(double *); // Function pointer definition
    double global_fitness;
    double global_solution[D];
    double x_low_bound;
    double x_up_bound;
};

// sum(X^2)
double calculate_fitness(double params[D])
{
    double result = 0;
    for (int i = 0; i < D; i++)
    {
        result += *(params + i) * *(params + i); // method one : pointer
        // result += var[i] * var[i];     // method two : array
    }
    return result;
}

// run DE
void runDE(double F, double CR, double x_low_bound, double x_up_bound, double (*calculate_fitness)(double *))
{
    struct DE de;
    de.F = F;
    de.CR = CR;
    de.x_low_bound = x_low_bound;
    de.x_up_bound = x_up_bound;
    de.function = calculate_fitness;

    srand((unsigned)time(NULL));

    // inital population
    for (int i = 0; i < NP; i++)
    {
        for (int j = 0; j < D; j++)
        {
            double randx = (double)rand() / RAND_MAX;
            de.X[i][j] = de.x_low_bound + (de.x_up_bound - de.x_low_bound) * randx;
        }
    }
    memcpy(&de.global_solution, &de.X[0], sizeof(double) * D);
    de.global_fitness = calculate_fitness(de.X[0]);
    double V[NP][D];
    double U[NP][D];
    /* End of initialization */

    for (int iter = 1; iter <= Iter; iter++)
    {
        // mutate
        for (int i = 0; i < NP; i++)
        {
            int r1, r2, r3;
            // r1,r2,r3 random and different
            do
            {
                r1 = (int)rand() % NP;
            } while (r1 == i);
            do
            {
                r2 = (int)rand() % NP;
            } while (r2 == i || r2 == r1);
            do
            {
                r3 = (int)rand() % NP;
            } while (r3 == i || r3 == r1 || r3 == r2);
            // do
            // {
            //     r1 = (int)rand() % NP;
            //     r2 = (int)rand() % NP;
            //     r3 = (int)rand() % NP;
            // } while (r1 == i || r2 == i || r3 == i || r1 == r2 || r2 == r3 || r1 == r3);
            for (int j = 0; j < D; j++)
            {
                V[i][j] = de.X[r1][j] + F * (de.X[r2][j] - de.X[r3][j]);
                // limit of bound
                V[i][j] = V[i][j] > de.x_up_bound ? de.x_up_bound : V[i][j];
                V[i][j] = V[i][j] < de.x_low_bound ? de.x_low_bound : V[i][j];
            }
        }

        //crossover
        for (int i = 0; i < NP; i++)
        {
            int randc = (int)rand() % D;
            for (int j = 0; j < D; j++)
            {
                double rand_cr = (double)rand() / RAND_MAX;
                U[i][j] = ((j == randc) || (rand_cr <= de.CR)) ? V[i][j] : de.X[i][j];
                // limit of bound
                U[i][j] = U[i][j] > de.x_up_bound ? de.x_up_bound : U[i][j];
                U[i][j] = U[i][j] < de.x_low_bound ? de.x_low_bound : U[i][j];
            }
        }

        // select
        for (int i = 0; i < NP; i++)
        {
            double x_fitness = calculate_fitness(de.X[i]);
            double u_fitness = calculate_fitness(U[i]);
            if (u_fitness < x_fitness)
            {
                // memcpy(&de.X[i], &U[i], sizeof(double) * D);
                for (int j = 0; j < D; j++)
                {
                    de.X[i][j] = U[i][j];
                }
            }
            x_fitness = calculate_fitness(de.X[i]);
            if (x_fitness < de.global_fitness)
            {
                de.global_fitness = x_fitness;
                // memcpy(&de.global_solution, &de.X[i], sizeof(double) * D);
                for (int j = 0; j < D; j++)
                    de.global_solution[j] = de.X[i][j];
            }
        }

        printf("This is %d iteration, globalfitness is %f\n ", iter, de.global_fitness);
    }
    printf("The iteration is end, show the soultion and fitness!\n");
    printf("global_fitness :%f\n", de.global_fitness);
    for (int i = 0; i < D; i++)
    {
        printf("%f,", de.global_solution[i]);
    }
}

void main(void)
{ // Test the running time of the program . including time.h
    printf("the program is running!\n");
    clock_t start, finished;
    start = clock();
    runDE(0.5, 0.4, -10.0, 10.0, calculate_fitness);
    finished = clock();
    double Total_time = (double)(finished - start) / CLOCKS_PER_SEC;
    // printf("\n");
    puts("");
    printf("The program ran for : %f second\n", Total_time);
    // system("pause");
}