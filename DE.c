/***************Differential Evolution Algorithm*************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define nVar 10  // Number of variables
#define NP 50    // Number of Population
#define Iter 200 // Maximum number of iterations

struct Individual
{
    double position[nVar];
    double fitness;
};

struct DE
{
    struct Individual X[NP];
    double F;
    double CR;
    double (*function)(double *); // Function pointer definition
    double global_fitness;
    double global_solution[nVar];
    double x_low_bound;
    double x_up_bound;
};

// sum(X^2)
double function_fitness(double *var)
{
    double result = 0;
    for (int i = 0; i < nVar; i++)
    {
        // result += *(var + i) * *(var + i); // method one : pointer
        result += var[i] * var[i];
                                            //result += var[i] * var[i];     // method two : array
    }
    return result;
}

// run DE
void runDE(double F, double CR, double x_low_bound, double x_up_bound, double (*function_fitness)(double *))
{

    struct DE de;
    de.F = F;
    de.CR = CR;
    de.x_low_bound = x_low_bound;
    de.x_up_bound = x_up_bound;
    de.function = function_fitness;

    srand((unsigned)time(NULL));

    // inital population
    for (int i = 0; i < NP; i++)
    {
        for (int j = 0; j < nVar; j++)
        {
            double randx = (double)rand() / RAND_MAX;
            de.X[i].position[j] = de.x_low_bound + (de.x_up_bound - de.x_low_bound) * randx;
        }
        de.X[i].fitness = de.function(de.X[i].position);
    }

    memcpy(&de.global_solution, &de.X[0].position, sizeof(double) * nVar);
    de.global_fitness = de.X[0].fitness;
    struct Individual mutate[NP];
    struct Individual crossover[NP];
    /* End of initialization */

    for (int iter = 1; iter <= Iter; iter++)
    {
        // mutate
        for (int i = 0; i < NP; i++)
        {
            // r1,r2,r3 random and different
            int r1 = (int)rand() % NP;
            int r2 = (int)rand() % NP;
            int r3 = (int)rand() % NP;
            if (r1 == i || r2 == i || r3 == i || r1 == r2 || r2 == r3)
            {
                r1 = (int)rand() % NP;
                r2 = (int)rand() % NP;
                r3 = (int)rand() % NP;
            }
            for (int j = 0; j < nVar; j++)
            {
                mutate[i].position[j] = de.X[r1].position[j] + F * (de.X[r2].position[j] - de.X[r3].position[j]);
                // up and low bound
                if (mutate[i].position[j] > de.x_up_bound)
                {
                    mutate[i].position[j] = de.x_up_bound;
                }
                if (mutate[i].position[j] < de.x_low_bound)
                {
                    mutate[i].position[j] = de.x_low_bound;
                }
            }
        }

        //crossover
        for (int i = 0; i < NP; i++)
        {
            int randc = (int)rand() % nVar;
            for (int j = 0; j < nVar; j++)
            {
                double rand_cr = (double)rand() / RAND_MAX;
                if ((j == randc) || (rand_cr <= de.CR))
                {
                    crossover[i].position[j] = mutate[i].position[j];
                }
                else
                {
                    crossover[i].position[j] = de.X[i].position[j];
                }
                // limit of variables of bound
                if (crossover[i].position[j] > de.x_up_bound)
                {
                    crossover[i].position[j] = de.x_up_bound;
                }
                if (crossover[i].position[j] < de.x_low_bound)
                {
                    crossover[i].position[j] = de.x_low_bound;
                }
            }
            crossover[i].fitness = de.function(crossover[i].position);
        }

        // select
        for (int i = 0; i < NP; i++)
        {
            if (crossover[i].fitness < de.X[i].fitness)
            {
                de.X[i].fitness = crossover[i].fitness;
                memcpy(&de.X[i].position, &crossover[i].position, sizeof(double) * nVar);

                if (de.X[i].fitness < de.global_fitness)
                {
                    de.global_fitness = de.X[i].fitness;
                    memcpy(&de.global_solution, &de.X[i].position, sizeof(double) * nVar);
                }
            }
        }
        // 一次迭代结束
        printf("This is %d iteration, globalfitness is %f\n ", iter, de.global_fitness);
    }
    printf("The iteration is end, show the soultion and fitness!\n");
    printf("global_fitness :%f\n", de.global_fitness);
    for (int i = 0; i < nVar; i++)
    {
        printf("%f,", de.global_solution[i]);
    }
}

void main(void)
{ // Test the running time of the program . including time.h
    printf("the program is running!\n");
    clock_t pro_start, pro_finished;
    pro_start = clock();
    runDE(0.5, 0.4, -10.0, 10.0, function_fitness);
    pro_finished = clock();
    double Total_time = (double)(pro_finished - pro_start) / CLOCKS_PER_SEC;
    printf("\n");
    printf("The program ran for : %f second\n", Total_time);
    // system("pause");
}