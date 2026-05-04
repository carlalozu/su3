
#define PLAQ_SUM_C

#include "lattice.h"
#include "su3prod.h"
#include "uflds.h"
#include "utils.h"
#include "global.h"

static su3_dble *udb;

#pragma omp declare target
static double plaq_dble(su3_dble *udb, int mu, int nu, int ix)
{
   int ip[4];
   su3_dble wd1, wd2;

   plaq_uidx(mu, nu, ix, ip);

   su3xsu3     (&wd1, udb+ip[0], udb+ip[1]);
   su3dagxsu3dag(&wd2, udb+ip[3], udb+ip[2]);

   return cm3x3_retr(&wd1, &wd2);
}
#pragma omp end declare target


static qflt local_plaq_sum_dble(int iw)
{
   int bc;
   double wp,pa=0.0;
   qflt rqsm;

   bc=bc_type();

   if (iw==0)
      wp=1.0;
   else
      wp=0.5;

   rqsm.q[0]=0.0;
   rqsm.q[1]=0.0;
   udb=udfld();
   // #pragma omp parallel private(k,ix,t,n,pa) reduction(sum_qflt : rqsm)
   #pragma omp target teams distribute parallel for reduction(+:pa)
   for (int ix=0;ix<VOLUME;ix++){
      for (int mu = 0; mu < 4; mu++) {
         for (int nu = mu+1; nu < 4; nu++) {
            double local_pa=0.0;
            int t=global_time(ix);

            if (mu<1)
            {
               if ((t<(N0-1))||(bc!=0))
                  local_pa+=plaq_dble(udb,mu,nu,ix);
            }
            else
            {
               if (((t>0)&&(t<(N0-1)))||(bc==3))
                  local_pa+=plaq_dble(udb,mu,nu,ix);
               else if ((t==0)||(bc==0))
               {
                  if (bc==1)
                     local_pa+=wp*3.0;
                  else
                     local_pa+=wp*plaq_dble(udb,mu,nu,ix);
               }
               else
               {
                  local_pa+=plaq_dble(udb,mu,nu,ix);
                  local_pa+=wp*3.0;
               }
            }
            pa+=local_pa;
         }
      }
   }
   #pragma omp target update from(pa)
   prof_end(&compute);
   acc_qflt(pa,rqsm.q);
   return rqsm;
}

/*
 * plaq_sum_dble(icom)
 *
 * Returns the sum of Re Tr[plaquette] over all sites and orientations.
 * icom is ignored (no MPI); present for openQCD API compatibility.
 */
double plaq_sum_dble(int icom)
{
   qflt rqsm;

   prof_begin(&compute);
   rqsm=local_plaq_sum_dble(0);
   return rqsm.q[0]+rqsm.q[1];
}
