#include "grip_motion.h"
#include <assert.h>
#include <stdio.h>
int main(void){
 float x=2,v=0,old;unsigned i;
 for(i=0;i<200;++i){old=v;gm_step(&x,&v,3.5,0,4,2,.002f);assert(fabsf(v)<=GM_MAX_SPEED+.00001f);assert(fabsf(v-old)<=GM_ACCELERATION*.002f+.00001f);assert(x>=0&&x<=4);}
 assert(v>0);
 for(i=0;i<500;++i){old=v;gm_step(&x,&v,1,0,4,2,.002f);assert(fabsf(v-old)<=GM_ACCELERATION*.002f+.00001f);assert(x>=0&&x<=4);}
 assert(v<0);
 for(i=0;i<2000;++i){gm_step(&x,&v,0,0,4,2,.002f);assert(x>=0&&x<=4);}
 assert(x<.001f);
 x=3.99f;v=0;for(i=0;i<100;++i){gm_step(&x,&v,4,0,4,2,.002f);assert(fabsf(v)<.15f);}
 puts("trajectory limits: acceleration, symmetric speed, reversal and endpoint bounds passed");
}
