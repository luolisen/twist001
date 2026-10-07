#ifndef GRIP_MOTION_H
#define GRIP_MOTION_H
#include <math.h>
#define GM_ACCELERATION 1.5f
#define GM_MAX_SPEED (25.0f / 18.0f)
#define GM_END_MIN_SPEED 0.1f
#define GM_END_BAND_MM 10.0f
/* Bounds apply to the trajectory, not a certification of measured motor speed. */
static inline float gm_towards(float x,float y,float step) {
 return x<y?fminf(x+step,y):fmaxf(x-step,y);
}
static inline void gm_step(float *position,float *velocity,float target,float low,float high,float requested,float dt) {
 const float acceleration=GM_ACCELERATION,maximum=GM_MAX_SPEED,minimum=GM_END_MIN_SPEED;
 if(!(dt>0.0f)||dt>0.01f||!isfinite(target)||!isfinite(*position)||!isfinite(*velocity))return;
 target=fminf(high,fmaxf(low,target));
 const float distance=target-*position;
 const float endpoint_distance=fmaxf(0.0f,fminf(*position-low,high-*position));
 const float end_fraction=fminf(1.0f,endpoint_distance/(GM_END_BAND_MM/18.0f));
 const float endpoint_limit=minimum+(maximum-minimum)*end_fraction;
 const float limit=fminf(fmaxf(0.0f,requested),fminf(maximum,endpoint_limit));
 const float braking_limit=sqrtf(2.0f*acceleration*fabsf(distance));
 const float desired=(distance>=0.0f?1.0f:-1.0f)*fminf(limit,braking_limit);
 const float next=gm_towards(*velocity,desired,acceleration*dt);
 float advanced=*position+0.5f*(*velocity+next)*dt;
 *velocity=next;
 /* While reversing, deceleration may continue briefly in the old direction. */
 if(advanced<low){advanced=low;*velocity=0.0f;}
 if(advanced>high){advanced=high;*velocity=0.0f;}
 if(fabsf(distance)<0.0001f && fabsf(next)<acceleration*dt){advanced=target;*velocity=0.0f;}
 *position=advanced;
}
#endif
