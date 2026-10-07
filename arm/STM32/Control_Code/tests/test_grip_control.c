#include "grip_control.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
 GP_Command m={123,1,1400,200,GP_GAP,0},decoded; uint8_t raw[GP_BYTES],ack[8],result;
 GP_Cache cache={0};GP_Assembly assembly={0};
 gp_encode(raw,&m);assert(gp_decode(raw,&decoded));
 assert(gp_gate(&m,123,1000,0,1,1,7463)==GP_OK);
 assert(gp_gate(&m,124,1000,0,1,1,7463)==GP_SESSION);
 assert(gp_gate(&m,123,1401,0,1,1,7463)==GP_EXPIRED);
 assert(gp_gate(&m,123,1000,-1,1,1,7463)==GP_OWNER);
 assert(gp_gate(&m,123,1000,0,0,1,7463)==GP_DISABLED);
 assert(gp_gate(&m,123,1000,0,1,0,7463)==GP_SENSOR);
 m.arg=8000;assert(gp_gate(&m,123,1000,0,1,1,7463)==GP_RANGE);m.arg=200;
 assert(!gp_duplicate(&cache,&m,raw,&result));gp_cache(&cache,&m,raw,GP_OK);
 assert(gp_duplicate(&cache,&m,raw,&result)&&result==GP_OK);
 raw[12]^=1;assert(!gp_decode(raw,&decoded));
 assert(gp_duplicate(&cache,&m,raw,&result)&&result==GP_CONFLICT);gp_encode(raw,&m);
 --m.seq;assert(gp_duplicate(&cache,&m,raw,&result)&&result==GP_STALE);++m.seq;
 gp_ack(ack,&m,GP_OK);assert(gp_ack_valid(ack,&m));ack[4]^=1;assert(!gp_ack_valid(ack,&m));
 assert(!gp_feed(&assembly,1,raw+8,1000));
 assert(!gp_feed(&assembly,0,raw,1000));assert(!gp_feed(&assembly,1,raw+8,1001));
 assert(gp_feed(&assembly,2,raw+16,1002));assert(gp_decode(assembly.raw,&decoded));
 assert(!gp_feed(&assembly,0,raw,1000));assert(!gp_feed(&assembly,2,raw+16,1051));
 m.op=GP_DISABLE;m.deadline=0;assert(gp_gate(&m,123,2000,-1,0,0,7463)==GP_OK);
 assert(gp_gate(&m,124,2000,-1,0,0,7463)==GP_SESSION);
 m.op=GP_GAP;m.deadline=2400;m.origin=1;assert(gp_gate(&m,123,2000,1,1,1,7463)==GP_OK);assert(gp_gate(&m,123,2000,0,1,1,7463)==GP_OWNER);
 m.origin=2;assert(gp_gate(&m,123,2000,2,1,1,7463)==GP_OK);
 puts("gripper control: CRC, expiry, reset session, ownership, disabled/sensor/range gates, duplicate conflict and fragment timeout passed");
}
