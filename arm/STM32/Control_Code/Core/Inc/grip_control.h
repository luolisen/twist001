#ifndef GRIP_CONTROL_H
#define GRIP_CONTROL_H
#include "grip_can.h"
#define GP_BYTES 24U
enum { GP_ENABLE=1, GP_DISABLE=2, GP_GAP=3, GP_PING=4, GP_RATE=5, GP_OPEN=6, GP_CLOSE=7 };
enum { GP_OK=0, GP_SESSION=1, GP_EXPIRED=2, GP_STALE=3, GP_CONFLICT=4, GP_DISABLED=5, GP_RANGE=6, GP_SENSOR=7, GP_OWNER=8 };
typedef struct {uint32_t session,seq,deadline;uint16_t arg;uint8_t op,origin;} GP_Command;
typedef struct {uint32_t seq;uint8_t raw[GP_BYTES],result,valid;} GP_Cache;
typedef struct {uint8_t raw[GP_BYTES],mask;uint32_t started;} GP_Assembly;
static inline void gp_encode(uint8_t *p,const GP_Command *m) {
    memset(p,0,GP_BYTES);gc_put32(p,m->session);gc_put32(p+4,m->seq);gc_put32(p+8,m->deadline);
    gc_put16(p+12,m->arg);p[14]=m->op;p[15]=(uint8_t)(2U|(m->origin<<4));gc_put16(p+16,gc_crc(p,16));
}
static inline int gp_decode(const uint8_t *p,GP_Command *m) {
    unsigned i;if((p[15]&15U)!=2U||(p[15]>>4)>2U||gc_get16(p+16)!=gc_crc(p,16))return 0;
    for(i=18;i<GP_BYTES;++i)if(p[i])return 0;
    m->session=gc_get32(p);m->seq=gc_get32(p+4);m->deadline=gc_get32(p+8);m->arg=gc_get16(p+12);m->op=p[14];m->origin=p[15]>>4;return 1;
}
static inline int gp_feed(GP_Assembly *a,unsigned part,const uint8_t *p,uint32_t now) {
    if(part>=3U)return 0;
    if(a->mask&&now-a->started>50U)a->mask=0;
    if(part==0U){a->mask=0;a->started=now;}
    if(!a->mask&&part!=0U)return 0;
    memcpy(a->raw+part*8U,p,8);a->mask|=(uint8_t)(1U<<part);
    if(a->mask==7U){a->mask=0;return 1;}return 0;
}
/* Negative: return cached/rejection result, no execution. Zero: new command. */
static inline int gp_duplicate(const GP_Cache *c,const GP_Command *m,const uint8_t *raw,uint8_t *result) {
    if(!c->valid)return 0;
    if(m->seq==c->seq){*result=memcmp(c->raw,raw,GP_BYTES)?GP_CONFLICT:c->result;return 1;}
    if(m->seq<c->seq){*result=GP_STALE;return 1;}return 0;
}
static inline void gp_cache(GP_Cache *c,const GP_Command *m,const uint8_t *raw,uint8_t result){c->seq=m->seq;memcpy(c->raw,raw,GP_BYTES);c->result=result;c->valid=1;}
static inline uint8_t gp_gate(const GP_Command *m,uint32_t session,uint32_t now,int owner,int enabled,int sensor_ok,uint16_t maxgap) {
    if(m->session!=session)return GP_SESSION;
    if(m->op==GP_DISABLE)return GP_OK;
    if(!m->deadline||m->deadline-now>500U||m->deadline==now)return GP_EXPIRED;
    if(!sensor_ok)return GP_SENSOR;
    if(enabled&&owner!=(int)m->origin)return GP_OWNER;
    switch(m->op){
      case GP_ENABLE: return GP_OK;
      case GP_PING: return enabled?GP_OK:GP_DISABLED;
      case GP_GAP: if(m->arg>maxgap)return GP_RANGE;return enabled?GP_OK:GP_DISABLED;
      case GP_RATE: if(m->arg<100U||m->arg>2000U)return GP_RANGE;return enabled?GP_OK:GP_DISABLED;
      case GP_OPEN: case GP_CLOSE: return enabled?GP_OK:GP_DISABLED;
      default:return GP_RANGE;
    }
}
static inline uint16_t gp_ack_crc(uint32_t session,const uint8_t *p){uint8_t b[10];gc_put32(b,session);memcpy(b+4,p,6);return gc_crc(b,10);}
static inline void gp_ack(uint8_t *p,const GP_Command *m,uint8_t result){gc_put32(p,m->seq);p[4]=result;p[5]=(uint8_t)(m->op|(m->origin<<6));gc_put16(p+6,gp_ack_crc(m->session,p));}
static inline int gp_ack_valid(const uint8_t *p,const GP_Command *m){return gc_get32(p)==m->seq&&p[5]==(uint8_t)(m->op|(m->origin<<6))&&gc_get16(p+6)==gp_ack_crc(m->session,p);}
#endif
