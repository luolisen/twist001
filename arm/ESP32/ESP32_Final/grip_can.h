#ifndef GRIP_CAN_H
#define GRIP_CAN_H
#include <stdint.h>
#include <string.h>
#define GC_NODE_ID 7U
#define GC_REQUEST_ID ((uint32_t)GC_NODE_ID << 8)
#define GC_REPLY_ID (GC_REQUEST_ID + 1U)
#define GC_PARTS 9U
#define GC_BYTES 36U
#define GC_CONTROL_ID (GC_REQUEST_ID + 0x10U)
#define GC_ACK_ID (GC_REQUEST_ID + 0x13U)
static inline uint16_t gc_crc(const uint8_t *p,unsigned n) {
    uint16_t c=0xffffU;unsigned i,b;
    for(i=0;i<n;++i){c^=(uint16_t)p[i]<<8;for(b=0;b<8;++b)c=(uint16_t)((c&0x8000U)?((uint32_t)c<<1)^0x1021U:((uint32_t)c<<1));}return c;
}
static inline void gc_put32(uint8_t *p,uint32_t x){p[0]=(uint8_t)x;p[1]=(uint8_t)(x>>8);p[2]=(uint8_t)(x>>16);p[3]=(uint8_t)(x>>24);}
static inline uint32_t gc_get32(const uint8_t *p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static inline void gc_put16(uint8_t *p,uint16_t x){p[0]=(uint8_t)x;p[1]=(uint8_t)(x>>8);}
static inline uint16_t gc_get16(const uint8_t *p){return (uint16_t)((uint16_t)p[0]|((uint16_t)p[1]<<8));}
static inline void gc_request(uint8_t *p,uint32_t seq){p[0]=0xA7;p[1]=1;gc_put32(p+2,seq);gc_put16(p+6,gc_crc(p,6));}
static inline int gc_request_valid(const uint8_t *p){return p[0]==0xA7&&p[1]==1&&gc_get16(p+6)==gc_crc(p,6);}
/* Whole-snapshot CRC also binds all five fragments to the request sequence. */
static inline uint16_t gc_snapshot_crc(uint32_t seq,const uint8_t *p){uint8_t b[GC_BYTES+2U];gc_put32(b,seq);memcpy(b+4,p,GC_BYTES-2U);return gc_crc(b,GC_BYTES+2U);}
static inline void gc_seal(uint32_t seq,uint8_t *p){gc_put16(p+GC_BYTES-2U,gc_snapshot_crc(seq,p));}
static inline int gc_valid(uint32_t seq,const uint8_t *p){return gc_get16(p+GC_BYTES-2U)==gc_snapshot_crc(seq,p);}
#endif
