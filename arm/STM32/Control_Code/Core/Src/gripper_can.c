#include "gripper_can.h"
#include "grip_control.h"
#include "can.h"
#include "usart.h"
#include <stdio.h>
#include <string.h>
#include <limits.h>
static uint32_t sequence,command_sequence;
static unsigned retries,invalid,stale,timeouts;
static uint8_t state[GC_BYTES];
static void host(const char *s){(void)HAL_UART_Transmit(&huart1,(uint8_t *)s,(uint16_t)strlen(s),100U);}
static int receive(CAN_RxHeaderTypeDef *rx,uint8_t *data){return HAL_CAN_GetRxFifoFillLevel(&hcan2,CAN_RX_FIFO1)&&HAL_CAN_GetRxMessage(&hcan2,CAN_RX_FIFO1,rx,data)==HAL_OK;}
static int drain(void){CAN_RxHeaderTypeDef rx;uint8_t d[8];unsigned n;for(n=0;n<32U&&receive(&rx,d);++n){}return n<32U;}
static int tx_frame(uint32_t id,uint8_t *data,uint32_t *mailbox){CAN_TxHeaderTypeDef tx={0};tx.IDE=CAN_ID_EXT;tx.RTR=CAN_RTR_DATA;tx.ExtId=id;tx.DLC=8;return HAL_CAN_AddTxMessage(&hcan2,&tx,data,mailbox)==HAL_OK;}
static int fetch(void){
 uint8_t req[8],data[8];uint16_t mask=0;CAN_RxHeaderTypeDef rx;uint32_t seq=++sequence,mailbox,start;unsigned attempt;
 if(!seq)seq=++sequence;
 if(!drain()){host("GRIP_CAN_RX_BUSY feedback_invalid=1\r\n");return 0;}
 gc_request(req,seq);
 for(attempt=0;attempt<3U;++attempt){
  if(attempt)++retries;
  if(!tx_frame(GC_REQUEST_ID,req,&mailbox))continue;
  start=HAL_GetTick();
  while(HAL_GetTick()-start<100U){
   if(!receive(&rx,data))continue;
   if(rx.IDE!=CAN_ID_EXT||rx.RTR!=CAN_RTR_DATA||rx.DLC!=8||rx.ExtId<GC_REPLY_ID||rx.ExtId>=GC_REPLY_ID+GC_PARTS){++invalid;continue;}
   if(gc_get32(data)!=seq){++stale;continue;}
   unsigned part=(unsigned)(rx.ExtId-GC_REPLY_ID);memcpy(state+4U*part,data+4,4);mask|=(uint16_t)(1U<<part);
   if(mask!=((1U<<GC_PARTS)-1U))continue;
   if(!gc_valid(seq,state)){++invalid;mask=0;continue;}return 1;
  }
  (void)HAL_CAN_AbortTxRequest(&hcan2,mailbox);
 }
 ++timeouts;host("GRIP_CAN_TIMEOUT feedback_invalid=1 no automatic motion\r\n");return 0;
}
void Gripper_QueryCAN(void){
 char text[450];if(!fetch())return;
 (void)snprintf(text,sizeof(text),"GRIP_CAN_STATUS seq=%lu boot=%u angle_mrad=%ld target_mrad=%ld gap_cmm=%u enabled=%u motion=%u arrived=%u boundary=%u i2c_error=%u control_max_period_us=%lu max_gap_cmm=%u session=%lu uptime_ms=%lu cmd_seq=%lu cmd_result=%u fault=%u retries=%u invalid=%u stale=%u timeouts=%u\r\n",
 (unsigned long)sequence,gc_get16(state+16),(long)(int32_t)gc_get32(state),(long)(int32_t)gc_get32(state+4),gc_get16(state+8),state[10]&1U,(state[10]>>1)&1U,(state[10]>>5)&1U,(state[10]>>2)&7U,state[11],(unsigned long)gc_get32(state+12),gc_get16(state+18),(unsigned long)gc_get32(state+20),(unsigned long)gc_get32(state+24),(unsigned long)gc_get32(state+28),state[32],state[33],retries,invalid,stale,timeouts);host(text);
}
static int number(const char *p,uint16_t *n){uint32_t v=0;if(!*p)return 0;while(*p){if(*p<'0'||*p>'9')return 0;v=v*10U+(unsigned)(*p++-'0');if(v>65535U)return 0;}*n=(uint16_t)v;return 1;}
int Gripper_CANCommand(const char *line){
 GP_Command m={0};uint8_t raw[GP_BYTES],data[8];CAN_RxHeaderTypeDef rx;uint32_t mb[3],start;unsigned attempt,part;char text[120];
 if(!strcmp(line,"ENABLE"))m.op=GP_ENABLE;
 else if(!strcmp(line,"DISABLE"))m.op=GP_DISABLE;
 else if(!strcmp(line,"PING"))m.op=GP_PING;
 else if(!strcmp(line,"OPEN"))m.op=GP_OPEN;
 else if(!strcmp(line,"CLOSE"))m.op=GP_CLOSE;
 else if(!strncmp(line,"GAP ",4)&&number(line+4,&m.arg))m.op=GP_GAP;
 else if(!strncmp(line,"RATE ",5)&&number(line+5,&m.arg))m.op=GP_RATE;
 else return 0;
 if(!fetch())return 1;
 m.session=gc_get32(state+20);m.deadline=gc_get32(state+24)+400U;
 uint32_t last=gc_get32(state+28);if(last>command_sequence)command_sequence=last;
 if(command_sequence==UINT32_MAX){host("GRIP_COMMAND_REJECT sequence exhausted\r\n");return 1;}
 m.seq=++command_sequence;m.origin=0;gp_encode(raw,&m);
 for(attempt=0;attempt<3U;++attempt){
  for(part=0;part<3U;++part){
   if(!tx_frame(GC_CONTROL_ID+part,raw+8U*part,&mb[part]))break;
  }
  if(part!=3U){for(unsigned j=0;j<part;++j)(void)HAL_CAN_AbortTxRequest(&hcan2,mb[j]);continue;}
  start=HAL_GetTick();
  while(HAL_GetTick()-start<50U){
   if(!receive(&rx,data))continue;
   if(rx.IDE!=CAN_ID_EXT||rx.RTR!=CAN_RTR_DATA||rx.DLC!=8||rx.ExtId!=GC_ACK_ID||!gp_ack_valid(data,&m))continue;
   (void)snprintf(text,sizeof(text),"GRIP_COMMAND_ACK cmd_seq=%lu op=%u result=%u accepted=%u\r\n",(unsigned long)m.seq,m.op,data[4],data[4]==GP_OK);host(text);return 1;
  }
  for(part=0;part<3U;++part)(void)HAL_CAN_AbortTxRequest(&hcan2,mb[part]);
 }
 host("GRIP_COMMAND_TIMEOUT outcome_unknown do not assume accepted\r\n");return 1;
}
