#include "gripper_can.h"
#include "grip_can.h"
#include "can.h"
#include "usart.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
CAN_HandleTypeDef hcan2;
UART_HandleTypeDef huart1,huart3;
static uint32_t tick,seq;
static unsigned sends,part,mode;
static uint8_t snapshot[GC_BYTES];
static char output[2000];
uint32_t HAL_GetTick(void){return tick++;}
int HAL_UART_Transmit(UART_HandleTypeDef *h,uint8_t *p,uint16_t n,uint32_t timeout){
    (void)h;(void)timeout;size_t l=strlen(output);assert(l+n<sizeof(output));memcpy(output+l,p,n);output[l+n]=0;return HAL_OK;
}
uint32_t HAL_CAN_GetRxFifoFillLevel(CAN_HandleTypeDef *h,uint32_t fifo){
    (void)h;assert(fifo==CAN_RX_FIFO1);return sends&&part<GC_PARTS&&mode!=3;
}
int HAL_CAN_AddTxMessage(CAN_HandleTypeDef *h,CAN_TxHeaderTypeDef *tx,uint8_t *p,uint32_t *mailbox){
    (void)h;assert(tx->ExtId==GC_REQUEST_ID&&tx->IDE==CAN_ID_EXT&&tx->DLC==8&&gc_request_valid(p));
    uint32_t next=gc_get32(p+2);if(sends)assert(next==seq);seq=next;++sends;part=0;*mailbox=1;
    gc_put32(snapshot,(uint32_t)(int32_t)-3600);gc_put32(snapshot+4,(uint32_t)(int32_t)-3601);
    gc_put16(snapshot+8,1527);snapshot[10]=1;snapshot[11]=0;gc_put32(snapshot+12,2400);gc_put16(snapshot+16,42);gc_seal(seq,snapshot);
    if(mode==2&&sends==1)snapshot[2]^=1;return HAL_OK;
}
int HAL_CAN_GetRxMessage(CAN_HandleTypeDef *h,uint32_t fifo,CAN_RxHeaderTypeDef *rx,uint8_t *p){
    (void)h;assert(fifo==CAN_RX_FIFO1);unsigned index=GC_PARTS-1-part;memset(rx,0,sizeof(*rx));rx->ExtId=GC_REPLY_ID+index;rx->IDE=CAN_ID_EXT;rx->DLC=8;
    gc_put32(p,mode==4&&sends==1?seq-1:seq);memcpy(p+4,snapshot+4*index,4);++part;
    if(mode==1&&sends==1&&part==GC_PARTS-1)part=GC_PARTS; /* one fragment lost */
    return HAL_OK;
}
int HAL_CAN_AbortTxRequest(CAN_HandleTypeDef *h,uint32_t mailbox){(void)h;(void)mailbox;return HAL_OK;}
static void run(unsigned m){tick=sends=part=0;mode=m;output[0]=0;Gripper_QueryCAN();}
int main(void){
    const uint8_t vector[]="123456789";assert(gc_crc(vector,9)==0x29b1);
    run(0);assert(sends==1&&strstr(output,"GRIP_CAN_STATUS")&&strstr(output,"angle_mrad=-3600"));
    run(1);assert(sends==2&&strstr(output,"GRIP_CAN_STATUS"));
    run(2);assert(sends==2&&strstr(output,"GRIP_CAN_STATUS"));
    run(3);assert(sends==3&&strstr(output,"GRIP_CAN_TIMEOUT"));
    run(4);assert(sends==2&&strstr(output,"GRIP_CAN_STATUS"));
    puts("CAN link: CRC vector, reordered fragments, lost fragment, corrupt snapshot, disconnect, stale sequence passed");
}
