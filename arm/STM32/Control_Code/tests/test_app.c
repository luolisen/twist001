#include "can.h"
#include "usart.h"
#include "readcan_app.h"
#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <string.h>
static CAN_Regs regs;
CAN_HandleTypeDef hcan2={&regs};
UART_HandleTypeDef huart1;
UART_HandleTypeDef huart3;
static jmp_buf done;
static const char *input;
static char output[20000];
static uint32_t tick,tx_count,aborts,grip_tx;
static int reply,pending,idle_sim,encoder_case;
static uint16_t encoder_value=19190;
static uint8_t reply_axis,reply_code,release_count,enable_count,enabled[6];
static int32_t simulated_position[6],cached_position[6];
static uint8_t fb_body[6][8];
static uint32_t fb_packets;
uint32_t HAL_GetTick(void) {return tick++;}
void HAL_Delay(uint32_t ms) {tick+=ms;}
uint32_t HAL_CAN_GetError(CAN_HandleTypeDef *h) {(void)h;return 0;}
uint32_t HAL_CAN_GetRxFifoFillLevel(CAN_HandleTypeDef *h,uint32_t fifo)
{(void)h;(void)fifo;return reply_axis!=0;}
int HAL_CAN_GetRxMessage(CAN_HandleTypeDef *h,uint32_t fifo,CAN_RxHeaderTypeDef *r,uint8_t *p)
{
    (void)h;(void)fifo;memset(r,0,sizeof(*r));
    r->IDE=CAN_ID_EXT;r->ExtId=(uint32_t)reply_axis<<8;
    if(reply_code==0x31) {
        r->DLC=4;p[0]=0x31;p[1]=encoder_value>>8;p[2]=encoder_value;p[3]=0x6B;
        switch(encoder_case) {
            case 1:r->ExtId=7U<<8;break;
            case 2:r->IDE=CAN_ID_STD;break;
            case 3:r->RTR=CAN_RTR_REMOTE;break;
            case 4:r->DLC=3;break;
            case 5:p[0]=0x36;break;
            case 6:p[3]=0;break;
        }
    }
    else if(reply_code==0x36) {uint32_t v=simulated_position[reply_axis-1];r->DLC=7;p[0]=0x36;p[1]=0;p[2]=v>>24;p[3]=v>>16;p[4]=v>>8;p[5]=v;p[6]=0x6B;}
    else if(reply_code==0x1A) {r->DLC=4;p[0]=0x1A;p[1]=0x04;p[2]=0x24;p[3]=0x6B;}
    else {r->DLC=3;p[0]=reply_code;p[1]=reply_code==0x3A ? enabled[reply_axis-1] : (reply_code==0x1A ? 0x04 : (reply==2 ? 0xE2 : 0x02));p[2]=0x6B;}
    reply_axis=0;return HAL_OK;
}
int HAL_CAN_AddTxMessage(CAN_HandleTypeDef *h,CAN_TxHeaderTypeDef *r,uint8_t *p,uint32_t *m)
{
    (void)h;
    assert(r->IDE==CAN_ID_EXT && r->RTR==CAN_RTR_DATA);
    uint8_t axis=(uint8_t)(r->ExtId>>8),packet=(uint8_t)r->ExtId;
    if(p[0]==0xFF) {assert(axis==0 && packet==0 && r->DLC==3 && p[1]==0x66 && p[2]==0x6B);memcpy(simulated_position,cached_position,sizeof(simulated_position));++tx_count;*m=1;return HAL_OK;}
    assert(axis>=1 && axis<=6);
    if(p[0]==0xFB) {
        if(packet==0) {assert(r->DLC==8);memcpy(fb_body[axis-1],p,8);++fb_packets;}
        else {assert(packet==1 && r->DLC==4 && p[1]==1 && p[2]<=1 && p[3]==0x6B);uint8_t *b=fb_body[axis-1];assert(b[1]<=1);uint16_t speed=((uint16_t)b[2]<<8)|b[3];assert(speed<=1750 && speed>0);uint32_t v=((uint32_t)b[4]<<24)|((uint32_t)b[5]<<16)|((uint32_t)b[6]<<8)|b[7];cached_position[axis-1]=b[1]?-(int32_t)v:(int32_t)v;if(!p[2])simulated_position[axis-1]=cached_position[axis-1];if(reply){reply_axis=axis;reply_code=0xFB;}}
    } else {
        assert(packet==0);
        if(p[0]==0x31 || p[0]==0x36 || p[0]==0x3A || p[0]==0x1A) {assert(r->DLC==2 && p[1]==0x6B);}
        else if(p[0]==0xFE) {assert(r->DLC==4 && p[1]==0x98 && p[2]==0 && p[3]==0x6B);}
        else {assert(p[0]==0xF3 && r->DLC==5 && p[1]==0xAB && p[2]<=1 && p[3]==0 && p[4]==0x6B);if(p[2])++enable_count;else ++release_count;if(reply!=2)enabled[axis-1]=p[2];}
        if(reply && p[0]!=0xFE) {reply_axis=axis;reply_code=p[0];}
    }
    ++tx_count;*m=1;
    return HAL_OK;
}
uint32_t HAL_CAN_IsTxMessagePending(CAN_HandleTypeDef *h,uint32_t m) {(void)h;(void)m;return pending;}
int HAL_CAN_AbortTxRequest(CAN_HandleTypeDef *h,uint32_t m) {(void)h;(void)m;++aborts;return HAL_OK;}
int HAL_UART_Transmit(UART_HandleTypeDef *h,uint8_t *p,uint16_t n,uint32_t timeout)
{
    (void)timeout;
    if(h==&huart3) { assert(n==2 && memcmp(p,"S\n",2)==0); ++grip_tx; return HAL_OK; }
    size_t len=strlen(output);assert(len+n<sizeof(output));
    memcpy(output+len,p,n);output[len+n]=0;return HAL_OK;
}
int HAL_UART_Receive(UART_HandleTypeDef *h,uint8_t *p,uint16_t n,uint32_t timeout)
{(void)n;(void)timeout;if(h==&huart3) return HAL_TIMEOUT;if(!*input) {if(idle_sim && !strstr(output,"PLAY_WATCHDOG")){tick+=500;return HAL_TIMEOUT;}longjmp(done,1);} *p=(uint8_t)*input++;return HAL_OK;}
static void run(const char *commands,int responses,int tx_pending)
{
    memset(enabled,1,sizeof(enabled));release_count=enable_count=0;fb_packets=0;for(int i=0;i<6;++i)simulated_position[i]=cached_position[i]=1971;input=commands;output[0]=0;grip_tx=0;tx_count=0;aborts=0;reply_axis=0;reply=responses;pending=tx_pending;
    if(setjmp(done)==0) {ReadCAN_Run();}
}
int main(void)
{
    tick=0;run("ID?\nEF\nREAD 0\nREAD 7\nA1,2,3,4,5,6B\n",1,0);
    assert(tx_count==0 && strstr(output,"UNHOMED") && strstr(output,"REJECT"));
    run("SCAN\n",1,0);assert(tx_count==6 && strstr(output,"POS J6 motor_deg=197.1") && strstr(output,"SCAN_DONE"));
    run("READ 1\nSTATE\n",0,0);assert(tx_count==1 && strstr(output,"REPLY_TIMEOUT J1"));
    assert(strstr(output,"valid=0 fresh=0 model=UNHOMED"));
    run("READ 1\n",0,1);assert(aborts==1 && strstr(output,"TX_TIMEOUT J1"));
    run("READ 1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\nLISTEN\n",1,0);
    assert(tx_count==0 && strstr(output,"REJECT invalid line") && strstr(output,"LISTEN_DONE"));
    run("GRIP?\n",0,0);assert(grip_tx==1 && tx_count==0 && strstr(output,"GRIP_TIMEOUT"));
    run("TEACH SAMPLE\n",1,0);assert(tx_count==0 && strstr(output,"teaching not started"));
    run("TEACH START\nTEACH SAMPLE\nTEACH STOP\nTEACH SAMPLE\n",1,0);
    assert(!strstr(output,"POS J") && !strstr(output,"TX t=") && tx_count==6 && strstr(output,"TEACH_FRAME,0,") && strstr(output,",63,1971,1971,1971,1971,1971,1971,") && strstr(output,"TEACH_STOPPED") && strstr(output,"teaching not started"));
    run("TEACH START\nTEACH SAMPLE\nTEACH STOP\n",0,0);
    assert(tx_count==6 && strstr(output,",0,NA,NA,NA,NA,NA,NA,"));
    run("TEACH BEGIN\nTEACH STOP\n",1,0);assert(release_count==6 && strstr(output,"released=6 confirmed=STATE_AND_ACK"));
    run("TEACH BEGIN\n",0,0);assert(release_count==0 && strstr(output,"no release sent"));
    run("TEACH BEGIN\n",2,0);assert(release_count==6 && strstr(output,"release unconfirmed"));
    run("PLAY STEP 1974 1974 1974 1974 1974 1974\n",1,0);assert(tx_count==0 && strstr(output,"not armed"));
    run("PLAY ARM\nPLAY STEP 1974 1974 1974 1974 1974 1974\nPLAY STOP\n",1,0);
    assert(enable_count==6 && fb_packets==12 && strstr(output,"PLAY_READY") && strstr(output,"PLAY_STEP_QUEUED") && strstr(output,"PLAY_STOPPED"));
    run("PLAY ARM\nPLAY STEP 1971 1971 1971 1971 1971 4000\n",1,0);
    assert(fb_packets==6 && strstr(output,"bounded-step check failed"));
    run("PLAY ARM\n",0,0);assert(enable_count==0 && strstr(output,"options unconfirmed"));
    idle_sim=1;run("PLAY ARM\n",1,0);assert(strstr(output,"PLAY_WATCHDOG"));idle_sim=0;
    run("PLAY ARM\nPLAY RATE 7\nPLAY STEP 1971 1971 1971 1971 1971 2005\nPLAY STOP\n",1,0);
    assert(fb_packets==12 && strstr(output,"PLAY_RATE_OK multiplier=7") && strstr(output,"PLAY_STEP_QUEUED"));
    run("PLAY ARM\nPLAY RATE 7\nPLAY STEP 1971 1971 1971 1971 1971 2007\n",1,0);
    assert(fb_packets==6 && strstr(output,"bounded-step check failed"));
    run("PLAY ARM\nPLAY RATE 7\nPLAY STREAM 1974 1974 1974 1974 1974 1974\nPLAY STOP\n",1,0);
    assert(fb_packets==12 && strstr(output,"PLAY_STREAM_QUEUED,") && !strstr(output,"PLAY_REPLY"));
    run("PLAY ARM\nPLAY RATE 10\nPLAY STREAM 1974 1974 1974 1974 1974 2019\nPLAY STOP\n",1,0);
    assert(fb_packets==12 && strstr(output,"PLAY_RATE_OK multiplier=10") && strstr(output,"PLAY_STREAM_QUEUED,"));
    run("PLAY ARM\nPLAY RATE 10\nPLAY SPEED 1072 822 1002 183 301 27\nPLAY STREAM 1974 1974 1974 1974 1974 1974\nPLAY STOP\n",1,0);
    assert(fb_packets==12 && strstr(output,"PLAY_SPEED_OK") && strstr(output,"PLAY_STREAM_QUEUED,"));
    run("PLAY ARM\nPLAY SPEED 1751 822 1002 183 301 27\n",1,0);assert(strstr(output,"speed range") && fb_packets==6);
    run("PLAY ARM\nGRIP BAD\nPLAY PING\nPLAY STOP\n",1,0);
    assert(strstr(output,"unsupported command") && strstr(output,"PLAY_ALIVE") && !strstr(output,"stop playback before"));
    run("ENC 0\nENC 7\n",1,0);assert(tx_count==0 && strstr(output,"REJECT"));
    run("ENC 5\n",1,0);assert(tx_count==1 && enable_count==0 && release_count==0 && fb_packets==0 && strstr(output,"ENC J5 raw=19190 ") && strstr(output,"single_turn=1 model=UNHOMED"));
    encoder_value=0;run("ENC 1\n",1,0);assert(strstr(output,"ENC J1 raw=0 "));
    encoder_value=65535;run("ENC 6\n",1,0);assert(strstr(output,"ENC J6 raw=65535 "));
    encoder_value=19190;
    for(encoder_case=1;encoder_case<=6;++encoder_case) {
        run("ENC 5\n",1,0);assert(strstr(output,"REPLY_TIMEOUT J5") && !strstr(output,"ENC J5 raw="));
    }
    encoder_case=0;
    run("ENC 5\n",0,0);assert(strstr(output,"REPLY_TIMEOUT J5") && !strstr(output,"ENC J5 raw="));
    run("ENC 5\n",0,1);assert(aborts==1 && strstr(output,"TX_TIMEOUT J5") && !strstr(output,"ENC J5 raw="));
    run("TEACH START\nENC 5\nTEACH STOP\n",1,0);assert(tx_count==0 && !strstr(output,"ENC J5 raw="));
    puts("encoder: exact ID/type/length/code/tail, boundary counts, timeouts and no drive passed");
    puts("app: read-only commands, six-axis scan, invalid frames and timeout checks passed");
}
