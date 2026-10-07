#include "gripper_link.h"
#include "usart.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
UART_HandleTypeDef huart1, huart3;
static uint32_t tick;
static const char *response;
static int sent, busy;
static char output[3000];
uint32_t HAL_GetTick(void) { return tick++; }
int HAL_UART_Transmit(UART_HandleTypeDef *h,uint8_t *p,uint16_t n,uint32_t timeout) {
 (void)timeout;
 if(h==&huart3) { assert(n==2 && memcmp(p,"S\n",2)==0); sent++; }
 else { size_t l=strlen(output); assert(l+n<sizeof(output)); memcpy(output+l,p,n);output[l+n]=0; }
 return HAL_OK;
}
int HAL_UART_Receive(UART_HandleTypeDef *h,uint8_t *p,uint16_t n,uint32_t timeout) {
 (void)h;(void)n;(void)timeout;
 if(busy) {*p='x';return HAL_OK;}
 if(sent && *response) {*p=(uint8_t)*response++; return HAL_OK;}
 return HAL_TIMEOUT;
}
static void run(const char *reply,int flooding) { tick=0;sent=0;busy=flooding;response=reply;output[0]=0;Gripper_QueryStatus(); }
int main(void) {
 run("version=v2.2.3-DENGFOC-STM32-RO enabled=0 gap_mm=42.00 i2c_error=0\r\n",0);
 assert(sent==1 && strstr(output,"GRIP_STATUS") && strstr(output,"gap_mm=42.00"));
 run("version=v2.2.1-old enabled=0 gap_mm=42.00 i2c_error=0\n",0);
 assert(sent==1 && strstr(output,"GRIP_TIMEOUT"));
 run("version=v2.2.3-DENGFOC-STM32-RO enabled=0\n",0);
 assert(strstr(output,"GRIP_TIMEOUT"));
 run("",0);assert(sent==1 && strstr(output,"GRIP_TIMEOUT"));
 run("",1);assert(sent==0 && strstr(output,"GRIP_RX_BUSY"));
 puts("gripper link: valid status, wrong version, incomplete reply, disconnect, busy line passed; TX limited to S");
}
