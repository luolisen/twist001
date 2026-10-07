#ifndef READCAN_USART_H
#define READCAN_USART_H
#include "main.h"
extern UART_HandleTypeDef huart1;
extern UART_HandleTypeDef huart3;
void MX_USART1_UART_Init(void);
void MX_USART3_UART_Init(void);
#endif
