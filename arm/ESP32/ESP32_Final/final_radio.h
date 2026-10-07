#ifndef FINAL_RADIO_H
#define FINAL_RADIO_H
#include <WiFi.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLE2902.h>
#include <mbedtls/md.h>
#include "private_config.h"
#include "grip_control.h"
// AFI1 framing: 20-byte header, <=512-byte payload, 16-byte HMAC-SHA256.
// Only the loop task may mutate the drive. Radio tasks use bounded queues.
struct FinalRadioCommand { uint8_t raw[24],origin; uint32_t seq,ticket; };
struct FinalRadioResult { uint32_t ticket; uint8_t ack[8]; };
struct FinalBLEFragment { uint8_t bytes[20],size; };
static QueueHandle_t radioInput=nullptr,radioOutput=nullptr,bleInput=nullptr;
static portMUX_TYPE radioMux=portMUX_INITIALIZER_UNLOCKED;
static uint8_t radioSnapshot[40];
static volatile uint32_t finalRadioDropped=0;
static BLECharacteristic *radioTX=nullptr;
static uint32_t radioTicket=0;
static WiFiServer radioServer(7420);
static void finalRadioPublish(const uint8_t *status) {
 portENTER_CRITICAL(&radioMux);memcpy(radioSnapshot,status,40);portEXIT_CRITICAL(&radioMux);
}
static bool finalRadioTake(FinalRadioCommand &command) {return radioInput&&xQueueReceive(radioInput,&command,0)==pdTRUE;}
static void finalRadioComplete(const FinalRadioCommand &command,const uint8_t *ack) {
 FinalRadioResult result={};result.ticket=command.ticket;memcpy(result.ack,ack,8);
 if(xQueueSend(radioOutput,&result,0)!=pdTRUE)++finalRadioDropped;
}
static void radioMAC(const uint8_t *data,size_t n,uint8_t *tag) {
 uint8_t full[32];mbedtls_md_hmac(mbedtls_md_info_from_type(MBEDTLS_MD_SHA256),RADIO_KEY,32,data,n,full);memcpy(tag,full,16);
}
static bool radioEqual(const uint8_t *a,const uint8_t *b) {uint8_t diff=0;for(unsigned i=0;i<16;++i)diff|=a[i]^b[i];return diff==0;}
static size_t radioExchange(const uint8_t *input,size_t length,uint8_t origin,uint8_t *output) {
 if(length<36||memcmp(input,"AFI1",4)||input[4]!=1)return 0;
 uint16_t size=gc_get16(input+6);if(size>512||length!=size+36U)return 0;
 uint8_t mac[16];radioMAC(input,length-16,mac);if(!radioEqual(mac,input+length-16))return 0;
 uint8_t snapshot[40];portENTER_CRITICAL(&radioMux);memcpy(snapshot,radioSnapshot,40);portEXIT_CRITICAL(&radioMux);
 uint32_t session=gc_get32(input+12),seq=gc_get32(input+8);uint16_t responseSize=0;
 memcpy(output,input,20);
 if(input[5]==1||input[5]==2) {
   if(size!=0)return 0;
   memcpy(output+20,snapshot,40);responseSize=40;
 } else if(input[5]==3) {
   GP_Command command;
   if(size!=24||!gp_decode(input+20,&command)||command.origin!=origin||command.seq!=seq||command.session!=session||command.deadline!=gc_get32(input+16))return 0;
   FinalRadioCommand request={};request.seq=seq;request.ticket=++radioTicket;request.origin=origin;memcpy(request.raw,input+20,24);
   if(xQueueSend(radioInput,&request,0)!=pdTRUE){++finalRadioDropped;return 0;}
   FinalRadioResult result;bool found=false;uint32_t started=millis();
   while(millis()-started<100U) {
     if(xQueueReceive(radioOutput,&result,pdMS_TO_TICKS(5))==pdTRUE&&result.ticket==request.ticket){found=true;break;}
   }
   if(!found)return 0;
   memcpy(output+20,result.ack,8);responseSize=8;
 } else {
   // Whole-arm motion stays with STM32 and its host bridge. No raw motor tunnel.
   output[20]=0xff;responseSize=1;
 }
 gc_put16(output+6,responseSize);radioMAC(output,responseSize+20,output+20+responseSize);return responseSize+36;
}
class FinalRXCallbacks:public BLECharacteristicCallbacks {
 void onWrite(BLECharacteristic *characteristic) override {
   auto value=characteristic->getValue();
   if(value.length()>20||value.length()<4){++finalRadioDropped;return;}
   FinalBLEFragment fragment={};fragment.size=value.length();memcpy(fragment.bytes,value.c_str(),fragment.size);
   if(xQueueSend(bleInput,&fragment,0)!=pdTRUE)++finalRadioDropped;
 }
};
class FinalServerCallbacks:public BLEServerCallbacks {
 void onDisconnect(BLEServer *server) override {server->startAdvertising();}
};
static void finalRadioTask(void *) {
 WiFi.mode(WIFI_AP_STA);WiFi.softAP("Alan-Robot",AP_PASSWORD);
 if(strlen(STA_SSID))WiFi.begin(STA_SSID,STA_PASSWORD);
 radioServer.begin();
 BLEDevice::init("Alan-Robot");
 BLEServer *server=BLEDevice::createServer();server->setCallbacks(new FinalServerCallbacks());
 BLEService *service=server->createService("8d1b1000-0920-4a97-a815-7d060a327007");
 BLECharacteristic *rx=service->createCharacteristic("8d1b1001-0920-4a97-a815-7d060a327007",BLECharacteristic::PROPERTY_WRITE);
 radioTX=service->createCharacteristic("8d1b1002-0920-4a97-a815-7d060a327007",BLECharacteristic::PROPERTY_NOTIFY);
 radioTX->addDescriptor(new BLE2902());rx->setCallbacks(new FinalRXCallbacks());service->start();
 BLEAdvertising *advertising=server->getAdvertising();advertising->addServiceUUID(service->getUUID());advertising->start();
 WiFiClient client;uint8_t tcp[548],ble[548],response[548];size_t tcpUsed=0;uint32_t tcpStarted=0,tcpActivity=0;
 uint8_t bleTag=0,bleTotal=0;uint64_t bleMask=0;uint8_t bleSizes[35]={};uint32_t bleStarted=0;
 for(;;) {
   if(!client||!client.connected()){client.stop();client=radioServer.accept();tcpUsed=0;tcpStarted=millis();tcpActivity=millis();}
   if(client&&client.connected()) {
     for(unsigned n=0;n<64&&client.available();++n) {
       if(tcpUsed==0)tcpStarted=millis();
       if(tcpUsed>=sizeof(tcp)){client.stop();break;}
       tcp[tcpUsed++]=client.read();tcpActivity=millis();
       if(tcpUsed>=20) {
         const size_t expected=36U+gc_get16(tcp+6);
         if(expected>sizeof(tcp)){client.stop();break;}
         if(tcpUsed==expected){size_t result=radioExchange(tcp,tcpUsed,1,response);if(result)client.write(response,result);else client.stop();tcpUsed=0;break;}
       }
     }
     if((tcpUsed&&millis()-tcpStarted>500U)||millis()-tcpActivity>2500U){client.stop();tcpUsed=0;}
   }
   FinalBLEFragment fragment;
   for(unsigned n=0;n<4&&xQueueReceive(bleInput,&fragment,0)==pdTRUE;++n) {
     uint8_t *f=fragment.bytes,tag=f[0],index=f[1],total=f[2],size=f[3];
     if(total==0||total>35||index>=total||size==0||size>16||fragment.size!=size+4U||(index+1<total&&size!=16)){++finalRadioDropped;continue;}
     if(!bleMask||tag!=bleTag||millis()-bleStarted>500U){bleMask=0;bleTag=tag;bleTotal=total;bleStarted=millis();}
     if(total!=bleTotal||index*16U+size>sizeof(ble)){bleMask=0;++finalRadioDropped;continue;}
     if((bleMask&(1ULL<<index))&&(bleSizes[index]!=size||memcmp(ble+16U*index,f+4,size))){bleMask=0;++finalRadioDropped;continue;}
     memcpy(ble+16U*index,f+4,size);bleSizes[index]=size;bleMask|=1ULL<<index;
     if(bleMask==((1ULL<<total)-1)) {
       size_t result=radioExchange(ble,(total-1)*16U+bleSizes[total-1],2,response);bleMask=0;
       if(result)for(unsigned part=0;part<(result+15)/16;++part){uint8_t out[20];size_t count=min((size_t)16,result-part*16);out[0]=tag;out[1]=part;out[2]=(result+15)/16;out[3]=count;memcpy(out+4,response+part*16,count);radioTX->setValue(out,count+4);radioTX->notify();vTaskDelay(pdMS_TO_TICKS(5));}
     }
   }
   vTaskDelay(pdMS_TO_TICKS(1));
 }
}
static void finalRadioBegin() {
 radioInput=xQueueCreate(4,sizeof(FinalRadioCommand));radioOutput=xQueueCreate(4,sizeof(FinalRadioResult));bleInput=xQueueCreate(40,sizeof(FinalBLEFragment));
 if(!radioInput||!radioOutput||!bleInput){++finalRadioDropped;return;}
 if(xTaskCreatePinnedToCore(finalRadioTask,"final-radio",8192,nullptr,1,nullptr,0)!=pdPASS)++finalRadioDropped;
}
#endif
