# ESP32 FinalIntegration v3.0 夹爪固件

配套 `arm/STM32/Control_Code/`。保留纯 P 控制、力矩保持、端点减速及反向制动；CAN 节点 7，另提供 Wi-Fi TCP 和 BLE GATT 夹爪接口，采用 HMAC、序号、租约、来源独占与有界队列。上电不自动张开。

硬件最大开口 **74.63 mm**，仿真为 **80 mm**，不能直接把两个量程混同。尚未完成无线实机及整臂闭环验收。ESP32 不提供无电脑的六轴 CAN 转发或手柄 HID 直连。

## 构建

安装 Arduino CLI 和 esp32 core 3.3.11。在仓库根目录生成本机独立配置；脚本保留已存在密钥，不打印密钥：

```sh
python3 arm/ESP32/tools/provision_wireless.py
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 --build-property build.partitions=huge_app --build-property upload.maximum_size=3145728 --build-path /tmp/twist001-esp32-build arm/ESP32/ESP32_Final
```

固件使用 esp32 core 提供的 Arduino、TWAI、Wire、Wi-Fi、BLE 与 mbedTLS 接口。配置在 `config/private/wireless.json`，生成的 `ESP32_Final/private_config.h` 不提交。配套客户端必须另行配置同一密钥；此次不发布原设备密钥或含密钥二进制。
使用 huge_app 3 MB 分区、无 OTA。此次只构建、不烧录；未来升级必须使用匹配分区表及启动文件。

本次 ESP32 发布源码与 FinalIntegration 原源文件逐字节相同，来源清单及联合验证见 `arm/firmware_tools/`。
