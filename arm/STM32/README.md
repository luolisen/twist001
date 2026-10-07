# STM32 FinalIntegration v3.0

当前 `Control_Code/` 全量替换为配套 ESP32 夹爪的 STM32F407VG 最终集成源码：六轴 X42S CAN 控制、示教/回放、有界同步命令、停止保持、夹爪 CAN 查询与控制。六轴回放期间允许夹爪链路，不自动归零。

在仓库根目录构建（只生成固件，不烧录）：

```sh
cmake -S arm/STM32/Control_Code -B /tmp/twist001-stm32-build -DCMAKE_TOOLCHAIN_FILE="$PWD/arm/STM32/Control_Code/cmake/arm-gcc.cmake"
cmake --build /tmp/twist001-stm32-build -j4
```

工具链可通过 `STM32_ARM_TOOLCHAIN` 指向包含 `bin/arm-none-eabi-gcc` 的目录。硬件为 STM32F407VG；链接脚本为 `STM32F407VG_FLASH.ld`。配套夹爪源码见 `arm/ESP32/`；联合回归见 `arm/firmware_tools/test_firmware.sh`。

当前源代码来自独立 FinalIntegration 工程，文件 SHA-256 见 `arm/firmware_tools/source_manifest.json`。`Control_Code/README.md` 与 `manifest.json` 是继承的原工程资料，路径和基线信息不代表当前发布验收；以本页和联合验证记录为准。
原 `jlc.epro2` 与 PCB/原理图图片原样保留，未在本次更新中核验硬件设计一致性。未烧录，未执行实机运动。完整主机网关与 ROS 桥接不在此次固件发布范围内。
