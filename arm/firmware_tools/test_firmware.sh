#!/bin/sh
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project="$base/../STM32/Control_Code"
sdk=$(xcrun --sdk macosx --show-sdk-path)
build=$(mktemp -d /tmp/readcan-tests.XXXXXX)
trap 'rm -rf "$build"' EXIT
clang -isysroot "$sdk" -std=c99 -Wall -Wextra -Werror -fsanitize=address,undefined -I "$project/Core/Inc" "$project/Core/Src/x42s_protocol.c" "$project/tests/test_protocol.c" -o "$build/protocol"
"$build/protocol"
clang -isysroot "$sdk" -std=c99 -Wall -Wextra -Werror -fsanitize=address,undefined -I "$project/tests/fake_hal" -I "$project/Core/Inc" "$project/Core/Src/x42s_protocol.c" "$project/Core/Src/readcan_app.c" "$project/Core/Src/gripper_link.c" "$project/Core/Src/gripper_can.c" "$project/tests/test_app.c" -o "$build/app"
"$build/app"

clang -isysroot "$sdk" -std=c99 -Wall -Wextra -Werror -fsanitize=address,undefined -I "$project/tests/fake_hal" -I "$project/Core/Inc" "$project/Core/Src/gripper_link.c" "$project/tests/test_gripper_link.c" -o "$build/gripper"
"$build/gripper"

clang -isysroot "$sdk" -std=c99 -Wall -Wextra -Werror -fsanitize=address,undefined -I "$project/tests/fake_hal" -I "$project/Core/Inc" "$project/Core/Src/gripper_can.c" "$project/tests/test_gripper_can.c" -o "$build/gripper-can"
"$build/gripper-can"
clang -isysroot "$sdk" -std=c99 -Wall -Wextra -Werror -fsanitize=address,undefined -I "$project/Core/Inc" "$project/tests/test_grip_control.c" -o "$build/gripper-control"
"$build/gripper-control"

clang -isysroot "$sdk" -std=c99 -Wall -Wextra -Werror -fsanitize=address,undefined -I "$base/../ESP32/ESP32_Final" "$base/../ESP32/ESP32_Final/tests/test_grip_motion.c" -o "$build/motion"
"$build/motion"
