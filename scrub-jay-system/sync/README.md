# sync — Raspberry Pi Pico Firmware

## `sync.cpp`

---

## Hardware wiring

| Peripheral | Pin |
|---|---|
| UART0 TX (serial monitor) | **GPIO 0** |
| UART0 RX | **GPIO 1** |
| I2C SDA → MPU-6050 | **GPIO 4** (I2C0 default) |
| I2C SCL → MPU-6050 | **GPIO 5** (I2C0 default) |
| Microphone (ADC) | **GPIO 27** (ADC channel 1) |
| Debug toggle (ADC timing) | GPIO 13 |
| Debug toggle (IMU timing) | GPIO 12 |
| MPU-6050 power | 3.3 V + GND |

---

## Prerequisites

You need the **Raspberry Pi Pico VS Code extension** or the SDK installed manually. The `CMakeLists.txt` expects SDK 2.1.1 at `~/.pico-sdk/`:

```bash
# Check if the SDK is present
ls ~/.pico-sdk/sdk/2.1.1
```

If missing, install via the [Pico VS Code extension](https://marketplace.visualstudio.com/items?itemName=raspberry-pi.raspberry-pi-pico) (recommended), or manually:

```bash
# ARM toolchain + SDK via Homebrew (macOS)
brew install cmake ninja
# Then install SDK 2.1.1 manually or via the extension
```

---

## Build

A `build/` directory already exists with a pre-built `sync.uf2`. To **rebuild from scratch**:

```bash
cd build
cmake .. -G Ninja
ninja
```

Or to do a clean rebuild:

```bash
rm -rf build && mkdir build && cd build
cmake .. -G Ninja
ninja
```

The output you care about is `build/sync.uf2`.

---

## Flash to Pico

**Option 1 — drag and drop (simplest):**

1. Hold the **BOOTSEL** button on the Pico and plug in USB.
2. It mounts as a mass storage device (`RPI-RP2`).
3. Copy the UF2:

```bash
cp build/sync.uf2 /Volumes/RPI-RP2/
```

The Pico reboots and starts running immediately.

**Option 2 — picotool (no unplug needed if already running):**

```bash
# Put the Pico into bootloader mode from software
picotool reboot -f -u
# Then flash
picotool load build/sync.uf2 --force
picotool reboot
```

---

## Monitor serial output

The firmware prints comma-separated ADC samples and IMU/classification frames over UART0 at **115200 baud**. Connect a USB-to-UART adapter on GPIO 0/1, then:

```bash
# Find the port
ls /dev/tty.usb*

# Open serial monitor
screen /dev/tty.usbserial-XXXX 115200
# or
python3 -m serial.tools.miniterm /dev/tty.usbserial-XXXX 115200
```

Press `Ctrl+A` then `K` to exit `screen`.
