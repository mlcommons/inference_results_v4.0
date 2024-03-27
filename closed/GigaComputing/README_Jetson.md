# MLPerf Inference v3.1 NVIDIA-Optimized Inference on Jetson Systems
This is a repository of NVIDIA-optimized implementations for the [MLPerf](https://mlcommons.org/en/) Inference Benchmark.
This README is a quickstart tutorial on how to setup the the Jetson systems as a public / external user.
Please also read README.md for general instructions on how to run the code.

---

NVIDIA Jetson is a platform for AI at the edge. Its high-performance, low-power computing for deep learning and computer vision makes it the ideal platform for compute-intensive projects. The Jetson platform includes a variety of Jetson modules together with NVIDIA JetPack™ SDK.
Each Jetson module is a computing system packaged as a plug-in unit (a System on Module (SOM)). NVIDIA offers a variety of Jetson modules with different capabilities.
JetPack bundles all of the Jetson platform software, starting with NVIDIA Jetson Linux. Jetson Linux provides the Linux kernel, bootloader, NVIDIA drivers, flashing utilities, sample file system, and more for the Jetson platform.

## NVIDIA Submissions

The Jetson AGX Orin / Orin NX submission supports:

- ResNet50 (Offline, Single Stream, and Multistream), at 99% of FP32 accuracy target
- RetinaNet (Offline, Single Stream, and Multistream), at 99% of FP32 accuracy target
- 3D-unet (Offline and Single Stream), at 99% and 99.9% of FP32 accuracy target
- bert (Offline and Single Stream), at 99% of FP32 accuracy target
- rnn-t (Offline and Single Stream), at 99% of FP32 accuracy target

To generate the preprocessed datasets, follow the benchmark-specific instructions described in the [README.md](README.md) files for each benchmark.

## Setup the Jetson AGX Orin/NX System

### Flash the board
Follow the the [Jetson Developer Guide](https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/IN/QuickStart.html#to-flash-the-jetson-developer-kit-operating-software) to flash the board with the r35.3.1 L4T

For replicating performance submission on AGX dev kit, flash the AGX board with MaxN config, e.g. `sudo ./flash.sh jetson-agx-orin-devkit-maxn mmcblk0p1`

For replicating power submission on AGX dev kit, flash the AGX board with non MaxN config, e.g. `sudo ./flash.sh jetson-agx-orin-devkit mmcblk0p1`

For replicating performance and power submission with NX on Nano kit + NVME, flash the Nano board with regular config, e.g. `sudo ./tools/kernel_flash/l4t_initrd_flash.sh --external-device nvme0n1p1 -c tools/kernel_flash/flash_l4t_external.xml -p "-c bootloader/t186ref/cfg/flash_t234_qspi.xml" --showlogs --network usb0 jetson-orin-nano-devkit internal`

### Enable PVA for RetinaNet Offline
Execute the following command to unblock PVA on Jetson systems
```
echo 0 | sudo tee /sys/kernel/debug/pva0/vpu_app_authentication
```

More information can be found from the README file inside the [PVA sample package](https://developer.nvidia.com/downloads/embedded/l4t/cupva-algos-gen2-2.0.0-cupva_algo_dlops.deb)

### Apply Custom 64k Page Size Kernel
NVIDIA's Orin submission systems uses a custom 64k Page size kernel on all systems.

After flashing the Jetson board, please follow the steps below to build apply a 64K page size custom kernel. All steps should be executed on the board instead of the host.

#### Step 1: Download and prepare the kernel source
Make sure you are on the target system (the board), and download the kernel source from [Jetson Linux Archive](https://developer.nvidia.com/embedded/jetson-linux-r3531). Choose Driver Package (BSP) Sources to download

Extract the kernel source by running
```
tar -xjf public_sources.tbz2
cd Linux_for_Tegra/source/public
tar –xjf kernel_src.tbz2
```

#### Step 2: Install software dependencies
Install necessary software to flash the board with
```
sudo apt-get install make build-essential bc libncurses-dev bison flex libssl-dev libelf-dev
```

#### Step 3: Enable 64k page size config and build the kernel
Modifiy the config file under `/Linux_for_Tegra/source/public/kernel/kernel-5.10/arch/arm64/configs/tegra_defconfig` by adding the following flag
```
CONFIG_ARM64_64K_PAGES=y
```

Compile the kernel
```
mkdir kernel_out
./nvbuild.sh -o $PWD/kernel_out
```

#### Step 4: Build the L4T modules
Enter the kernel build directory and build the modules
```
cd kernel_out
sudo make -j8 INSTALL_MOD_PATH=<some-path-to-store-your-module-build> O= modules_install
```

#### Step 5: Apply the image and modules
Apply the image to the system
```
sudo cp -v arch/arm64/boot/Image /boot/Image
```

Apply the modules to the system
```
cd <some-path-to-store-your-module-build>/lib/modules/5.10.104-tegra
sudo cp -rv kernel /lib/modules/5.10.104-tegra/
sudo cp -rv module* /lib/modules/5.10.104-tegra/
```

#### Step 6: Reboot
```
sudo reboot
```

Wait for a few minutes and you should be able to access the system again. You can verify that 64k page size kernel has taken affect with
```
getconf PAGESIZE
65536
```

#### Tips
The built kernel Image and the modules and be reused to patch the system again after reflashing. Please make sure those files' permission is root:root.

#### Alternative cross compile option
For user interested in cross compiling the kernel, please check [kernel customization](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/Kernel/KernelCustomization.html#building-the-kernel) and ask NVIDIA L4T team for any questions.


### Troubleshooting

- System Non-operational After Flashing the L4T Image`

Make sure the following dependencies are installed on your host system

```
sudo apt update && sudo apt install -y libxml2-utils & sudo apt install -y qemu-user-static
```

- No GPU detected after applying the custom kernel

Make sure you have the modules built and replaced the system modules following the instructions in Step5 above.


## Apply Perf Optmization for AGX/NX

### Lock the max clocks

The best ML performance can be achieved with highest power settings. In NVIDIA published Orin benchmarks, the board were configured to use MAXN power profile. The MAXN mode unlock GPU/DLA/CPU/EMC TDP restrictions and is available through flash config switch.

Run the following command under closed/NVIDIA/ to lock the max clocks
```
sudo jetson_clocks
sudo jetson_clocks --fan
OUTSIDE_MLPINF_ENV=1 make apply_jetson_oc_limit
```

### USB-C Power Adapters

For performance submission systems, Dell 130.0W Adapter (HA130PM170) were used for Jetson AGX.

## Apply Perf/W Optmization for AGX

### Disabling Wi-Fi
```
echo 14100000.pcie > /sys/bus/platform/drivers/tegra194-pcie/unbind
```

### USB-C Ethernet
For the MaxQ submission, USB-C Ethernet (RNDIS) was utilized. This feature should function seamlessly upon setup. By connecting the USB-C cable between the PC and the Orin device, the L4T RNDIS service will initiate. The Orin device can access the host PC via the IP address 192.168.55.100, and conversely, the PC can reach the Orin device via the address 192.168.55.1. If necessary, adjustments to the default address assignments can be made by editing the configuration script located at the following path:

```
/opt/nvidia/l4t-usb-device-mode/nv-l4t-usb-device-mode-config.sh
```

To deactivate the service:

```
sudo systemctl stop nv-l4t-usb-device-mode.service
sudo systemctl disable nv-l4t-usb-device-mode.service
```
To reactivate the service:
```
sudo systemctl enable nv-l4t-usb-device-mode.service
```

### Limit display clock
```
echo 1 > /sys/kernel/debug/bpmp/debug/clk/nafll_dce/mrq_rate_locked
echo 115200000 > /sys/kernel/debug/bpmp/debug/clk/nafll_dce/rate
```

The MaxQ measurements were collected with the display disabled completely.

```
sudo systemctl disable gdm3
```

### Limit DRAM clock
```
echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked
echo 2133000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate
```

### Fan profile
The default Orin fan profiles can be tuned to optmize chip leakage and fan power tradeoff. A custom default fan profile 'MLPerf' can be created in /etc/nvpower/nvfancontrol/nvfancontrol_p3701_0000.conf to minimize overall system power for MaxQ submission. Note that the ambient temperature strongly influences the operating temperature, the rule of thumb is to maintain between 55-75C for the best result.

```
...
	FAN_PROFILE MLPerf {
		#TEMP 	HYST	PWM	RPM
		0	0 	255 	2900
		10	0 	255	2900
		11	0	171	1940
		23	0	171	1940
		60	0	0	0
		105	0	0	0
	}
...
FAN_DEFAULT_CONTROL open_loop
FAN_DEFAULT_PROFILE MLPerf
FAN_DEFAULT_GOVERNOR pid
```

### USB-C Power Adapters
Taking advantage of the USB-C PD features, Orin results were collected on high efficiency third party USB-C adapters. For power submission systems, Anker 715 Charger (Nano II 65W) were used.
To enhance power reproducibility, users may employ a USB-C PD to barrel jack converter to manually adjust the voltage requirement. The majority of the workload can operate within the 9-12V range for the Orin AGX development kit, with the exception of Bert, which necessitates a minimum of 15V. For the Orin NX, 9V is adequate to handle all submission workloads. A recommended cable can be procured from Adafruit (Product ID: 5449).

## Apply Perf/W Optmization for NX

### Orin NX NVME ASPM
Unlike the Orin AGX devkit, the Orin NX module does not come with a flash storage device. Users may enable PCI-E ASPM to improve idle and low utilization power. Below is an example of how a user can enable NVME ASPM in the Orin NX devkit. A KIOXIA NVME drive is used as an example to obtain the device address binding using the following commands:

```
sudo lspci | grep KIOXIA | cut -d' ' -f1 | xargs -I{} find /sys/devices/platform -name {} | grep pcie | cut -d'/' -f5
# Output: 141e0000.pcie
```
Next, we decompiled the device tree binary and searched for the corresponding address:
```
sudo dtc -I dtb -O dts /boot/dtb/kernel_tegra234-p3767-0000-p3768-0000-a0.dtb > pcie_text.dts
```
After that, we edited the following lines in the __pcie_text.dts__ file:

```
pcie_ep@141e0000 {
	...
	nvidia,disable-aspm-states = <0x00>;
	...
}
pcie@141e0000 {
	...
	nvidia,disable-aspm-states = <0x00>;
	...
}
```
Once the file was edited, we saved it and then compiled and saved the device tree binary:
```
dtc -I dts -O dtb  pcie_text.dts > pcie_bin.dtb
sudo cp pcie_bin.dtb /boot/dtb/kernel_tegra234-p3767-0000-p3768-0000-a0.dtb
sudo sync
```

## Download Data, Model and Preprocess the data

The data downloading, model downloading and data preprocessing have to be done on **a x86 system**. Please follow the section **Setting up the Scratch Spaces**, **Download the Datasets**, **Downloading the Model files**, **Preprocessing the datasets for inferences** described in [README.md](README.md) to perform the procedure.

## Running a Benchmark

As noted in [README.md](README.md) all benchmarks need to run inside a docker container. Follow the steps in **Running your first benchmark** in the main [README.md](README.md) for instructions on running the benchmarks.

## FAQ
- Is it a must to apply the 64k page kernel?

To replocate NVIDIA's Jetson AGX Orin submission results, it is necessary to have the 64k page kernel.

- Orin system is not detected. The code throws error of `/bin/bash: nvidia-smi: command not found`

Please make sure you add `OUTSIDE_MLPINF_ENV=1` in front of the make command or run `export OUTSIDE_MLPINF_ENV=1`

- I encountered error when launching the docker. The code throws error `Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
nvidia docker is not installed`

Please make sure you install [NVIDIA container Toolkit](https://github.com/NVIDIA/nvidia-docker) and set it as the default container runtime. You can also set the container runtime by adding `--runtime=nvidia` to the `DOCKER_ARGS`. E.g. `DOCKER_ARGS="--runtime=nvidia"`

- I encountered cpu permission error when launching the docker. The code throws error `ocker: Error response from daemon: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: unable to apply cgroup configuration: failed to write "0-11"`

Please check the value of `/sys/fs/cgroup/cpuset/docker/cpuset.cpus` and `/sys/fs/cgroup/cpuset/cpuset.cpus`. You need to reset the value of `/sys/fs/cgroup/cpuset/docker/cpuset.cpus` with the value of `/sys/fs/cgroup/cpuset/cpuset.cpus`. E.g. `sudo echo "0-11" > /sys/fs/cgroup/cpuset/docker/cpuset.cpus`
