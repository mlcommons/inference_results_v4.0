# Boot/BIOS Firmware Settings

## AMD CBS

### NBIO Common Options
#### SMU Common Options
##### Determinism Control: Auto
##### cTDP Control: Auto
##### EfficiencyModeEn: Auto
##### Package Power Limit  Control: Auto
##### DF Cstates: Auto

### DF Common Options

#### Scrubber
##### DRAM scrub time: Auto
##### Poisson scrubber control: Auto
##### Redirect scrubber control: Auto

#### Memory Addressing
##### NUMA nodes per socket: NPS1

### CPU Common Options
#### Performance
##### SMT Control: Disable

# Management Firmware Settings

Out-of-the-box.

# Fan Settings

#### Offline
##### ResNet50 (9,300 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>125</b> 0xFF
 0a 3c 00
</pre>

##### RetinaNet (6,750 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>75</b> 0xFF
 0a 3c 00
</pre>

##### BERT-99 (8,100 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>100</b> 0xFF
 0a 3c 00
</pre>


#### SingleStream and MultiStream
##### ResNet50, RetinaNet, BERT-99 (5,550 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>50</b> 0xFF
 0a 3c 00
</pre>

# Maximum Frequency

The maximum chip frequency is controlled through a variable called `vc`.

#### Offline
##### ResNet50 (vc=15)
##### RetinaNet (vc=13)
##### BERT-99 (vc=15)

#### SingleStream and MultiStream
##### ResNet50, RetinaNet, BERT-99 (vc=17)


The CPU frequency policy is controlled through CPU governors.

#### ResNet50 (performance)

<pre>
<b>&dollar;</b> sudo cpupower frequency-set --governor <b>performance</b>
 0a 3c 00
</pre>

#### RetinaNet and BERT-99 (powersave)

<pre>
<b>&dollar;</b> sudo cpupower frequency-set --governor <b>powersave</b>
 0a 3c 00
</pre>


