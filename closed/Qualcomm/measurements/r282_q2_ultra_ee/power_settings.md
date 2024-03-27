# Boot/BIOS Firmware Settings

## AMD CBS

### CPU Common Options

#### Performance
##### SMT Control: Disabled
#### Core Performance Boost: Disabled (Offline) / Auto (SingleStream, MultiStream)

### NBIO Common Options

#### SMU Common Options
##### EfficiencyModeEn: Enabled (Offline) / Auto (SingleStream, MultiStream)

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
##### ResNet50 (vc=3)
##### RetinaNet (vc=4)
##### BERT-99 (vc=3)

#### SingleStream and MultiStream
##### ResNet50, RetinaNet, BERT-99 (vc=11)


The CPU frequency policy is controlled through CPU governors.

#### Offline (powersave)

<pre>
<b>&dollar;</b> sudo cpupower frequency-set --governor <b>powersave</b>
</pre>

#### SingleStream and MultiStream (performance)

<pre>
<b>&dollar;</b> sudo cpupower frequency-set --governor <b>performance</b>
</pre>
