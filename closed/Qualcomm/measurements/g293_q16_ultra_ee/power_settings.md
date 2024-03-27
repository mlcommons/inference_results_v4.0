# Boot/BIOS Firmware Settings

## AMD CBS

### CPU Common Options

#### Performance
##### CCD/Core/Thread Enablement
###### CCD Control: 2 CCDs (BERT-99 and BERT-99.9) / 4 CCDs (RetinaNet) / Auto (ResNet50)
###### Core Control: FOUR (4+0) (BERT-99 and BERT-99.9) / Auto (RetinaNet and ResNet50)
###### SMT Control: Disabled
#### Core Performance Boost: Disabled (BERT-99, BERT-99.9, RetinaNet, ResNet50 Offline) / Auto (ResNet50 Server)

### NBIO Common Options

#### SMU Common Options
##### Power Profile Selection : Efficiency Mode (BERT-99, BERT-99.9, RetinaNet, ResNet50 Offline) / High Performance Mode (ResNet50 Server)


# Management Firmware Settings

Out-of-the-box.

# Fan Settings

#### Offline

##### ResNet50 (12,600 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>175</b> 0xFF
 0a 3c 00
</pre>

##### RetinaNet (9,300 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>125</b> 0xFF
 0a 3c 00
</pre>


##### BERT-99 (9,300 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>125</b> 0xFF
 0a 3c 00
</pre>


##### BERT-99.9 (12,600 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>175</b> 0xFF
 0a 3c 00
</pre>

#### Server

##### ResNet50 (18,100 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>250</b> 0xFF
 0a 3c 00
</pre>

##### RetinaNet (10,800 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>150</b> 0xFF
 0a 3c 00
</pre>


##### BERT-99 (9,300 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>125</b> 0xFF
 0a 3c 00
</pre>


##### BERT-99.9 (12,600 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>175</b> 0xFF
 0a 3c 00
</pre>


# Maximum Frequency

The maximum chip frequency is controlled through a variable called `vc`.

#### Offline
##### ResNet50 (vc=3)
##### RetinaNet (vc=5)
##### BERT-99 (vc=4)
##### BERT-99.9 (vc=1)

#### Server
##### ResNet50 (vc=3)
##### RetinaNet (vc=5)
##### BERT-99 (vc=3)
##### BERT-99.9 (vc=1)


The CPU frequency policy is controlled through CPU governors.

#### ResNet50 Offline, RetinaNet, BERT-99, BERT-99.9 (powersave)

<pre>
<b>&dollar;</b> sudo cpupower frequency-set --governor <b>powersave</b>
</pre>

#### ResNet50 Server (performance)

<pre>
<b>&dollar;</b> sudo cpupower frequency-set --governor <b>performance</b>
</pre>
