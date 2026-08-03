- First benchmark, runner.py -> 220GB/s achieved, 69%(0.0573 ms) of T4's 320GB/s spec sheet.() 

- First Triton kernel. Vector add, masked, correct vs torch (exact 0 diff).
Sweep 4K→134M: plateaus at 240 GB/s = 75% of T4 peak. Matches torch everywhere >16K.
Small sizes launch-bound (7 GB/s at 4K). Single-size benchmarks are meaningless.

vector-add-performance:
           size  Triton (GB/s)  Torch (GB/s)
0        4096.0       8.000000      7.958549
1        8192.0      15.593909     24.000000
2       16384.0      41.795919     44.043011
3       32768.0      69.033707     66.421623
4       65536.0     108.984479    113.253460
5      131072.0     153.600004    153.600004
6      262144.0     189.776067    188.321833
7      524288.0     210.501068    209.268761
8     1048576.0     222.281519    221.655016
9     2097152.0     230.490042    230.152770
10    4194304.0     234.057145    234.057145
11    8388608.0     235.935494    235.423444
12   16777216.0     237.341786    237.449270
13   33554432.0     238.258583    238.024212
14   67108864.0     238.527318    238.447082
15  134217728.0     238.808555    239.169982