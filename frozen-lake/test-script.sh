#!/bin/bash
echo "running 4x4 no prior"
date
date > 4x4_CS
python main.py -4x4 -cs -png _CS >> 4x4_CS
date > 4x4_10k
python main.py -4x4 -png _10k >> 4x4_10k
date > 4x4_7k
python main.py -4x4 -ms 7000 -png _7k >> 4x4_7k
date > 4x4_5k
python main.py -4x4 -ms 5000 -png _5k >> 4x4_5k

echo "running 8x8 no prior"
date
date > 8x8_CS
python main.py -cs -png _CS >> 8x8_CS
date > 8x8_10k
python main.py -png _10k >> 8x8_10k
date > 8x8_7k
python main.py -ms 7000 -png _7k >> 8x8_7k
date > 8x8_5k
python main.py -ms 5000 -png _5k >> 8x8_5k

echo "running 4x4 with prior"
date
date > 4x4_CS_P
python main.py -4x4 -cs -png _CS_P -p >> 4x4_CS_P
date > 4x4_10k_P
python main.py -4x4 -png _10k_P -p >> 4x4_10k_P
date > 4x4_7k_P
python main.py -4x4 -ms 7000 -png _7k_P -p >> 4x4_7k_P
date > 4x4_5k_P
python main.py -4x4 -ms 5000 -png _5k_P -p >> 4x4_5k_P

echo "running 8x8 no prior"
date
date > 8x8_CS_P
python main.py -cs -png _CS_P -p >> 8x8_CS_P
date > 8x8_10k_P
python main.py -png _10k_P -p >> 8x8_10k_P
date > 8x8_7k_P
python main.py -ms 7000 -png _7k_P -p >> 8x8_7k_P
date > 8x8_5k_P
python main.py -ms 5000 -png _5k_P-p  >> 8x8_5k_P