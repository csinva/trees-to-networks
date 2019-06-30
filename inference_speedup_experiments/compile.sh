cython pred.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.6 -o pred.so pred.c
