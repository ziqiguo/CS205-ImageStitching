Compile on Ubuntu:

g++ -std=c++11 -fpermissive -o test main.cpp fasthessian.cpp integral.cpp ipoint.cpp surf.cpp utils.cpp `pkg-config opencv --cflags --libs`

=======

Compile on MacOS:
g++-7 -std=c++11 -fpermissive -o test main.cpp fasthessian.cpp integral.cpp ipoint.cpp surf.cpp utils.cpp `pkg-config opencv --cflags --libs`

./test [mode] [openacc mem copy mode] [blend_mode(only required in mode1 and mode2)]

first mode argument: 
	0 -> extract features from single image
	1 -> extract features and compute matches
	2 -> extract features from video frames

openacc mem copy mode:
	0 -> mempry copy from host to device per response layer
	1 -> one single mem copy

blend_mode:
	0 -> off
	1 -> on
