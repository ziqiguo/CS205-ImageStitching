To compile: 
	pgc++ -acc -ta=tesla:cc60 -Minfo -o test ex2.cpp harris.cpp util.cpp `pkg-config opencv --cflags --libs`

To run:
	./test [path-to-img]