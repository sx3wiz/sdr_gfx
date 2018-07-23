all:
	g++ fft_test.cpp src/glad.c -Iinclude -lglfw -ldl -lGLEW -lGL -lfreetype -lSoapySDR -lfftw -lpthread -std=gnu++11 -O2 -o fft_test
