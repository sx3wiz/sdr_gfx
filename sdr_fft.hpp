#include <iostream>
#include <thread>
#include <mutex>
#include <array>
#include <vector>
#include <atomic>
#include <complex>
#include <algorithm>

#include "SoapySDR/Device.hpp"
#include "SoapySDR/Logger.h"

#include "fftw3.h"

using namespace SoapySDR;

template<typename T, int N>
std::vector<T>
conv(std::array<T, N> const &f, std::array<T, N> const &g) {
  int const nf = N;
  int const ng = N;
  int const n  = nf + ng - 1;
  std::vector<T> out(n, T());
  for(auto i(0); i < n; ++i) {
    int const jmn = (i >= ng - 1)? i - (ng - 1) : 0;
    int const jmx = (i <  nf - 1)? i            : nf - 1;
    for(auto j(jmn); j <= jmx; ++j) {
      out[i] += (f[j] * g[i - j]);
    }
  }
  return out;
}

template <int N>
class SDR_FFT {
public:
  SDR_FFT()
  {
    shutdown = false;

    //p = fftwf_create_plan(8192, FFTW_FORWARD, FFTW_ESTIMATE);
    p = fftwf_plan_dft_1d(N, reinterpret_cast<fftwf_complex*>(&fft_in[0]),
      reinterpret_cast<fftwf_complex*>(&fft_out[0]), FFTW_FORWARD, FFTW_MEASURE);

    for(int i = 0;i < N;i++)
    {
      hanning_window[i] = 0.5 * (1 - cos((2.0 * 3.14 * static_cast<float>(i)) / static_cast<float>(N) - 1.0f));
    }
  }
  ~SDR_FFT()
  {
    shutdown = true;
    stream_thread.join();

    fftwf_destroy_plan(p);
  }
  void init()
  {
    SoapySDR_setLogLevel(SoapySDRLogLevel::SOAPY_SDR_INFO);

    Kwargs limesdr_args;
    limesdr_args["driver"] = "rtlsdr";

    sdr = Device::make(limesdr_args);

    sdr->setAntenna(SOAPY_SDR_RX, 0, "RX");

    sdr->setGain(SOAPY_SDR_RX, 0, "TUNER", 30.0f);

    sdr->setSampleRate(SOAPY_SDR_RX, 0, 1e6);

    sdr->setFrequency(SOAPY_SDR_RX, 0, 105.7e6, Kwargs());

    std::vector<size_t> channels = {0};
    stream = sdr->setupStream(SOAPY_SDR_RX, "CF32", channels, Kwargs());

    sdr->activateStream(stream);

    stream_thread = std::thread(&SDR_FFT::stream_data, this);
  }
  std::vector<float> get_fft()
  {
    {
      std::lock_guard<std::mutex> guard(buffer_mutex);
      fft_in = buffer;
    }

    filter(fft_in);

    //std::cout << "/" << std::flush;

    fftwf_execute(p);

    std::vector<float> zabs(N);
    for(int i = 0;i < (N / 2);i++)
    {
      zabs[i + (N / 2)] = 20.0 * log10(std::abs(fft_out[i]));
      zabs[i] = 20.0 * log10(std::abs(fft_out[i + (N / 2)]));
    }

    /*
    std::vector<float> out(N, 0);
    for(int i = 10;i < N;i++)
    {
      double t = 0;
      for(int j = 0;j < 10;j++)
      {
        t += zabs[i - j];
      }
      out[i - 2] = 0.1f * t;
    }
    */

    return zabs;
  }

  void stream_data()
  {
    while(!shutdown)
    {
      std::array<float, N*2> buff;

      void* buffs[] = {&buff[0]};

      int flags;
      long long timeN;
      sdr->readStream(stream, buffs, N, flags, timeN);

      std::lock_guard<std::mutex> guard(buffer_mutex);
      for(int i = 0; i < N;i++)
      {
        buffer[i] = std::complex<float>(buff[(i * 2) + 0], buff[(i * 2) + 0]);
      }

      //std::cout << "." << std::flush;
    }
  }
  void filter(std::array<std::complex<float>, N>& data)
  {
    for(int i = 0;i < N;i++)
    {
      data[i] = std::complex<float>(data[i].real()*hanning_window[i], data[i].imag()*hanning_window[i]);
    }
  }
protected:
  std::array<std::complex<float>, N> buffer, fft_in, fft_out;
  std::array<float, N> hanning_window;

  Device* sdr;
  Stream* stream;

  std::atomic_bool shutdown;
  std::mutex buffer_mutex;

  std::thread stream_thread;

  fftwf_plan p;
};
