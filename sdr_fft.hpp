#include <iostream>
#include <thread>
#include <mutex>
#include <array>
#include <vector>
#include <atomic>

#include "SoapySDR/Device.hpp"
#include "SoapySDR/Logger.h"

#include "fftw.h"

using namespace SoapySDR;

class SDR_FFT {
public:
  SDR_FFT()
  {
    shutdown = false;
  }
  ~SDR_FFT()
  {
    shutdown = true;
    stream_thread.join();
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
    std::array<std::complex<double>, 8192> fft_buffer;
    {
      std::lock_guard<std::mutex> guard(buffer_mutex);
      fft_buffer = buffer;
    }

    //std::cout << "/" << std::flush;

    std::vector<std::complex<double>> samps_out(8192);

    fftw_plan p;
    p = fftw_create_plan(8192, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_one(p, reinterpret_cast<fftw_complex*>(&fft_buffer[0]), reinterpret_cast<fftw_complex*>(&samps_out[0]));

    fftw_destroy_plan(p);

    std::vector<float> zabs(8192);
    for(int i = 0;i < 4096;i++)
    {
      zabs[i + 4096] = 20.0 * log10(std::abs(samps_out[i]));
      zabs[i] = 20.0 * log10(std::abs(samps_out[i + 4096]));
    }

    return zabs;
  }

  void stream_data()
  {
    while(!shutdown)
    {
      std::array<float, 8192*2> buff;

      void* buffs[] = {&buff[0]};

      int flags;
      long long timeN;
      sdr->readStream(stream, buffs, 8192, flags, timeN);

      std::lock_guard<std::mutex> guard(buffer_mutex);
      for(int i = 0; i < 8192;i++)
      {
        buffer[i] = std::complex<double>(buff[(i * 2) + 0], buff[(i * 2) + 0]);
      }

      //std::cout << "." << std::flush;
    }
  }
protected:
  std::array<std::complex<double>, 8192> buffer;

  Device* sdr;
  Stream* stream;

  std::atomic_bool shutdown;
  std::mutex buffer_mutex;

  std::thread stream_thread;
};
