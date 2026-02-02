#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>

#include <hip/hip_runtime.h>

namespace gcore::inference {

enum class TraceLevel : int { Off = 0, Stats = 1 };

struct TraceConfig {
  TraceLevel level = TraceLevel::Off;
  int layer_from = 0;
  int layer_to = 999999;
  int every_n = 1;
  bool profile = false;
};

class Tracer {
public:
  Tracer() = default;

  void init_from_env() {
    cfg_.level = (TraceLevel)env_i("GRETA_TRACE_LEVEL", 0);
    cfg_.layer_from = env_i("GRETA_TRACE_LAYER_FROM", 0);
    cfg_.layer_to = env_i("GRETA_TRACE_LAYER_TO", 999999);
    cfg_.every_n = env_i("GRETA_TRACE_EVERY_N", 1);
    cfg_.profile = env_i("GRETA_PROFILE_BLOCKS", 0) != 0;
  }

  bool enabled() const { return cfg_.level != TraceLevel::Off; }
  bool profile_enabled() const { return cfg_.profile; }
  bool should_trace_layer(int layer, int step) const {
    if (!enabled())
      return false;
    if (layer < cfg_.layer_from || layer > cfg_.layer_to)
      return false;
    if (cfg_.every_n <= 1)
      return true;
    return (step % cfg_.every_n) == 0;
  }

  const TraceConfig &cfg() const { return cfg_; }

private:
  static int env_i(const char *k, int def) {
    const char *v = std::getenv(k);
    if (!v || !*v)
      return def;
    const char *end;
    return (int)std::strtol(v, nullptr, 10);
  }

  TraceConfig cfg_{};
};

} // namespace gcore::inference
