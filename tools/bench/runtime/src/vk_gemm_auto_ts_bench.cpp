#include "gcore/rt/vk/backend.hpp"
#include "vk_autotune.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using greta::vk_autotune::Candidate;
using greta::vk_autotune::RunArgs;

static void print_usage(const char *exe) {
  std::cout
      << "Uso:\n"
      << "  " << exe << " [--m M] [--n N] [--k K] [--iters I] [--batch B]\n"
      << "\nEnv vars (autotune):\n"
      << "  GRETA_VK_AUTOTUNE_DISABLE=1            (no autotunea; elige el "
         "primer candidato viable)\n"
      << "  GRETA_VK_AUTOTUNE_FORCE=<name>         (fuerza winner por nombre: "
         "tiled_vec2_32x8, etc.)\n"
      << "  GRETA_VK_AUTOTUNE_RETUNE=1             (ignora cache y re-tunea)\n"
      << "  GRETA_VK_AUTOTUNE_CLEAR=1              (limpia cache en memoria "
         "antes de resolver)\n"
      << "  GRETA_VK_AUTOTUNE_NO_WRITE=1           (no escribe cache a disco)\n"
      << "  GRETA_VK_AUTOTUNE_PERSIST_FORCE=1      (si FORCE está seteado, "
         "persiste en cache)\n"
      << "  GRETA_VK_AUTOTUNE_MARGIN=1.03          (rerun top2 si best/second "
         "< margin)\n"
      << "  GRETA_VK_AUTOTUNE_RERUN_ITERS=60       (iters del rerun top2)\n"
      << "\nVulkan:\n"
      << "  VK_ICD_FILENAMES=...                   (selecciona ICD Vulkan)\n";
}

static std::string exe_dir(char **argv) {
  std::filesystem::path p = std::filesystem::path(argv[0]);
  if (p.has_parent_path())
    return p.parent_path().string();
  return ".";
}

// “Viable” básico para candidatos que requieren subgroup 32
static bool
candidate_viable_for_device(const Candidate &c,
                            const greta::vk_autotune::DeviceInfo &di) {
  if (!c.requires_subgroup32)
    return true;

  // Si no pudimos probe min_subgroup_size (0), dejamos que intente igual.
  if (di.min_subgroup_size == 0)
    return true;

  // Requiere soporte para subgroup size 32.
  return di.min_subgroup_size <= 32;
}

int main(int argc, char **argv) {
  RunArgs args{};
  // defaults: 1024/1024/1024, iters=30, batch=100

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const char *k) {
      if (i + 1 >= argc) {
        std::cerr << "Falta valor para " << k << "\n";
        std::exit(1);
      }
    };

    if (a == "--m") {
      need("--m");
      args.M = uint32_t(std::stoul(argv[++i]));
    } else if (a == "--n") {
      need("--n");
      args.N = uint32_t(std::stoul(argv[++i]));
    } else if (a == "--k") {
      need("--k");
      args.K = uint32_t(std::stoul(argv[++i]));
    } else if (a == "--iters") {
      need("--iters");
      args.iters = std::stoi(argv[++i]);
    } else if (a == "--batch") {
      need("--batch");
      args.batch = std::stoi(argv[++i]);
    } else if (a == "--help" || a == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Argumento desconocido: " << a << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  std::cout << "GRETA CORE Runtime Bench: vk_gemm_auto_ts_bench\n";
  std::cout << "M=" << args.M << " N=" << args.N << " K=" << args.K
            << " iters=" << args.iters << " batch=" << args.batch << "\n";

  gcore::rt::vk::Backend backend;
  std::string berr;
  if (!backend.init(&berr)) {
    std::cerr << "Backend init failed: " << berr << "\n";
    std::cout << "STATUS=FAILED reason=\"backend_init_failed\"\n";
    return 1;
  }
  auto info = backend.device_info();
  std::cout << "Selected device:\n";
  std::cout << "  vendor_id=0x" << std::hex << info.vendor_id << std::dec
            << "\n";
  std::cout << "  device_id=0x" << std::hex << info.device_id << std::dec
            << "\n";
  std::cout << "  device_type=" << info.type << "\n";
  std::cout << "  name=" << info.name << "\n";
  std::cout << "  driver_name=" << info.driver_name << "\n";
  backend.print_diagnostics(std::cout);

  if (backend.gpu_blacklisted()) {
    std::cout << "SKIPPED: GPU blacklisted: " << backend.blacklist_reason()
              << "\n";
    std::cout << "STATUS=SKIPPED reason=\"gpu_blacklisted\"\n";
    return 0;
  }

  const std::string dir = exe_dir(argv);

  // Lista de candidatos (nombres estables -> cache)
  // exe: nombre del binario relativo a dir
  std::vector<Candidate> candidates = {
      {"tiled_f16acc32", "vk_gemm_f16acc32_tiled_ts_bench", false},
      {"tiled_vec2", "vk_gemm_f16acc32_tiled_vec2_ts_bench", false},
      {"tiled_vec2_32x8", "vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench", false},
      {"tiled_vec2_db", "vk_gemm_f16acc32_tiled_vec2_db_ts_bench", false},
      {"subgroup", "vk_gemm_f16acc32_subgroup_ts_bench", true},
  };

  if (!backend.fp16_enabled()) {
    std::cout << "SKIPPED: FP16 not enabled: " << backend.fp16_status_reason()
              << "\n";
    std::cout << "STATUS=SKIPPED reason=\"fp16_not_enabled\"\n";
    return 0;
  }

  // Skip FP16 autotune if device is blacklisted by FP16 healthcheck
  auto di_opt = greta::vk_autotune::probe_device();
  if (di_opt && greta::vk_autotune::fp16_blacklisted(*di_opt) &&
      !greta::vk_autotune::env_flag_true("GRETA_VK_FP16_ALLOW_UNSAFE")) {
    const std::string reason = greta::vk_autotune::fp16_blacklist_reason(*di_opt);
    greta::vk_autotune::tag_fp16_blacklist_cache(*di_opt, reason);
    std::cout << "SKIPPED: FP16 blacklisted by healthcheck\n";
    std::cout << "  fp16_blacklist_path="
              << greta::vk_autotune::fp16_blacklist_path_string() << "\n";
    std::cout << "  device_key=" << di_opt->key_string() << "\n";
    std::cout << "  hint=set GRETA_VK_FP16_ALLOW_UNSAFE=1 to override\n";
    std::cout << "STATUS=SKIPPED reason=\"fp16_blacklisted\"\n";
    return 0;
  }

  // Opción: desactivar autotune y elegir el primer candidato viable
  if (greta::vk_autotune::env_flag_true("GRETA_VK_AUTOTUNE_DISABLE")) {
    if (!di_opt) {
      std::cerr
          << "AUTOTUNE_DISABLE: probe_device() falló (no hay Vulkan device)\n";
      return 1;
    }
    const auto &di = *di_opt;

    std::string chosen;
    for (const auto &c : candidates) {
      if (candidate_viable_for_device(c, di)) {
        chosen = c.name;
        break;
      }
    }

    if (chosen.empty()) {
      std::cerr
          << "AUTOTUNE_DISABLE: ningún candidato viable para este device\n";
      return 1;
    }

    // Cache path (solo para mostrar)
    greta::vk_autotune::Cache cache;
    cache.load();

    std::cout << "AUTOTUNE DISABLED: selected first viable candidate\n";
    std::cout << "AUTOTUNE FINAL WINNER:\n";
    std::cout << "  winner=" << chosen << "\n";
    std::cout << "  cache_path=" << cache.path() << "\n";
    std::cout << "STATUS=OK\n";
    return 0;
  }

  // Resolver con autotune (cache + retune + force + rerun top2)
  auto rr = greta::vk_autotune::resolve_winner(args, dir, candidates);

  // Si winner vacío -> error (Vulkan probe falló o todo dio 0 y se consideró
  // inválido)
  if (rr.winner.empty()) {
    std::cerr << "AUTOTUNE: no se pudo resolver winner\n";
    std::cout << "STATUS=FAILED reason=\"autotune_failed\"\n";
    return 1;
  }

  // Salida final consistente (siempre)
  std::cout << "\nAUTOTUNE FINAL WINNER:\n";
  std::cout << "  winner=" << rr.winner << "\n";
  std::cout << "  cache_path=" << rr.cache_path << "\n";
  std::cout << "STATUS=OK\n";
  return 0;
}
