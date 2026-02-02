#include "gcore/inference/block_scheduler.hpp"
#include "gcore/inference/generator.hpp"
#include "gcore/inference/model_config.hpp"
#include "gcore/inference/tokenizer.hpp"
#include "gcore/inference/weight_loader.hpp"

#include <cstring>
#include <iostream>

void print_usage() {
  std::cout
      << "Usage: greta_infer [options]\n"
      << "Options:\n"
      << "  --model <path>      Path to model weights (GGUF format)\n"
      << "  --prompt <text>     Input prompt\n"
      << "  --batch-size <n>    Batch size for inference (default: 1)\n"
      << "  --max-tokens <n>    Maximum tokens to generate (default: 32)\n"
      << "  --temperature <t>   Sampling temperature (default: 1.0)\n"
      << "  --top-k <k>         Top-K sampling (default: 50)\n"
      << "  --greedy            Use greedy decoding\n"
      << "  --help              Show this help\n";
}

int main(int argc, char *argv[]) {
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║           GRETA CORE - LLM Inference Engine              ║\n";
  std::cout << "║                    Phase 3 Preview                       ║\n";
  std::cout
      << "╚══════════════════════════════════════════════════════════╝\n\n";

  // Default parameters
  std::string model_path;
  std::string prompt = "Hello, I am a language model";
  int batch_size = 1;
  gcore::inference::SamplingParams params;
  params.max_tokens = 32;
  params.temperature = 1.0f;
  params.top_k = 50;
  params.greedy = false;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
      prompt = argv[++i];
    } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
      batch_size = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
      params.max_tokens = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      params.temperature = std::atof(argv[++i]);
    } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
      params.top_k = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--greedy") == 0) {
      params.greedy = true;
    } else if (strcmp(argv[i], "--help") == 0) {
      print_usage();
      return 0;
    }
  }

  std::cout << "Configuration:\n";
  std::cout << "  Model: " << (model_path.empty() ? "(demo mode)" : model_path)
            << "\n";
  std::cout << "  Prompt: \"" << prompt << "\"\n";
  std::cout << "  Max tokens: " << params.max_tokens << "\n";
  std::cout << "  Temperature: " << params.temperature << "\n";
  std::cout << "  Top-K: " << params.top_k << "\n";
  std::cout << "  Greedy: " << (params.greedy ? "yes" : "no") << "\n";

  const char *verbose_info = std::getenv("GRETA_VERBOSE_INFO");
  if (verbose_info && std::string(verbose_info) == "1") {
    int hip_ver = 0;
    (void)hipRuntimeGetVersion(&hip_ver);
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, 0);

    const char *graph_env = std::getenv("GRETA_HIP_GRAPH");
    const char *prof_env = std::getenv("GRETA_PROFILE_BLOCKS");

    std::cout << "\nSystem Info (VERBOSE):\n";
    std::cout << "  GPU: " << prop.name << "\n";
    std::cout << "  HIP Runtime Version: " << hip_ver << "\n";
    std::cout << "  GRETA_HIP_GRAPH: " << (graph_env ? graph_env : "0") << "\n";
    std::cout << "  GRETA_PROFILE_BLOCKS: " << (prof_env ? prof_env : "0")
              << "\n";
  }
  std::cout << "\n";

  std::string err;
  if (gcore::rt::GretaContext::instance().initialize() !=
      gcore::rt::GretaResult::SUCCESS) {
    std::cerr << "Failed to initialize GRETA context\n";
    return 1;
  }

  // Initialize model config
  auto config = gcore::inference::ModelConfig::llama2_7b();
  std::cout << "Model: Llama-2-7B (" << config.param_count() / 1e9
            << "B params)\n";

  // Initialize scheduler
  std::cout << "[GRETA_MAIN] Initializing scheduler..." << std::endl;
  gcore::inference::BlockScheduler scheduler;
  if (!scheduler.init(config, &err)) {
    std::cerr << "Scheduler init failed: " << err << "\n";
    return 1;
  }
  std::cout << "[GRETA_MAIN] Initialized scheduler for "
            << scheduler.num_layers() << " layers\n";

  // Allocate buffers
  std::cout << "Allocating buffers...\n";
  if (!scheduler.allocate_weights(&err)) {
    std::cerr << "Weight allocation failed: " << err << "\n";
    return 1;
  }
  if (!scheduler.allocate_activations(
          batch_size, 2048, &err)) { // Support up to 2k context for bench
    std::cerr << "Activation allocation failed: " << err << "\n";
    return 1;
  }
  std::cout << "Buffers allocated\n";

  // Load weights from model file if provided
  if (!model_path.empty()) {
    std::cout << "\nLoading weights from: " << model_path << "\n";
    auto loader = gcore::inference::create_weight_loader(model_path, &err);
    if (!loader) {
      std::cerr << "Failed to open model: " << err << "\n";
      return 1;
    }
    if (!scheduler.load_weights(*loader, &err)) {
      std::cerr << "Weight loading failed: " << err << "\n";
      return 1;
    }
    config = loader->get_config();
    std::cout << "Weights loaded and config updated (vocab size: "
              << config.vocab_size << ")\n";
  }

  // Initialize tokenizer
  gcore::inference::Tokenizer tokenizer;
  if (!tokenizer.load("tokenizer.json", &err)) {
    std::cout << "Using fallback tokenizer (demo mode)\n";
  }

  // Initialize generator
  gcore::inference::Generator generator;
  if (!generator.init(config, &scheduler, &err)) {
    std::cerr << "Generator init failed: " << err << "\n";
    return 1;
  }
  std::cout << "Generator initialized\n\n";

  // Generate
  std::cout << "═══════════════════════════════════════════════════════════\n";
  std::cout << "Generating...\n\n";

  gcore::inference::GenerationStats stats;
  std::string output = generator.generate(
      prompt, params, &stats, [](int32_t id, const std::string &text) {
        // Streaming callback (would print each token)
      });

  std::cout << "Prompt: " << prompt << "\n";
  std::cout << "Generated: " << output << "\n\n";
  std::cout << "═══════════════════════════════════════════════════════════\n";

  // Print stats
  std::cout << "Statistics:\n";
  std::cout << "  Prompt tokens: " << stats.prompt_tokens << "\n";
  std::cout << "  Generated tokens: " << stats.generated_tokens << "\n";
  std::cout << "  Total time: " << stats.total_time_ms << " ms\n";
  std::cout << "  Time to first token: " << stats.time_to_first_token_ms
            << " ms\n";
  std::cout << "  Tokens/second: " << stats.tokens_per_second << "\n";

  std::cout << "\nSTATUS=OK\n";
  return 0;
}
