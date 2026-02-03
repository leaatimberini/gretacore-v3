#include "gcore/inference/block_scheduler.hpp"
#include "gcore/inference/generator.hpp"
#include "gcore/inference/model_config.hpp"
#include "gcore/inference/tokenizer.hpp"
#include "gcore/inference/weight_loader.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

void print_usage() {
  std::cout
      << "Usage: greta_infer [options]\n"
      << "Options:\n"
      << "  --model <path>      Path to model weights (GGUF format)\n"
      << "  --prompt <text>     Input prompt\n"
      << "  --prompt-file <path> Read prompt from file\n"
      << "  --batch-size <n>    Batch size for inference (default: 1)\n"
      << "  --max-tokens <n>    Maximum tokens to generate (default: 32)\n"
      << "  --temperature <t>   Sampling temperature (default: 1.0)\n"
      << "  --top-k <k>         Top-K sampling (default: 50)\n"
      << "  --greedy            Use greedy decoding\n"
      << "  --demo-tokenizer    Force fallback ASCII tokenizer\n"
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

  bool force_demo_tokenizer = false;
  bool enable_alignment = false;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
      prompt = argv[++i];
    } else if (strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc) {
      std::ifstream f(argv[++i]);
      if (f.is_open()) {
        std::stringstream ss;
        ss << f.rdbuf();
        prompt = ss.str();
      }
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
    } else if (strcmp(argv[i], "--demo-tokenizer") == 0) {
      force_demo_tokenizer = true;
    } else if (strcmp(argv[i], "--alignment") == 0) {
      enable_alignment = true;
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
  std::unique_ptr<gcore::inference::WeightLoader> loader;
  if (!model_path.empty()) {
    loader = gcore::inference::create_weight_loader(model_path, &err);
    if (!loader) {
      std::cerr << "Failed to open model: " << err << "\n";
      return 1;
    }
    config = loader->get_config();
    if (config.num_heads_kv == 0)
      config.num_heads_kv = config.num_heads;
    if (config.num_heads > 0)
      config.head_dim = config.dim / config.num_heads;
  }

  std::cout << "Model config: layers=" << config.num_layers
            << ", dim=" << config.dim
            << ", heads=" << config.num_heads
            << ", hidden=" << config.hidden_dim
            << ", vocab=" << config.vocab_size
            << ", params=" << (config.param_count() / 1e9) << "B\n";

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
  size_t max_seq_len = 2048; // Default context for bench
  if (const char *max_seq_env = std::getenv("GRETA_MAX_SEQ_LEN")) {
    char *end = nullptr;
    long v = std::strtol(max_seq_env, &end, 10);
    if (end != max_seq_env && v > 0) {
      max_seq_len = static_cast<size_t>(v);
      std::cout << "[GRETA_MAIN] GRETA_MAX_SEQ_LEN=" << max_seq_len
                << std::endl;
    }
  }
  if (!scheduler.allocate_activations(batch_size, max_seq_len,
                                      &err)) { // Configurable max_seq_len
    std::cerr << "Activation allocation failed: " << err << "\n";
    return 1;
  }
  std::cout << "Buffers allocated\n";

  // Load weights from model file if provided
  if (!model_path.empty()) {
    std::cout << "\nLoading weights from: " << model_path << "\n";
    if (!loader) {
      std::cerr << "Failed to open model: " << err << "\n";
      return 1;
    }
    if (!scheduler.load_weights(*loader, &err)) {
      std::cerr << "Weight loading failed: " << err << "\n";
      return 1;
    }
    std::cout << "Weights loaded (vocab size: "
              << config.vocab_size << ")\n";
  }

  // Initialize tokenizer
  gcore::inference::Tokenizer tokenizer;
  if (force_demo_tokenizer) {
    std::cout << "[TOKENIZER] Forced ASCII fallback (--demo-tokenizer)\n";
    tokenizer.use_ascii_fallback();
  } else if (!config.vocabulary.empty()) {
    tokenizer.set_vocabulary(config.vocabulary);
    std::cout << "[TOKENIZER] Loaded GGUF vocab: "
              << config.vocabulary.size() << "\n";
  } else {
    // Try to find .model file near the GGUF model
    std::string tokenizer_path = "tokenizer.model";
    if (!model_path.empty()) {
      size_t last_slash = model_path.find_last_of("/\\");
      if (last_slash != std::string::npos) {
        tokenizer_path =
            model_path.substr(0, last_slash + 1) + "tokenizer.model";
      }
    }
    if (!tokenizer.load(tokenizer_path, &err)) {
      std::cout << "[TOKENIZER] Info: Loading failed (" << err
                << "). Falling back to ASCII.\n";
    }
  }
  std::cout << "[TOKENIZER] Mode: "
            << (tokenizer.is_using_sentencepiece() ? "SentencePiece"
                : (tokenizer.vocab_size() > 0 ? "GGUF vocab"
                                             : "ASCII Fallback"))
            << "\n";

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

  gcore::inference::AlignmentCallback align_cb = nullptr;
  if (enable_alignment) {
    align_cb = [](const gcore::inference::AlignmentStep &step) {
      std::cout << "[ALIGNMENT_STEP] {\"step\":" << step.step
                << ",\"token_id\":" << step.token_id
                << ",\"logit\":" << step.logit
                << ",\"stats\":{\"min\":" << step.logit_min
                << ",\"max\":" << step.logit_max
                << ",\"avg\":" << step.logit_mean
                << ",\"nan\":" << step.nan_count
                << ",\"inf\":" << step.inf_count << "},\"topk_ids\":[";
      for (size_t i = 0; i < step.topk_ids.size(); ++i) {
        std::cout << step.topk_ids[i] << (i == 9 ? "" : ",");
      }
      std::cout << "]}" << std::endl;
    };
  }

  gcore::inference::GenerationStats stats;
  std::string output = generator.generate(
      prompt, params, &stats, [](int32_t id, const std::string &text) {},
      align_cb);

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
