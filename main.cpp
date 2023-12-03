#include <iostream>
#include <string>
#include <json.hpp>
#include <stdio.h>
#include <ctime>
#include <random>
#include "util.h"
#include "ggml.h"
#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#if defined(_WIN32)
#include <windows.h>
#endif

using std::string;
using std::vector;

void print_usage(int argc, const char *argv[])
{
    printf("usage: %s [arguments]\n", argv[0]);
    printf("\n");
    printf("arguments:\n");
    printf("  -h, --help                         show this help message and exit\n");
    printf("  -M, --mode [txt2img or img2img]    generation mode (default: txt2img)\n");
    printf("  -t, --threads N                    number of threads to use during computation (default: -1).\n");
    printf("                                     If threads <= 0, then threads will be set to the number of CPU physical cores\n");
    printf("  -m, --model [MODEL]                path to model\n");
    printf("  --vae [VAE]                        path to vae\n");
    printf("  --taesd [TAESD_PATH]               path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)\n");
    printf("  --type [TYPE]                      weight type (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)\n");
    printf("                                     If not specified, the default is the type of the weight file.");
    printf("  --lora-model-dir [DIR]             lora model directory\n");
    printf("  -i, --init-img [IMAGE]             path to the input image, required by img2img\n");
    printf("  -o, --output OUTPUT                path to write result image to (default: ./output.png)\n");
    printf("  -p, --prompt [PROMPT]              the prompt to render\n");
    printf("  -n, --negative-prompt PROMPT       the negative prompt (default: \"\")\n");
    printf("  --cfg-scale SCALE                  unconditional guidance scale: (default: 7.0)\n");
    printf("  --strength STRENGTH                strength for noising/unnoising (default: 0.75)\n");
    printf("                                     1.0 corresponds to full destruction of information in init image\n");
    printf("  -H, --height H                     image height, in pixel space (default: 512)\n");
    printf("  -W, --width W                      image width, in pixel space (default: 512)\n");
    printf("  --sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, lcm}\n");
    printf("                                     sampling method (default: \"euler_a\")\n");
    printf("  --steps  STEPS                     number of sample steps (default: 20)\n");
    printf("  --rng {std_default, cuda}          RNG (default: cuda)\n");
    printf("  -s SEED, --seed SEED               RNG seed (default: 42, use random seed for < 0)\n");
    printf("  -b, --batch-count COUNT            number of images to generate.\n");
    printf("  --schedule {discrete, karras}      Denoiser sigma schedule (default: discrete)\n");
    printf("  -v, --verbose                      print extra info\n");
}
const char *rng_type_to_str[] = {
    "std_default",
    "cuda",
};

// Names of the sampler method, same order as enum sample_method in stable-diffusion.h
const char *sample_method_str[] = {
    "euler_a",
    "euler",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "lcm",
};
// Names of the sigma schedule overrides, same order as sample_schedule in stable-diffusion.h
const char *schedule_str[] = {
    "default",
    "discrete",
    "karras",
};
const char *modes_str[] = {
    "txt2img",
    "img2img",
};
enum SDMode
{
    TXT2IMG,
    IMG2IMG,
    MODE_COUNT
};

struct SDParams
{
    int n_threads = -1;
    SDMode mode = TXT2IMG;

    std::string model_path;
    std::string vae_path;
    std::string taesd_path;
    ggml_type wtype = GGML_TYPE_COUNT;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string input_path;

    std::string prompt;
    std::string negative_prompt;
    float cfg_scale = 7.0f;
    int width = 512;
    int height = 512;
    int batch_count = 1;

    SampleMethod sample_method = EULER_A;
    Schedule schedule = DEFAULT;
    int sample_steps = 20;
    float strength = 0.75f;
    RNGType rng_type = CUDA_RNG;
    int64_t seed = 42;
    bool verbose = false;
};

void parse_args(int argc, const char **argv, SDParams &params)
{
    bool invalid_arg = false;
    std::string arg;
    for (int i = 1; i < argc; i++)
    {
        arg = argv[i];

        if (arg == "-t" || arg == "--threads")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        }
        else if (arg == "-M" || arg == "--mode")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            const char *mode_selected = argv[i];
            int mode_found = -1;
            for (int d = 0; d < MODE_COUNT; d++)
            {
                if (!strcmp(mode_selected, modes_str[d]))
                {
                    mode_found = d;
                }
            }
            if (mode_found == -1)
            {
                fprintf(stderr, "error: invalid mode %s, must be one of [txt2img, img2img]\n",
                        mode_selected);
                exit(1);
            }
            params.mode = (SDMode)mode_found;
        }
        else if (arg == "-m" || arg == "--model")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.model_path = argv[i];
        }
        else if (arg == "--vae")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.vae_path = argv[i];
        }
        else if (arg == "--taesd")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.taesd_path = argv[i];
        }
        else if (arg == "--type")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            std::string type = argv[i];
            if (type == "f32")
            {
                params.wtype = GGML_TYPE_F32;
            }
            else if (type == "f16")
            {
                params.wtype = GGML_TYPE_F16;
            }
            else if (type == "q4_0")
            {
                params.wtype = GGML_TYPE_Q4_0;
            }
            else if (type == "q4_1")
            {
                params.wtype = GGML_TYPE_Q4_1;
            }
            else if (type == "q5_0")
            {
                params.wtype = GGML_TYPE_Q5_0;
            }
            else if (type == "q5_1")
            {
                params.wtype = GGML_TYPE_Q5_1;
            }
            else if (type == "q8_0")
            {
                params.wtype = GGML_TYPE_Q8_0;
            }
            else
            {
                fprintf(stderr, "error: invalid weight format %s, must be one of [f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0]\n",
                        type.c_str());
                exit(1);
            }
        }
        else if (arg == "--lora-model-dir")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.lora_model_dir = argv[i];
        }
        else if (arg == "-i" || arg == "--init-img")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.input_path = argv[i];
        }
        else if (arg == "-o" || arg == "--output")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.output_path = argv[i];
        }
        else if (arg == "-p" || arg == "--prompt")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.prompt = argv[i];
        }
        else if (arg == "-n" || arg == "--negative-prompt")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.negative_prompt = argv[i];
        }
        else if (arg == "--cfg-scale")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.cfg_scale = std::stof(argv[i]);
        }
        else if (arg == "--strength")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.strength = std::stof(argv[i]);
        }
        else if (arg == "-H" || arg == "--height")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.height = std::stoi(argv[i]);
        }
        else if (arg == "-W" || arg == "--width")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.width = std::stoi(argv[i]);
        }
        else if (arg == "--steps")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.sample_steps = std::stoi(argv[i]);
        }
        else if (arg == "-b" || arg == "--batch-count")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.batch_count = std::stoi(argv[i]);
        }
        else if (arg == "--rng")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            std::string rng_type_str = argv[i];
            if (rng_type_str == "std_default")
            {
                params.rng_type = STD_DEFAULT_RNG;
            }
            else if (rng_type_str == "cuda")
            {
                params.rng_type = CUDA_RNG;
            }
            else
            {
                invalid_arg = true;
                break;
            }
        }
        else if (arg == "--schedule")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            const char *schedule_selected = argv[i];
            int schedule_found = -1;
            for (int d = 0; d < N_SCHEDULES; d++)
            {
                if (!strcmp(schedule_selected, schedule_str[d]))
                {
                    schedule_found = d;
                }
            }
            if (schedule_found == -1)
            {
                invalid_arg = true;
                break;
            }
            params.schedule = (Schedule)schedule_found;
        }
        else if (arg == "-s" || arg == "--seed")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            params.seed = std::stoll(argv[i]);
        }
        else if (arg == "--sampling-method")
        {
            if (++i >= argc)
            {
                invalid_arg = true;
                break;
            }
            const char *sample_method_selected = argv[i];
            int sample_method_found = -1;
            for (int m = 0; m < N_SAMPLE_METHODS; m++)
            {
                if (!strcmp(sample_method_selected, sample_method_str[m]))
                {
                    sample_method_found = m;
                }
            }
            if (sample_method_found == -1)
            {
                invalid_arg = true;
                break;
            }
            params.sample_method = (SampleMethod)sample_method_found;
        }
        else if (arg == "-h" || arg == "--help")
        {
            print_usage(argc, argv);
            exit(0);
        }
        else if (arg == "-v" || arg == "--verbose")
        {
            params.verbose = true;
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv);
            exit(1);
        }
    }
    if (invalid_arg)
    {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv);
        exit(1);
    }
    if (params.n_threads <= 0)
    {
        params.n_threads = get_num_physical_cores();
    }

    if (params.prompt.length() == 0)
    {
        fprintf(stderr, "error: the following arguments are required: prompt\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.model_path.length() == 0)
    {
        fprintf(stderr, "error: the following arguments are required: model_path\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.mode == IMG2IMG && params.input_path.length() == 0)
    {
        fprintf(stderr, "error: when using the img2img mode, the following arguments are required: init-img\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.output_path.length() == 0)
    {
        fprintf(stderr, "error: the following arguments are required: output_path\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.width <= 0 || params.width % 64 != 0)
    {
        fprintf(stderr, "error: the width must be a multiple of 64\n");
        exit(1);
    }

    if (params.height <= 0 || params.height % 64 != 0)
    {
        fprintf(stderr, "error: the height must be a multiple of 64\n");
        exit(1);
    }

    if (params.sample_steps <= 0)
    {
        fprintf(stderr, "error: the sample_steps must be greater than 0\n");
        exit(1);
    }

    if (params.strength < 0.f || params.strength > 1.f)
    {
        fprintf(stderr, "error: can only work with strength in [0.0, 1.0]\n");
        exit(1);
    }

    if (params.seed < 0)
    {
        srand((int)time(NULL));
        params.seed = rand();
    }
}

std::string get_image_params(SDParams params, int64_t seed)
{
    std::string parameter_string = params.prompt + "\n";
    if (params.negative_prompt.size() != 0)
    {
        parameter_string += "Negative prompt: " + params.negative_prompt + "\n";
    }
    parameter_string += "Steps: " + std::to_string(params.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(params.cfg_scale) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(params.width) + "x" + std::to_string(params.height) + ", ";
    parameter_string += "Model: " + basename(params.model_path) + ", ";
    parameter_string += "RNG: " + std::string(rng_type_to_str[params.rng_type]) + ", ";
    parameter_string += "Sampler: " + std::string(sample_method_str[params.sample_method]);
    if (params.schedule == KARRAS)
    {
        parameter_string += " karras";
    }
    parameter_string += ", ";
    parameter_string += "Version: stable-diffusion.cpp";
    return parameter_string;
}

int main(int argc, const char *argv[])
{

#if defined(_WIN32)
    // 设置标准输出为UTF-8
    SetConsoleOutputCP(CP_UTF8);
#endif

    string line;

    vector<const char *> args;
    for (int i = 0; i < argc; i++)
    {
        args.push_back(argv[i]);
    }

    args.push_back("--prompt");
    args.push_back("1girl");
    SDParams params;

    parse_args(args.size(), args.data(), params);

    if (params.verbose)
    {
        printf("%s", sd_get_system_info().c_str());
        set_sd_log_level(SDLogLevel::DEBUG);
    }
    
    bool vae_decode_only = false;

    StableDiffusion sd(params.n_threads, vae_decode_only, params.taesd_path, false, params.lora_model_dir, params.rng_type);

    if (!sd.load_from_file(params.model_path, params.vae_path, params.wtype, params.schedule))
    {
        fprintf(stderr, "load model from '%s' failed\n", params.model_path.c_str());
        return 1;
    }

    vector<string> keys = {"cfg_scale", "width", "height", "sample_method", "sample_steps", "strength", "seed", "output", "prompt", "negative_prompt"};

    std::vector<uint8_t *> results;

    // {"cfg_scale":"1", "width":"256", "height":"256", "sample_method":"LCM", "sample_steps":"5", "strength":"1", "seed":"-1", "output":"output.png", "prompt":"<lora:lcm-lora-sdv1-5:1>1girl", "negative_prompt":"text"}
    while (std::getline(std::cin, line))
    {
        try
        {
            results.clear();

            // 解析JSON
            nlohmann::json json = nlohmann::json::parse(line);

            // {"p": "hello world"}

            auto check_keys = 1;
            for (int i = 0; i < keys.size(); i++)
            {

                // if (json.find("p") == json.end() || json["p"].is_null() || json["p"].empty() || json["p"].type() != nlohmann::json::value_t::string)
                if (json.find(keys[i]) == json.end() || json[keys[i]].is_null() || json[keys[i]].empty() || json[keys[i]].type() != nlohmann::json::value_t::string)
                {
                    std::cout << "error: " << keys[i] << " is required" << std::endl;
                    check_keys = 0;
                    break;
                }
            }

            if (check_keys == 0)
            {
                continue;
            }

            auto prompt = json["prompt"].get<string>();
            params.prompt = prompt;
            std::cout << "prompt: " << prompt << std::endl;

            auto negative_prompt = json["negative_prompt"].get<string>();
            params.negative_prompt = negative_prompt;
            std::cout << "negative_prompt: " << negative_prompt << std::endl;

            auto output = json["output"].get<string>();
            params.output_path = output;
            std::cout << "output: " << output << std::endl;

            auto cfg_scale_str = json["cfg_scale"].get<string>();
            float cfg_scale = std::stof(cfg_scale_str);
            params.cfg_scale = cfg_scale;
            std::cout << "cfg_scale: " << cfg_scale << std::endl;

            auto width_str = json["width"].get<string>();
            int width = std::stoi(width_str);
            params.width = width;
            std::cout << "width: " << width << std::endl;

            auto height_str = json["height"].get<string>();
            int height = std::stoi(height_str);
            params.height = height;
            std::cout << "height: " << height << std::endl;

            auto sample_steps_str = json["sample_steps"].get<string>();
            int sample_steps = std::stoi(sample_steps_str);
            params.sample_steps = sample_steps;
            std::cout << "sample_steps: " << sample_steps << std::endl;

            auto strength_str = json["strength"].get<string>();
            float strength = std::stof(strength_str);
            params.strength = strength;
            std::cout << "strength: " << strength << std::endl;

            auto seed_str = json["seed"].get<string>();
            int64_t seed = std::stoll(seed_str);
            params.seed = seed;
            std::cout << "seed: " << seed << std::endl;

            auto sample_method = json["sample_method"].get<string>();
            // EULER_A,
            // EULER,
            // HEUN,
            // DPM2,
            // DPMPP2S_A,
            // DPMPP2M,
            // DPMPP2Mv2,
            // LCM,
            // N_SAMPLE_METHODS
            if (sample_method == "EULER_A")
            {
                params.sample_method = EULER_A;
            }
            else if (sample_method == "EULER")
            {
                params.sample_method = EULER;
            }
            else if (sample_method == "HEUN")
            {
                params.sample_method = HEUN;
            }
            else if (sample_method == "DPM2")
            {
                params.sample_method = DPM2;
            }
            else if (sample_method == "DPMPP2S_A")
            {
                params.sample_method = DPMPP2S_A;
            }
            else if (sample_method == "DPMPP2M")
            {
                params.sample_method = DPMPP2M;
            }
            else if (sample_method == "DPMPP2Mv2")
            {
                params.sample_method = DPMPP2Mv2;
            }
            else if (sample_method == "LCM")
            {
                params.sample_method = LCM;
            }
            else
            {
                std::cout << "error: sample_method is invalid" << std::endl;
                return 1;
            }

            uint8_t *input_image_buffer = NULL;

            if (json.find("input_path") == json.end() || json["input_path"].is_null() || json["input_path"].empty() || json["input_path"].type() != nlohmann::json::value_t::string)
            {
                params.mode = TXT2IMG;
            }
            else
            {
                params.mode = IMG2IMG;
                params.input_path = json["input_path"].get<string>();
            }

            if (params.mode == TXT2IMG)
            {
                results = sd.txt2img(params.prompt,
                                     params.negative_prompt,
                                     params.cfg_scale,
                                     params.width,
                                     params.height,
                                     params.sample_method,
                                     params.sample_steps,
                                     params.seed,
                                     params.batch_count);
            }
            else
            {

                vae_decode_only = false;

                int c = 0;
                input_image_buffer = stbi_load(params.input_path.c_str(), &params.width, &params.height, &c, 3);
                if (input_image_buffer == NULL)
                {
                    fprintf(stderr, "load image from '%s' failed\n", params.input_path.c_str());
                    continue;
                }
                if (c != 3)
                {
                    fprintf(stderr, "input image must be a 3 channels RGB image, but got %d channels\n", c);
                    free(input_image_buffer);
                    continue;
                }
                if (params.width <= 0 || params.width % 64 != 0)
                {
                    std::cout << "error: the width of image must be a multiple of 64:" << params.width << std::endl;
                    free(input_image_buffer);
                    continue;
                }
                if (params.height <= 0 || params.height % 64 != 0)
                {
                    fprintf(stderr, "error: the height of image must be a multiple of 64\n");
                    free(input_image_buffer);
                    continue;
                }
                results = sd.img2img(input_image_buffer,
                                     params.prompt,
                                     params.negative_prompt,
                                     params.cfg_scale,
                                     params.width,
                                     params.height,
                                     params.sample_method,
                                     params.sample_steps,
                                     params.strength,
                                     params.seed);
                free(input_image_buffer);
            }

            if (results.size() == 0 || results.size() != params.batch_count)
            {
                // fprintf(stderr, "generate failed\n");
                // return 1;

                std::cout << "error: generate failed" << std::endl;
            }

            size_t last = params.output_path.find_last_of(".");
            std::string dummy_name = last != std::string::npos ? params.output_path.substr(0, last) : params.output_path;
            for (int i = 0; i < params.batch_count; i++)
            {
                std::string final_image_path = i > 0 ? dummy_name + "_" + std::to_string(i + 1) + ".png" : dummy_name + ".png";
                stbi_write_png(final_image_path.c_str(), params.width, params.height, 3, results[i], 0, get_image_params(params, params.seed + i).c_str());
                // printf("save result image to '%s'\n", final_image_path.c_str());
                std::cout << "save result image to '" << final_image_path << "'" << std::endl;
            }
        }
        catch (const nlohmann::json::parse_error &e)
        {
            std::cout << "error: " << e.what() << std::endl;
        }
    }

    return 0;
}
