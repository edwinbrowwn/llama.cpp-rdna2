#include "common.h"
#include "log.h"
#include "llama.h"
#include "speculative-content.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct sample_case {
    const char * name;
    const char * intent;
    std::string text;
};

static const std::vector<sample_case> & builtin_samples() {
    static const std::vector<sample_case> samples = {
        {
            "plain-prose",
            "should remain at +0",
            "The old library was quiet in the afternoon. We reviewed the notes, discussed the tradeoffs, and agreed to revisit the decision tomorrow."
        },
        {
            "numbers-in-prose",
            "small measurements should not create a sustained boost",
            "The trial used 3 samples and reached 92.4 percent accuracy. That result is encouraging, although the group was too small for a firm conclusion."
        },
        {
            "inline-json-in-prose",
            "the tiny data fragment may spike briefly, then return to prose",
            "The service returned {\"status\":\"ok\",\"count\":3}, after which the report continued with an ordinary explanation of the result."
        },
        {
            "parenthetical-data",
            "parentheses and percentages should not look like a code block",
            "The temperature rose from 18.2 C to 21.7 C (about 19 percent) during the afternoon, then gradually returned to normal."
        },
        {
            "inline-code-in-prose",
            "inline code should be localized rather than boost the whole paragraph",
            "Call `cache.clear()` before retrying, but keep the surrounding explanation concise and readable for operators."
        },
        {
            "fenced-python",
            "the fenced body should be strongly detected and trailing prose should decay",
            "Here is the implementation:\n\n```python\ndef clamp(value, low, high):\n    if value < low:\n        return low\n    return min(value, high)\n```\n\nThis version keeps the boundary behavior explicit."
        },
        {
            "fenced-json",
            "the complete JSON block should be strongly detected",
            "The response payload is:\n\n```json\n{\n  \"status\": \"ok\",\n  \"items\": [1, 2, 3],\n  \"retry\": false\n}\n```\n\nThe prose resumes here."
        },
        {
            "unfenced-code",
            "multi-line syntax should be detected without a fence",
            "const total = values.reduce((sum, value) => sum + value, 0);\nif (total > limit) {\n    return total - limit;\n}\n"
        },
        {
            "markdown-table",
            "tabular structure should receive moderate structural confidence",
            "The measurements were:\n\n| mode | tokens/s | accepted |\n| --- | ---: | ---: |\n| base | 95.2 | 6989 |\n| test | 72.8 | 5100 |\n\nThe table is followed by prose."
        },
        {
            "math-in-prose",
            "a small formula should not classify the entire paragraph as code",
            "For this estimate, x = (a + b) / 2 and the error stays below 0.01. The rest of the argument is descriptive rather than symbolic."
        },
        {
            "tiny-braced-fragment",
            "a single braced symbol in prose should decay immediately",
            "We use {x} as a placeholder in this paragraph, not as a serialized object or a block of source code."
        },
        {
            "url-in-prose",
            "a URL and query value should not create a sustained code classification",
            "See https://example.com/docs?id=3 for the background material, then continue reading the ordinary explanation."
        },
        {
            "key-values-in-prose",
            "two short key/value facts should create at most a brief low boost",
            "The note recorded status=ok and retries=2 before describing why the operation eventually succeeded."
        },
        {
            "path-in-prose",
            "a filesystem path should remain prose",
            "The archive is stored under /var/lib/example/data and can be reviewed later by the operations team."
        },
        {
            "fenced-shell",
            "a fenced command and comment should remain detected until the closing fence",
            "Run the following check:\n\n```sh\n# inspect the service\nsystemctl status example.service\nif [ $? -ne 0 ]; then\n    exit 1\nfi\n```\n\nContinue with the written procedure."
        },
    };
    return samples;
}

static void set_env(const char * name, const std::string & value) {
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static std::string escaped(const std::string & value) {
    std::string result;
    result.reserve(value.size() + 8);
    static const char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        switch (c) {
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            case '"': result += "\\\""; break;
            default:
                if (c < 0x20 || c == 0x7f) {
                    result += "\\x";
                    result += hex[c >> 4];
                    result += hex[c & 0x0f];
                } else {
                    result.push_back((char) c);
                }
                break;
        }
    }
    return result;
}

static std::string flag_names(uint16_t flags) {
    struct flag_name { uint16_t flag; const char * name; };
    static const flag_name names[] = {
        { STF_ALPHA, "alpha" }, { STF_DIGIT, "digit" }, { STF_WHITESPACE, "space" },
        { STF_NEWLINE, "newline" }, { STF_QUOTE, "quote" }, { STF_COLON, "colon" },
        { STF_COMMA, "comma" }, { STF_BRACE, "brace" }, { STF_BRACKET, "bracket" },
        { STF_PAREN, "paren" }, { STF_OPERATOR, "operator" }, { STF_CODE_HINT, "code" },
        { STF_MATH_HINT, "math" }, { STF_FENCE, "fence" }, { STF_COMMENT_HINT, "comment" },
        { STF_STRONG_ANCHOR, "anchor" },
    };
    std::string result;
    for (const auto & item : names) {
        if ((flags & item.flag) == 0) continue;
        if (!result.empty()) result += '|';
        result += item.name;
    }
    return result.empty() ? "none" : result;
}

static const spec_content_roll_entry * newest_entry(
        const common_speculative_content_state & state, int window) {
    if (state.history_size == 0 || window <= 0) return nullptr;
    const int index = state.history_size < (uint8_t) window
            ? (int) state.history_size - 1
            : ((int) state.history_pos + window - 1) % window;
    return &state.history[index];
}

static int raw_level(int score) {
    if (score >= SPEC_CONTENT_SCORE_LEVEL_3) return 3;
    if (score >= SPEC_CONTENT_SCORE_LEVEL_2) return 2;
    if (score >= SPEC_CONTENT_SCORE_LEVEL_1) return 1;
    return 0;
}

struct sample_result {
    std::array<int, 4> levels = {};
    int transitions = 0;
    int tokens = 0;
    int score_max = -999;
    int score_min = 999;
};

static sample_result classify_sample(
        common_speculative_content & classifier, const llama_vocab * vocab,
        const sample_case & sample, int window, int base_width,
        int max_boost, bool detail) {
    classifier.reset(0);
    const auto tokens = common_tokenize(vocab, sample.text, false, true);

    sample_result result;
    int previous = 0;
    if (detail) {
        std::printf("\nDETAIL sample=%s intent=\"%s\" window=%d tokens=%zu\n",
                sample.name, sample.intent, window, tokens.size());
        std::printf(" idx token  piece                              flags                                      score raw sel fence inline draft target\n");
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
        const uint8_t selected = classifier.prepare(0, tokens.data(), i, tokens[i]);
        const auto * state = classifier.state(0);
        if (state == nullptr) continue;
        const auto * entry = newest_entry(*state, window);
        const int raw = raw_level(state->structure_score);
        const int level = std::min<int>(selected, max_boost);
        result.levels[(size_t) level]++;
        result.transitions += i > 0 && level != previous;
        result.tokens++;
        result.score_max = std::max(result.score_max, (int) state->structure_score);
        result.score_min = std::min(result.score_min, (int) state->structure_score);
        previous = level;

        if (detail) {
            std::string piece = escaped(common_token_to_piece(vocab, tokens[i], true));
            if (piece.size() > 34) piece = piece.substr(0, 31) + "...";
            const std::string flags = flag_names(entry != nullptr ? entry->flags : (uint16_t) STF_NONE);
            std::printf("%4zu %6d  %-34s %-42s %5d  %d   %d    %d      %d      %d      %d\n",
                    i, tokens[i], piece.c_str(), flags.c_str(),
                    (int) state->structure_score, raw, level,
                    state->fence_open ? 1 : 0, state->inline_code_open ? 1 : 0,
                    base_width + level, base_width + level + 1);
        }
    }
    return result;
}

static std::vector<int> parse_windows(const std::string & text) {
    std::vector<int> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const int value = std::atoi(item.c_str());
        if (value >= 4 && value <= SPEC_CONTENT_WINDOW_MAX &&
                std::find(result.begin(), result.end(), value) == result.end()) {
            result.push_back(value);
        }
    }
    return result;
}

static void usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s -m MODEL [--windows 4,8,12,16,24,32] [--detail-window 8] "
            "[--base-width 4] [--max-boost 1] [--file TEXT]\n", argv0);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string file_path;
    std::vector<int> windows = { 4, 8, 12, 16, 24, 32 };
    int detail_window = 8;
    int base_width = 4;
    int max_boost = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "-m" || arg == "--model") model_path = value(arg.c_str());
        else if (arg == "--file") file_path = value(arg.c_str());
        else if (arg == "--windows") windows = parse_windows(value(arg.c_str()));
        else if (arg == "--detail-window") detail_window = std::atoi(value(arg.c_str()));
        else if (arg == "--base-width") base_width = std::atoi(value(arg.c_str()));
        else if (arg == "--max-boost") max_boost = std::atoi(value(arg.c_str()));
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", arg.c_str()); usage(argv[0]); return 2; }
    }

    if (model_path.empty() || windows.empty() || detail_window < 4 ||
            detail_window > SPEC_CONTENT_WINDOW_MAX || base_width < 1 ||
            max_boost < 1 || max_boost > 3) {
        usage(argv[0]);
        return 2;
    }
    if (std::find(windows.begin(), windows.end(), detail_window) == windows.end()) {
        windows.push_back(detail_window);
        std::sort(windows.begin(), windows.end());
    }

    common_init();
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load vocabulary from %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<sample_case> samples;
    if (!file_path.empty()) {
        std::ifstream input(file_path, std::ios::binary);
        if (!input) {
            std::fprintf(stderr, "failed to open %s\n", file_path.c_str());
            llama_model_free(model);
            return 1;
        }
        std::stringstream buffer;
        buffer << input.rdbuf();
        samples.push_back({ "external", "user-supplied visual sample", buffer.str() });
    } else {
        samples = builtin_samples();
    }

    std::printf("MODEL %s\nBASE draft=%d max_boost=+%d; target_rows=draft+1\n",
            model_path.c_str(), base_width, max_boost);
    std::printf("SUMMARY columns: window sample tokens +0 +1 +2 +3 boosted%% transitions score[min,max]\n");
    for (int window : windows) {
        set_env("SPEC_CONTENT_BOOST", "1");
        set_env("SPEC_CONTENT_MAX_BOOST", std::to_string(max_boost));
        set_env("SPEC_CONTENT_WINDOW", std::to_string(window));
        set_env("SPEC_CONTENT_TRACE", "0");
        set_env("SPEC_CONTENT_PROVISIONAL", "0");
        common_speculative_content classifier;
        classifier.init(vocab, 1, true, base_width);

        for (const auto & sample : samples) {
            const bool detail = window == detail_window;
            const auto result = classify_sample(
                    classifier, vocab, sample, window, base_width, max_boost, detail);
            const int boosted = result.tokens - result.levels[0];
            const double boosted_pct = result.tokens > 0 ? 100.0 * boosted / result.tokens : 0.0;
            std::printf("SUMMARY %2d %-22s %4d %4d %4d %4d %4d %7.2f %4d [%d,%d]  # %s\n",
                    window, sample.name, result.tokens,
                    result.levels[0], result.levels[1], result.levels[2], result.levels[3],
                    boosted_pct, result.transitions, result.score_min, result.score_max,
                    sample.intent);
        }
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
