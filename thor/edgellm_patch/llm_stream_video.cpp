/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// Persistent single-frame streaming driver for live video / camera captioning.
//
// Loads the LLM + multimodal engines ONCE, then loops reading one path per
// line from stdin. Each line must point to a small llm_inference-style JSON
// request file (single request, single image). Prints one JSON line per
// response to stdout: {"ok":bool,"latency_ms":double,"output_text":str,
// "finish_reason":str}. A line containing exactly "QUIT" ends the loop.
//
// Usage:
//   llm_stream_video --engineDir DIR --multimodalEngineDir DIR
//
// Intended to be driven by a wrapper (e.g. a Python script) that extracts
// frames from a video or camera, writes a per-frame request JSON, and reads
// back the caption for that frame — without paying engine-load cost per frame.

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "requestFileParser.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

int main(int argc, char** argv)
{
    std::string engineDir;
    std::string multimodalEngineDir;
    for (int i = 1; i < argc; ++i)
    {
        std::string const a = argv[i];
        if (a == "--engineDir" && i + 1 < argc)
        {
            engineDir = argv[++i];
        }
        else if (a == "--multimodalEngineDir" && i + 1 < argc)
        {
            multimodalEngineDir = argv[++i];
        }
    }
    if (engineDir.empty())
    {
        std::cerr << "Usage: " << argv[0] << " --engineDir DIR [--multimodalEngineDir DIR]\n";
        return 1;
    }

    // Keep stdout limited to the one-JSON-object-per-line response protocol;
    // the library logger sends INFO (and below) to stdout, WARNING+ to stderr.
    gLogger.setLevel(nvinfer1::ILogger::Severity::kWARNING);

    auto pluginHandles = loadEdgellmPluginLib();

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::unique_ptr<rt::LLMInferenceRuntime> runtime;
    try
    {
        runtime = std::make_unique<rt::LLMInferenceRuntime>(
            engineDir, multimodalEngineDir, std::unordered_map<std::string, std::string>{}, stream);
    }
    catch (std::exception const& e)
    {
        std::cerr << "Runtime init failed: " << e.what() << "\n";
        return 2;
    }

    if (!runtime->captureDecodingCUDAGraph(stream))
    {
        std::cerr << "Warning: CUDA graph capture failed, proceeding without.\n";
    }

    // Signal readiness on stderr so a driving process can wait for engine load
    // to finish (engine load takes a couple seconds; per-frame calls after
    // this point do not pay that cost again).
    std::cerr << "READY" << std::endl;

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            continue;
        }
        if (line == "QUIT")
        {
            break;
        }

        auto const t0 = std::chrono::steady_clock::now();

        std::unordered_map<std::string, std::string> loraMap;
        std::vector<rt::LLMGenerationRequest> batches;
        try
        {
            std::tie(loraMap, batches) = exampleUtils::parseRequestFile(line, /*batchSizeOverride=*/1,
                /*maxGenerateLengthOverride=*/-1);
        }
        catch (std::exception const& e)
        {
            Json err;
            err["ok"] = false;
            err["error"] = e.what();
            std::cout << err.dump() << std::endl;
            continue;
        }

        if (batches.empty() || batches.front().requests.empty())
        {
            Json err;
            err["ok"] = false;
            err["error"] = "no requests parsed from request file";
            std::cout << err.dump() << std::endl;
            continue;
        }

        rt::LLMGenerationResponse response;
        bool const ok = runtime->handleRequest(batches.front(), response, stream);

        auto const t1 = std::chrono::steady_clock::now();
        double const latencyMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        Json out;
        out["ok"] = ok;
        out["latency_ms"] = latencyMs;
        out["output_text"] = (ok && !response.outputTexts.empty()) ? response.outputTexts.front() : std::string();
        out["finish_reason"]
            = (ok && !response.finishReasons.empty()) ? rt::finishReasonName(response.finishReasons.front()) : "error";
        std::cout << out.dump() << std::endl;
        std::cout.flush();
    }

    runtime.reset();
    cudaStreamDestroy(stream);
    return 0;
}
