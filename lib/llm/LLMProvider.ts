import {
  UnsupportedModelError,
  UnsupportedModelProviderError,
} from "@/types/stagehandErrors";
import { LanguageModel } from "ai";
import { LogLine } from "../../types/log";
import {
  AvailableModel,
  ClientOptions,
  ModelProvider,
} from "../../types/model";
import { LLMCache } from "../cache/LLMCache";
import { AISdkClient } from "./aisdk";
import { AnthropicClient } from "./AnthropicClient";
import { CerebrasClient } from "./CerebrasClient";
import { GoogleClient } from "./GoogleClient";
import { GroqClient } from "./GroqClient";
import { LLMClient } from "./LLMClient";
import { OpenAIClient } from "./OpenAIClient";
import { openai } from "@ai-sdk/openai";
import { anthropic } from "@ai-sdk/anthropic";
import { google } from "@ai-sdk/google";

const modelToProviderMap: { [key in AvailableModel]: ModelProvider } = {
  "gpt-4.1": "openai",
  "gpt-4.1-mini": "openai",
  "gpt-4.1-nano": "openai",
  "o4-mini": "openai",
  //prettier-ignore
  "o3": "openai",
  "o3-mini": "openai",
  //prettier-ignore
  "o1": "openai",
  "o1-mini": "openai",
  "gpt-4o": "openai",
  "gpt-4o-mini": "openai",
  "gpt-4o-2024-08-06": "openai",
  "gpt-4.5-preview": "openai",
  "o1-preview": "openai",
  "claude-3-5-sonnet-latest": "anthropic",
  "claude-3-5-sonnet-20240620": "anthropic",
  "claude-3-5-sonnet-20241022": "anthropic",
  "claude-3-7-sonnet-20250219": "anthropic",
  "claude-3-7-sonnet-latest": "anthropic",
  "cerebras-llama-3.3-70b": "cerebras",
  "cerebras-llama-3.1-8b": "cerebras",
  "groq-llama-3.3-70b-versatile": "groq",
  "groq-llama-3.3-70b-specdec": "groq",
  "gemini-1.5-flash": "google",
  "gemini-1.5-pro": "google",
  "gemini-1.5-flash-8b": "google",
  "gemini-2.0-flash-lite": "google",
  "gemini-2.0-flash": "google",
  "gemini-2.5-flash-preview-04-17": "google",
  "gemini-2.5-pro-preview-03-25": "google",
  "aisdk/anthropic/claude-3-5-sonnet-latest": "aisdk",
  "aisdk/anthropic/claude-3-5-sonnet-20240620": "aisdk",
  "aisdk/anthropicclaude-3-5-sonnet-20241022": "aisdk",
  "aisdk/anthropic/claude-3-7-sonnet-20250219": "aisdk",
  "aisdk/anthropic/claude-3-7-sonnet-latest": "aisdk",
  "aisdk/google/gemini-1.5-flash": "aisdk",
  "aisdk/google/gemini-1.5-pro": "aisdk",
  "aisdk/google/gemini-1.5-flash-8b": "aisdk",
  "aisdk/google/gemini-2.0-flash-lite": "aisdk",
  "aisdk/google/gemini-2.0-flash": "aisdk",
  "aisdk/google/gemini-2.5-flash-preview-04-17": "aisdk",
  "aisdk/google/gemini-2.5-pro-preview-03-25": "aisdk",
  "aisdk/openai/gpt-4.1": "aisdk",
  "aisdk/openai/gpt-4.1-mini": "aisdk",
  "aisdk/openai/gpt-4.1-nano": "aisdk",
  "aisdk/openai/o4-mini": "aisdk",
  "aisdk/openai/o3": "aisdk",
  "aisdk/openai/o3-mini": "aisdk",
  "aisdk/openai/o1": "aisdk",
  "aisdk/openai/o1-mini": "aisdk",
  "aisdk/openai/gpt-4o": "aisdk",
  "aisdk/openai/gpt-4o-mini": "aisdk",
  "aisdk/openai/gpt-4o-2024-08-06": "aisdk",
  "aisdk/openai/gpt-4.5-preview": "aisdk",
  "aisdk/openai/o1-preview": "aisdk",
};

export class LLMProvider {
  private logger: (message: LogLine) => void;
  private enableCaching: boolean;
  private cache: LLMCache | undefined;

  constructor(logger: (message: LogLine) => void, enableCaching: boolean) {
    this.logger = logger;
    this.enableCaching = enableCaching;
    this.cache = enableCaching ? new LLMCache(logger) : undefined;
  }

  cleanRequestCache(requestId: string): void {
    if (!this.enableCaching) {
      return;
    }

    this.logger({
      category: "llm_cache",
      message: "cleaning up cache",
      level: 1,
      auxiliary: {
        requestId: {
          value: requestId,
          type: "string",
        },
      },
    });
    this.cache.deleteCacheForRequestId(requestId);
  }

  getClient(
    modelName: AvailableModel,
    clientOptions?: ClientOptions,
  ): LLMClient {
    const provider = modelToProviderMap[modelName];
    if (!provider) {
      throw new UnsupportedModelError(Object.keys(modelToProviderMap));
    }

    if (provider === "aisdk") {
      const parts = modelName.split("/");
      if (parts.length !== 3) {
        throw new Error(`Invalid aisdk model format: ${modelName}`);
      }

      const [, subProvider, subModelName] = parts;
      let languageModel: LanguageModel;

      switch (subProvider) {
        case "openai":
          languageModel = openai(subModelName);
          break;
        case "anthropic":
          languageModel = anthropic(subModelName);
          break;
        case "google":
          languageModel = google(subModelName);
          break;
        default:
          throw new Error(`Unsupported aisdk sub-provider: ${subProvider}`);
      }

      return new AISdkClient({
        model: languageModel,
        logger: this.logger,
        enableCaching: this.enableCaching,
        cache: this.cache,
      });
    }

    const availableModel = modelName as AvailableModel;
    switch (provider) {
      case "openai":
        return new OpenAIClient({
          logger: this.logger,
          enableCaching: this.enableCaching,
          cache: this.cache,
          modelName: availableModel,
          clientOptions,
        });
      case "anthropic":
        return new AnthropicClient({
          logger: this.logger,
          enableCaching: this.enableCaching,
          cache: this.cache,
          modelName: availableModel,
          clientOptions,
        });
      case "cerebras":
        return new CerebrasClient({
          logger: this.logger,
          enableCaching: this.enableCaching,
          cache: this.cache,
          modelName: availableModel,
          clientOptions,
        });
      case "groq":
        return new GroqClient({
          logger: this.logger,
          enableCaching: this.enableCaching,
          cache: this.cache,
          modelName: availableModel,
          clientOptions,
        });
      case "google":
        return new GoogleClient({
          logger: this.logger,
          enableCaching: this.enableCaching,
          cache: this.cache,
          modelName: availableModel,
          clientOptions,
        });
      default:
        throw new UnsupportedModelProviderError([
          ...new Set(Object.values(modelToProviderMap)),
        ]);
    }
  }

  static getModelProvider(modelName: AvailableModel): ModelProvider {
    const provider = modelToProviderMap[modelName];

    return provider;
  }
}
