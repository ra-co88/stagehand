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

function modelToProvider(
  modelName: AvailableModel | LanguageModel,
): ModelProvider {
  if (typeof modelName === "string") {
    const provider = modelToProviderMap[modelName];
    if (!provider) {
      throw new UnsupportedModelError(Object.keys(modelToProviderMap));
    }
    return provider;
  }
  return "aisdk";
}

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
    modelName: AvailableModel | LanguageModel,
    clientOptions?: ClientOptions,
  ): LLMClient {
    const provider = modelToProvider(modelName);
    if (!provider) {
      throw new UnsupportedModelError(Object.keys(modelToProviderMap));
    }

    if (provider === "aisdk") {
      return new AISdkClient({
        model: modelName as LanguageModel,
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
