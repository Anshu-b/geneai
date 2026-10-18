export interface Classification {
  domain: "shopping" | "cloud_scheduling" | "quant" | null;
  supported: boolean;
  title: string;
  reason: string;
}

export function classifyGoal(prompt: string): Classification {
  const text = prompt.toLowerCase();
  if (!text.trim()) {
    return {
      domain: null,
      supported: false,
      title: "Waiting for a goal",
      reason: "GeneAI only launches a domain that has an evaluator. An empty prompt does not start a run.",
    };
  }
  if (/(outfit|wardrobe|clothing|clothes|catalog|lululemon|dress code)/.test(text)) {
    return {
      domain: "shopping",
      supported: true,
      title: "Shopping adapter",
      reason:
        "This maps to the shopping evaluator: catalog outfits under hard constraints. It will not invent products or a checkout.",
    };
  }
  if (/(schedul|queue|cloud|server|latency|workload|datacenter|data center|kubernetes)/.test(text)) {
    return {
      domain: "cloud_scheduling",
      supported: true,
      title: "Cloud scheduling adapter",
      reason:
        "This maps to the cloud scheduler. Fitness comes from the simulator, not from the model's confidence.",
    };
  }
  if (/(trad(e|ing)|equity|sharpe|portfolio|backtest|ticker|stock)/.test(text)) {
    return {
      domain: "quant",
      supported: false,
      title: "Quant is not available",
      reason:
        "The engine has no quant adapter in this build. GeneAI will not assign a backtest score to this prompt.",
    };
  }
  return {
    domain: null,
    supported: false,
    title: "No evaluator for this goal",
    reason:
      "Ask GeneAI only runs a registered adapter. This prompt does not match shopping or cloud scheduling, so there is no fitness to evolve.",
  };
}

export interface LaunchRequest {
  objective: {
    domain: string;
    prompt: string;
    constraints: Record<string, unknown>;
    evaluation_version: string;
  };
  config: {
    seed: number;
    population_size: number;
    generations: number;
    elite_count: number;
  };
}

export function launchBody(prompt: string, domain: "shopping" | "cloud_scheduling"): LaunchRequest {
  return {
    objective: {
      domain,
      prompt,
      constraints: {},
      evaluation_version: "unspecified",
    },
    config: {
      seed: 42,
      population_size: 4,
      generations: 2,
      elite_count: 1,
    },
  };
}
