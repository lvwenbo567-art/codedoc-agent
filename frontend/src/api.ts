import type {
  AgentFeedbackCreateRequest,
  BuildIndexRequest,
  BuildIndexResponse,
  HITLAgentResponse,
  HumanReviewDecision,
  ProjectConfig,
  ScanProjectRequest,
  ScanProjectResponse,
  SSELogItem,
  UploadProjectZipResponse,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function buildUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;

  if (!response.ok) {
    const message =
      typeof data?.error?.message === "string"
        ? data.error.message
        : typeof data?.message === "string"
          ? data.message
          : typeof data?.detail?.message === "string"
            ? data.detail.message
            : Array.isArray(data?.detail)
              ? data.detail
                  .map((item: { msg?: string }) => item.msg)
                  .filter(Boolean)
                  .join("；")
              : typeof data?.detail === "string"
                ? data.detail
                : JSON.stringify(data, null, 2);

    throw new Error(message || `HTTP ${response.status}`);
  }

  return data as T;
}

function buildBasePayload(config: ProjectConfig) {
  return {
    project_id: config.project_id,
    thread_id: config.thread_id,
    project_root: config.project_root,
    chunks_path: config.chunks_path,
    index_path: config.index_path,
    embedding_provider: config.embedding_provider,
    embedding_model: config.embedding_model,
    embedding_base_url: config.embedding_base_url,
    rerank_provider: config.rerank_provider,
    rerank_model: config.rerank_model,
    rerank_local_files_only: config.rerank_local_files_only,
    enable_human_review: config.enable_human_review,
    approval_required_tools: config.approval_required_tools,
    recursion_limit: config.recursion_limit,
  };
}

export async function startHitlAgent(
  query: string,
  config: ProjectConfig,
): Promise<HITLAgentResponse> {
  const response = await fetch(buildUrl("/langgraph/hitl/start"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ...buildBasePayload(config),
      query,
    }),
  });

  return parseJsonResponse<HITLAgentResponse>(response);
}

export async function resumeHitlAgent(
  decision: HumanReviewDecision,
  config: ProjectConfig,
): Promise<HITLAgentResponse> {
  const response = await fetch(buildUrl("/langgraph/hitl/resume"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ...buildBasePayload(config),
      decision,
    }),
  });

  return parseJsonResponse<HITLAgentResponse>(response);
}

export async function submitFeedback(
  payload: AgentFeedbackCreateRequest,
): Promise<unknown> {
  const response = await fetch(buildUrl("/agent-quality/feedback"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return parseJsonResponse<unknown>(response);
}

export async function listFeedback(projectId: number): Promise<unknown> {
  const response = await fetch(
    buildUrl(`/agent-quality/feedback?project_id=${projectId}&limit=20`),
  );

  return parseJsonResponse<unknown>(response);
}

export async function listBadCases(projectId: number): Promise<unknown> {
  const response = await fetch(
    buildUrl(`/agent-quality/bad-cases?project_id=${projectId}&limit=20`),
  );

  return parseJsonResponse<unknown>(response);
}

export async function fetchVersion(): Promise<unknown> {
  const response = await fetch(buildUrl("/version"));

  return parseJsonResponse<unknown>(response);
}

export async function scanProject(
  payload: ScanProjectRequest,
): Promise<ScanProjectResponse> {
  const response = await fetch(buildUrl("/scan"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return parseJsonResponse<ScanProjectResponse>(response);
}

export async function buildVectorIndex(
  payload: BuildIndexRequest,
): Promise<BuildIndexResponse> {
  const response = await fetch(buildUrl("/index"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return parseJsonResponse<BuildIndexResponse>(response);
}

export async function uploadProjectZip(
  file: File,
  projectName: string,
): Promise<UploadProjectZipResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("project_name", projectName);

  const response = await fetch(buildUrl("/project-upload/zip"), {
    method: "POST",
    body: formData,
  });

  return parseJsonResponse<UploadProjectZipResponse>(response);
}

function parseSSEBlock(block: string): { event: string; data: unknown } | null {
  const lines = block.split("\n");
  const eventLine = lines.find((line) => line.startsWith("event:"));
  const dataLines = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trim());

  if (!eventLine && dataLines.length === 0) {
    return null;
  }

  const event = eventLine?.slice("event:".length).trim() || "message";
  const rawData = dataLines.join("\n");

  try {
    return {
      event,
      data: rawData ? JSON.parse(rawData) : null,
    };
  } catch {
    return {
      event,
      data: rawData,
    };
  }
}

export async function streamHitlAgent(
  query: string,
  config: ProjectConfig,
  onEvent: (item: SSELogItem) => void,
): Promise<void> {
  const response = await fetch(buildUrl("/langgraph/hitl/stream"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ...buildBasePayload(config),
      query,
    }),
  });

  if (!response.ok || !response.body) {
    await parseJsonResponse(response);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const parsed = parseSSEBlock(block);

      if (!parsed) {
        continue;
      }

      onEvent({
        id: crypto.randomUUID(),
        event: parsed.event,
        data: parsed.data,
        receivedAt: new Date().toLocaleTimeString(),
      });
    }
  }
}
