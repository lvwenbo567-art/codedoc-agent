export type ToolCallHistoryItem = {
  sequence?: number;
  tool_call_id?: string;
  tool_name?: string;
  arguments?: Record<string, unknown>;
  signature?: string;
  repeat_index?: number;
  model_call_index?: number;
};

export type MessageTraceItem = {
  type?: string;
  content?: string;
  tool_calls?: Array<{
    id?: string;
    name?: string;
    args?: Record<string, unknown>;
    type?: string;
  }>;
  tool_call_id?: string;
  name?: string;
};

export type HITLInterrupt = {
  type?: string;
  request_id?: string;
  project_id?: number;
  thread_id?: string;
  effective_thread_id?: string;
  query?: string;
  tool_calls?: Array<{
    id: string;
    name: string;
    args: Record<string, unknown>;
  }>;
  instructions?: string;
};

export type HITLAgentResponse = {
  query: string;
  project_id: number;
  thread_id: string;
  effective_thread_id: string;
  run_id: string;
  answer: string;
  status: string;
  success: boolean;
  completed: boolean;
  stop_reason: string;
  interrupts: HITLInterrupt[];
  approval_status: string;
  review_history: Array<Record<string, unknown>>;
  turn_index: number;
  model_call_count: number;
  tool_call_count: number;
  message_count: number;
  message_trace: MessageTraceItem[];
  tool_call_history: ToolCallHistoryItem[];
  execution_steps: string[];
  checkpoint_id?: string | null;
  total_duration_ms: number;
  error_message?: string | null;
  allowed_tools: string[];
  provider: string;
  model_name: string;
  evidence?: Array<Record<string, unknown>>;
  citations?: Array<Record<string, unknown>>;
  answer_quality?: Record<string, unknown>;
};

export type ProjectConfig = {
  project_id: number;
  thread_id: string;
  project_name: string;
  project_root: string;
  chunks_path: string;
  index_path: string;
  embedding_provider: string;
  embedding_model: string;
  embedding_base_url: string;
  rerank_provider: string;
  rerank_model: string;
  rerank_local_files_only: boolean;
  enable_human_review: boolean;
  approval_required_tools: string[];
  recursion_limit: number;
};

export type HumanReviewDecision =
  | {
      decision: "approve";
      feedback?: string;
    }
  | {
      decision: "reject";
      feedback?: string;
    }
  | {
      decision: "edit";
      feedback?: string;
      edited_tool_calls: Array<{
        id: string;
        name: string;
        args: Record<string, unknown>;
      }>;
    };

export type SSELogItem = {
  id: string;
  event: string;
  data: unknown;
  receivedAt: string;
};

export type AgentFeedbackCreateRequest = {
  project_id: number;
  thread_id: string;
  run_id?: string | null;
  query: string;
  answer: string;
  rating: -1 | 0 | 1;
  issue_tags: string[];
  comment?: string | null;
  corrected_answer?: string | null;
};

export type EvalSummary = {
  total_cases?: number;
  passed_cases?: number;
  failed_cases?: number;
  task_success_rate?: number;
  tool_exact_match_rate?: number;
  average_tool_precision?: number;
  average_tool_recall?: number;
  average_tool_f1?: number;
  first_tool_accuracy?: number;
  forbidden_tool_safety_rate?: number;
  average_answer_term_coverage?: number;
  completion_rate?: number;
  latency_pass_rate?: number;
  average_latency_ms?: number;
  p95_latency_ms?: number;
};

export type ScanProjectRequest = {
  project_path: string;
  chunk_size: number;
  overlap: number;
  save_chunks: boolean;
  output_path: string;
  save_to_db: boolean;
};

export type ScanProjectResponse = {
  success?: boolean;
  data?: {
    project_path: string;
    project_id: number | null;
    file_count: number;
    chunk_count: number;
    chunk_stats?: Record<string, unknown>;
    files?: Array<Record<string, unknown>>;
    chunk_previews?: Array<Record<string, unknown>>;
    saved_path?: string | null;
    db_path?: string | null;
  };
};

export type BuildIndexRequest = {
  chunks_path: string;
  output_path: string;
  embedding_provider: string;
  embedding_model: string;
  embedding_base_url: string;
  embedding_api_key?: string;
  batch_size: number;
  incremental: boolean;
};

export type BuildIndexResponse = {
  success?: boolean;
  data?: {
    chunks_path: string;
    output_path: string;
    embedding_provider: string;
    embedding_model: string;
    dimension: number;
    chunk_count: number;
    vector_count: number;
    incremental: boolean;
    build_stats?: Record<string, unknown>;
    update_stats?: Record<string, unknown>;
  };
};

export type ProjectSetupState = {
  projectName: string;
  projectPath: string;
  chunkSize: number;
  overlap: number;
  chunksOutputPath: string;
  indexOutputPath: string;
};

export type UploadProjectZipResponse = {
  success?: boolean;
  data?: {
    project_name: string;
    upload_id: string;
    filename: string;
    project_path: string;
    zip_path: string;
    extracted_file_count: number;
  };
};

export type AppView = "setup" | "workspace" | "evaluation" | "settings";

export type SkillDefinition = {
  skill_name: string;
  display_name: string;
  description: string;
  intent_keywords: string[];
  example_queries: string[];
  recommended_tools: string[];
  output_sections: string[];
  requires_human_review_tools: string[];
};

export type SkillRouteResponse = {
  success?: boolean;
  data?: {
    route: {
      query: string;
      skill_name: string;
      display_name: string;
      confidence: number;
      matched_keywords: string[];
      recommended_tools: string[];
      output_sections: string[];
      reason: string;
    };
    plan: Record<string, unknown>;
  };
};

export type McpTool = {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
};

export type McpResource = {
  uri: string;
  name: string;
  description: string;
  mime_type: string;
};

export type McpPrompt = {
  name: string;
  description: string;
  template: string;
};

export type IngestionJobResponse = {
  job_id: string;
  project_id: number;
  status: string;
  stage: string;
  progress: number;
  attempt: number;
  request_data: Record<string, unknown>;
  result_data: Record<string, unknown>;
  parent_job_id: string | null;
  cancel_requested: boolean;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  finished_at: string | null;
  error_type: string | null;
  error_message: string | null;
};
