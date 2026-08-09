import { FormEvent, ReactNode, useEffect, useMemo, useState } from "react";
import {
  buildVectorIndex,
  fetchVersion,
  listBadCases,
  listFeedback,
  promoteFeedbackToBadCase,
  resumeHitlAgent,
  scanProject,
  startHitlAgent,
  streamHitlAgent,
  submitFeedback,
  uploadProjectZip,
} from "./api";
import type {
  AppView,
  BuildIndexResponse,
  EvalSummary,
  HITLAgentResponse,
  HITLInterrupt,
  HumanReviewDecision,
  ProjectConfig,
  ProjectSetupState,
  ScanProjectResponse,
  SSELogItem,
  ToolCallHistoryItem,
} from "./types";

type ChatRole = "user" | "assistant" | "system";
type ChatStatus = "idle" | "running" | "streaming" | "interrupted" | "completed" | "failed";

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  status?: ChatStatus;
  response?: HITLAgentResponse | null;
};

type ChatSession = {
  id: string;
  title: string;
  threadId: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
  latestResponse: HITLAgentResponse | null;
  events: SSELogItem[];
};

type ProjectWorkspaceRecord = {
  id: string;
  projectName: string;
  projectPath: string;
  projectId: number;
  chunksPath: string;
  indexPath: string;
  uploadedFilename?: string;
  fileCount?: number;
  chunkCount?: number;
  vectorCount?: number;
  dimension?: number;
  status: "uploaded" | "scanned" | "indexed";
  createdAt: string;
  updatedAt: string;
};

type WorkspaceSessionState = {
  sessions: ChatSession[];
  activeSessionId: string;
};

const SESSION_STORAGE_KEY = "codedoc-agent.frontend.sessions.v3";
const ACTIVE_SESSION_KEY = "codedoc-agent.frontend.active-session.v3";
const WORKSPACE_SESSION_STORAGE_KEY = "codedoc-agent.frontend.workspace-sessions.v1";
const PROJECT_RECORD_STORAGE_KEY = "codedoc-agent.frontend.project-records.v1";

const defaultSetup: ProjectSetupState = {
  projectName: "test_project",
  projectPath: "test_project",
  chunkSize: 1200,
  overlap: 120,
  chunksOutputPath: "outputs/test_project_chunks.json",
  indexOutputPath: "outputs/test_project_vector_index_bge_m3.json",
};

const defaultConfig: ProjectConfig = {
  project_id: 1,
  project_name: "test_project",
  thread_id: "frontend-demo",
  project_root: "test_project",
  chunks_path: defaultSetup.chunksOutputPath,
  index_path: defaultSetup.indexOutputPath,
  embedding_provider: "ollama",
  embedding_model: "bge-m3",
  embedding_base_url: "http://localhost:11434",
  rerank_provider: "sentence_transformers",
  rerank_model: "D:/models/bge-reranker-v2-m3",
  rerank_local_files_only: true,
  enable_human_review: true,
  approval_required_tools: ["run_project_tests"],
  recursion_limit: 40,
};

const exampleQuestions = [
  "keyword_score 在哪里定义？",
  "项目有哪些主要模块？",
  "README 里怎么启动项目？",
  "运行 tests/test_search.py",
];

function nowText(): string {
  return new Date().toLocaleTimeString();
}

function createId(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }

  return `${prefix}-${Date.now()}-${Math.round(Math.random() * 100000)}`;
}

function createThreadId(prefix = "chat"): string {
  const stamp = new Date()
    .toISOString()
    .replace(/[-:.TZ]/g, "")
    .slice(0, 14);

  return `${prefix}-${stamp}`;
}

function removeZipSuffix(filename: string): string {
  return filename.replace(/\.zip$/i, "").trim();
}

function normalizeProjectName(value: string): string {
  const cleaned = value
    .trim()
    .replace(/[\\/:*?"<>|]/g, "_")
    .replace(/\s+/g, "_");

  return cleaned || "uploaded_project";
}

function createSession(threadId = createThreadId("chat")): ChatSession {
  return {
    id: createId("session"),
    title: "新会话",
    threadId,
    createdAt: nowText(),
    updatedAt: nowText(),
    messages: [
      {
        id: createId("msg"),
        role: "system",
        content:
          "已创建会话。你可以上传/选择项目，完成扫描和索引后开始进行代码仓库问答。",
        createdAt: nowText(),
        status: "completed",
      },
    ],
    latestResponse: null,
    events: [],
  };
}

function loadSessions(): ChatSession[] {
  try {
    const raw = localStorage.getItem(SESSION_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;

    if (Array.isArray(parsed) && parsed.length > 0) {
      return parsed;
    }
  } catch {
    // localStorage 数据损坏时直接重新创建，不影响后端。
  }

  return [createSession(defaultConfig.thread_id)];
}

function loadProjectRecords(): ProjectWorkspaceRecord[] {
  try {
    const raw = localStorage.getItem(PROJECT_RECORD_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;

    if (Array.isArray(parsed)) {
      return parsed.filter(
        (item): item is ProjectWorkspaceRecord =>
          Boolean(item?.id && item?.projectName && item?.projectPath),
      );
    }
  } catch {
    // 项目记录损坏时不影响核心问答功能，直接返回空列表。
  }

  return [];
}

function loadWorkspaceSessionMap(): Record<string, WorkspaceSessionState> {
  try {
    const raw = localStorage.getItem(WORKSPACE_SESSION_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;

    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, WorkspaceSessionState>;
    }
  } catch {
    // 会话分组损坏时回退为空，不影响项目继续使用。
  }

  return {};
}

function getWorkspaceKeyFromSetup(setup: ProjectSetupState): string {
  return setup.projectPath || setup.projectName || "default";
}

function getWorkspaceKeyFromRecord(record: ProjectWorkspaceRecord): string {
  return record.projectPath || record.projectName || record.id;
}

function createWorkspaceSessionState(
  threadPrefix = "chat",
): WorkspaceSessionState {
  const session = createSession(createThreadId(threadPrefix));

  return {
    sessions: [session],
    activeSessionId: session.id,
  };
}

function buildProjectRecordFromState({
  setup,
  config,
  scanResult,
  indexResult,
  uploadedFilename,
  status,
}: {
  setup: ProjectSetupState;
  config: ProjectConfig;
  scanResult: ScanProjectResponse | null;
  indexResult: BuildIndexResponse | null;
  uploadedFilename?: string;
  status: ProjectWorkspaceRecord["status"];
}): ProjectWorkspaceRecord {
  const scanData = scanResult?.data;
  const indexData = indexResult?.data;
  const stamp = nowText();

  return {
    id: setup.projectPath || setup.projectName,
    projectName: setup.projectName,
    projectPath: setup.projectPath,
    projectId: scanData?.project_id ?? (status === "uploaded" ? 0 : config.project_id ?? 1),
    chunksPath: scanData?.saved_path || setup.chunksOutputPath,
    indexPath: indexData?.output_path || setup.indexOutputPath,
    uploadedFilename,
    fileCount: scanData?.file_count,
    chunkCount: scanData?.chunk_count,
    vectorCount: indexData?.vector_count,
    dimension: indexData?.dimension,
    status,
    createdAt: stamp,
    updatedAt: stamp,
  };
}

function mergeProjectRecord(
  records: ProjectWorkspaceRecord[],
  record: ProjectWorkspaceRecord,
): ProjectWorkspaceRecord[] {
  const existing = records.find(
    (item) => item.projectPath === record.projectPath || item.id === record.id,
  );
  const merged: ProjectWorkspaceRecord = existing
    ? {
        ...existing,
        ...record,
        id: existing.id,
        createdAt: existing.createdAt,
        updatedAt: nowText(),
      }
    : record;

  return [merged, ...records.filter((item) => item.id !== merged.id)].slice(0, 20);
}

function formatJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function shortText(text: string, maxLength = 34): string {
  const normalized = text.trim().replace(/\s+/g, " ");

  if (normalized.length <= maxLength) {
    return normalized || "新会话";
  }

  return `${normalized.slice(0, maxLength)}...`;
}

function percent(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return `${Math.round(value * 100)}%`;
}

function numberText(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) {
    return "-";
  }

  return String(Math.round(value));
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function readString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function readNestedAnswer(value: unknown): string {
  const mapping = asRecord(value);
  const directAnswer = readString(mapping.answer);

  if (directAnswer) {
    return directAnswer;
  }

  const outputAnswer = readString(asRecord(mapping.output).answer);

  if (outputAnswer) {
    return outputAnswer;
  }

  for (const nestedValue of Object.values(mapping)) {
    const nestedAnswer = readString(asRecord(nestedValue).answer);

    if (nestedAnswer) {
      return nestedAnswer;
    }
  }

  return "";
}

function summarizeNodeUpdate(data: unknown): string {
  const mapping = asRecord(data);
  const nodeNames = Object.keys(mapping).filter((key) => key !== "__interrupt__");

  if (!nodeNames.length) {
    return "收到工作流状态更新";
  }

  return `节点更新：${nodeNames.join("、")}`;
}

function appendStreamingLine(content: string, line: string): string {
  const base = content.startsWith("### 实时执行进度")
    ? content
    : "### 实时执行进度\n\n- 已连接 SSE，等待 Agent 输出...";

  if (base.includes(line)) {
    return base;
  }

  return `${base}\n- ${line}`;
}

function renderInlineMarkdown(text: string): ReactNode[] {
  const parts = text.split(/(\[[^\]]+\]\([^)]+\)|`[^`]+`|\*\*[^*]+\*\*)/g);

  return parts.map((part, index) => {
    const linkMatch = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(part);

    if (linkMatch) {
      return (
        <a href={linkMatch[2]} key={index} rel="noreferrer" target="_blank">
          {linkMatch[1]}
        </a>
      );
    }

    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={index}>{part.slice(1, -1)}</code>;
    }

    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }

    return <span key={index}>{part}</span>;
  });
}

function renderHeading(level: number, children: ReactNode[], key: string): ReactNode {
  if (level <= 1) {
    return <h2 key={key}>{children}</h2>;
  }

  if (level === 2) {
    return <h3 key={key}>{children}</h3>;
  }

  if (level === 3) {
    return <h4 key={key}>{children}</h4>;
  }

  if (level === 4) {
    return <h5 key={key}>{children}</h5>;
  }

  return <h6 key={key}>{children}</h6>;
}

function isHorizontalRule(line: string): boolean {
  return /^([-*_]\s*){3,}$/.test(line.trim());
}

function splitTableRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isTableSeparator(line: string): boolean {
  const cells = splitTableRow(line);

  return cells.length >= 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function isTableStart(currentLine: string, nextLine: string | undefined): boolean {
  return Boolean(nextLine && currentLine.includes("|") && isTableSeparator(nextLine));
}

function renderMarkdownTable(tableLines: string[], key: string): ReactNode {
  const headerCells = splitTableRow(tableLines[0]);
  const bodyRows = tableLines.slice(2).map(splitTableRow);

  return (
    <div className="markdown-table-wrap" key={key}>
      <table>
        <thead>
          <tr>
            {headerCells.map((cell, index) => (
              <th key={`th-${index}`}>{renderInlineMarkdown(cell)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bodyRows.map((row, rowIndex) => (
            <tr key={`tr-${rowIndex}`}>
              {headerCells.map((_, cellIndex) => (
                <td key={`td-${rowIndex}-${cellIndex}`}>
                  {renderInlineMarkdown(row[cellIndex] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MarkdownRenderer({ content }: { content: string }) {
  const lines = content.split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    if (isHorizontalRule(trimmed)) {
      blocks.push(<hr key={`hr-${index}`} />);
      index += 1;
      continue;
    }

    if (isTableStart(trimmed, lines[index + 1])) {
      const tableLines = [trimmed, lines[index + 1].trim()];
      index += 2;

      while (index < lines.length && lines[index].trim().includes("|")) {
        tableLines.push(lines[index].trim());
        index += 1;
      }

      blocks.push(renderMarkdownTable(tableLines, `table-${index}`));
      continue;
    }

    if (trimmed.startsWith("```")) {
      const codeLines: string[] = [];
      index += 1;

      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }

      if (index < lines.length) {
        index += 1;
      }

      blocks.push(
        <pre className="markdown-code" key={`code-${index}`}>
          <code>{codeLines.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    const headingMatch = /^(#{1,6})\s+(.+)$/.exec(trimmed);

    if (headingMatch) {
      blocks.push(
        renderHeading(
          headingMatch[1].length,
          renderInlineMarkdown(headingMatch[2]),
          `h-${index}`,
        ),
      );
      index += 1;
      continue;
    }

    if (trimmed.startsWith(">")) {
      const quoteLines: string[] = [];

      while (index < lines.length && lines[index].trim().startsWith(">")) {
        quoteLines.push(lines[index].trim().replace(/^>\s?/, ""));
        index += 1;
      }

      blocks.push(
        <blockquote key={`quote-${index}`}>
          {renderInlineMarkdown(quoteLines.join(" "))}
        </blockquote>,
      );
      continue;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      const items: ReactNode[] = [];

      while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) {
        const itemText = lines[index].trim().replace(/^[-*]\s+/, "");
        items.push(<li key={`li-${index}`}>{renderInlineMarkdown(itemText)}</li>);
        index += 1;
      }

      blocks.push(<ul key={`ul-${index}`}>{items}</ul>);
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items: ReactNode[] = [];

      while (index < lines.length && /^\d+\.\s+/.test(lines[index].trim())) {
        const itemText = lines[index].trim().replace(/^\d+\.\s+/, "");
        items.push(<li key={`oli-${index}`}>{renderInlineMarkdown(itemText)}</li>);
        index += 1;
      }

      blocks.push(<ol key={`ol-${index}`}>{items}</ol>);
      continue;
    }

    const paragraphLines = [trimmed];
    index += 1;

    while (
      index < lines.length &&
      lines[index].trim() &&
      !lines[index].trim().startsWith("#") &&
      !lines[index].trim().startsWith("```") &&
      !lines[index].trim().startsWith(">") &&
      !isHorizontalRule(lines[index].trim()) &&
      !isTableStart(lines[index].trim(), lines[index + 1]) &&
      !/^[-*]\s+/.test(lines[index].trim()) &&
      !/^\d+\.\s+/.test(lines[index].trim())
    ) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }

    blocks.push(
      <p key={`p-${index}`}>{renderInlineMarkdown(paragraphLines.join(" "))}</p>,
    );
  }

  return <div className="markdown-answer">{blocks}</div>;
}

function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function NavTabs({ view, onChange }: { view: AppView; onChange: (view: AppView) => void }) {
  const items: Array<{ key: AppView; label: string }> = [
    { key: "setup", label: "项目" },
    { key: "workspace", label: "问答" },
    { key: "evaluation", label: "评测" },
    { key: "settings", label: "设置" },
  ];

  return (
    <nav className="nav-tabs">
      {items.map((item) => (
        <button
          className={view === item.key ? "nav-tab active" : "nav-tab"}
          key={item.key}
          onClick={() => onChange(item.key)}
        >
          {item.label}
        </button>
      ))}
    </nav>
  );
}

function ProjectSetupPage({
  setup,
  config,
  scanResult,
  indexResult,
  projectRecords,
  running,
  systemMessage,
  onSetupChange,
  onConfigChange,
  onScan,
  onBuildIndex,
  onUseDemo,
  onUploadZip,
  onEnterWorkspace,
  onSelectProjectRecord,
  onOpenProjectRecord,
  onDeleteProjectRecord,
}: {
  setup: ProjectSetupState;
  config: ProjectConfig;
  scanResult: ScanProjectResponse | null;
  indexResult: BuildIndexResponse | null;
  projectRecords: ProjectWorkspaceRecord[];
  running: boolean;
  systemMessage: string;
  onSetupChange: (setup: ProjectSetupState) => void;
  onConfigChange: (config: ProjectConfig) => void;
  onScan: () => void;
  onBuildIndex: () => void;
  onUseDemo: () => void;
  onUploadZip: (file: File) => void;
  onEnterWorkspace: () => void;
  onSelectProjectRecord: (record: ProjectWorkspaceRecord) => void;
  onOpenProjectRecord: (record: ProjectWorkspaceRecord) => void;
  onDeleteProjectRecord: (recordId: string) => void;
}) {
  const scanData = scanResult?.data;
  const indexData = indexResult?.data;

  function updateSetup<K extends keyof ProjectSetupState>(
    key: K,
    value: ProjectSetupState[K],
  ) {
    onSetupChange({ ...setup, [key]: value });
  }

  function updateConfig<K extends keyof ProjectConfig>(key: K, value: ProjectConfig[K]) {
    onConfigChange({ ...config, [key]: value });
  }

  return (
    <section className="setup-grid">
      <div className="card setup-hero">
        <p className="eyebrow dark">Project</p>
        <h2>接入代码仓库</h2>
        <p className="muted">
          上传 ZIP 或选择本地路径，构建索引后即可开始问答。
        </p>
        <div className="capability-strip">
          <span>AST</span>
          <span>BM25 / Vector</span>
          <span>Rerank</span>
          <span>Agent Tools</span>
        </div>
        <div className="hero-stat-grid">
          <Metric label="已扫描文件" value={scanData?.file_count ?? "-"} />
          <Metric label="知识 chunks" value={scanData?.chunk_count ?? "-"} />
          <Metric label="向量数量" value={indexData?.vector_count ?? "-"} />
          <Metric
            label="Embedding"
            value={config.embedding_model || "bge-m3"}
          />
        </div>
        <p className="setup-message">{systemMessage}</p>
        <div className="hero-actions">
          <button disabled={running} onClick={onUseDemo}>
            使用示例项目
          </button>
          <button className="secondary-button" disabled={running} onClick={onEnterWorkspace}>
            进入问答
          </button>
        </div>
      </div>

      <div className="card upload-card">
        <h3>选择项目</h3>
        <label className="drop-zone">
          <input
            accept=".zip"
            type="file"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                onUploadZip(file);
              }
            }}
          />
          <strong>上传 ZIP</strong>
          <span>支持包含 README、src、tests 的 Python 项目</span>
        </label>

        <div className="form-grid compact">
          <label>
            项目名称
            <input
              value={setup.projectName}
              onChange={(event) => updateSetup("projectName", event.target.value)}
            />
          </label>
          <label>
            本地项目路径
            <input
              value={setup.projectPath}
              onChange={(event) => updateSetup("projectPath", event.target.value)}
            />
          </label>
          <label>
            chunks 输出
            <input
              value={setup.chunksOutputPath}
              onChange={(event) => updateSetup("chunksOutputPath", event.target.value)}
            />
          </label>
          <label>
            索引输出
            <input
              value={setup.indexOutputPath}
              onChange={(event) => updateSetup("indexOutputPath", event.target.value)}
            />
          </label>
        </div>
      </div>

      <div className="card project-history-card">
        <div className="section-title-row">
          <div>
            <p className="eyebrow dark">Recent</p>
            <h3>项目历史</h3>
          </div>
          <small>{projectRecords.length} 个记录</small>
        </div>

        {projectRecords.length === 0 ? (
          <p className="muted">
            上传或扫描后会自动保存。
          </p>
        ) : (
          <div className="project-record-list">
            {projectRecords.map((record) => (
              <article className="project-record-item" key={record.id}>
                <div>
                  <strong>{record.projectName}</strong>
                  <span>{record.projectPath}</span>
                  <small>
                    {record.status} · project_id={record.projectId} · chunks=
                    {record.chunkCount ?? "-"} · vectors={record.vectorCount ?? "-"}
                  </small>
                </div>
                <div className="record-actions">
                  <button
                    className="secondary-button"
                    disabled={running}
                    onClick={() => onSelectProjectRecord(record)}
                  >
                    选择
                  </button>
                  <button
                    className="secondary-button"
                    disabled={running}
                    onClick={() => onOpenProjectRecord(record)}
                  >
                    打开
                  </button>
                  <button
                    className="ghost-button"
                    disabled={running}
                    onClick={() => onDeleteProjectRecord(record.id)}
                  >
                    删除
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>

      <div className="card">
        <h3>构建索引</h3>
        <div className="step-action-row">
          <button disabled={running} onClick={onScan}>
            扫描
          </button>
          <button
            className="secondary-button"
            disabled={running || !scanData}
            title={!scanData ? "请先扫描项目生成 chunks" : undefined}
            onClick={onBuildIndex}
          >
            构建索引
          </button>
        </div>

        <div className="status-steps">
          <StatusStep
            done={Boolean(scanData)}
            title="扫描与切分"
            detail={
              scanData
                ? `${scanData.file_count} 个文件，${scanData.chunk_count} 个 chunks`
                : "未扫描"
            }
          />
          <StatusStep
            done={Boolean(indexData)}
            title="向量索引"
            detail={
              indexData
                ? `${indexData.vector_count} 条向量，维度 ${indexData.dimension}`
                : "未构建"
            }
          />
          <StatusStep
            done={Boolean(config.project_id)}
            title="项目工作区"
            detail={`project_id=${config.project_id}，thread_id=${config.thread_id}`}
          />
        </div>

        <details>
          <summary>原始结果</summary>
          <pre>{formatJson({ scanResult, indexResult })}</pre>
        </details>

        <div className="form-grid compact">
          <label>
            Embedding Provider
            <input
              value={config.embedding_provider}
              onChange={(event) => updateConfig("embedding_provider", event.target.value)}
            />
          </label>
          <label>
            Embedding Model
            <input
              value={config.embedding_model}
              onChange={(event) => updateConfig("embedding_model", event.target.value)}
            />
          </label>
          <label>
            Embedding Base URL
            <input
              value={config.embedding_base_url}
              onChange={(event) => updateConfig("embedding_base_url", event.target.value)}
            />
          </label>
          <label>
            Rerank Model
            <input
              value={config.rerank_model}
              onChange={(event) => updateConfig("rerank_model", event.target.value)}
            />
          </label>
        </div>
      </div>
    </section>
  );
}

function StatusStep({ done, title, detail }: { done: boolean; title: string; detail: string }) {
  return (
    <div className={done ? "status-step done" : "status-step"}>
      <span>{done ? "✓" : "•"}</span>
      <div>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>
    </div>
  );
}

function ProjectReadinessBadge({
  scanResult,
  indexResult,
}: {
  scanResult: ScanProjectResponse | null;
  indexResult: BuildIndexResponse | null;
}) {
  if (indexResult?.data) {
    return <span className="badge badge-ok">已索引，可完整问答</span>;
  }

  if (scanResult?.data) {
    return <span className="badge badge-warn">已扫描，待构建索引</span>;
  }

  return <span className="badge badge-muted">未扫描，仅可查看目录类问题</span>;
}

function WorkspacePage({
  setup,
  config,
  scanResult,
  indexResult,
  projectRecords,
  sessions,
  activeSession,
  query,
  loading,
  streaming,
  onQueryChange,
  onAsk,
  onStream,
  onNewThread,
  onSwitchSession,
  onDeleteSession,
  onClearCurrentSession,
  onOpenSetup,
  onOpenSettings,
  onClearEvents,
  onOpenProjectRecord,
  onSelectFeedbackResponse,
}: {
  setup: ProjectSetupState;
  config: ProjectConfig;
  scanResult: ScanProjectResponse | null;
  indexResult: BuildIndexResponse | null;
  projectRecords: ProjectWorkspaceRecord[];
  sessions: ChatSession[];
  activeSession: ChatSession;
  query: string;
  loading: boolean;
  streaming: boolean;
  onQueryChange: (query: string) => void;
  onAsk: () => void;
  onStream: () => void;
  onNewThread: () => void;
  onSwitchSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  onClearCurrentSession: () => void;
  onOpenSetup: () => void;
  onOpenSettings: () => void;
  onClearEvents: () => void;
  onOpenProjectRecord: (record: ProjectWorkspaceRecord) => void;
  onSelectFeedbackResponse: (response: HITLAgentResponse) => void;
}) {
  const canUseFullRag = Boolean(indexResult?.data);
  const currentProjectPath = setup.projectPath || config.project_root;
  const otherProjectRecords = projectRecords.filter(
    (record) => record.projectPath !== currentProjectPath,
  );

  return (
    <section className="product-workspace">
      <aside className="session-sidebar card">
        <div className="sidebar-title">
          <div>
            <p className="eyebrow dark">Workspace</p>
            <h3>{config.project_name}</h3>
          </div>
          <button className="tiny" onClick={onNewThread}>
            新建
          </button>
        </div>

        <div className="project-summary">
          <span>状态</span>
          <strong>
            <ProjectReadinessBadge indexResult={indexResult} scanResult={scanResult} />
          </strong>
          <span>路径</span>
          <strong>{setup.projectPath || config.project_root}</strong>
          <span>chunks</span>
          <strong>{config.chunks_path}</strong>
          <span>索引</span>
          <strong>{config.index_path}</strong>
          <span>审核</span>
          <strong>{config.approval_required_tools.join(", ") || "无"}</strong>
        </div>

        <div className="session-list">
          {sessions.map((session) => (
            <button
              className={
                session.id === activeSession.id ? "session-item active" : "session-item"
              }
              key={session.id}
              onClick={() => onSwitchSession(session.id)}
            >
              <span>{session.title}</span>
              <small>{session.threadId}</small>
            </button>
          ))}
        </div>

        <div className="sidebar-actions">
          <button className="secondary-button wide" onClick={onClearCurrentSession}>
            清空会话
          </button>
          <button className="secondary-button wide" onClick={onOpenSetup}>
            项目设置
          </button>
          <button className="ghost-button wide" onClick={onOpenSettings}>
            运行设置
          </button>
          <button
            className="ghost-button wide"
            disabled={sessions.length <= 1}
            onClick={() => onDeleteSession(activeSession.id)}
          >
            删除会话
          </button>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-title compact-title">
            <div>
              <p className="eyebrow dark">Projects</p>
              <h3>项目历史</h3>
            </div>
            <small>{projectRecords.length}</small>
          </div>
          {otherProjectRecords.length ? (
            <div className="compact-project-list">
              {otherProjectRecords.slice(0, 5).map((record) => (
                <button
                  className="compact-project-item"
                  key={record.id}
                  onClick={() => onOpenProjectRecord(record)}
                >
                  <span>{record.projectName}</span>
                  <small>
                    {record.status} · chunks={record.chunkCount ?? "-"} · vectors=
                    {record.vectorCount ?? "-"}
                  </small>
                </button>
              ))}
            </div>
          ) : (
            <p className="muted small-muted">
              暂无其它项目。
            </p>
          )}
        </div>
      </aside>

      <main className="chat-product card">
        <div className="chat-header">
          <div>
            <p className="eyebrow dark">Chat</p>
            <h2>问代码，查证据</h2>
            <p className="muted">
              {activeSession.threadId}
            </p>
            {!canUseFullRag ? (
              <p className="workspace-warning">
                当前项目未完成索引，代码/文档检索能力受限。
              </p>
            ) : null}
          </div>
          <ResponseStatus response={activeSession.latestResponse} running={loading || streaming} />
        </div>

        <div className="workflow-strip">
          <span className="workflow-dot done">项目</span>
          <span className={scanResult?.data ? "workflow-dot done" : "workflow-dot"}>
            chunks
          </span>
          <span className={indexResult?.data ? "workflow-dot done" : "workflow-dot"}>
            index
          </span>
          <span className="workflow-dot done">tools</span>
          <span className="workflow-dot done">memory</span>
          <span className="workflow-dot done">HITL</span>
        </div>

        <div className="quick-prompts">
          {exampleQuestions.map((item) => (
            <button className="chip" key={item} onClick={() => onQueryChange(item)}>
              {item}
            </button>
          ))}
        </div>

        <div className="message-list">
          {activeSession.messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              onSelectFeedbackResponse={onSelectFeedbackResponse}
            />
          ))}
        </div>

        <form
          className="composer docked"
          onSubmit={(event: FormEvent) => {
            event.preventDefault();
            onAsk();
          }}
        >
          <textarea
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="问一个和当前仓库有关的问题..."
          />
          <div className="composer-actions">
            <button disabled={loading || streaming || !query.trim()} type="submit">
              {loading ? "运行中..." : "发送"}
            </button>
            <button
              className="secondary-button"
              disabled={loading || streaming || !query.trim()}
              type="button"
              onClick={onStream}
            >
              流式
            </button>
          </div>
        </form>
      </main>

      <aside className="inspector card">
        <Inspector
          events={activeSession.events}
          response={activeSession.latestResponse}
          onClearEvents={onClearEvents}
        />
      </aside>
    </section>
  );
}

function ResponseStatus({
  response,
  running,
}: {
  response: HITLAgentResponse | null;
  running?: boolean;
}) {
  if (running) {
    return <span className="badge badge-warn">运行中</span>;
  }

  if (!response) {
    return <span className="badge badge-muted">未运行</span>;
  }

  if (response.status === "interrupted") {
    return <span className="badge badge-warn">等待审核</span>;
  }

  if (response.success) {
    return <span className="badge badge-ok">成功</span>;
  }

  return <span className="badge badge-error">失败</span>;
}

function MessageBubble({
  message,
  onSelectFeedbackResponse,
}: {
  message: ChatMessage;
  onSelectFeedbackResponse: (response: HITLAgentResponse) => void;
}) {
  return (
    <article className={`message-bubble ${message.role}`}>
      <div className="message-meta">
        <strong>
          {message.role === "user"
            ? "你"
            : message.role === "assistant"
              ? "CodeDoc Agent"
              : "系统"}
        </strong>
        <span>{message.createdAt}</span>
        {message.status ? <em>{message.status}</em> : null}
        {message.role === "assistant" && message.response ? (
          <button
            className="message-feedback-button"
            onClick={() => onSelectFeedbackResponse(message.response as HITLAgentResponse)}
            type="button"
          >
            反馈此回答
          </button>
        ) : null}
      </div>
      <div className="message-content">
        {message.role === "assistant" ? (
          <MarkdownRenderer content={message.content || "等待输出..."} />
        ) : (
          <p>{message.content}</p>
        )}
      </div>
      {message.response?.tool_call_history?.length ? (
        <details className="message-details">
          <summary>本轮工具调用 {message.response.tool_call_history.length} 次</summary>
          <pre>{formatJson(message.response.tool_call_history)}</pre>
        </details>
      ) : null}
    </article>
  );
}

function Inspector({
  response,
  events,
  onClearEvents,
}: {
  response: HITLAgentResponse | null;
  events: SSELogItem[];
  onClearEvents: () => void;
}) {
  const tools = response?.tool_call_history ?? [];
  const messages = response?.message_trace ?? [];
  const evidence = response?.evidence ?? [];
  const citations = response?.citations ?? [];

  return (
    <>
      <h3>运行状态</h3>
      <div className="metric-grid">
        <Metric label="模型调用" value={response?.model_call_count ?? "-"} />
        <Metric label="工具调用" value={response?.tool_call_count ?? "-"} />
        <Metric
          label="耗时"
          value={response ? `${Math.round(response.total_duration_ms)} ms` : "-"}
        />
        <Metric label="stop" value={response?.stop_reason ?? "-"} />
      </div>

      <div className="step-list">
        {(response?.execution_steps ?? []).map((step, index) => (
          <span key={`${step}-${index}`}>{step}</span>
        ))}
      </div>

      <h3>工具调用</h3>
      {tools.length ? (
        tools.map((tool) => (
          <ToolCard key={`${tool.sequence}-${tool.tool_call_id}`} tool={tool} />
        ))
      ) : (
        <p className="muted">暂无工具调用。</p>
      )}

      <h3>证据与引用</h3>
      {evidence.length || citations.length ? (
        <div className="details-list">
          {evidence.slice(0, 6).map((item, index) => (
            <details key={`e-${index}`}>
              <summary>Evidence #{index + 1}</summary>
              <pre>{formatJson(item)}</pre>
            </details>
          ))}
          {citations.slice(0, 6).map((item, index) => (
            <details key={`c-${index}`}>
              <summary>Citation #{index + 1}</summary>
              <pre>{formatJson(item)}</pre>
            </details>
          ))}
        </div>
      ) : (
        <p className="muted">暂无结构化 evidence/citations，可查看 message_trace。</p>
      )}

      <div className="card-title-row inspector-subtitle">
        <h3>SSE 事件流</h3>
        <button className="ghost-button tiny" onClick={onClearEvents}>
          清空
        </button>
      </div>
      <div className="event-timeline">
        {events.length ? (
          events.map((event) => (
            <details
              key={event.id}
              open={event.event === "interrupt" || event.event === "completed"}
            >
              <summary>
                <span>{event.event}</span>
                <small>{event.receivedAt}</small>
              </summary>
              <pre>{formatJson(event.data)}</pre>
            </details>
          ))
        ) : (
          <p className="muted">点击“SSE 流式执行”后，这里会实时显示节点更新、token、interrupt 和 completed。</p>
        )}
      </div>

      <details>
        <summary>message_trace</summary>
        <pre>{formatJson(messages)}</pre>
      </details>
    </>
  );
}

function ToolCard({ tool }: { tool: ToolCallHistoryItem }) {
  return (
    <div className="tool-card">
      <div className="tool-title">
        <strong>
          #{tool.sequence ?? "-"} {tool.tool_name}
        </strong>
        <code>{tool.tool_call_id}</code>
      </div>
      <pre>{formatJson(tool.arguments ?? {})}</pre>
    </div>
  );
}

function ReviewDialog({
  interrupt,
  loading,
  onDecision,
  onClose,
}: {
  interrupt: HITLInterrupt | null;
  loading: boolean;
  onDecision: (decision: HumanReviewDecision) => void;
  onClose: () => void;
}) {
  const [feedback, setFeedback] = useState("");
  const [editedText, setEditedText] = useState("");

  if (!interrupt) {
    return null;
  }

  const toolCalls = interrupt.tool_calls ?? [];

  function prepareEdit() {
    setEditedText(formatJson(toolCalls));
  }

  function submitEdit() {
    try {
      const edited = JSON.parse(editedText);
      onDecision({
        decision: "edit",
        feedback,
        edited_tool_calls: edited,
      });
    } catch (error) {
      alert(`工具参数不是合法 JSON：${String(error)}`);
    }
  }

  return (
    <div className="dialog-backdrop">
      <section className="review-dialog">
        <div className="dialog-header">
          <div>
            <p className="eyebrow dark">Human-in-the-loop</p>
            <h2>Agent 请求执行需要审核的工具</h2>
          </div>
          <button className="ghost-button" onClick={onClose}>
            收起
          </button>
        </div>
        <p className="muted">{interrupt.instructions}</p>

        <div className="tool-card-list">
          {toolCalls.map((toolCall) => (
            <div className="tool-card" key={toolCall.id}>
              <div className="tool-title">
                <strong>{toolCall.name}</strong>
                <code>{toolCall.id}</code>
              </div>
              <pre>{formatJson(toolCall.args)}</pre>
            </div>
          ))}
        </div>

        <label>
          审核备注
          <input
            value={feedback}
            onChange={(event) => setFeedback(event.target.value)}
            placeholder="例如：允许运行该测试文件"
          />
        </label>

        <div className="dialog-actions">
          <button
            disabled={loading}
            onClick={() => onDecision({ decision: "approve", feedback })}
          >
            approve
          </button>
          <button
            className="danger-button"
            disabled={loading}
            onClick={() => onDecision({ decision: "reject", feedback })}
          >
            reject
          </button>
          <button className="secondary-button" disabled={loading} onClick={prepareEdit}>
            修改参数
          </button>
        </div>

        <textarea
          className="json-editor"
          value={editedText}
          onChange={(event) => setEditedText(event.target.value)}
          placeholder="点击“修改参数”后可编辑工具调用 JSON。"
        />
        <button
          className="secondary-button"
          disabled={loading || !editedText}
          onClick={submitEdit}
        >
          提交 edit 并继续
        </button>
      </section>
    </div>
  );
}


function EvaluationPage({
  feedbackResponse,
  config,
}: {
  feedbackResponse: HITLAgentResponse | null;
  config: ProjectConfig;
}) {
  const [rating, setRating] = useState<-1 | 0 | 1>(1);
  const [issueTags, setIssueTags] = useState<string[]>([]);
  const [comment, setComment] = useState("");
  const [correctedAnswer, setCorrectedAnswer] = useState("");
  const [requiredTerms, setRequiredTerms] = useState("");
  const [feedbackId, setFeedbackId] = useState<number | null>(null);
  const [feedbackMessage, setFeedbackMessage] = useState("");
  const [badCaseMessage, setBadCaseMessage] = useState("");
  const [qualityData, setQualityData] = useState<unknown>(null);
  const [evalSummary, setEvalSummary] = useState<EvalSummary | null>(null);
  const [retrievalMethods, setRetrievalMethods] = useState<
    Array<{
      method: string;
      summary: Record<string, number>;
    }>
  >([]);
  const latestToolNames = useMemo(
    () =>
      Array.from(
        new Set(
          (feedbackResponse?.tool_call_history || [])
            .map((item) => item.tool_name)
            .filter((name): name is string => Boolean(name)),
        ),
      ),
    [feedbackResponse],
  );
  const latestCitations = Array.isArray(feedbackResponse?.citations)
    ? feedbackResponse?.citations || []
    : [];
  const suggestedIssueTags = [
    { value: "retrieval_error", label: "检索不准" },
    { value: "tool_routing_error", label: "工具选错" },
    { value: "answer_generation_error", label: "回答生成问题" },
    { value: "citation_error", label: "引用/证据问题" },
    { value: "should_refuse", label: "应拒答未拒答" },
    { value: "latency_issue", label: "响应太慢" },
  ];
  const badCaseId = feedbackResponse
    ? `frontend-${feedbackResponse.project_id}-${feedbackResponse.run_id || Date.now()}`
        .replace(/[^a-zA-Z0-9_-]/g, "-")
        .slice(0, 160)
    : "";

  useEffect(() => {
    setRating(1);
    setIssueTags([]);
    setComment("");
    setCorrectedAnswer("");
    setRequiredTerms("");
    setFeedbackId(null);
    setFeedbackMessage("");
    setBadCaseMessage("");
  }, [feedbackResponse?.run_id]);

  function toggleIssueTag(tag: string) {
    setIssueTags((current) =>
      current.includes(tag)
        ? current.filter((item) => item !== tag)
        : [...current, tag],
    );
  }

  async function submitCurrentFeedback() {
    if (!feedbackResponse) {
      alert("还没有可反馈的回答。");
      return;
    }

    const result = await submitFeedback({
      project_id: config.project_id,
      thread_id: feedbackResponse.thread_id,
      run_id: feedbackResponse.run_id,
      query: feedbackResponse.query,
      answer: feedbackResponse.answer || feedbackResponse.error_message || "无回答",
      rating,
      issue_tags: rating < 0 && issueTags.length === 0 ? ["frontend_feedback"] : issueTags,
      comment: comment || null,
      corrected_answer: correctedAnswer || null,
    });

    setQualityData(result);
    const createdFeedbackId = Number(asRecord(result).feedback_id);
    setFeedbackId(Number.isFinite(createdFeedbackId) ? createdFeedbackId : null);
    setFeedbackMessage(
      rating < 0
        ? "负反馈已保存。你可以继续一键沉淀为 Bad Case。"
        : "反馈已保存，后续可以在反馈列表中查看。",
    );
  }

  async function promoteCurrentBadCase() {
    if (!feedbackResponse || feedbackId === null) {
      alert("请先提交当前回答反馈，再提升为 Bad Case。");
      return;
    }

    const expectedTools = latestToolNames.length ? latestToolNames : [];
    const requiredAnswerTerms = requiredTerms
      .split(/[,，\n]/)
      .map((item) => item.trim())
      .filter(Boolean);
    const result = await promoteFeedbackToBadCase(feedbackId, {
      case_id: badCaseId || `frontend-feedback-${feedbackId}`,
      name: shortText(feedbackResponse.query, 80),
      expected_tool_names: expectedTools,
      forbidden_tool_names: [],
      required_answer_terms: requiredAnswerTerms,
      accepted_stop_reasons: ["completed"],
      notes: comment || "由前端反馈中心沉淀的 Bad Case。",
    });

    setQualityData(result);
    setBadCaseMessage("已提升为 Bad Case，可用于后续回归测试。");
  }

  async function importEvalReport(file: File) {
    const text = await file.text();
    const data = JSON.parse(text);
    setEvalSummary(data.summary ?? null);
    setRetrievalMethods(Array.isArray(data.methods) ? data.methods : []);
    setQualityData(data);
  }

  return (
    <section className="evaluation-layout">
      <div className="card evaluation-hero">
        <p className="eyebrow dark">Evaluation</p>
        <h2>评测与反馈</h2>
        <p className="muted">
          导入报告，反馈回答，沉淀 Bad Case。
        </p>
        <div className="capability-strip">
          <span>Recall@5</span>
          <span>MRR</span>
          <span>NDCG@5</span>
          <span>Tool Accuracy</span>
          <span>Bad Case</span>
        </div>
      </div>

      <div className="card">
        <h2>评测报告</h2>
        <p className="muted">
          支持 Retrieval 和 Agent JSON 报告。
        </p>
        <label className="file-button">
          导入报告
          <input
            accept="application/json,.json"
            type="file"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void importEvalReport(file);
              }
            }}
          />
        </label>
        {!evalSummary && !retrievalMethods.length ? (
          <div className="empty-state">
            <strong>暂无报告</strong>
            <p>
              选择本地 JSON 评测报告即可查看指标。
            </p>
          </div>
        ) : null}
        <div className="metric-grid large">
          <Metric label="total" value={evalSummary?.total_cases ?? "-"} />
          <Metric label="passed" value={evalSummary?.passed_cases ?? "-"} />
          <Metric label="success" value={percent(evalSummary?.task_success_rate)} />
          <Metric label="tool F1" value={percent(evalSummary?.average_tool_f1)} />
          <Metric label="precision" value={percent(evalSummary?.average_tool_precision)} />
          <Metric label="recall" value={percent(evalSummary?.average_tool_recall)} />
          <Metric
            label="avg latency"
            value={`${numberText(evalSummary?.average_latency_ms)} ms`}
          />
          <Metric
            label="p95 latency"
            value={`${numberText(evalSummary?.p95_latency_ms)} ms`}
          />
        </div>

        {retrievalMethods.length ? (
          <>
            <h3>检索策略对比</h3>
            <div className="retrieval-method-grid">
              {retrievalMethods.map((method) => (
                <article className="method-card" key={method.method}>
                  <strong>{method.method}</strong>
                  <span>Recall@5 {percent(method.summary.recall_at_k)}</span>
                  <span>MRR {percent(method.summary.mrr)}</span>
                  <span>NDCG@5 {percent(method.summary.ndcg_at_k)}</span>
                  <span>Avg {numberText(method.summary.average_latency_ms)} ms</span>
                </article>
              ))}
            </div>
          </>
        ) : null}
      </div>

      <div className="card">
        <h2>反馈</h2>
        {!feedbackResponse ? (
          <div className="empty-state">
            <strong>暂无可反馈回答</strong>
            <p>
              在问答页点击“反馈此回答”。
            </p>
          </div>
        ) : null}
        {feedbackResponse ? (
          <div className="feedback-target-card">
            <div>
              <span>评价对象</span>
              <strong>{shortText(feedbackResponse.query, 90)}</strong>
            </div>
            <div>
              <span>状态</span>
              <strong>{feedbackResponse.status || feedbackResponse.stop_reason}</strong>
            </div>
            <div>
              <span>工具</span>
              <strong>{latestToolNames.length ? latestToolNames.join(" / ") : "无"}</strong>
            </div>
            <div>
              <span>证据</span>
              <strong>{latestCitations.length} 条</strong>
            </div>
          </div>
        ) : null}

        <div className="feedback-box product-feedback-box">
          <div className="rating-choice-grid">
            {[
              { value: 1 as const, title: "可用", desc: "保留正样本" },
              { value: 0 as const, title: "一般", desc: "需要补充" },
              { value: -1 as const, title: "有问题", desc: "转为 Bad Case" },
            ].map((item) => (
              <button
                className={rating === item.value ? "rating-card active" : "rating-card"}
                key={item.value}
                onClick={() => {
                  setRating(item.value);
                  if (item.value >= 0) {
                    setBadCaseMessage("");
                  }
                }}
                type="button"
              >
                <strong>{item.title}</strong>
                <span>{item.desc}</span>
              </button>
            ))}
          </div>

          <div>
            <label>问题类型</label>
            <div className="issue-tag-grid">
              {suggestedIssueTags.map((tag) => (
                <button
                  className={issueTags.includes(tag.value) ? "issue-tag active" : "issue-tag"}
                  key={tag.value}
                  onClick={() => toggleIssueTag(tag.value)}
                  type="button"
                >
                  {tag.label}
                </button>
              ))}
            </div>
          </div>

          <div className="step-action-row">
            <button className="secondary-button" onClick={submitCurrentFeedback}>
              保存反馈
            </button>
            <button
              className="secondary-button danger-lite"
              disabled={rating >= 0 || feedbackId === null}
              onClick={promoteCurrentBadCase}
            >
              一键沉淀 Bad Case
            </button>
            <button
              className="ghost-button"
              onClick={() => void listFeedback(config.project_id).then(setQualityData)}
            >
              读取反馈
            </button>
            <button
              className="ghost-button"
              onClick={() => void listBadCases(config.project_id).then(setQualityData)}
            >
              读取 Bad Cases
            </button>
          </div>

          {feedbackMessage ? <p className="success-hint">{feedbackMessage}</p> : null}
          {badCaseMessage ? <p className="success-hint">{badCaseMessage}</p> : null}

          <label>
            反馈备注
            <textarea
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              placeholder="例如：回答没有使用正确工具 / 缺少证据 / 应该拒答 / 定位到错误文件..."
            />
          </label>

          <label>
            期望回答或修正要点
            <textarea
              value={correctedAnswer}
              onChange={(event) => setCorrectedAnswer(event.target.value)}
              placeholder="可选。写下你希望 Agent 正确回答的要点，后续可作为回归测试参考。"
            />
          </label>

          {rating < 0 ? (
            <div className="bad-case-draft">
              <div>
                <span>Bad Case ID</span>
                <strong>{badCaseId || "提交反馈后生成"}</strong>
              </div>
              <label>
                期望答案关键词
                <input
                  value={requiredTerms}
                  onChange={(event) => setRequiredTerms(event.target.value)}
                  placeholder="例如：keyword_score, search.py, 关键词计数"
                />
              </label>
              <p className="muted">
                提升为 Bad Case 后，系统会保存 query、原回答、期望工具和关键词，
                后续可加入 JSONL 评测集做回归测试。
              </p>
            </div>
          ) : null}
        </div>
        <details open>
          <summary>原始数据</summary>
          <pre>{qualityData ? formatJson(qualityData) : "暂无数据。"}</pre>
        </details>
      </div>
    </section>
  );
}

function SettingsPage({
  config,
  onChange,
  onPing,
  systemMessage,
}: {
  config: ProjectConfig;
  onChange: (config: ProjectConfig) => void;
  onPing: () => void;
  systemMessage: string;
}) {
  function update<K extends keyof ProjectConfig>(key: K, value: ProjectConfig[K]) {
    onChange({ ...config, [key]: value });
  }

  return (
    <section className="settings-layout">
      <div className="card">
        <div className="card-title-row">
          <h2>运行设置</h2>
          <button className="secondary-button" onClick={onPing}>
            检查后端
          </button>
        </div>
        <p className="muted">{systemMessage}</p>
        <div className="form-grid">
          <label>
            project_id
            <input
              type="number"
              value={config.project_id}
              onChange={(event) => update("project_id", Number(event.target.value))}
            />
          </label>
          <label>
            thread_id
            <input
              value={config.thread_id}
              onChange={(event) => update("thread_id", event.target.value)}
            />
          </label>
          <label>
            project_root
            <input
              value={config.project_root}
              onChange={(event) => update("project_root", event.target.value)}
            />
          </label>
          <label>
            chunks_path
            <input
              value={config.chunks_path}
              onChange={(event) => update("chunks_path", event.target.value)}
            />
          </label>
          <label>
            index_path
            <input
              value={config.index_path}
              onChange={(event) => update("index_path", event.target.value)}
            />
          </label>
          <label>
            recursion_limit
            <input
              type="number"
              value={config.recursion_limit}
              onChange={(event) => update("recursion_limit", Number(event.target.value))}
            />
          </label>
        </div>
      </div>

      <div className="card">
        <h2>模型与工具</h2>
        <div className="form-grid">
          <label>
            embedding_provider
            <input
              value={config.embedding_provider}
              onChange={(event) => update("embedding_provider", event.target.value)}
            />
          </label>
          <label>
            embedding_model
            <input
              value={config.embedding_model}
              onChange={(event) => update("embedding_model", event.target.value)}
            />
          </label>
          <label>
            embedding_base_url
            <input
              value={config.embedding_base_url}
              onChange={(event) => update("embedding_base_url", event.target.value)}
            />
          </label>
          <label>
            rerank_provider
            <input
              value={config.rerank_provider}
              onChange={(event) => update("rerank_provider", event.target.value)}
            />
          </label>
          <label>
            rerank_model
            <input
              value={config.rerank_model}
              onChange={(event) => update("rerank_model", event.target.value)}
            />
          </label>
          <label className="checkbox-row">
            <input
              checked={config.enable_human_review}
              type="checkbox"
              onChange={(event) => update("enable_human_review", event.target.checked)}
            />
            启用人工审核
          </label>
        </div>
        <p className="muted">
          当前需要人工审核的工具：{config.approval_required_tools.join("、") || "无"}
        </p>
      </div>
    </section>
  );
}

function buildSyntheticResponse({
  query,
  config,
  status,
  answer = "",
  interrupts = [],
  stopReason = status,
}: {
  query: string;
  config: ProjectConfig;
  status: "interrupted" | "completed" | "failed";
  answer?: string;
  interrupts?: HITLInterrupt[];
  stopReason?: string;
}): HITLAgentResponse {
  return {
    query,
    project_id: config.project_id,
    thread_id: config.thread_id,
    effective_thread_id: `project:${config.project_id}:thread:${config.thread_id}`,
    run_id: createId("stream-run"),
    answer,
    status,
    success: status === "completed",
    completed: status === "completed" || status === "failed",
    stop_reason: stopReason,
    interrupts,
    approval_status: status === "interrupted" ? "pending" : "not_required",
    review_history: [],
    turn_index: 0,
    model_call_count: 0,
    tool_call_count: 0,
    message_count: 0,
    message_trace: [],
    tool_call_history: [],
    execution_steps: [],
    checkpoint_id: null,
    total_duration_ms: 0,
    error_message: null,
    allowed_tools: [],
    provider: "openai_compatible",
    model_name: "qwen3.5:4b",
  };
}

export default function App() {
  const [view, setView] = useState<AppView>("setup");
  const [setup, setSetup] = useState<ProjectSetupState>(defaultSetup);
  const [config, setConfig] = useState<ProjectConfig>(defaultConfig);
  const [sessions, setSessions] = useState<ChatSession[]>(loadSessions);
  const [activeSessionId, setActiveSessionId] = useState(
    () => localStorage.getItem(ACTIVE_SESSION_KEY) || "",
  );
  const [activeWorkspaceKey, setActiveWorkspaceKey] = useState(() =>
    getWorkspaceKeyFromSetup(defaultSetup),
  );
  const [workspaceSessionMap, setWorkspaceSessionMap] =
    useState<Record<string, WorkspaceSessionState>>(loadWorkspaceSessionMap);
  const [query, setQuery] = useState(exampleQuestions[0]);
  const [scanResult, setScanResult] = useState<ScanProjectResponse | null>(null);
  const [indexResult, setIndexResult] = useState<BuildIndexResponse | null>(null);
  const [projectRecords, setProjectRecords] =
    useState<ProjectWorkspaceRecord[]>(loadProjectRecords);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [systemMessage, setSystemMessage] = useState("后端未检查");
  const [reviewVisible, setReviewVisible] = useState(true);
  const [selectedFeedbackResponse, setSelectedFeedbackResponse] =
    useState<HITLAgentResponse | null>(null);

  const activeSession = useMemo(() => {
    return sessions.find((session) => session.id === activeSessionId) ?? sessions[0];
  }, [activeSessionId, sessions]);

  useEffect(() => {
    if (!activeSession) {
      const next = createSession(defaultConfig.thread_id);
      setSessions([next]);
      setActiveSessionId(next.id);
      return;
    }

    setConfig((current) => ({
      ...current,
      thread_id: activeSession.threadId,
    }));
  }, [activeSession?.id]);

  useEffect(() => {
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    if (activeSessionId) {
      localStorage.setItem(ACTIVE_SESSION_KEY, activeSessionId);
    }
  }, [activeSessionId]);

  useEffect(() => {
    setSelectedFeedbackResponse(null);
  }, [activeSessionId, activeWorkspaceKey]);

  useEffect(() => {
    if (!activeWorkspaceKey || sessions.length === 0) {
      return;
    }

    setWorkspaceSessionMap((current) => ({
      ...current,
      [activeWorkspaceKey]: {
        sessions,
        activeSessionId,
      },
    }));
  }, [activeWorkspaceKey, activeSessionId, sessions]);

  useEffect(() => {
    localStorage.setItem(
      WORKSPACE_SESSION_STORAGE_KEY,
      JSON.stringify(workspaceSessionMap),
    );
  }, [workspaceSessionMap]);

  useEffect(() => {
    localStorage.setItem(PROJECT_RECORD_STORAGE_KEY, JSON.stringify(projectRecords));
  }, [projectRecords]);

  const currentInterrupt = useMemo(() => {
    const response = activeSession?.latestResponse;

    if (response?.status !== "interrupted") {
      return null;
    }

    return response.interrupts[0] ?? null;
  }, [activeSession]);

  useEffect(() => {
    if (currentInterrupt) {
      setReviewVisible(true);
      return;
    }

    if (!loading && !streaming) {
      setReviewVisible(false);
    }
  }, [currentInterrupt, loading, streaming]);

  function updateActiveSession(updater: (session: ChatSession) => ChatSession) {
    setSessions((current) =>
      current.map((session) =>
        session.id === activeSession.id ? updater(session) : session,
      ),
    );
  }

  function appendMessage(message: ChatMessage) {
    updateActiveSession((session) => ({
      ...session,
      title: message.role === "user" ? shortText(message.content) : session.title,
      updatedAt: nowText(),
      messages: [...session.messages, message],
    }));
  }

  function patchMessage(messageId: string, patch: Partial<ChatMessage>) {
    updateActiveSession((session) => ({
      ...session,
      updatedAt: nowText(),
      messages: session.messages.map((message) =>
        message.id === messageId ? { ...message, ...patch } : message,
      ),
      latestResponse: patch.response === undefined ? session.latestResponse : patch.response,
    }));
  }

  function appendEvent(item: SSELogItem) {
    updateActiveSession((session) => ({
      ...session,
      events: [...session.events, item],
      updatedAt: nowText(),
    }));
  }

  function selectFeedbackResponse(response: HITLAgentResponse) {
    setSelectedFeedbackResponse(response);
    setView("evaluation");
  }

  function buildEffectiveConfig(): ProjectConfig {
    return {
      ...config,
      thread_id: activeSession.threadId,
    };
  }

  function switchWorkspaceSessions(workspaceKey: string, threadPrefix = "chat") {
    const saved = workspaceSessionMap[workspaceKey];
    const nextState =
      saved && saved.sessions.length > 0
        ? saved
        : createWorkspaceSessionState(threadPrefix);

    setActiveWorkspaceKey(workspaceKey);
    setSessions(nextState.sessions);
    setActiveSessionId(nextState.activeSessionId || nextState.sessions[0].id);
    setQuery(exampleQuestions[0]);
    setReviewVisible(true);
  }

  function applyProjectToConfig(
    nextSetup = setup,
    nextScanResult = scanResult,
    nextIndexResult = indexResult,
  ) {
    setConfig((current) => {
      const projectId = nextScanResult?.data?.project_id ?? current.project_id;

      return {
        ...current,
        project_id: projectId || current.project_id || 1,
        project_name: nextSetup.projectName,
        project_root: nextSetup.projectPath || ".",
        chunks_path: nextScanResult?.data?.saved_path || nextSetup.chunksOutputPath,
        index_path: nextIndexResult?.data?.output_path || nextSetup.indexOutputPath,
        thread_id: activeSession.threadId,
      };
    });
  }

  function saveProjectRecord(
    nextSetup: ProjectSetupState,
    nextScanResult: ScanProjectResponse | null,
    nextIndexResult: BuildIndexResponse | null,
    status: ProjectWorkspaceRecord["status"],
    uploadedFilename?: string,
  ) {
    const record = buildProjectRecordFromState({
      setup: nextSetup,
      config,
      scanResult: nextScanResult,
      indexResult: nextIndexResult,
      uploadedFilename,
      status,
    });

    setProjectRecords((current) => mergeProjectRecord(current, record));
  }

  function openProjectRecord(
    record: ProjectWorkspaceRecord,
    enterWorkspace = true,
  ) {
    const workspaceKey = getWorkspaceKeyFromRecord(record);
    const savedSessionState =
      workspaceSessionMap[workspaceKey] || createWorkspaceSessionState("chat");
    const savedActiveSession =
      savedSessionState.sessions.find(
        (session) => session.id === savedSessionState.activeSessionId,
      ) || savedSessionState.sessions[0];
    const nextSetup: ProjectSetupState = {
      projectName: record.projectName,
      projectPath: record.projectPath,
      chunkSize: setup.chunkSize,
      overlap: setup.overlap,
      chunksOutputPath: record.chunksPath,
      indexOutputPath: record.indexPath,
    };

    setSetup(nextSetup);
    setScanResult(
      record.status === "uploaded"
        ? null
        : {
            success: true,
            data: {
              project_path: record.projectPath,
              project_id: record.projectId,
              file_count: record.fileCount ?? 0,
              chunk_count: record.chunkCount ?? 0,
              saved_path: record.chunksPath,
            },
          },
    );
    setIndexResult(
      record.status !== "indexed"
        ? null
        : {
            success: true,
            data: {
              chunks_path: record.chunksPath,
              output_path: record.indexPath,
              embedding_provider: config.embedding_provider,
              embedding_model: config.embedding_model,
              dimension: record.dimension ?? 0,
              chunk_count: record.chunkCount ?? 0,
              vector_count: record.vectorCount ?? 0,
              incremental: true,
            },
          },
    );
    setConfig((current) => ({
      ...current,
      project_id: record.projectId || current.project_id || 1,
      project_name: record.projectName,
      project_root: record.projectPath || ".",
      chunks_path: record.chunksPath,
      index_path: record.indexPath,
      thread_id: savedActiveSession.threadId,
    }));
    setSystemMessage(`已切换到项目工作台：${record.projectName}`);
    setActiveWorkspaceKey(workspaceKey);
    setSessions(savedSessionState.sessions);
    setActiveSessionId(savedActiveSession.id);
    setQuery(exampleQuestions[0]);
    setReviewVisible(true);

    if (enterWorkspace) {
      setView("workspace");
    } else {
      setView("setup");
    }
  }

  function deleteProjectRecord(recordId: string) {
    setProjectRecords((current) => current.filter((record) => record.id !== recordId));
  }

  function createNewThread() {
    const next = createSession(createThreadId("chat"));
    setSessions((current) => [next, ...current]);
    setActiveSessionId(next.id);
    setQuery("");
    setReviewVisible(true);
  }

  function deleteSession(sessionId: string) {
    setSessions((current) => {
      if (current.length <= 1) {
        return current;
      }

      const filtered = current.filter((session) => session.id !== sessionId);

      if (sessionId === activeSession.id) {
        setActiveSessionId(filtered[0].id);
      }

      return filtered;
    });
  }

  function clearCurrentSession() {
    updateActiveSession((session) => ({
      ...createSession(session.threadId),
      id: session.id,
      title: "新会话",
    }));
  }

  function useDemoProject() {
    const workspaceKey = getWorkspaceKeyFromSetup(defaultSetup);
    setSetup(defaultSetup);
    setConfig(defaultConfig);
    setScanResult({
      success: true,
      data: {
        project_path: "test_project",
        project_id: 1,
        file_count: 0,
        chunk_count: 0,
        saved_path: defaultSetup.chunksOutputPath,
      },
    });
    setIndexResult({
      success: true,
      data: {
        chunks_path: defaultSetup.chunksOutputPath,
        output_path: defaultSetup.indexOutputPath,
        embedding_provider: "ollama",
        embedding_model: "bge-m3",
        dimension: 1024,
        chunk_count: 0,
        vector_count: 0,
        incremental: true,
      },
    });
    saveProjectRecord(
      defaultSetup,
      {
        success: true,
        data: {
          project_path: "test_project",
          project_id: 1,
          file_count: 0,
          chunk_count: 0,
          saved_path: defaultSetup.chunksOutputPath,
        },
      },
      {
        success: true,
        data: {
          chunks_path: defaultSetup.chunksOutputPath,
          output_path: defaultSetup.indexOutputPath,
          embedding_provider: "ollama",
          embedding_model: "bge-m3",
          dimension: 1024,
          chunk_count: 0,
          vector_count: 0,
          incremental: true,
        },
      },
      "indexed",
    );
    switchWorkspaceSessions(workspaceKey, "demo");
    setView("workspace");
  }

  async function runScan() {
    setLoading(true);

    try {
      const result = await scanProject({
        project_path: setup.projectPath,
        chunk_size: setup.chunkSize,
        overlap: setup.overlap,
        save_chunks: true,
        output_path: setup.chunksOutputPath,
        save_to_db: true,
      });

      setScanResult(result);
      applyProjectToConfig(setup, result, indexResult);
      saveProjectRecord(setup, result, indexResult, "scanned");
      setSystemMessage(
        `扫描成功：${result.data?.file_count ?? 0} 个文件，${result.data?.chunk_count ?? 0} 个 chunks`,
      );
    } catch (error) {
      setSystemMessage(`扫描失败：${String(error)}`);
    } finally {
      setLoading(false);
    }
  }

  async function runUploadZip(file: File) {
    setLoading(true);

    try {
      const zipProjectName = normalizeProjectName(
        removeZipSuffix(file.name) || setup.projectName,
      );
      const result = await uploadProjectZip(file, zipProjectName);
      const projectPath = result.data?.project_path;

      if (!projectPath) {
        throw new Error("上传成功，但后端没有返回 project_path");
      }

      const projectName = normalizeProjectName(
        result.data?.project_name || zipProjectName,
      );
      const nextSetup = {
        ...setup,
        projectPath,
        projectName,
        chunksOutputPath: `outputs/${projectName}_${result.data?.upload_id}_chunks.json`,
        indexOutputPath: `outputs/${projectName}_${result.data?.upload_id}_vector_index.json`,
      };

      setSetup(nextSetup);
      applyProjectToConfig(nextSetup, null, null);
      saveProjectRecord(nextSetup, null, null, "uploaded", result.data?.filename);
      switchWorkspaceSessions(getWorkspaceKeyFromSetup(nextSetup), "chat");
      setSystemMessage(
        `上传成功：${result.data?.filename}，解压文件数 ${result.data?.extracted_file_count}`,
      );
    } catch (error) {
      setSystemMessage(`上传失败：${String(error)}`);
    } finally {
      setLoading(false);
    }
  }

  async function runBuildIndex() {
    if (!scanResult?.data?.saved_path && !setup.chunksOutputPath) {
      setSystemMessage("请先扫描项目生成 chunks，再构建向量索引。");
      return;
    }

    if (!scanResult?.data) {
      setSystemMessage("当前项目还没有扫描结果。请先点击“扫描并生成 chunks”。");
      return;
    }

    setLoading(true);

    try {
      const chunksPath = scanResult?.data?.saved_path || setup.chunksOutputPath;
      const result = await buildVectorIndex({
        chunks_path: chunksPath,
        output_path: setup.indexOutputPath,
        embedding_provider: config.embedding_provider,
        embedding_model: config.embedding_model,
        embedding_base_url: config.embedding_base_url,
        embedding_api_key: "",
        batch_size: 16,
        incremental: true,
      });

      setIndexResult(result);
      applyProjectToConfig(setup, scanResult, result);
      saveProjectRecord(setup, scanResult, result, "indexed");
      setSystemMessage(
        `索引构建成功：${result.data?.vector_count ?? 0} 条向量，维度 ${result.data?.dimension ?? "-"}`,
      );
    } catch (error) {
      setSystemMessage(`构建索引失败：${String(error)}`);
    } finally {
      setLoading(false);
    }
  }

  async function runAsk() {
    const normalizedQuery = query.trim();

    if (!normalizedQuery) {
      return;
    }

    const effectiveConfig = buildEffectiveConfig();
    const userMessage: ChatMessage = {
      id: createId("msg"),
      role: "user",
      content: normalizedQuery,
      createdAt: nowText(),
      status: "completed",
    };
    const assistantId = createId("msg");

    appendMessage(userMessage);
    appendMessage({
      id: assistantId,
      role: "assistant",
      content: "Agent 正在分析问题、选择工具并组织回答...",
      createdAt: nowText(),
      status: "running",
    });

    setLoading(true);
    setReviewVisible(false);
    setQuery("");

    try {
      const result = await startHitlAgent(normalizedQuery, effectiveConfig);
      const answer =
        result.answer ||
        result.error_message ||
        (result.status === "interrupted"
          ? "Agent 已暂停，等待你审核工具调用。"
          : "模型没有返回有效回答。");

      patchMessage(assistantId, {
        content: answer,
        status: result.status === "interrupted" ? "interrupted" : result.success ? "completed" : "failed",
        response: result,
      });
      setReviewVisible(result.status === "interrupted");
    } catch (error) {
      patchMessage(assistantId, {
        content: `请求失败：${String(error)}`,
        status: "failed",
        response: null,
      });
      setReviewVisible(false);
    } finally {
      setLoading(false);
    }
  }

  async function runResume(decision: HumanReviewDecision) {
    const effectiveConfig = buildEffectiveConfig();
    const assistantId = createId("msg");

    appendMessage({
      id: createId("msg"),
      role: "system",
      content: `人工审核：${decision.decision}${decision.feedback ? `，备注：${decision.feedback}` : ""}`,
      createdAt: nowText(),
      status: "completed",
    });
    appendMessage({
      id: assistantId,
      role: "assistant",
      content: "已收到审核决定，Agent 正在继续执行...",
      createdAt: nowText(),
      status: "running",
    });

    setLoading(true);

    try {
      const result = await resumeHitlAgent(decision, effectiveConfig);
      patchMessage(assistantId, {
        content: result.answer || result.error_message || "Agent 已继续执行，但没有返回有效回答。",
        status: result.status === "interrupted" ? "interrupted" : result.success ? "completed" : "failed",
        response: result,
      });
    } catch (error) {
      patchMessage(assistantId, {
        content: `继续执行失败：${String(error)}`,
        status: "failed",
        response: null,
      });
    } finally {
      setLoading(false);
    }
  }

  async function runStream() {
    const normalizedQuery = query.trim();

    if (!normalizedQuery) {
      return;
    }

    const effectiveConfig = buildEffectiveConfig();
    const assistantId = createId("msg");

    appendMessage({
      id: createId("msg"),
      role: "user",
      content: normalizedQuery,
      createdAt: nowText(),
      status: "completed",
    });
    appendMessage({
      id: assistantId,
      role: "assistant",
      content: "### 实时执行进度\n\n- 已连接 SSE，等待 Agent 输出...",
      createdAt: nowText(),
      status: "streaming",
    });

    setStreaming(true);
    setReviewVisible(false);
    setQuery("");
    updateActiveSession((session) => ({ ...session, events: [] }));

    try {
      await streamHitlAgent(normalizedQuery, effectiveConfig, (item) => {
        appendEvent(item);

        const data = asRecord(item.data);

        if (item.event === "connected") {
          updateActiveSession((session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    content: appendStreamingLine(
                      message.content,
                      `${item.receivedAt} 已建立流式连接`,
                    ),
                    status: "streaming",
                  }
                : message,
            ),
          }));
        }

        if (item.event === "token") {
          const token = readString(data.text);

          if (token) {
            updateActiveSession((session) => ({
              ...session,
              messages: session.messages.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: message.content.startsWith("### 实时执行进度")
                        ? token
                        : `${message.content}${token}`,
                    }
                  : message,
              ),
            }));
          }
        }

        if (item.event === "node_update") {
          const answer = readNestedAnswer(data);

          if (answer) {
            patchMessage(assistantId, {
              content: answer,
              status: "streaming",
            });
          } else {
            updateActiveSession((session) => ({
              ...session,
              messages: session.messages.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: appendStreamingLine(
                        message.content,
                        `${item.receivedAt} ${summarizeNodeUpdate(data)}`,
                      ),
                      status: "streaming",
                    }
                  : message,
              ),
            }));
          }
        }

        if (item.event === "interrupt") {
          const interrupts = Array.isArray(data.interrupts)
            ? (data.interrupts as HITLInterrupt[])
            : [];
          const response = buildSyntheticResponse({
            query: normalizedQuery,
            config: effectiveConfig,
            status: "interrupted",
            answer: "Agent 已暂停，等待你审核工具调用。",
            interrupts,
          });

          patchMessage(assistantId, {
            content: response.answer,
            status: "interrupted",
            response,
          });
        }

        if (item.event === "completed") {
          const answer = readNestedAnswer(data) || "Agent 执行完成，但没有返回有效回答。";
          const response = buildSyntheticResponse({
            query: normalizedQuery,
            config: effectiveConfig,
            status: "completed",
            answer,
            stopReason: readString(data.stop_reason) || "completed",
          });

          patchMessage(assistantId, {
            content: answer,
            status: "completed",
            response,
          });
        }

        if (item.event === "error") {
          patchMessage(assistantId, {
            content: `流式执行失败：${readString(data.message) || formatJson(data)}`,
            status: "failed",
            response: buildSyntheticResponse({
              query: normalizedQuery,
              config: effectiveConfig,
              status: "failed",
              answer: readString(data.message) || "流式执行失败。",
              stopReason: "stream_error",
            }),
          });
        }
      });
    } catch (error) {
      patchMessage(assistantId, {
        content: `SSE 请求失败：${String(error)}`,
        status: "failed",
        response: null,
      });
    } finally {
      setStreaming(false);
    }
  }

  async function pingBackend() {
    try {
      const version = await fetchVersion();
      setSystemMessage(`后端连接成功：${JSON.stringify(version)}`);
    } catch (error) {
      setSystemMessage(`后端连接失败：${String(error)}`);
    }
  }

  function enterWorkspace() {
    applyProjectToConfig();
    setView("workspace");
  }

  if (!activeSession) {
    return <main className="app-shell">正在初始化会话...</main>;
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">CodeDoc Research Agent</p>
          <h1>代码仓库问答工作台</h1>
          <p className="topbar-subtitle">
            上传仓库，构建索引，用 Agent 查询代码与文档。
          </p>
        </div>
        <NavTabs view={view} onChange={setView} />
      </header>

      {view === "setup" ? (
        <ProjectSetupPage
          config={config}
          indexResult={indexResult}
          projectRecords={projectRecords}
          running={loading}
          scanResult={scanResult}
          setup={setup}
          systemMessage={systemMessage}
          onBuildIndex={runBuildIndex}
          onConfigChange={setConfig}
          onDeleteProjectRecord={deleteProjectRecord}
          onEnterWorkspace={enterWorkspace}
          onOpenProjectRecord={openProjectRecord}
          onSelectProjectRecord={(record) => openProjectRecord(record, false)}
          onScan={runScan}
          onSetupChange={setSetup}
          onUploadZip={runUploadZip}
          onUseDemo={useDemoProject}
        />
      ) : null}

      {view === "workspace" ? (
        <WorkspacePage
          activeSession={activeSession}
          config={config}
          indexResult={indexResult}
          loading={loading}
          projectRecords={projectRecords}
          query={query}
          scanResult={scanResult}
          sessions={sessions}
          setup={setup}
          streaming={streaming}
          onAsk={runAsk}
          onClearCurrentSession={clearCurrentSession}
          onClearEvents={() => updateActiveSession((session) => ({ ...session, events: [] }))}
          onDeleteSession={deleteSession}
          onNewThread={createNewThread}
          onOpenSetup={() => setView("setup")}
          onOpenSettings={() => setView("settings")}
          onOpenProjectRecord={openProjectRecord}
          onQueryChange={setQuery}
          onSelectFeedbackResponse={selectFeedbackResponse}
          onStream={runStream}
          onSwitchSession={setActiveSessionId}
        />
      ) : null}

      {view === "evaluation" ? (
        <EvaluationPage
          config={config}
          feedbackResponse={selectedFeedbackResponse || activeSession.latestResponse}
        />
      ) : null}

      {view === "settings" ? (
        <SettingsPage
          config={config}
          systemMessage={systemMessage}
          onChange={setConfig}
          onPing={pingBackend}
        />
      ) : null}

      {reviewVisible ? (
        <ReviewDialog
          interrupt={currentInterrupt}
          loading={loading}
          onClose={() => setReviewVisible(false)}
          onDecision={runResume}
        />
      ) : null}
    </main>
  );
}

