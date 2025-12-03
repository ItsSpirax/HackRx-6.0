"use client";

import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";

type PipelineStatus = "pending" | "active" | "done" | "cached" | "skipped";

interface PipelineStep {
  id: string;
  label: string;
  status: PipelineStatus;
  detail?: string;
  duration_ms?: number;
}

interface Citation {
  id: number;
  label: string;
  text: string;
  score?: number;
  logit?: number;
  confidence?: number | null;
  source_url?: string;
}

type CitationFragment = string | { ids: number[] };

interface AnswerPayload {
  question: string;
  text: string;
  citations: Citation[];
  rerank?: {
    model?: string;
    confidence?: number | null;
    considered?: number;
    top_logit?: number | null;
  };
  timing?: {
    search_ms?: number;
    rerank_ms?: number;
    generation_ms?: number;
    total_ms?: number;
  };
  cache_hit?: boolean;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  mode?: "single" | "batch";
  questions?: string[];
  answers?: AnswerPayload[];
  answer?: AnswerPayload;
  timestamp: string;
}

interface DocumentSummary {
  url: string;
  doc_hash: string;
  doc_type: string;
  status: string;
  cached: boolean;
  force_refresh: boolean;
  chunk_count: number;
  average_tokens: number;
  max_tokens?: number;
  min_tokens?: number;
}

interface UploadState {
  status: "idle" | "uploading" | "success" | "error";
  tempUrl: string;
  error: string | null;
}

interface ToastState {
  variant: "success" | "error" | "info";
  message: string;
}

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "https://adithvm.centralindia.cloudapp.azure.com";
const WS_BASE = API_BASE.replace(/^http/, "ws");

const DEFAULT_PIPELINE_STEPS: PipelineStep[] = [
  { id: "extract", label: "Extracting & Chunking Document", status: "pending" },
  { id: "embed", label: "Generating Vector Embeddings", status: "pending", detail: "0/0 chunks" },
  { id: "index", label: "Indexing in Vector Store", status: "pending" },
  { id: "ready", label: "Document Ready to Query", status: "pending" },
];

const FALLBACK_PRELOADED: { label: string; url: string }[] = [
  {
    label: "Student Resource Book",
    url: "https://engineering.nmims.edu/docs/srbbti21.pdf",
  },
  {
    label: "Apple Q4 2024 Financial Report",
    url: "https://s2.q4cdn.com/470004039/files/doc_earnings/2024/q4/filing/10-Q4-2024-As-Filed.pdf",
  },
  {
    label: "NMIMS Website",
    url: "https://engineering.nmims.edu/",
  },
  {
    label: "Constitution of India",
    url: "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2024/07/20240716890312078.pdf",
  },
  {
    label: "Titanic Dataset",
    url: "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv",
  },
];

const createId = () => {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
};

const formatTimestamp = (iso: string) => {
  const date = new Date(iso);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
};

const mergePipelineSteps = (incoming?: PipelineStep[]): PipelineStep[] => {
  if (!incoming || incoming.length === 0) {
    return DEFAULT_PIPELINE_STEPS;
  }
  return DEFAULT_PIPELINE_STEPS.map((step) => {
    const match = incoming.find((candidate) => candidate.id === step.id);
    return match ? { ...step, ...match } : step;
  });
};

const statusStyles: Record<PipelineStatus, string> = {
  pending: "border-slate-800/60 bg-[#141020] text-slate-400",
  active: "border-orange-500 bg-orange-950/60 text-orange-300 shadow-lg shadow-orange-500/20",
  done: "border-amber-500 bg-amber-900/40 text-amber-300 shadow-lg shadow-amber-500/20",
  cached: "border-orange-400 bg-orange-900/40 text-orange-300 shadow-lg shadow-orange-400/20",
  skipped: "border-slate-700 bg-slate-900 text-slate-500",
};

const confidenceLabel = (confidence?: number | null) => {
  if (confidence === null || confidence === undefined) {
    return "Confidence unavailable";
  }
  const percent = Math.round(confidence * 100);
  if (percent >= 85) return `Relevance: ${percent}% (High)`;
  if (percent >= 60) return `Relevance: ${percent}% (Medium)`;
  return `Relevance: ${percent}% (Low)`;
};

const confidenceColor = (confidence?: number | null) => {
  if (confidence === null || confidence === undefined) {
    return "bg-slate-700";
  }
  if (confidence >= 0.85) return "bg-amber-400";
  if (confidence >= 0.6) return "bg-orange-400";
  return "bg-rose-500";
};

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const deriveConfidenceValue = (answer: AnswerPayload): number | null => {
  const direct = toFiniteNumber(answer.rerank?.confidence);
  if (direct !== null) {
    return clamp01(direct);
  }

  const topLogit = toFiniteNumber(answer.rerank?.top_logit);
  if (topLogit !== null) {
    const logistic = 1 / (1 + Math.exp(-topLogit));
    return clamp01(logistic);
  }

  const citationConfidences = answer.citations
    .map((citation) => toFiniteNumber(citation.confidence))
    .filter((value): value is number => value !== null);
  if (citationConfidences.length > 0) {
    return clamp01(Math.max(...citationConfidences));
  }

  const citationScores = answer.citations
    .map((citation) => toFiniteNumber(citation.score))
    .filter((value): value is number => value !== null);
  if (citationScores.length > 0) {
    const maxScore = Math.max(...citationScores);
    if (!Number.isFinite(maxScore)) {
      return null;
    }
    return clamp01(maxScore);
  }

  return null;
};

export default function Home() {
  const [sessionId] = useState(() => createId());
  const [documentUrl, setDocumentUrl] = useState("");
  const [documentDisplay, setDocumentDisplay] = useState("");
  const [forceRefresh, setForceRefresh] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [preloadedDocs, setPreloadedDocs] = useState(FALLBACK_PRELOADED);
  const [pipelineSteps, setPipelineSteps] = useState(DEFAULT_PIPELINE_STEPS);
  const [pipelineActive, setPipelineActive] = useState(false);
  const [documentInfo, setDocumentInfo] = useState<DocumentSummary | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [composerMode, setComposerMode] = useState<"single" | "batch">("single");
  const [composerInput, setComposerInput] = useState("");
  const [isSubmittingDoc, setIsSubmittingDoc] = useState(false);
  const [isSubmittingQuestion, setIsSubmittingQuestion] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadState, setUploadState] = useState<UploadState>({ status: "idle", tempUrl: "", error: null });
  const [activeCitation, setActiveCitation] = useState<{ answer: AnswerPayload; citation: Citation } | null>(null);
  const [activeMetrics, setActiveMetrics] = useState<AnswerPayload | null>(null);
  const [lastRunMeta, setLastRunMeta] = useState<{ timing: Record<string, unknown>; metrics: Record<string, unknown> } | null>(null);
  const [toast, setToast] = useState<ToastState | null>(null);
  const [hoveredCitation, setHoveredCitation] = useState<{
    answer: AnswerPayload;
    citation: Citation;
    position: { x: number; y: number };
    placement: "above" | "below";
  } | null>(null);
  const [currentDocumentHash, setCurrentDocumentHash] = useState<string>("");

  const citationTooltipTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const chatContainerRef = useRef<HTMLDivElement | null>(null);

  const setPipelineState = (incoming?: PipelineStep[], options?: { active?: boolean; embedDetail?: string }) => {
    const merged = mergePipelineSteps(incoming);
    const withDetail = options?.embedDetail
      ? merged.map((step) =>
          step.id === "embed"
            ? { ...step, detail: options.embedDetail }
            : step
        )
      : merged;
    setPipelineSteps(withDetail);
    if (typeof options?.active === "boolean") {
      setPipelineActive(options.active);
    }
  };

  useEffect(() => {
    const fetchPreloaded = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/v1/hackrx/preloaded`);
        if (!response.ok) {
          throw new Error("Failed to fetch preloaded documents");
        }
        const data = await response.json();
        if (Array.isArray(data.documents) && data.documents.length > 0) {
          setPreloadedDocs(data.documents);
        }
      } catch (error) {
        console.warn("Using fallback preloaded documents", error);
      }
    };

    fetchPreloaded();
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timeout = setTimeout(() => setToast(null), 3200);
    return () => clearTimeout(timeout);
  }, [toast]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const wsUrl = `${WS_BASE}/api/v1/hackrx/ws?session_id=${sessionId}`;

    try {
      const socket = new WebSocket(wsUrl);
      websocketRef.current = socket;

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data?.type === "pipeline") {
            const steps = Array.isArray(data.steps) ? (data.steps as PipelineStep[]) : undefined;
            const active = typeof data.active === "boolean" ? data.active : undefined;
            const embedDetail = typeof data.embed_detail === "string" ? data.embed_detail : undefined;
            setPipelineState(steps, { active, embedDetail });
          } else if (data?.type === "toast" && typeof data.message === "string") {
            const variant = data.variant === "error" || data.variant === "success" ? data.variant : "info";
            handleToast(variant, data.message);
          }
        } catch (error) {
          console.warn("Unable to parse websocket message", error);
        }
      };

      socket.onerror = () => {
        console.warn("Pipeline websocket encountered an error");
      };

      socket.onclose = () => {
        if (websocketRef.current === socket) {
          websocketRef.current = null;
        }
      };

      return () => {
        if (websocketRef.current === socket) {
          websocketRef.current = null;
        }
        socket.close(1000, "component unmounted");
      };
    } catch (error) {
      console.warn("Unable to establish pipeline websocket", error);
    }
  }, [sessionId]);

  const pipelineProgress = useMemo(() => {
    const completed = pipelineSteps.filter((step) => step.status === "done" || step.status === "cached").length;
    return Math.round((completed / pipelineSteps.length) * 100);
  }, [pipelineSteps]);

  const clearCitationTooltipTimer = () => {
    if (citationTooltipTimer.current) {
      clearTimeout(citationTooltipTimer.current);
      citationTooltipTimer.current = null;
    }
  };

  const showCitationTooltip = (answer: AnswerPayload, citation: Citation, element: HTMLElement) => {
    clearCitationTooltipTimer();
    const rect = element.getBoundingClientRect();
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
    const clampedX = Math.min(viewportWidth - 24, Math.max(24, rect.left + rect.width / 2));
    const spaceAbove = rect.top;
    const spaceBelow = viewportHeight - rect.bottom;
    const placement: "above" | "below" = spaceAbove > spaceBelow ? "above" : "below";
    const rawY = placement === "above" ? rect.top : rect.bottom;
    const clampedY = Math.min(viewportHeight - 24, Math.max(24, rawY));
    setHoveredCitation({
      answer,
      citation,
      position: {
        x: clampedX,
        y: clampedY,
      },
      placement,
    });
  };

  const scheduleHideCitationTooltip = (delay = 120) => {
    clearCitationTooltipTimer();
    citationTooltipTimer.current = setTimeout(() => {
      setHoveredCitation(null);
    }, delay);
  };

  const handleToast = (variant: ToastState["variant"], message: string) => {
    setToast({ variant, message });
  };

  const updateDocumentSource = (actual: string, display?: string) => {
    // Clear chat history if switching to a different document
    if (actual !== documentUrl && documentUrl !== "") {
      setChatMessages([]);
    }
    setDocumentUrl(actual);
    setDocumentDisplay(display ?? actual);
  };

  const handleDocumentInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    setDocumentDisplay(value);
    setDocumentUrl(value);
    setSelectedPreset(null);
  };

  const handleFileSelection = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setUploadFile(event.target.files[0]);
      setUploadState({ status: "idle", tempUrl: "", error: null });
    }
  };

  const uploadSelectedFile = async () => {
    if (!uploadFile) {
      handleToast("error", "Select a file before uploading.");
      return;
    }
    const formData = new FormData();
    formData.append("file", uploadFile);

    setUploadState({ status: "uploading", tempUrl: "", error: null });

    try {
      const response = await fetch(`${API_BASE}/api/v1/hackrx/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();
      const actualUrl = data.internal_path ?? data.relative_path ?? data.temp_url ?? "";
      const originalName = uploadFile?.name?.split(/[/\\]/).pop() ?? "";

      updateDocumentSource(actualUrl, originalName || actualUrl);
      setSelectedPreset(null);
      setUploadState({ status: "success", tempUrl: data.temp_url, error: null });
      handleToast("success", "Upload complete. Document ready for processing.");
    } catch (error) {
      console.error(error);
      setUploadState({ status: "error", tempUrl: "", error: "Upload failed. Try again." });
      handleToast("error", "Unable to upload the file.");
    }
  };

  const submitDocument = async (targetUrl?: string) => {
    const candidateUrl = targetUrl ?? documentUrl;
    const resolvedUrl = candidateUrl.trim();
    if (!resolvedUrl) {
      handleToast("error", "Provide a document URL or upload a file.");
      return;
    }

    if (!resolvedUrl.startsWith("/api/uploads/") && targetUrl === undefined) {
      setDocumentDisplay(resolvedUrl);
    }

    // Clear chat history when processing a new document
    if (resolvedUrl !== currentDocumentHash) {
      setChatMessages([]);
    }

    setIsSubmittingDoc(true);
    setPipelineState(
      [
        {
          id: "extract",
          label: "Extracting & Chunking Document",
          status: "active",
          detail: forceRefresh ? "Re-processing document" : "Fetching document",
        },
        {
          id: "embed",
          label: "Generating Vector Embeddings",
          status: "pending",
          detail: forceRefresh ? "Awaiting fresh embeddings" : "Checking cache"
        },
      ],
      { active: true }
    );

    try {
      const response = await fetch(`${API_BASE}/api/v1/hackrx/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          documents: resolvedUrl,
          questions: [],
          force_refresh: forceRefresh,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Document processing failed with status ${response.status}`);
      }

      const data = await response.json();
      setPipelineState(data.pipeline, {
        active: false,
        embedDetail: data.embedding_progress?.label,
      });
      setDocumentInfo(data.document ?? null);
      setLastRunMeta({ timing: data.timing ?? {}, metrics: data.metrics ?? {} });

      const canonicalUrl = data.document?.url ?? resolvedUrl;
      setDocumentUrl(canonicalUrl);
      setCurrentDocumentHash(data.document?.doc_hash ?? canonicalUrl);
      if (!canonicalUrl.startsWith("/api/uploads/")) {
        setDocumentDisplay(canonicalUrl);
      }
      handleToast("success", "Document ready to query.");
    } catch (error) {
      console.error(error);
      setPipelineState(undefined, { active: false });
      handleToast("error", "Document processing failed. Check the logs and try again.");
    } finally {
      setIsSubmittingDoc(false);
    }
  };

  const handleQuestionsSubmit = async () => {
    const normalizedInput = documentUrl.trim();
    const docSource = normalizedInput || documentInfo?.url || "";

    if (!docSource) {
      handleToast("error", "Provide or upload a document before asking questions.");
      return;
    }

    const preparedQuestions = composerMode === "batch"
      ? composerInput.split(/\n+/).map((line) => line.trim()).filter(Boolean)
      : [composerInput.trim()].filter(Boolean);

    if (preparedQuestions.length === 0) {
      handleToast("info", "Enter at least one question.");
      return;
    }

    if (normalizedInput && normalizedInput !== documentInfo?.url) {
      setSelectedPreset(null);
    }

    const timestamp = new Date().toISOString();
    if (composerMode === "batch") {
      const batchMessage: ChatMessage = {
        id: createId(),
        role: "user",
        content: preparedQuestions.join("\n"),
        questions: preparedQuestions,
        timestamp,
        mode: "batch",
      };
      setChatMessages((prev) => [...prev, batchMessage]);
    } else {
      setChatMessages((prev) => [
        ...prev,
        {
          id: createId(),
          role: "user",
          content: preparedQuestions[0],
          timestamp,
          mode: "single",
        },
      ]);
    }
    setComposerInput("");
    setIsSubmittingQuestion(true);
    const previousDocumentInfo = documentInfo;
    const isNewDocument = !documentInfo || (normalizedInput && normalizedInput !== documentInfo.url);
    if (isNewDocument) {
      setDocumentInfo(null);
    }
    setPipelineState(
      [
        {
          id: "extract",
          label: "Extracting & Chunking Document",
          status: "active",
          detail: isNewDocument ? "Fetching document" : "Collecting document context",
        },
        {
          id: "embed",
          label: "Generating Vector Embeddings",
          status: "pending",
          detail: isNewDocument ? "Generating embeddings" : "Matching relevant chunks",
        },
      ],
      { active: true }
    );

    try {
      const response = await fetch(`${API_BASE}/api/v1/hackrx/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          documents: docSource,
          questions: preparedQuestions,
          force_refresh: false,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Answer generation failed with status ${response.status}`);
      }

      const data = await response.json();
      setPipelineState(data.pipeline, {
        active: false,
        embedDetail: data.embedding_progress?.label,
      });
      setLastRunMeta({ timing: data.timing ?? {}, metrics: data.metrics ?? {} });

      if (data.document) {
        setDocumentInfo(data.document);
        if (typeof data.document.url === "string" && data.document.url.trim()) {
          const canonicalUrl = data.document.url.trim();
          setDocumentUrl(canonicalUrl);
          if (!canonicalUrl.startsWith("/api/uploads/")) {
            setDocumentDisplay(canonicalUrl);
          }
        }
      }

      const answers: AnswerPayload[] = Array.isArray(data.answers) ? data.answers : [];
      const answerTimestamp = new Date().toISOString();

      if (composerMode === "batch") {
        const batchResponse: ChatMessage = {
          id: createId(),
          role: "assistant",
          content: answers.map((answer) => answer.text).join("\n"),
          answers,
          timestamp: answerTimestamp,
          mode: "batch",
        };
        setChatMessages((prev) => [...prev, batchResponse]);
      } else {
        const answerMessages: ChatMessage[] = answers.map((answer) => ({
          id: createId(),
          role: "assistant",
          content: answer.text,
          answer,
          timestamp: answerTimestamp,
          mode: "single",
        }));
        setChatMessages((prev) => [...prev, ...answerMessages]);
      }
    } catch (error) {
      console.error(error);
      handleToast("error", "Unable to fetch answers for your questions.");
      if (isNewDocument) {
        setDocumentInfo(previousDocumentInfo ?? null);
      }
      setPipelineState(undefined, { active: false });
    } finally {
      setIsSubmittingQuestion(false);
    }
  };

  const renderAnswerText = (answer: AnswerPayload) => {
    const citationRegex = /\[CITE:([^\]]+)\]/g;
    const fragments: CitationFragment[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = citationRegex.exec(answer.text)) !== null) {
      if (match.index > lastIndex) {
        fragments.push(answer.text.slice(lastIndex, match.index));
      }
      const idMatches = match[1]?.match(/\d+/g) ?? [];
      const ids = idMatches
        .map((value) => Number.parseInt(value, 10))
        .filter((value) => Number.isFinite(value));
      if (ids.length > 0) {
        fragments.push({ ids });
      }
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < answer.text.length) {
      fragments.push(answer.text.slice(lastIndex));
    }

    return fragments.map((fragment, index) => {
      if (typeof fragment === "string") {
        return (
          <span key={`fragment-${index}`} className="whitespace-pre-wrap">
            {fragment}
          </span>
        );
      }

      return (
        <span
          key={`citation-group-${index}`}
          className="mx-1 inline-flex flex-wrap items-center align-middle"
        >
          {fragment.ids.map((citationId, groupIndex) => {
            const citation = answer.citations.find((item) => item.id === citationId);
            if (!citation) {
              return (
                <span
                  key={`citation-missing-${index}-${groupIndex}`}
                  className="inline-flex items-center"
                >
                  {groupIndex > 0 && (
                    <span className="px-1 text-xs font-semibold text-orange-300/60">,</span>
                  )}
                  <span className="inline-flex cursor-not-allowed items-center rounded-full bg-slate-800 px-2 py-0.5 text-xs font-medium text-slate-400">
                    C{citationId}
                  </span>
                </span>
              );
            }

            return (
              <span key={`citation-${index}-${citation.id}`} className="inline-flex items-center">
                {groupIndex > 0 && (
                  <span className="px-1 text-xs font-semibold text-orange-300/60">,</span>
                )}
                <button
                  type="button"
                  className="inline-flex items-center rounded-full bg-gradient-to-r from-orange-500 to-amber-500 px-2 py-0.5 text-xs font-semibold text-white shadow-lg shadow-orange-500/30 transition hover:from-orange-400 hover:to-amber-400 hover:shadow-orange-400/40 focus:outline-none focus:ring-2 focus:ring-orange-400/60"
                  onClick={() => {
                    setActiveCitation({ answer, citation });
                    setHoveredCitation(null);
                  }}
                  onMouseEnter={(event) => showCitationTooltip(answer, citation, event.currentTarget)}
                  onMouseLeave={() => scheduleHideCitationTooltip()}
                  onFocus={(event) => showCitationTooltip(answer, citation, event.currentTarget)}
                  onBlur={() => scheduleHideCitationTooltip()}
                >
                  C{citation.id}
                </button>
              </span>
            );
          })}
        </span>
      );
    });
  };

  const renderChatMessage = (message: ChatMessage) => {
    if (message.role === "user") {
      if (message.mode === "batch" && message.questions) {
        return (
          <div key={message.id} className="flex justify-end">
            <div className="max-w-2xl space-y-3 rounded-2xl border border-orange-500/60 bg-[#1c0f2d] px-5 py-4 text-sm shadow-xl shadow-orange-500/10">
              <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wide text-orange-300">
                <span>Batch Questions</span>
                <span className="rounded-full bg-orange-500/20 px-2 py-0.5">{message.questions.length} items</span>
              </div>
              <ol className="space-y-2 text-left text-sm">
                {message.questions.map((question, index) => (
                  <li
                    key={`${message.id}-q-${index}`}
                    className="rounded-xl border border-orange-500/30 bg-[#2a1640] px-3 py-2.5 leading-5 text-orange-100"
                  >
                    <span className="mr-2 text-xs font-semibold text-orange-400">Q{index + 1}.</span>
                    {question}
                  </li>
                ))}
              </ol>
              <div className="text-right text-[11px] uppercase tracking-wide text-orange-300/70">
                {formatTimestamp(message.timestamp)}
              </div>
            </div>
          </div>
        );
      }

      return (
        <div key={message.id} className="flex justify-end">
          <div className="max-w-xl rounded-2xl border border-orange-400/50 bg-gradient-to-br from-orange-500 via-orange-400 to-amber-500 px-4 py-3 text-sm shadow-lg shadow-orange-500/30">
            <div className="font-bold leading-6 text-white">{message.content}</div>
            <div className="mt-1 text-right text-[11px] font-medium uppercase tracking-wide text-white/80">
              {formatTimestamp(message.timestamp)}
            </div>
          </div>
        </div>
      );
    }

    if (message.mode === "batch" && message.answers) {
      return (
        <div key={message.id} className="flex justify-start">
          <div className="w-full max-w-2xl space-y-4 rounded-2xl border border-orange-900/50 bg-[#050b17]/95 p-5 text-sm text-slate-100 shadow-xl">
            <div className="flex items-center gap-3 text-xs font-semibold uppercase tracking-wide text-orange-300/80">
              <span className="rounded-full bg-gradient-to-r from-orange-500/20 to-amber-500/20 px-3 py-1 text-orange-200 shadow-inner">Deep Intel</span>
              <span>Batch Response</span>
              <span className="ml-auto text-slate-400">{formatTimestamp(message.timestamp)}</span>
            </div>
            <div className="space-y-4">
              {message.answers.map((answer, index) => {
                const confidence = deriveConfidenceValue(answer);
                return (
                  <div
                    key={`${message.id}-answer-${index}`}
                    className="space-y-3 rounded-2xl border border-orange-900/40 bg-slate-900/80 p-4 shadow-inner"
                  >
                    <div className="text-xs font-semibold uppercase tracking-wide text-orange-300/80">
                      Q{index + 1}: {answer.question}
                    </div>
                    <div className="text-base leading-6 text-slate-100">
                      {renderAnswerText(answer)}
                    </div>
                    <div className="flex flex-wrap items-center gap-4">
                      <div className="flex items-center gap-3">
                        <div className="h-2 w-24 rounded-full bg-slate-800">
                          <div
                            className={`h-2 rounded-full shadow-inner transition-all ${confidenceColor(confidence)}`}
                            style={{ width: `${Math.min(100, Math.round((confidence ?? 0) * 100))}%` }}
                          />
                        </div>
                        <span className="text-xs font-medium text-slate-300/80">
                          {confidenceLabel(confidence)}
                        </span>
                      </div>
                      {answer.cache_hit && (
                        <span className="rounded-full border border-orange-400/40 bg-orange-500/10 px-3 py-1 text-xs font-semibold text-orange-200">
                          Cache Hit
                        </span>
                      )}
                      <button
                        type="button"
                        className="ml-auto rounded-full border border-orange-500/40 bg-orange-500/5 px-3 py-1 text-xs font-semibold text-orange-200 transition hover:border-orange-400 hover:bg-orange-500/10 hover:text-orange-100"
                        onClick={() => setActiveMetrics(answer)}
                      >
                        View RAG Metrics
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      );
    }

    if (!message.answer) {
      return null;
    }

    const { answer } = message;
    const confidence = deriveConfidenceValue(answer);

    return (
      <div key={message.id} className="flex justify-start">
        <div className="w-full max-w-2xl space-y-4 rounded-2xl border border-orange-900/50 bg-[#050b17]/95 p-5 text-sm text-slate-100 shadow-xl">
          <div className="flex items-center gap-3 text-xs font-semibold uppercase tracking-wide text-orange-300/80">
            <span className="rounded-full bg-gradient-to-r from-orange-500/20 to-amber-500/20 px-3 py-1 text-orange-200 shadow-inner">Deep Intel</span>
            <span>Answer</span>
            {answer.cache_hit && (
              <span className="rounded-full border border-orange-400/40 bg-orange-500/10 px-3 py-1 text-xs font-semibold text-orange-200">
                Cache Hit
              </span>
            )}
            <span className="ml-auto text-slate-400">{formatTimestamp(message.timestamp)}</span>
          </div>
          <div className="space-y-3 rounded-2xl border border-orange-900/40 bg-slate-900/80 p-4 shadow-inner">
            <div className="text-xs font-semibold uppercase tracking-wide text-orange-300/80">
              Question
            </div>
            <div className="text-base font-medium text-slate-100">
              {answer.question}
            </div>
            <div className="pt-2 text-base leading-6 text-slate-100">
              {renderAnswerText(answer)}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="h-2 w-28 rounded-full bg-slate-800">
                <div
                  className={`h-2 rounded-full shadow-inner transition-all ${confidenceColor(confidence)}`}
                  style={{ width: `${Math.min(100, Math.round((confidence ?? 0) * 100))}%` }}
                />
              </div>
              <span className="text-xs font-medium text-slate-300/80">
                {confidenceLabel(confidence)}
              </span>
            </div>
            <button
              type="button"
              className="ml-auto rounded-full border border-orange-500/40 bg-orange-500/5 px-3 py-1 text-xs font-semibold text-orange-200 transition hover:border-orange-400 hover:bg-orange-500/10 hover:text-orange-100"
              onClick={() => setActiveMetrics(answer)}
            >
              View RAG Metrics
            </button>
          </div>
        </div>
      </div>
    );
  };

  // Auto-scroll when chat messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-[#0b0613] via-[#12061f] to-[#040108] text-orange-100">
      <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
        <div className="halloween-cobweb cobweb-top-left" />
        <div className="halloween-cobweb cobweb-top-right" />
        <div className="halloween-candle candle-left" />
        <div className="halloween-candle candle-right" />
      </div>
      <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-6 py-10 lg:flex-row">
        <aside className="w-full space-y-6 rounded-3xl border border-slate-900/70 bg-[#170a28]/85 p-6 shadow-[0_45px_60px_-35px_rgba(0,0,0,0.85)] backdrop-blur lg:max-w-sm">
          <div className="space-y-1">
            <h1 className="text-3xl font-bold text-orange-100">Deep Intel</h1>
            <p className="text-sm text-orange-200/70">Process knowledge bases, monitor the pipeline, and interrogate them with full transparency.</p>
          </div>

          <div className="space-y-4 rounded-2xl border border-slate-900/60 bg-[#14071f]/90 p-4 shadow-inner">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-orange-300/80">Document Source</h2>
            <label className="block text-xs font-medium text-orange-200/80">Public Document URL</label>
            <input
              className="w-full rounded-xl border border-slate-800 bg-[#1c0f2d] px-3 py-2 text-sm text-orange-100 shadow-sm transition focus:border-orange-500 focus:outline-none focus:ring-2 focus:ring-orange-500/40"
              placeholder="https://..."
              value={documentDisplay}
              onChange={handleDocumentInputChange}
            />
            <div className="rounded-xl border border-dashed border-slate-800 p-3 text-xs text-orange-200/70">
              <p className="font-semibold text-orange-200">Upload a file</p>
              <p className="mt-1">Supported formats: PDF, DOCX, HTML, CSV, PPT, Images.</p>
              <div className="mt-3 flex items-center gap-3">
                <input
                  type="file"
                  onChange={handleFileSelection}
                  className="w-full text-xs text-orange-300/70"
                />
                <button
                  type="button"
                  onClick={uploadSelectedFile}
                  disabled={!uploadFile || uploadState.status === "uploading"}
                  className="rounded-lg bg-orange-500 px-3 py-2 text-xs font-semibold text-[#1a0d0d] shadow transition hover:bg-orange-400 disabled:cursor-not-allowed disabled:bg-orange-600/60"
                >
                  {uploadState.status === "uploading" ? "Uploading..." : "Upload Document"}
                </button>
              </div>
              {uploadState.status === "success" && (
                <p className="mt-2 text-xs text-emerald-400">Temporary URL ready.</p>
              )}
              {uploadState.status === "error" && (
                <p className="mt-2 text-xs text-rose-400">{uploadState.error}</p>
              )}
            </div>

            <label className="flex items-center gap-2 text-xs font-medium text-orange-200">
              <input
                type="checkbox"
                checked={forceRefresh}
                onChange={(event) => setForceRefresh(event.target.checked)}
                className="h-4 w-4 rounded border-slate-700 bg-[#1c0f2d] text-orange-500 focus:ring-orange-500"
              />
              Bypass cache and re-process document
            </label>

            <button
              type="button"
              onClick={() => submitDocument()}
              disabled={isSubmittingDoc}
              className="w-full rounded-xl bg-orange-500 px-4 py-2 text-sm font-semibold text-[#1a0d0d] shadow-lg transition hover:bg-orange-400 disabled:cursor-not-allowed disabled:bg-orange-600/70"
            >
              {isSubmittingDoc ? "Processing..." : "Process Document"}
            </button>
          </div>

          <div className="space-y-3 rounded-2xl border border-slate-900/60 bg-[#14071f]/90 p-4 shadow-inner">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-orange-300/80">Example Documents</h2>
            <div className="space-y-2">
              {preloadedDocs.map((doc) => (
                <button
                  key={doc.url}
                  type="button"
                  className={`w-full rounded-xl border px-3 py-2 text-left text-xs font-semibold transition ${selectedPreset === doc.url ? "border-orange-500 bg-orange-500/15 text-orange-200" : "border-slate-800 bg-[#1c0f2d] text-orange-200/70 hover:border-orange-400 hover:text-orange-200"}`}
                  onClick={() => {
                    setSelectedPreset(doc.url);
                    updateDocumentSource(doc.url, doc.label);
                  }}
                >
                  {doc.label}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-3 rounded-2xl border border-slate-900/60 bg-[#14071f]/90 p-4 shadow-inner">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-orange-300/80">Current Document</h2>
            {documentInfo ? (
              <div className="space-y-2 text-xs text-orange-200/80">
                <div className="font-semibold text-orange-200">{documentInfo.doc_type.toUpperCase()} • {documentInfo.cached ? "Cached" : "Fresh"}</div>
                <div className="truncate text-orange-300/70" title={documentInfo.url}>{documentInfo.url}</div>
                <div>Chunks: {documentInfo.chunk_count}</div>
                <div>Average Tokens: {Math.round(documentInfo.average_tokens || 0)}</div>
                <div>Doc Hash: <span className="font-mono text-[11px] text-orange-200">{documentInfo.doc_hash.slice(0, 10)}...</span></div>
              </div>
            ) : (
              <p className="text-xs text-orange-300/70">No document processed yet.</p>
            )}
          </div>
        </aside>

        <main className="flex w-full flex-1 flex-col gap-6 pb-10">
          {toast && (
            <div
              className={`mx-auto w-full max-w-2xl rounded-full border px-4 py-2 text-sm font-medium shadow ${toast.variant === "success" ? "border-emerald-500/50 bg-emerald-900/40 text-emerald-200" : toast.variant === "error" ? "border-rose-500/50 bg-rose-900/40 text-rose-200" : "border-orange-500/60 bg-orange-900/40 text-orange-200"}`}
            >
              {toast.message}
            </div>
          )}

          <section className="rounded-3xl border border-slate-900/70 bg-[#14071f]/90 p-6 shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wide text-orange-300/80">Pipeline Transparency</h2>
                <p className="text-xs text-orange-200/70">{pipelineActive ? "Processing document through the RAG pipeline..." : "Latest pipeline status."}</p>
              </div>
              <div className="text-xs font-medium text-orange-200/70">Progress: {pipelineProgress}%</div>
            </div>
            <div className="mt-5 grid gap-3 md:grid-cols-4">
              {pipelineSteps.map((step) => (
                <div key={step.id} className={`rounded-2xl border px-4 py-3 text-xs font-semibold transition ${statusStyles[step.status]}`}>
                  <div>{step.label}</div>
                  {step.detail && (
                    <div className="mt-1 text-[11px] font-medium uppercase text-orange-200/70">{step.detail}</div>
                  )}
                  {step.duration_ms && (
                    <div className="mt-1 text-[11px] text-orange-200/70">{step.duration_ms} ms</div>
                  )}
                </div>
              ))}
            </div>
          </section>

          <section className="flex flex-1 flex-col gap-4 rounded-3xl border border-slate-900/70 bg-[#0b121f]/90 p-6 shadow-xl">
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div>
                <h2 className="text-lg font-semibold text-orange-100">Ask with Confidence</h2>
                <p className="text-sm text-orange-200/70">Submit single or batched questions to interrogate the indexed context.</p>
              </div>
              <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-[#1c0f2d] px-2 py-1 text-xs font-semibold text-orange-200/60">
                <button
                  type="button"
                  onClick={() => setComposerMode("single")}
                  className={`rounded-full px-3 py-1 transition ${composerMode === "single" ? "bg-orange-500 text-[#1a0d0d]" : "hover:text-orange-300"}`}
                >
                  Single Question
                </button>
                <button
                  type="button"
                  onClick={() => setComposerMode("batch")}
                  className={`rounded-full px-3 py-1 transition ${composerMode === "batch" ? "bg-orange-500 text-[#1a0d0d]" : "hover:text-orange-300"}`}
                >
                  Batch Questions
                </button>
              </div>
            </div>

            <div className="flex h-[420px] flex-col gap-4 overflow-y-auto rounded-2xl border border-slate-900/70 bg-[#050b17] p-4" ref={chatContainerRef}>
              {chatMessages.length === 0 ? (
                <div className="flex h-full flex-col items-center justify-center text-center text-sm text-orange-200/70">
                  <p>Load a document, then ask targeted questions or paste a batch to see the RAG pipeline at work.</p>
                </div>
              ) : (
                chatMessages.map((message) => renderChatMessage(message))
              )}
            </div>

            <textarea
              className="min-h-[120px] w-full flex-1 rounded-2xl border border-slate-800 bg-[#1c0f2d] px-4 py-3 text-sm text-orange-100 shadow-inner transition focus:border-orange-500 focus:outline-none focus:ring-2 focus:ring-orange-500/40"
              placeholder={composerMode === "single" ? "Ask a focused question about the processed document." : "Enter multiple questions, each on a new line."}
              value={composerInput}
              onChange={(event) => setComposerInput(event.target.value)}
            />

            <div className="flex items-center justify-between">
              <p className="text-xs text-orange-200/70">Citations become interactive chips once the answer generates.</p>
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setComposerInput("")}
                  className="rounded-full border border-slate-800 px-3 py-1 text-xs font-semibold text-orange-200/70 transition hover:border-rose-500 hover:text-rose-400"
                >
                  Clear
                </button>
                <button
                  type="button"
                  onClick={handleQuestionsSubmit}
                  disabled={isSubmittingQuestion}
                  className="rounded-full bg-orange-500 px-4 py-2 text-xs font-semibold text-[#1a0d0d] shadow-lg transition hover:bg-orange-400 disabled:cursor-not-allowed disabled:bg-orange-600/70"
                >
                  {isSubmittingQuestion ? "Generating..." : composerMode === "single" ? "Submit Question" : "Submit Batch"}
                </button>
              </div>
            </div>
          </section>
        </main>
      </div>

      {hoveredCitation && (
        <div
          className="pointer-events-none fixed z-40"
          style={{
            top: hoveredCitation.position.y,
            left: hoveredCitation.position.x,
            transform:
              hoveredCitation.placement === "above"
                ? "translate(-50%, calc(-100% - 16px))"
                : "translate(-50%, 16px)",
          }}
        >
          <div
            className="relative pointer-events-auto w-80 max-w-sm rounded-xl border border-orange-500/40 bg-[#0f172a]/95 p-4 text-sm text-slate-100 shadow-2xl shadow-orange-500/20"
            onMouseEnter={clearCitationTooltipTimer}
            onMouseLeave={() => scheduleHideCitationTooltip(180)}
          >
            <div className="max-h-60 overflow-y-auto pr-1">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-orange-300/90">
                {hoveredCitation.citation.label}
              </div>
              <p className="mt-2 text-sm leading-5 text-slate-100/90">
                {hoveredCitation.citation.text}
              </p>
            </div>
            <div className="mt-3 flex flex-wrap gap-3 text-[10px] uppercase tracking-wide text-orange-300/70">
              {typeof hoveredCitation.citation.confidence === "number" && (
                <span>Confidence {Math.round(hoveredCitation.citation.confidence * 100)}%</span>
              )}
              {hoveredCitation.citation.score !== undefined && (
                <span>Score {hoveredCitation.citation.score?.toFixed(2)}</span>
              )}
            </div>
            <div className="mt-2 text-[10px] text-slate-300/70">
              Question: {hoveredCitation.answer.question}
            </div>
            {hoveredCitation.placement === "above" ? (
              <div className="pointer-events-none absolute left-1/2 top-full -mt-[1px] -translate-x-1/2">
                <div className="h-3 w-3 rotate-45 border border-orange-500/40 bg-[#0f172a]" />
              </div>
            ) : (
              <div className="pointer-events-none absolute left-1/2 bottom-full -mb-[1px] -translate-x-1/2">
                <div className="h-3 w-3 rotate-45 border border-orange-500/40 bg-[#0f172a]" />
              </div>
            )}
          </div>
        </div>
      )}

      {activeCitation && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 px-4">
          <div className="w-full max-w-2xl space-y-4 rounded-2xl border border-slate-900 bg-[#12081c] p-6 text-sm text-orange-100 shadow-2xl">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-semibold text-orange-200">{activeCitation.citation.label}</h3>
                <p className="text-xs text-orange-200/70">Question: {activeCitation.answer.question}</p>
              </div>
              <button
                type="button"
                onClick={() => setActiveCitation(null)}
                className="rounded-full border border-slate-800 px-3 py-1 text-xs font-semibold text-orange-200/70 hover:border-orange-400 hover:text-orange-100"
              >
                Close
              </button>
            </div>
            <div className="rounded-xl border border-slate-900 bg-[#190c2c] p-4 text-sm leading-6 text-orange-100">
              {activeCitation.citation.text}
            </div>
            <div className="flex flex-wrap items-center gap-4 text-xs text-orange-200/70">
              {typeof activeCitation.citation.confidence === "number" && (
                <span>Confidence: {Math.round(activeCitation.citation.confidence * 100)}%</span>
              )}
              {activeCitation.citation.score !== undefined && (
                <span>Score: {activeCitation.citation.score?.toFixed(3)}</span>
              )}
              {activeCitation.citation.source_url && (
                <a
                  href={activeCitation.citation.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-full border border-orange-500 px-3 py-1 font-semibold text-orange-200 hover:bg-orange-500/10"
                >
                  Open Source
                </a>
              )}
            </div>
          </div>
        </div>
      )}

      {activeMetrics && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 px-4">
          <div className="w-full max-w-2xl space-y-5 rounded-2xl border border-slate-900 bg-[#14071f] p-6 text-sm text-orange-100 shadow-2xl">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-semibold text-orange-200">RAG Metrics</h3>
                <p className="text-xs text-orange-200/70">Question: {activeMetrics.question}</p>
              </div>
              <button
                type="button"
                onClick={() => setActiveMetrics(null)}
                className="rounded-full border border-slate-800 px-3 py-1 text-xs font-semibold text-orange-200/70 hover:border-orange-400 hover:text-orange-100"
              >
                Close
              </button>
            </div>

            <div className="space-y-3 rounded-xl border border-slate-900 bg-[#190c2c] p-4">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-orange-300/80">Answer Timing</h4>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="rounded-lg border border-slate-800/70 bg-[#0f051d] p-3">
                  <div className="font-semibold text-orange-200">Search</div>
                  <div>{activeMetrics.timing?.search_ms ?? 0} ms</div>
                </div>
                <div className="rounded-lg border border-slate-800/70 bg-[#0f051d] p-3">
                  <div className="font-semibold text-orange-200">Re-rank</div>
                  <div>{activeMetrics.timing?.rerank_ms ?? 0} ms</div>
                </div>
                <div className="rounded-lg border border-slate-800/70 bg-[#0f051d] p-3">
                  <div className="font-semibold text-orange-200">Generation</div>
                  <div>{activeMetrics.timing?.generation_ms ?? 0} ms</div>
                </div>
                <div className="rounded-lg border border-slate-800/70 bg-[#0f051d] p-3">
                  <div className="font-semibold text-orange-200">Answer Total</div>
                  <div>{activeMetrics.timing?.total_ms ?? 0} ms</div>
                </div>
              </div>
            </div>

            {lastRunMeta && (
              <div className="space-y-3 rounded-xl border border-slate-900 bg-[#190c2c] p-4">
                <h4 className="text-xs font-semibold uppercase tracking-wide text-orange-300/80">Request Summary</h4>
                <div className="grid gap-2 text-xs text-orange-200/80">
                  {Object.entries(lastRunMeta.timing).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between">
                      <span className="font-medium capitalize text-orange-200/70">{key.replace(/_/g, " ")}</span>
                      <span>{value as unknown as string}</span>
                    </div>
                  ))}
                  {Object.entries(lastRunMeta.metrics).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between">
                      <span className="font-medium capitalize text-orange-200/70">{key.replace(/_/g, " ")}</span>
                      <span>{typeof value === "number" ? value : JSON.stringify(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
