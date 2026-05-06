export type AnyRecord = Record<string, any>;

export type LogEntry = {
  id: string;
  ts: number;
  level: "DEBUG" | "INFO" | "WARN" | "ERROR";
  category: string;
  message: string;
  details?: AnyRecord;
};

export type PromptProfile = {
  slug: string;
  name: string;
  description: string;
  active: boolean;
  updated_at?: number;
};

export type StatusResponse = {
  health: AnyRecord;
  esp: AnyRecord;
  pipeline: AnyRecord;
  stt: AnyRecord;
  llm: AnyRecord;
  tts: AnyRecord;
  config: AnyRecord;
};

