import type { AnyRecord } from "./types";

export function getStoredToken() {
  return window.localStorage.getItem("alice_panel_token") || "";
}

export function setStoredToken(token: string) {
  if (token) {
    window.localStorage.setItem("alice_panel_token", token);
    document.cookie = `alice_panel_token=${encodeURIComponent(token)}; path=/; SameSite=Lax`;
  } else {
    window.localStorage.removeItem("alice_panel_token");
    document.cookie = "alice_panel_token=; Max-Age=0; path=/";
  }
}

export async function apiFetch<T>(path: string, options: RequestInit = {}, token = getStoredToken()): Promise<T> {
  const headers = new Headers(options.headers || {});
  if (token) headers.set("X-Alice-Token", token);
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const resp = await fetch(path, { ...options, headers });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  const contentType = resp.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return (await resp.json()) as T;
  return (await resp.text()) as T;
}

export function wsUrl(path: string, token = getStoredToken()) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const url = new URL(`${proto}://${window.location.host}${path}`);
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

export function setDeep(target: AnyRecord, path: string, value: any) {
  const keys = path.split(".");
  const clone = structuredClone(target || {});
  let cursor = clone;
  keys.slice(0, -1).forEach((key) => {
    cursor[key] = typeof cursor[key] === "object" && cursor[key] !== null ? cursor[key] : {};
    cursor = cursor[key];
  });
  cursor[keys[keys.length - 1]] = value;
  return clone;
}

export function downloadJson(filename: string, value: any) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

