function renderValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (Array.isArray(value)) return value.map(String).join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function KeyValues({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data).filter(
    ([, v]) => renderValue(v) !== "—" && renderValue(v) !== "",
  );
  if (entries.length === 0) {
    return <p className="text-sm text-zinc-500">Nothing recorded yet.</p>;
  }
  return (
    <dl className="grid gap-2 sm:grid-cols-2">
      {entries.map(([key, value]) => (
        <div key={key} className="rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2">
          <dt className="text-[11px] uppercase tracking-wide text-zinc-500">{key}</dt>
          <dd className="mt-0.5 break-words text-sm text-zinc-200">{renderValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}
