interface ProgressProps {
  value: number; // 0–100
  className?: string;
}

export function Progress({ value, className = '' }: ProgressProps) {
  const clamped = Math.min(100, Math.max(0, value));
  return (
    <div className={`w-full bg-gray-200 rounded-full h-2 ${className}`}>
      <div
        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}