interface MetricsCardProps {
  title: string;
  value: string | number;
  className?: string;
}

export default function MetricsCard({ title, value, className }: MetricsCardProps) {
  return (
    <div className={`bg-card border border-border rounded-lg p-4 ${className || ""}`}>
      <h3 className="text-sm font-medium text-muted-foreground mb-2">{title}</h3>
      <p className="text-2xl font-bold text-primary">{value}</p>
    </div>
  );
}