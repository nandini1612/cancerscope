import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, ScatterChart, Scatter, Tooltip } from "recharts";

const benignMalignantData = [
  { name: "Benign", value: 82, color: "hsl(var(--chart-benign))" },
  { name: "Malignant", value: 38, color: "hsl(var(--chart-malignant))" },
];

const featuresData = [
  { name: "Area (Worst)", value: 92 },
  { name: "Concave Points (Mean)", value: 87 },
  { name: "Radius (Worst)", value: 83 },
  { name: "Perimeter (Worst)", value: 78 },
  { name: "Concave Points (Worst)", value: 75 },
];

const scatterData = [
  { x: -2.5, y: 1.2, type: "Benign" },
  { x: -1.8, y: 0.8, type: "Benign" },
  { x: -2.1, y: 1.5, type: "Benign" },
  { x: -1.5, y: 0.5, type: "Benign" },
  { x: -2.3, y: 1.1, type: "Benign" },
  { x: -1.9, y: 0.9, type: "Benign" },
  { x: -2.0, y: 1.3, type: "Benign" },
  { x: -1.7, y: 0.7, type: "Benign" },
  { x: -2.4, y: 1.4, type: "Benign" },
  { x: -1.6, y: 0.6, type: "Benign" },
  { x: 1.8, y: -0.8, type: "Malignant" },
  { x: 2.2, y: -1.2, type: "Malignant" },
  { x: 1.5, y: -0.5, type: "Malignant" },
  { x: 2.0, y: -1.0, type: "Malignant" },
  { x: 2.5, y: -1.5, type: "Malignant" },
  { x: 1.7, y: -0.7, type: "Malignant" },
  { x: 2.3, y: -1.3, type: "Malignant" },
  { x: 1.9, y: -0.9, type: "Malignant" },
  { x: 2.1, y: -1.1, type: "Malignant" },
  { x: 1.6, y: -0.6, type: "Malignant" },
];

export default function Learn() {
  const totalCases = benignMalignantData.reduce((sum, entry) => sum + entry.value, 0);

  return (
    <div className="space-y-8 p-4 sm:p-6 lg:p-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-foreground">Understanding Breast Cancer</h1>
        <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
          Breast cancer is a disease in which cells in the breast grow out of control.
          Understanding the data behind it is the first step towards early detection and better outcomes.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-card border border-border rounded-lg p-6">
          <h2 className="text-xl font-semibold text-foreground mb-4">Benign vs. Malignant Balance</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={benignMalignantData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {benignMalignantData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  cursor={{ fill: 'hsla(var(--muted-foreground), 0.1)' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const entry = payload[0];
                      return (
                        <div className="rounded-md border border-border bg-popover px-3 py-2 shadow-lg text-sm min-w-[100px]">
                          <div className="flex items-center space-x-2">
                            <span
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: entry.payload.color, display: "inline-block" }}
                            ></span>
                            <span className="font-medium text-foreground">{entry.payload.name}</span>
                            <span className="ml-auto text-muted-foreground">{entry.payload.value}</span>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                  wrapperStyle={{
                    outline: 'none',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center space-x-6 mt-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-chart-benign rounded-full"></div>
              <span className="text-sm text-muted-foreground">Benign ({Math.round((benignMalignantData[0].value / totalCases) * 100)}%)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-chart-malignant rounded-full"></div>
              <span className="text-sm text-muted-foreground">Malignant ({Math.round((benignMalignantData[1].value / totalCases) * 100)}%)</span>
            </div>
          </div>
        </div>

        <div className="bg-card border border-border rounded-lg p-6">
          <h2 className="text-xl font-semibold text-foreground mb-4">Top 5 Predictive Features</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={featuresData}
                margin={{ top: 20, right: 30, left: 120, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--chart-grid))" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="name" type="category" width={100} />
                <Tooltip
                  cursor={{ fill: 'hsla(var(--muted-foreground), 0.1)' }}
                  contentStyle={{
                    backgroundColor: 'hsl(var(--background))',
                    borderColor: 'hsl(var(--border))'
                  }}
                />
                <Bar dataKey="value" fill="hsl(var(--chart-malignant))" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-xl font-semibold text-foreground mb-4">PCA 2D Scatter Plot</h2>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart
              margin={{ top: 20, right: 20, bottom: 60, left: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--chart-grid))" />
              <XAxis
                type="number"
                dataKey="x"
                name="Principal Component 1"
                label={{ value: 'Principal Component 1', position: 'insideBottom', offset: -10 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="Principal Component 2"
                label={{ value: 'Principal Component 2', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const entry = payload[0];
                    return (
                      <div className="rounded-md border border-border bg-popover px-3 py-2 shadow-lg text-sm min-w-[120px]">
                        <div className="flex flex-col space-y-1">
                          <span className="font-medium text-foreground">{entry.payload.type}</span>
                          <span className="text-muted-foreground">
                            PC1: {entry.payload.x}, PC2: {entry.payload.y}
                          </span>
                          <span
                            className="inline-block w-3 h-3 rounded-full mt-1"
                            style={{
                              backgroundColor:
                                entry.payload.type === "Benign"
                                  ? "hsl(var(--chart-benign))"
                                  : "hsl(var(--chart-malignant))",
                            }}
                          ></span>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
                wrapperStyle={{
                  outline: 'none',
                }}
              />
              <Scatter
                name="Benign"
                data={scatterData.filter(d => d.type === "Benign")}
                fill="hsl(var(--chart-benign))"
              />
              <Scatter
                name="Malignant"
                data={scatterData.filter(d => d.type === "Malignant")}
                fill="hsl(var(--chart-malignant))"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="flex justify-center space-x-6 mt-4">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-chart-benign rounded-full"></div>
            <span className="text-sm text-muted-foreground">Benign</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-chart-malignant rounded-full"></div>
            <span className="text-sm text-muted-foreground">Malignant</span>
          </div>
        </div>
      </div>

      <div className="bg-warning/20 border border-warning rounded-lg p-4">
        <p className="text-sm text-warning-foreground text-center">
          ⚠️ This tool is for informational purposes only and not a substitute for professional medical advice.
        </p>
      </div>
    </div>
  );
}