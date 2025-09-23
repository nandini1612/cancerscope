import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Download, Eye, Loader2 } from "lucide-react";
import MetricsCard from "@/components/MetricsCard";

interface PredictionResult {
  patientId: string;
  predictedClass: "Malignant" | "Benign";
  probability: number; // backend gives 0-1, we'll format as %
  shapPlotUrl?: string;
  topFeatures?: string[];
  explanationText?: string;
}

interface Metrics {
  sensitivity: number;
  specificity: number;
  rocAuc: number;
  prAuc: number;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';

// Helper function to format percentages
const formatPercent = (value: number, decimals: number = 2) => 
  `${(value * 100).toFixed(decimals)}%`;

export default function RiskPrediction() {
  const [hasUploaded, setHasUploaded] = useState(false);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedShapUrl, setSelectedShapUrl] = useState<string | null>(null);
  const [selectedExplanation, setSelectedExplanation] = useState<string | null>(null);

  // Fetch metrics on component mount
  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/predict-tabular/metrics`);
      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
      }
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
    }
  };

  const handleUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Always request SHAP explanations by adding the shap parameter
      const response = await fetch(`${API_BASE_URL}/predict-tabular/shap`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process CSV file');
      }

      const data = await response.json();
      
      // Handle both single prediction and multiple predictions
      if (data.predictions) {
        setPredictions(data.predictions);
      } else if (Array.isArray(data)) {
        setPredictions(data);
      } else {
        // Single prediction - wrap in array
        setPredictions([data]);
      }
      
      setHasUploaded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/predict-tabular/download`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'predictions.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (err) {
      console.error('Failed to download predictions:', err);
    }
  };

  const handleViewShap = (prediction: PredictionResult) => {
    if (prediction.shapPlotUrl) {
      setSelectedShapUrl(`${API_BASE_URL}${prediction.shapPlotUrl}`);
      setSelectedExplanation(prediction.explanationText || 'SHAP explanation not available');
    }
  };

  const resetUpload = () => {
    setHasUploaded(false);
    setPredictions([]);
    setSelectedShapUrl(null);
    setSelectedExplanation(null);
    setError(null);
  };

  return (
    <div className="space-y-8 px-4 sm:px-6 lg:px-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-foreground">Risk Prediction (Tabular Data)</h1>
        <p className="text-lg text-muted-foreground">
          Upload patient diagnostic features to explore predictions with SHAP explanations. Educational use only.
        </p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 text-sm">{error}</p>
        </div>
      )}

      <div className="space-y-8">
        {!hasUploaded ? (
          <UploadCard
            title="Upload CSV"
            description="File must include diagnostic features from UCI Breast Cancer Dataset."
            buttonText={isLoading ? "Processing..." : "Select CSV File"}
            onUpload={handleUpload}
            accept=".csv"
            disabled={isLoading}
          />
        ) : (
          <div className="space-y-6">
            <div className="bg-card border border-border rounded-lg p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold text-foreground">Prediction Results</h2>
                <Button 
                  onClick={handleDownload}
                  variant="outline"
                  size="sm"
                  className="flex items-center space-x-2"
                >
                  <Download className="w-4 h-4" />
                  <span>Download Predictions (CSV)</span>
                </Button>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Patient ID</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Predicted Class</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Probability</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Explanation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictions.map((result, index) => (
                      <tr key={result.patientId || `patient-${index}`} className="border-b border-border/50">
                        <td className="py-3 px-4 text-sm text-foreground">{result.patientId || `Patient ${index + 1}`}</td>
                        <td className="py-3 px-4 text-sm">
                          <span className={`font-medium ${
                            result.predictedClass === "Malignant" 
                              ? "text-red-600" 
                              : "text-green-600"
                          }`}>
                            {result.predictedClass}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-sm text-foreground">
                          {formatPercent(result.probability)}
                        </td>
                        <td className="py-3 px-4">
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={() => handleViewShap(result)}
                            disabled={!result.shapPlotUrl}
                          >
                            <Eye className="w-4 h-4 mr-1" />
                            {result.shapPlotUrl ? 'View SHAP' : 'No SHAP'}
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {selectedShapUrl && (
              <div className="bg-card border border-border rounded-lg p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-foreground">SHAP Explanation</h3>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => {
                      setSelectedShapUrl(null);
                      setSelectedExplanation(null);
                    }}
                  >
                    Close
                  </Button>
                </div>
                <div className="text-center mb-4">
                  <img 
                    src={selectedShapUrl} 
                    alt="SHAP Explanation"
                    className="max-w-full h-auto rounded-lg border mx-auto"
                    style={{ maxHeight: '600px' }}
                  />
                </div>
                <div className="space-y-4">
                  <div className="text-sm text-muted-foreground">
                    <p><strong>How to interpret SHAP waterfall plots:</strong></p>
                    <ul className="list-disc list-inside space-y-1 mt-2">
                      <li>Red bars push the prediction toward <strong>Malignant</strong></li>
                      <li>Blue bars push the prediction toward <strong>Benign</strong></li>
                      <li>Longer bars indicate stronger influence on the final prediction</li>
                      <li>The final prediction is shown at the bottom right</li>
                    </ul>
                  </div>
                  {selectedExplanation && (
                    <div className="bg-muted/50 rounded-lg p-4">
                      <h4 className="font-medium mb-2">Detailed Explanation:</h4>
                      <pre className="text-sm text-muted-foreground whitespace-pre-wrap">
                        {selectedExplanation}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            )}

            <Button 
              variant="outline"
              onClick={resetUpload}
              className="w-full"
            >
              Upload Another CSV File
            </Button>
          </div>
        )}

        {/* Model Performance Metrics - Always Visible */}
        <div>
          <h2 className="text-xl font-semibold text-foreground mb-4">Model Performance Metrics</h2>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {metrics ? (
              <>
                <MetricsCard
                  title="Sensitivity/Recall"
                  value={formatPercent(metrics.sensitivity, 1)}
                />
                <MetricsCard
                  title="Specificity"
                  value={formatPercent(metrics.specificity, 1)}
                />
                <MetricsCard
                  title="ROC-AUC"
                  value={metrics.rocAuc.toFixed(3)}
                />
                <MetricsCard title="PR-AUC" value={metrics.prAuc.toFixed(3)} />
              </>
            ) : (
              <div className="col-span-2 lg:col-span-4 bg-card border border-border rounded-lg p-4 text-center">
                <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
                <p className="text-sm text-muted-foreground">Loading metrics...</p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="bg-warning/20 border border-warning rounded-lg p-4">
        <p className="text-sm text-warning-foreground text-center">
          <strong>Disclaimer:</strong> CancerScope is a research tool and should
          not be used for clinical diagnosis. It is intended for informational
          purposes only. Always consult with a qualified healthcare professional
          for medical advice, diagnosis, and treatment.
        </p>
      </div>
    </div>
  );
}

interface UploadCardProps {
  title: string;
  description: string;
  buttonText: string;
  onUpload: (file: File) => void | Promise<void>;
  accept?: string;
  disabled?: boolean;
}

function UploadCard({ title, description, buttonText, onUpload, accept, disabled }: UploadCardProps) {
  const [fileName, setFileName] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileChange = (files: FileList | null) => {
    if (files && files.length > 0) {
      const file = files[0];
      setFileName(file.name);
      onUpload(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    handleFileChange(files);
  };

  return (
    <div
      className={`border-2 rounded-lg p-6 transition-all duration-200 flex flex-col justify-center items-center text-center
      ${disabled ? "opacity-50 pointer-events-none" : ""}
      ${isDragOver ? "border-primary bg-primary/10" : "border-dashed border-border bg-card"}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <h2 className="text-lg font-semibold text-foreground mb-2">{title}</h2>
      <p className="text-sm text-muted-foreground mb-4">{description}</p>
      <Button
        variant="outline"
        size="sm"
        className="flex items-center space-x-2"
        onClick={() => document.getElementById("file-input")?.click()}
        disabled={disabled}
      >
        {disabled ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <span>{buttonText}</span>
        )}
      </Button>
      <input
        id="file-input"
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => handleFileChange(e.target.files)}
      />
      {fileName && (
        <p className="text-sm text-foreground mt-2">
          <strong>Selected file:</strong> {fileName}
        </p>
      )}
      <p className="text-xs text-muted-foreground text-center mt-2">
        {disabled
          ? "Processing CSV, please wait..."
          : "Supported format: CSV. Max size: 10MB."}
      </p>
    </div>
  );
}