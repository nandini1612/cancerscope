import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Eye, Loader2 } from "lucide-react";
import MetricsCard from "@/components/MetricsCard";

interface ImagePredictionResult {
  predictedClass: "Malignant" | "Benign";
  confidence: number; // backend gives 0–1
  gradCamUrl?: string;
}

interface Metrics {
  sensitivity: number;
  specificity: number;
  rocAuc: number;
  prAuc: number;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:3001";

// helper for formatting percentages
const formatPercent = (value: number, decimals: number = 2) =>
  `${(value * 100).toFixed(decimals)}%`;

export default function ImageAnalysis() {
  const [hasUploaded, setHasUploaded] = useState(false);
  const [prediction, setPrediction] = useState<ImagePredictionResult | null>(
    null
  );
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [grayscaleImage, setGrayscaleImage] = useState<string | null>(null);

  // Fetch metrics on component mount
  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/metrics`);
      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
      }
    } catch (err) {
      console.error("Failed to fetch metrics:", err);
    }
  };

  // Function to convert image to grayscale
  const convertToGrayscale = (imageUrl: string): Promise<string> => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();
      
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        
        // Draw original image
        ctx!.drawImage(img, 0, 0);
        
        // Get image data and convert to grayscale
        const imageData = ctx!.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        for (let i = 0; i < data.length; i += 4) {
          const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
          data[i] = gray;     // Red
          data[i + 1] = gray; // Green
          data[i + 2] = gray; // Blue
          // Alpha channel (data[i + 3]) remains unchanged
        }
        
        ctx!.putImageData(imageData, 0, 0);
        resolve(canvas.toDataURL());
      };
      
      img.src = imageUrl;
    });
  };

  const handleUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);

    // Create a local URL for the uploaded image
    const imageUrl = URL.createObjectURL(file);
    setUploadedImage(imageUrl);

    // Convert to grayscale
    try {
      const grayscaleUrl = await convertToGrayscale(imageUrl);
      setGrayscaleImage(grayscaleUrl);
    } catch (err) {
      console.error("Failed to convert to grayscale:", err);
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/predict-image`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process image file");
      }

      const data = await response.json();
      setPrediction(data);
      setHasUploaded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      // Clean up the image URL if there's an error
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
        setUploadedImage(null);
        setGrayscaleImage(null);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const resetUpload = () => {
    setHasUploaded(false);
    setPrediction(null);
    setError(null);
    if (uploadedImage) {
      URL.revokeObjectURL(uploadedImage);
      setUploadedImage(null);
      setGrayscaleImage(null);
    }
  };

  return (
    <div className="space-y-8 px-4 sm:px-6 lg:px-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-foreground">Image Analysis</h1>
        <p className="text-lg text-muted-foreground">
          Upload histopathology images for breast cancer analysis with Grad-CAM explanations. Educational
          use only.
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
            title="Upload an Image"
            description="Drag and drop or click to select a histopathology image for analysis."
            buttonText={isLoading ? "Processing..." : "Upload Image"}
            onUpload={handleUpload}
            accept="image/*"
            disabled={isLoading}
          />
        ) : (
          <div className="space-y-8">
            {/* Two Images Side by Side */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Original Image */}
              <div className="bg-card border border-border rounded-lg p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Original Image</h3>
                <div className="aspect-square bg-muted rounded-lg overflow-hidden">
                  {uploadedImage ? (
                    <img
                      src={uploadedImage}
                      alt="Original histopathology image"
                      className="w-full h-full object-cover rounded-lg"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <div className="text-center text-muted-foreground">
                        <div className="w-16 h-16 bg-muted-foreground/20 rounded-lg mx-auto mb-2"></div>
                        <p className="text-sm">Original Image</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Grayscale Image with Results */}
              <div className="bg-card border border-border rounded-lg p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Analysis Result</h3>
                <div className="aspect-square bg-muted rounded-lg overflow-hidden mb-4">
                  {grayscaleImage ? (
                    <div className="relative w-full h-full">
                      {/* Grayscale base image */}
                      <img
                        src={grayscaleImage}
                        alt="Grayscale histopathology image"
                        className="w-full h-full object-cover rounded-lg"
                      />
                      {/* Grad-CAM heatmap overlay with vibrant colors */}
                      {prediction?.gradCamUrl && (
                        <div 
                          className="absolute inset-0 w-full h-full rounded-lg"
                          style={{
                            backgroundImage: `url(${API_BASE_URL}${prediction.gradCamUrl})`,
                            backgroundSize: 'cover',
                            backgroundPosition: 'center',
                            mixBlendMode: 'screen',
                            opacity: 0.9,
                            filter: 'contrast(2) saturate(3) brightness(1.2)'
                          }}
                        />
                      )}
                    </div>
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <div className="text-center text-muted-foreground">
                        <div className="w-16 h-16 bg-muted-foreground/20 rounded-lg mx-auto mb-2"></div>
                        <p className="text-sm">Processed Image</p>
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Results Display */}
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">
                      Prediction:
                    </span>
                    <span
                      className={`text-lg font-semibold ${
                        prediction?.predictedClass === "Malignant"
                          ? "text-red-600"
                          : "text-green-600"
                      }`}
                    >
                      {prediction?.predictedClass || "Unknown"}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">
                      Confidence:
                    </span>
                    <span className="text-lg font-semibold text-foreground">
                      {prediction
                        ? formatPercent(prediction.confidence, 2)
                        : "0.00%"}
                    </span>
                  </div>
                  {prediction?.gradCamUrl && (
                    <div className="mt-4 text-sm text-muted-foreground space-y-2">
                      <p><strong>Grad-CAM Analysis Applied:</strong></p>
                      <p>Heat map overlay shows regions that influenced the model's decision. Warmer areas indicate higher importance for the prediction.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <Button
              variant="outline"
              onClick={resetUpload}
              className="w-full"
            >
              Upload Another Image
            </Button>
          </div>
        )}

        {/* Model Performance Metrics - Always Visible */}
        <div>
          <h2 className="text-xl font-semibold text-foreground mb-4">
            Model Performance Metrics
          </h2>
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

      <div className="bg-warning/20 border border-warning rounded-lg p-4 mt-8">
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

function UploadCard({
  title,
  description,
  buttonText,
  onUpload,
  accept,
  disabled,
}: UploadCardProps) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileChange = (files: FileList | null) => {
    if (files && files.length > 0) {
      const file = files[0];
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
      <p className="text-xs text-muted-foreground text-center mt-2">
        {disabled
          ? "Processing image, please wait..."
          : "Supported formats: JPG, PNG, GIF. Max size: 10MB."}
      </p>
    </div>
  );
}