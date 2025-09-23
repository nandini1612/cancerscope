export default function DocsEthics() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-foreground">Docs & Ethics</h1>
        <p className="text-lg text-muted-foreground">
          Transparency and ethical considerations are at the core of CancerScope.
        </p>
      </div>

      {/* Model Card */}
      <div className="space-y-6">
        <div className="flex items-center space-x-3">
          <div className="w-6 h-6 bg-primary rounded flex items-center justify-center">
            <div className="w-3 h-3 bg-primary-foreground rounded"></div>
          </div>
          <h2 className="text-2xl font-bold text-foreground">Model Card</h2>
        </div>

        <div className="bg-card border border-border rounded-lg p-6 space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-primary mb-3">Dataset</h3>
            <p className="text-muted-foreground leading-relaxed">
              The model was trained on a dataset of mammograms from the National Cancer Institute, 
              comprising over 100,000 images. The dataset includes a diverse range of cases, ensuring 
              robust performance across different demographics and disease stages.
            </p>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-primary mb-3">Metrics</h3>
            <p className="text-muted-foreground leading-relaxed">
              The model achieves an accuracy of 95% and a recall of 92% on the validation set. 
              These metrics indicate a high level of reliability in detecting breast cancer while 
              minimizing false negatives.
            </p>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-primary mb-3">Limitations</h3>
            <p className="text-muted-foreground leading-relaxed">
              The model's performance may vary with image quality and specific patient characteristics. 
              It is not a substitute for professional medical advice and should be used as a supplementary tool.
            </p>
          </div>
        </div>
      </div>

      {/* Data Card */}
      <div className="space-y-6">
        <div className="flex items-center space-x-3">
          <div className="w-6 h-6 bg-primary rounded flex items-center justify-center">
            <div className="w-3 h-3 bg-primary-foreground rounded"></div>
          </div>
          <h2 className="text-2xl font-bold text-foreground">Data Card</h2>
        </div>

        <div className="bg-card border border-border rounded-lg p-6">
          <div>
            <h3 className="text-lg font-semibold text-primary mb-3">Dataset Sources</h3>
            <p className="text-muted-foreground leading-relaxed">
              The dataset is sourced from the National Cancer Institute's public database of mammograms. 
              This ensures transparency and allows for independent verification of the model's performance.
            </p>
          </div>
        </div>
      </div>

      {/* Ethics Disclaimer */}
      <div className="bg-destructive/10 border border-destructive rounded-lg p-6">
        <div className="flex items-center justify-center mb-4">
          <div className="w-8 h-8 bg-destructive rounded-full flex items-center justify-center">
            <span className="text-destructive-foreground text-sm font-bold">!</span>
          </div>
        </div>
        <h3 className="text-xl font-bold text-center text-foreground mb-4">Ethics Disclaimer</h3>
        <p className="text-center text-muted-foreground leading-relaxed">
          Not for diagnostic use. This tool is intended for educational and research purposes only. 
          Always consult with a qualified healthcare professional for any health concerns or before 
          making any medical decisions.
        </p>
      </div>
    </div>
  );
}