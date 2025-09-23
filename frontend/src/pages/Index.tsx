import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { Activity, Upload, BarChart3, FileText } from "lucide-react";

export default function Index() {
  return (
    <div className="space-y-16 px-4 sm:px-6 lg:px-8">
      {/* Hero Section */}
      <div className="text-center space-y-6">
        <h1 className="text-5xl font-bold text-foreground">
          CancerScope
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Advanced medical imaging analysis platform for researchers and healthcare professionals
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button asChild size="lg" className="bg-primary hover:bg-primary/90">
            <Link to="/learn">Get Started</Link>
          </Button>
          <Button asChild variant="outline" size="lg">
            <Link to="/about">Learn More</Link>
          </Button>
        </div>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Link 
          to="/learn" 
          className="group bg-card border border-border rounded-lg p-6 hover:border-primary transition-colors"
        >
          <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
            <BarChart3 className="w-6 h-6 text-primary" />
          </div>
          <h3 className="text-lg font-semibold text-foreground mb-2">Learn</h3>
          <p className="text-sm text-muted-foreground">
            Explore breast cancer data patterns and visualization insights.
          </p>
        </Link>

        <Link 
          to="/scan" 
          className="group bg-card border border-border rounded-lg p-6 hover:border-primary transition-colors"
        >
          <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
            <Upload className="w-6 h-6 text-primary" />
          </div>
          <h3 className="text-lg font-semibold text-foreground mb-2">Scan</h3>
          <p className="text-sm text-muted-foreground">
            Upload medical images for AI-powered analysis and predictions.
          </p>
        </Link>

        <Link 
          to="/risk-prediction" 
          className="group bg-card border border-border rounded-lg p-6 hover:border-primary transition-colors"
        >
          <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
            <Activity className="w-6 h-6 text-primary" />
          </div>
          <h3 className="text-lg font-semibold text-foreground mb-2">Risk Prediction</h3>
          <p className="text-sm text-muted-foreground">
            Analyze patient data to predict cancer risk probabilities.
          </p>
        </Link>

        <Link 
          to="/about" 
          className="group bg-card border border-border rounded-lg p-6 hover:border-primary transition-colors"
        >
          <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
            <FileText className="w-6 h-6 text-primary" />
          </div>
          <h3 className="text-lg font-semibold text-foreground mb-2">About</h3>
          <p className="text-sm text-muted-foreground">
            Documentation, ethics, and model transparency information.
          </p>
        </Link>
      </div>

      {/* Disclaimer */}
      <div className="bg-warning/20 border border-warning rounded-lg p-6">
        <h3 className="text-lg font-semibold text-warning-foreground mb-2">Important Notice</h3>
        <p className="text-warning-foreground">
          This tool is designed for educational and research purposes only. It is not intended for clinical 
          diagnosis or medical decision-making. Always consult with qualified healthcare professionals for 
          medical advice and treatment.
        </p>
      </div>
    </div>
  );
}
