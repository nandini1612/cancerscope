import { Upload } from "lucide-react";
import { Button } from "@/components/ui/button";

interface UploadCardProps {
  title: string;
  description: string;
  buttonText: string;
  onUpload: () => void;
  accept?: string;
}

export default function UploadCard({ title, description, buttonText, onUpload, accept }: UploadCardProps) {
  return (
    <div className="bg-card border-2 border-dashed border-border rounded-lg p-8 text-center">
      <div className="mb-4">
        <Upload className="w-12 h-12 text-primary mx-auto" />
      </div>
      <h3 className="text-lg font-semibold text-foreground mb-2">{title}</h3>
      <p className="text-muted-foreground mb-6 max-w-md mx-auto">
        {description}
      </p>
      <Button onClick={onUpload} className="bg-primary hover:bg-primary/90 text-primary-foreground">
        {buttonText}
      </Button>
    </div>
  );
}