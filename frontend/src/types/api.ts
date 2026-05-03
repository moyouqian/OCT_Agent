export type UploadedFile = {
  file_id: string;
  original_name: string;
  path: string;
  variables: Array<{
    name: string;
    shape: number[];
    dtype: string;
  }>;
};

export type ResultRef = {
  result_id: string;
  file_id?: string;
  file_path?: string;
  result_key: string;
  result_path?: string;
  kind: "array" | "bnn";
  outputs?: Record<string, { shape: number[] }>;
};

export type HeatmapPayload = {
  result_id: string;
  name: string;
  shape: number[];
  min: number;
  max: number;
  data: number[][];
};

export type PhysicalParams = {
  wavelength: number;
  bandwidth: number;
  refractiveIndex: number;
};

export type MethodSettings = {
  visualizationEnabled: boolean;
  showThinking: boolean;
  vector: boolean;
  cnn: boolean;
  bnn: boolean;
  Nx: number;
  Nz: number;
  g: number;
  MC_test: number;
  physical: PhysicalParams;
};

export type DisplayMessage = {
  id: string;
  role: "human" | "ai";
  content: string;
};
