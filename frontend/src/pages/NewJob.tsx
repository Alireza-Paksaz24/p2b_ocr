import { useForm, Controller } from 'react-hook-form';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { api, queryKeys } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { FileUpload } from '@/components/ui/FileUpload';
import { Select } from '@/components/ui/Select';
import { toast } from 'sonner';
import { useState } from 'react';

interface FormValues {
  model_id: string;
  output_formats: string[];
  files: File[];
}

const FORMAT_OPTIONS = [
  { label: 'Markdown', value: 'md' },
  { label: 'HTML', value: 'html' },
  { label: 'DOCX', value: 'docx' },
];

export default function NewJob() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const { data: modelsData } = useQuery({
    queryKey: queryKeys.models,
    queryFn: api.getModels,
  });

  const downloadedModels = (modelsData?.models ?? []).filter((m) => m.downloaded);

  const {
    register,
    handleSubmit,
    setValue,
    watch,
    control,
    formState: { errors },
  } = useForm<FormValues>({
    defaultValues: {
      model_id: '',
      output_formats: [],
      files: [],
    },
  });

  const outputFormats = watch('output_formats');

  const submitMutation = useMutation({
    mutationFn: (formData: FormData) => api.submitJob(formData),
    onSuccess: (data) => {
      toast.success('Job submitted successfully!');
      queryClient.invalidateQueries({ queryKey: queryKeys.jobs });
      navigate(`/jobs/${data.job_id}`);
    },
    onError: (error: Error) => {
      toast.error(`Submission failed: ${error.message}`);
    },
  });

  const onSubmit = (data: FormValues) => {
    const formData = new FormData();
    formData.append('model_id', data.model_id);
    formData.append('output_formats', data.output_formats.join(','));
    selectedFiles.forEach((file) => formData.append('files', file));
    submitMutation.mutate(formData);
  };

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">New OCR Job</h1>
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6 max-w-2xl">
        {/* Model Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
          <Controller
            name="model_id"
            control={control}
            rules={{ required: 'Please select a model' }}
            render={({ field }) => (
              <Select
                value={field.value}
                onValueChange={(value) => field.onChange(value)}
                options={downloadedModels.map((m) => ({
                  label: `${m.name} (${m.type})`,
                  value: m.id,
                }))}
                placeholder="Choose a downloaded model"
              />
            )}
          />
          {errors.model_id && <p className="text-red-500 text-sm mt-1">{errors.model_id.message}</p>}
        </div>

        {/* Output Formats */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Output Formats</label>
          <div className="flex gap-4">
            {FORMAT_OPTIONS.map((format) => (
              <label key={format.value} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  value={format.value}
                  {...register('output_formats', {
                    validate: (value) => value.length > 0 || 'Select at least one format',
                  })}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm">{format.label}</span>
              </label>
            ))}
          </div>
          {errors.output_formats && (
            <p className="text-red-500 text-sm mt-1">{errors.output_formats.message}</p>
          )}
        </div>

        {/* File Upload */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Files</label>
          <FileUpload
            accept="image/*,.pdf,.zip"
            onFilesSelected={(files) => {
              setSelectedFiles(files);
              setValue('files', files);
            }}
          />
          {selectedFiles.length > 0 && (
            <ul className="mt-2 space-y-1">
              {selectedFiles.map((file, idx) => (
                <li key={idx} className="text-sm text-gray-600">
                  {file.name} ({(file.size / 1024).toFixed(1)} KB)
                </li>
              ))}
            </ul>
          )}
          {errors.files && <p className="text-red-500 text-sm mt-1">{errors.files.message}</p>}
        </div>

        <Button type="submit" disabled={submitMutation.isPending}>
          {submitMutation.isPending ? 'Submitting...' : 'Submit Job'}
        </Button>
      </form>
    </div>
  );
}