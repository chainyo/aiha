//! Base traits, structs and enums for models.

/// Enumerate the different model libraries available on the Hugging Face Hub
#[derive(Clone, Debug, PartialEq)]
pub enum ModelLibraries {
    /// Adapter Transformers library
    AdapterTransformers,
    /// allenNLP library
    AllenNLP,
    /// Asteroid library
    Asteroid,
    /// Core ML library
    CoreML,
    /// Diffusers library
    Diffusers,
    /// ESPnet library
    ESPnet,
    /// Fairseq library
    Fairseq,
    /// fastai library
    Fastai,
    /// fastText library
    FastText,
    /// Flair library
    Flair,
    /// Flax library
    Flax,
    /// Graphcore library
    Graphcore,
    /// Habana library
    Habana,
    /// JAX library
    Jax,
    /// Joblib library
    Joblib,
    /// Keras library
    Keras,
    /// ML-Agents library
    MLAgents,
    /// NeMo library
    NeMo,
    /// OpenCLIP library
    OpenCLIP,
    /// OpenVINO library
    OpenVINO,
    /// ONNX library
    Onnx,
    /// PaddleNLP library
    PaddleNLP,
    /// PaddlePaddle library
    PaddlePaddle,
    /// pyannote.audio library
    PyannoteAudio,
    /// Pythae library
    Pythae,
    /// PyTorch library
    PyTorch,
    /// Rust library
    Rust,
    /// Safetensors library
    Safetensors,
    /// Sample Factory library
    SampleFactory,
    /// Scikit-learn library
    ScikitLearn,
    /// Sentence Transformers
    SentenceTransformers,
    /// spaCy library
    Spacy,
    /// SpanMarker library
    SpanMarker,
    /// speechbrain library
    Speechbrain,
    /// Stable-Baselines3 library
    StableBaselines3,
    /// Stanza library
    Stanza,
    /// TensorBoard library
    TensorBoard,
    /// TensorFlow library
    TensorFlow,
    /// TensorFlowTTS library
    TensorFlowTTS,
    /// TFLite library
    TFLite,
    /// Timm library
    Timm,
    /// Transformers library
    Transformers,
}

/// Generic trait for Hugging Face models
pub trait ModelConfig {
    /// Returns the model hidden size
    fn hidden_size(&self) -> &i32;
    /// Returns the model intermediate size
    fn intermediate_size(&self) -> &i32;
    /// Returns the model max position embeddings
    fn max_position_embeddings(&self) -> &i32;
    /// Returns the model number of attention heads
    fn num_attention_heads(&self) -> &i32;
    /// Returns the model number of hidden layers
    fn num_hidden_layers(&self) -> &i32;
    /// Returns the model repository name
    fn repo_name(&self) -> &str;
    /// Returns the model type
    fn model_type(&self) -> &str;
    /// Returns the model libraries
    fn available_libraries(&self) -> &[ModelLibraries];
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    struct MockModelConfig;

    impl ModelConfig for MockModelConfig {
        fn hidden_size(&self) -> &i32 {
            &1024
        }

        fn intermediate_size(&self) -> &i32 {
            &4096
        }

        fn max_position_embeddings(&self) -> &i32 {
            &512
        }

        fn num_attention_heads(&self) -> &i32 {
            &16
        }

        fn num_hidden_layers(&self) -> &i32 {
            &12
        }

        fn repo_name(&self) -> &str {
            "mock"
        }

        fn model_type(&self) -> &str {
            "mock"
        }

        fn available_libraries(&self) -> &[ModelLibraries] {
            &[ModelLibraries::PyTorch]
        }
    }

    #[test]
    fn test_hub_model_config() {
        let config = MockModelConfig;
        assert_eq!(*config.hidden_size(), 1024);
        assert_eq!(*config.intermediate_size(), 4096);
        assert_eq!(*config.max_position_embeddings(), 512);
        assert_eq!(*config.num_attention_heads(), 16);
        assert_eq!(*config.num_hidden_layers(), 12);
        assert_eq!(config.repo_name(), "mock");
        assert_eq!(config.model_type(), "mock");
        assert_eq!(config.available_libraries(), vec![ModelLibraries::PyTorch]);
    }

    #[test]
    fn test_model_libraries_equality() {
        let lib1 = ModelLibraries::PyTorch;
        let lib2 = ModelLibraries::PyTorch;
        let lib3 = ModelLibraries::TensorFlow;
        assert_eq!(lib1, lib2);
        assert_ne!(lib1, lib3);
    }

    #[test]
    fn test_model_libraries_display() {
        let lib1 = ModelLibraries::PyTorch;
        let lib2 = ModelLibraries::TensorFlow;
        assert_eq!(format!("{:?}", lib1), "PyTorch");
        assert_eq!(format!("{:?}", lib2), "TensorFlow");
    }

    #[test]
    fn test_model_libraries_exhaustiveness() {
        let libraries = vec![
            ModelLibraries::AdapterTransformers,
            ModelLibraries::AllenNLP,
            ModelLibraries::Asteroid,
            ModelLibraries::CoreML,
            ModelLibraries::Diffusers,
            ModelLibraries::ESPnet,
            ModelLibraries::Fairseq,
            ModelLibraries::Fastai,
            ModelLibraries::FastText,
            ModelLibraries::Flair,
            ModelLibraries::Flax,
            ModelLibraries::Graphcore,
            ModelLibraries::Habana,
            ModelLibraries::Jax,
            ModelLibraries::Joblib,
            ModelLibraries::Keras,
            ModelLibraries::MLAgents,
            ModelLibraries::NeMo,
            ModelLibraries::OpenCLIP,
            ModelLibraries::OpenVINO,
            ModelLibraries::Onnx,
            ModelLibraries::PaddleNLP,
            ModelLibraries::PaddlePaddle,
            ModelLibraries::PyannoteAudio,
            ModelLibraries::Pythae,
            ModelLibraries::PyTorch,
            ModelLibraries::Rust,
            ModelLibraries::Safetensors,
            ModelLibraries::SampleFactory,
            ModelLibraries::ScikitLearn,
            ModelLibraries::SentenceTransformers,
            ModelLibraries::Spacy,
            ModelLibraries::SpanMarker,
            ModelLibraries::Speechbrain,
            ModelLibraries::StableBaselines3,
            ModelLibraries::Stanza,
            ModelLibraries::TensorBoard,
            ModelLibraries::TensorFlow,
            ModelLibraries::TensorFlowTTS,
            ModelLibraries::TFLite,
            ModelLibraries::Timm,
            ModelLibraries::Transformers,
        ];
        assert_eq!(libraries.len(), 42);
    }
}
