# See full reference at https://devenv.sh/reference/options/
{pkgs, ...}:
# let
#   openai-whisper-cpp = pkgs.openai-whisper-cpp.overrideAttrs (oldAttrs: {
#     installPhase =
#       oldAttrs.installPhase
#       + ''
#         cp models/convert-whisper-to-coreml.py $out/bin/convert-whisper-to-coreml.py
#       '';
#   });
# in
{
  packages = with pkgs; [
    openai-whisper-cpp
    ffmpeg
  ];

  scripts.generate-coreml-model.exec =
    /*
    sh
    */
    ''
      mname="$1"
      python3 models/convert-whisper-to-coreml.py --model $mname --encoder-only True  --optimize-ane True
      xcrun coremlc compile models/coreml-encoder-''${mname}.mlpackage models/
      rm -rf models/ggml-''${mname}-encoder.mlmodelc
      mv -v models/coreml-encoder-''${mname}.mlmodelc models/ggml-''${mname}-encoder.mlmodelc
    '';

  dotenv.enable = true;

  languages.python = {
    enable = true;
    venv = {
      enable = true;
      requirements = ''
        yt-dlp

        llama-index
        llama-index-llms-gemini
        llama-index-embeddings-gemini
        llama-index-vector-stores-pinecone
        pinecone-client

        ane_transformers
        openai-whisper
        coremltools

        basedpyright
      '';
    };
  };
}
