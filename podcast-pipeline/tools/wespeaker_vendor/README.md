# WeSpeaker model definitions (vendored)

Copied from https://github.com/wenet-e2e/wespeaker (branch master, fetched 2026-10-31):
wespeaker/models/samresnet.py, pooling_layers.py, multi_view_attention.py. Licence: Apache-2.0,
the header of each file is kept. The only change: the `import wespeaker.models...` lines became
relative imports, so the `wespeaker` package is not needed.

Used by tools/embedders.py for the SimAM-ResNet100 checkpoint (model.safetensors + config.json).
