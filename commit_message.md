adas-lite: FCW + BSD stack on PipeGen FP16 perception

- adas_fusion (C++17): FCW TTC ladder + BSD zone persistence from v3.3
  reference parameters, monocular range fusion (MiDaS relative x bbox pinhole
  anchor), warning banners and FPS overlays
- pipelines/perception.py: TensorRT FP16 runner (polygraphy) producing
  PipeGen-schema results.json + annotated video
- FP16 engines: yolov8m (55MB), midas_v21_small (36MB) on RTX 5060 (sm_120)
- demos: front FCW (dashcam), rear BSD, CARLA validation clip
- docs/screenshots: full PipeGen session capture for submission evidence
