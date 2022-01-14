from composer import trainer, algorithms
trainer_hparams = trainer.load("resnet50")
trainer_hparams.algorithms = algorithms.load_multiple(
  "blurpool",
  "scale_schedule")
trainer = trainer_hparams.initialize_object()
trainer.fit()
