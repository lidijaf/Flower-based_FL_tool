class PreprocessingPipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit_transform(self, bundle, context):
        for step in self.steps:
            bundle = step.fit_transform(bundle, context)
        return bundle

    def transform(self, bundle, context):
        for step in self.steps:
            bundle = step.transform(bundle, context)
        return bundle
