from data_preprocessing.base import PreprocessingStep


class SortByColumnsStep(PreprocessingStep):
    def __init__(self, by, ascending=True, reset_index=True):
        self.by = by
        self.ascending = ascending
        self.reset_index = reset_index

    def transform(self, bundle, context):
        bundle.df = bundle.df.sort_values(by=self.by, ascending=self.ascending)
        if self.reset_index:
            bundle.df = bundle.df.reset_index(drop=True)
        return bundle
