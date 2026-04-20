from data_preprocessing.base import PreprocessingStep


class PartitionByGroupStep(PreprocessingStep):
    def __init__(self, group_col):
        self.group_col = group_col

    def transform(self, bundle, context):
        df = bundle.df

        if self.group_col not in df.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in DataFrame.")

        grouped = list(df.groupby(self.group_col))

        clients = {}
        client_mapping = {}

        for idx, (group_value, group_df) in enumerate(grouped, start=1):
            client_name = f"client{idx}"
            clients[client_name] = group_df.reset_index(drop=True)
            client_mapping[client_name] = str(group_value)

        bundle.clients = clients
        context.metadata["group_partition"] = {
            "group_col": self.group_col,
            "client_mapping": client_mapping,
        }

        return bundle
