from data_preprocessing.base import PreprocessingStep


class PartitionBalancedGroupsStep(PreprocessingStep):
    def __init__(self, group_col, num_clients):
        if num_clients < 1:
            raise ValueError("num_clients must be at least 1")

        self.group_col = group_col
        self.num_clients = num_clients

    def transform(self, bundle, context):
        df = bundle.df

        if self.group_col not in df.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in DataFrame.")

        # Count rows per group
        group_sizes = (
            df.groupby(self.group_col)
            .size()
            .sort_values(ascending=False)
        )

        unique_groups = group_sizes.index.tolist()

        if self.num_clients > len(unique_groups):
            raise ValueError(
                f"Requested {self.num_clients} clients, but only {len(unique_groups)} "
                f"unique groups found in column '{self.group_col}'."
            )

        # Greedy balancing
        client_groups = {f"client{i+1}": [] for i in range(self.num_clients)}
        client_loads = {f"client{i+1}": 0 for i in range(self.num_clients)}

        for group_value, group_count in group_sizes.items():
            target_client = min(client_loads, key=client_loads.get)
            client_groups[target_client].append(group_value)
            client_loads[target_client] += int(group_count)

        # Build per-client dataframes
        clients = {}
        for client_name, assigned_groups in client_groups.items():
            client_df = df[df[self.group_col].isin(assigned_groups)].copy().reset_index(drop=True)
            clients[client_name] = client_df

        bundle.clients = clients
        context.metadata["balanced_partition"] = {
            "group_col": self.group_col,
            "num_clients": self.num_clients,
            "client_loads": client_loads,
            "client_groups": client_groups,
        }

        return bundle
