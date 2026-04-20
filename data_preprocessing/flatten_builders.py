def make_flatten_fn(schema):
    meta_field = schema["input"]["meta_field"]
    payload_field = schema["input"]["payload_field"]
    field_map = schema["flatten"]["field_map"]

    def flatten_event(e):
        meta = e.get(meta_field)
        payload = e.get(payload_field)

        if not isinstance(meta, dict):
            meta = {}

        if not isinstance(payload, dict):
            payload = {}

        result = {}

        for out_name, spec in field_map.items():
            source = spec.get("source", "root")

            if source == "meta":
                container = meta
            elif source == "payload":
                container = payload
            else:
                container = e

            special = spec.get("special")

            if special == "keys_join":
                if isinstance(container, dict):
                    value = ",".join(container.keys())
                else:
                    value = ""
            elif special == "stringify_container":
                value = str(container)
            else:
                key = spec.get("key")
                value = container.get(key) if isinstance(container, dict) else None

            if spec.get("join_list", False):
                if isinstance(value, list):
                    value = ",".join(str(v) for v in value)
                elif value is None:
                    value = ""
                else:
                    value = str(value)

            result[out_name] = value

        return result

    return flatten_event
