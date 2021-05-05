

def shard(n_shard, model):
    shards = [set() for _ in range(n_shard)]
    key2shard = {}
    for key, param in enumerate(model.parameters()):
        if param.requires_grad:
            server = key % n_shard
            shards[server].add(key)
            key2shard[key] = server
    return shards, key2shard
