import docker
from config import POSTGRES_CONTAINER_NAME
import json

stats_list = []

client = docker.from_env()
container = client.containers.get(POSTGRES_CONTAINER_NAME)
stats = container.stats(stream=False)

memory_stats = stats['memory_stats']
print(memory_stats)

with open("container_stats.json", 'w') as file:
    json.dump(memory_stats, file)

"""
# Container Memory Statistics Explanation

## Top-level metrics

- `limit`: コンテナに割り当てられたメモリの上限 (8,219,820,032 bytes ≈ 7.7 GB)
- `usage`: コンテナが現在使用しているメモリの総量 (59,748,352 bytes ≈ 57 MB)

## Detailed memory statistics

### Anonymous Memory (プロセスが直接利用するメモリ)

- `active_anon`: アクティブな匿名メモリ。最近アクセスされたプロセスのメモリページ (19,501,056 bytes ≈ 18.6 MB)
- `inactive_anon`: 非アクティブな匿名メモリ。最近アクセスされていないプロセスのメモリページ (0 bytes)
- `anon`: 総匿名メモリ使用量 (4,485,120 bytes ≈ 4.3 MB)
- `anon_thp`: Transparent Huge Pages (THP) として割り当てられた匿名メモリ (0 bytes)

### File-backed Memory (ファイルシステムに関連するメモリ)

- `active_file`: アクティブなファイルキャッシュメモリ。最近アクセスされたファイルデータ (30,711,808 bytes ≈ 29.3 MB)
- `inactive_file`: 非アクティブなファイルキャッシュメモリ。最近アクセスされていないファイルデータ (6,774,784 bytes ≈ 6.5 MB)
- `file`: 総ファイルキャッシュメモリ使用量 (52,502,528 bytes ≈ 50.1 MB)
- `file_dirty`: ディスクに書き込まれていない修正されたファイルページ (0 bytes)
- `file_mapped`: メモリにマップされたファイル (26,562,560 bytes ≈ 25.3 MB)
- `file_writeback`: ディスクに書き込み中のファイルページ (0 bytes)

### Shared Memory

- `shmem`: 共有メモリ使用量（tmpfsを含む） (15,015,936 bytes ≈ 14.3 MB)

### Kernel Memory

- `kernel_stack`: カーネルスタックメモリ使用量 (98,304 bytes ≈ 96 KB)
- `slab`: カーネルのSlab割り当て合計 (1,658,848 bytes ≈ 1.6 MB)
- `slab_reclaimable`: 再利用可能なSlabメモリ (883,672 bytes ≈ 863 KB)
- `slab_unreclaimable`: 再利用不可能なSlabメモリ (775,176 bytes ≈ 757 KB)

### Page Faults and Swapping

- `pgfault`: マイナーページフォールトの数 (345,378)
- `pgmajfault`: メジャーページフォールトの数 (219)
- `pgactivate`: アクティブリストに移動されたページ数 (0)
- `pgdeactivate`: 非アクティブリストに移動されたページ数 (0)
- `pglazyfree`: 遅延解放されたページ数 (0)
- `pglazyfreed`: 遅延解放処理によって解放されたページ数 (0)
- `pgrefill`: 再充填されたページ数 (0)
- `pgscan`: スキャンされたページ数 (0)
- `pgsteal`: 回収されたページ数 (0)

### Transparent Huge Pages (THP)

- `thp_fault_alloc`: THPフォールトによって割り当てられたページ数 (2)
- `thp_collapse_alloc`: 既存のページのTHPへの昇格によって割り当てられたページ数 (0)

### Working Set

- `workingset_activate`: ワーキングセットからアクティブにされたページ数 (0)
- `workingset_nodereclaim`: 再利用不可能と判断されたページ数 (0)
- `workingset_refault`: ワーキングセットに再フォールトしたページ数 (0)

### Other

- `sock`: ネットワークソケットバッファに使用されているメモリ (0 bytes)
- `unevictable`: 退避不可能なメモリページ数 (0)
"""
