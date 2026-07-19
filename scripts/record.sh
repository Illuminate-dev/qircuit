#!/usr/bin/env bash
set -euo pipefail

COMMIT="${1:-$(git rev-parse --short HEAD)}"
DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
CSV="perf.csv"

echo "=== Recording benchmark for commit $COMMIT ==="
cargo bench --bench gates_bench 2>&1 | tee /tmp/qircuit_bench.out

to_ns() {
    local val="$1" unit="$2"
    case "$unit" in
        ns)  echo "$val" ;;
        µs|us) awk "BEGIN {printf \"%.0f\", $val * 1000}" ;;
        ms)  awk "BEGIN {printf \"%.0f\", $val * 1000000}" ;;
        s)   awk "BEGIN {printf \"%.0f\", $val * 1000000000}" ;;
        *)   echo "$val" ;;
    esac
}

if [ ! -f "$CSV" ]; then
    printf "commit,date,benchmark,qubits,time_ns\n" > "$CSV"
fi

grep 'time:' /tmp/qircuit_bench.out | while IFS= read -r line; do
    bench=$(echo "$line" | awk '{print $1}' | sed -E 's|^([^/]+)/[0-9]+q.*|\1|')
    qubits=$(echo "$line" | awk '{print $1}' | sed -E 's|^[^/]+/([0-9]+)q.*|\1|')
    median=$(echo "$line" | awk '{print $5}')
    unit=$(echo "$line" | awk '{print $6}')
    ns=$(to_ns "$median" "$unit")
    printf "%s,%s,%s,%s,%s\n" "$COMMIT" "$DATE" "$bench" "$qubits" "$ns" >> "$CSV"
done

echo "Appended $COMMIT to $CSV"
