#!/bin/bash
# Testing bash and piping
while IFS= read -r test; do
    [[ -z "$test" ]] && continue
    echo "$test"
done
