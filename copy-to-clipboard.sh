#!/bin/bash
while IFS= read -r copy; do
    wl-copy "$copy"
done

