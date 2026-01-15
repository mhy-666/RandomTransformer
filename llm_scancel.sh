#!/bin/bash
for job in {41892946..41892988}; do
    scancel $job
done