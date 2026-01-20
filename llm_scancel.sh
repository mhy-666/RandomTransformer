#!/bin/bash
for job in {42213240..42213250}; do
    scancel $job
done