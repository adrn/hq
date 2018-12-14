#!/bin/bash

rsync -zvr --max-size=200m --exclude "*~" \
perseus:/tigress/adrianp/projects/twoface/scripts/exp1/plots ~/projects/twoface/scripts/exp1

rsync -zvr --max-size=200m --exclude "*~" \
perseus:/tigress/adrianp/projects/twoface/scripts/exp1/cache ~/projects/twoface/scripts/exp1
