#!/bin/bash

image="apptainer/maestro.sif"
apptainer exec --writable-tmpfs --cleanenv --env OPENAI_API_KEY=$OPENAI_API_KEY $image bash --rcfile /etc/bash.bashrc